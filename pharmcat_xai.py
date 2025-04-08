import json
import re

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def vcf_to_dataframe(vcf_file, target_genes):
    """Convert VCF file to a pandas DataFrame focusing on target genes."""
    variants = []

    with open(vcf_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue

            parts = line.strip().split('\t')
            if len(parts) < 8:
                continue

            chrom = parts[0]
            pos = parts[1]
            rsid = parts[2]
            ref = parts[3]
            alt = parts[4]
            info = parts[7]

            # Check if this variant is for one of our target genes
            gene_match = re.search(r'PX=([^;]+)', info)
            if not gene_match:
                continue

            gene = gene_match.group(1)
            if gene not in target_genes:
                continue

            # Extract genotype information
            genotype = "0/0"  # Default
            if len(parts) > 9:
                format_fields = parts[8].split(':')
                sample_fields = parts[9].split(':')

                if 'GT' in format_fields:
                    gt_index = format_fields.index('GT')
                    if gt_index < len(sample_fields):
                        genotype = sample_fields[gt_index]

            # Convert genotype to numeric representation
            gt_numeric = np.nan
            if genotype == '0/0':
                gt_numeric = 0  # Reference homozygous
            elif genotype in ['0/1', '1/0', '0/2', '2/0']:
                gt_numeric = 1  # Heterozygous
            elif genotype in ['1/1', '2/2', '1/2', '2/1']:
                gt_numeric = 2  # Alternate homozygous or compound heterozygous

            variants.append({
                'gene': gene,
                'rsid': rsid,
                'position': pos,
                'ref': ref,
                'alt': alt,
                'genotype': genotype,
                'genotype_value': gt_numeric
            })

    df = pd.DataFrame(variants)
    df.to_csv('vcf_converted.csv', index=False)
    return df


def generate_synthetic_samples(vcf_df, n_samples=100):
    """Generate synthetic samples by perturbing the original genotypes."""
    # Create feature matrix where columns are gene_rsid combinations
    features = {}
    for _, row in vcf_df.iterrows():
        key = f"{row['gene']}_{row['rsid']}"
        features[key] = row['genotype_value']

    # Create synthetic samples
    all_samples = []
    all_samples.append(features.copy())  # Include original sample

    for _ in range(n_samples):
        sample = features.copy()

        # Randomly perturb some genotypes
        for key in sample:
            if np.random.random() < 0.3:  # 30% chance of mutation
                current = sample[key]
                if not np.isnan(current):
                    # Choose a different genotype
                    options = [g for g in [0, 1, 2] if g != current]
                    sample[key] = np.random.choice(options)

        all_samples.append(sample)

    return pd.DataFrame(all_samples)


def extract_pharmcat_data(match_file, phenotype_file, target_genes):
    """Extract relevant data from pharmcat output files."""
    match_df = pd.read_csv(match_file)
    phenotype_df = pd.read_csv(phenotype_file)

    # Extract phenotype information
    gene_data = {}

    for gene in target_genes:
        gene_phenotypes = phenotype_df[phenotype_df['gene'] == gene]
        gene_matches = match_df[match_df['gene'] == gene]

        if gene_phenotypes.empty:
            continue

        # Extract phenotype
        phenotype = gene_phenotypes['phenotype'].iloc[0] if 'phenotype' in gene_phenotypes.columns and not pd.isna(
            gene_phenotypes['phenotype'].iloc[0]) else "Unknown"

        # Extract variants
        variants = []
        if not gene_matches.empty and 'variant_rsids' in gene_matches.columns:
            for _, row in gene_matches.iterrows():
                if pd.notna(row['variant_rsids']) and row['variant_rsids']:
                    variants.extend(row['variant_rsids'].split(';'))

        # Extract diplotype info
        diplotype = "Unknown"
        allele1 = "Unknown"
        allele2 = "Unknown"

        if not gene_phenotypes.empty:
            diplotype = gene_phenotypes['diplotype_label'].iloc[
                0] if 'diplotype_label' in gene_phenotypes.columns and not pd.isna(
                gene_phenotypes['diplotype_label'].iloc[0]) else "Unknown"
            allele1 = gene_phenotypes['allele1_name'].iloc[
                0] if 'allele1_name' in gene_phenotypes.columns and not pd.isna(
                gene_phenotypes['allele1_name'].iloc[0]) else "Unknown"
            allele2 = gene_phenotypes['allele2_name'].iloc[
                0] if 'allele2_name' in gene_phenotypes.columns and not pd.isna(
                gene_phenotypes['allele2_name'].iloc[0]) else "Unknown"

        gene_data[gene] = {
            'phenotype': phenotype,
            'diplotype': diplotype,
            'allele1': allele1,
            'allele2': allele2,
            'variants': list(set(variants))
        }

    return gene_data


def get_ground_truth(ground_truth_file, target_genes):
    """Extract ground truth phenotypes from the ground truth file."""
    ground_truth = {}

    try:
        gt_df = pd.read_csv(ground_truth_file)

        # Assuming the first row contains our sample data
        sample_row = gt_df.iloc[0]

        for gene in target_genes:
            if gene in gt_df.columns:
                ground_truth[gene] = sample_row[gene]
            else:
                ground_truth[gene] = "Unknown"
    except Exception as e:
        print(f"Error reading ground truth file: {e}")
        # Set defaults if file read fails
        for gene in target_genes:
            ground_truth[gene] = "Unknown"

    return ground_truth


def run_shap_analysis(synthetic_samples, pharmcat_data, ground_truth, vcf_df):
    """Perform SHAP analysis on the synthetic data."""
    results = {}

    for gene, gene_info in pharmcat_data.items():
        print(f"Analyzing gene: {gene}")

        # Get variants for this gene
        gene_variants = vcf_df[vcf_df['gene'] == gene]

        # Get columns for this gene
        gene_columns = [col for col in synthetic_samples.columns if col.startswith(f"{gene}_")]

        if not gene_columns:
            print(f"  No features found for {gene}, skipping...")
            continue

        # Prepare X data
        X = synthetic_samples[gene_columns].copy()

        # For each sample, simulate whether it would get the same phenotype as the original
        # This is a simplified simulation based on similarity to the original
        y = []
        original_sample = synthetic_samples.iloc[0]

        for _, sample in synthetic_samples.iterrows():
            # Count how many variants match the original sample
            matching_variants = sum(
                1 for col in gene_columns
                if sample[col] == original_sample[col] and not np.isnan(sample[col])
            )
            total_variants = sum(
                1 for col in gene_columns
                if not np.isnan(original_sample[col])
            )

            similarity = matching_variants / total_variants if total_variants > 0 else 0

            # If very similar to original, assign same phenotype outcome
            if similarity > 0.7:
                y.append(1 if gene_info['phenotype'] == ground_truth[gene] else 0)
            else:
                # Otherwise flip the outcome
                y.append(0 if gene_info['phenotype'] == ground_truth[gene] else 1)

        y = np.array(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a surrogate model
        model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        # Get SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Calculate feature importance
        if isinstance(shap_values, list):
            # For multi-class output
            importance = np.abs(shap_values[1]).mean(axis=0)
        else:
            # For single output
            importance = np.abs(shap_values).mean(axis=0)

        # Map importance to variants
        feature_importance = {}
        for i, col in enumerate(gene_columns):
            rsid = col.split('_', 1)[1]
            feature_importance[rsid] = float(importance[i])

        # Sort features by importance
        sorted_importance = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))

        # Get variant details
        variant_details = {}
        for rsid in feature_importance.keys():
            variant_row = gene_variants[gene_variants['rsid'] == rsid]
            if not variant_row.empty:
                variant_details[rsid] = {
                    'position': variant_row['position'].iloc[0],
                    'ref': variant_row['ref'].iloc[0],
                    'alt': variant_row['alt'].iloc[0],
                    'genotype': variant_row['genotype'].iloc[0]
                }

        # Store results
        results[gene] = {
            'feature_importance': sorted_importance,
            'variant_details': variant_details,
            'pharmcat_phenotype': gene_info['phenotype'],
            'ground_truth_phenotype': ground_truth[gene],
            'match_accuracy': 1 if gene_info['phenotype'] == ground_truth[gene] else 0,
            'diplotype': gene_info['diplotype'],
            'allele1': gene_info['allele1'],
            'allele2': gene_info['allele2'],
            'pharmcat_variants': gene_info['variants']
        }

    return results


def analyze_decision_pathways(results, vcf_df, match_df, phenotype_df):
    """Analyze the decision pathways from variants to phenotype."""

    for gene, gene_info in results.items():
        # Get the top variants by importance
        top_variants = list(gene_info['feature_importance'].keys())[:3]

        # Get the corresponding rows from match and phenotype dataframes
        gene_matches = match_df[match_df['gene'] == gene]
        gene_phenotypes = phenotype_df[phenotype_df['gene'] == gene]

        # Extract diplotype and haplotype information
        diplotypes = []
        if not gene_matches.empty and 'diplotype_name' in gene_matches.columns:
            diplotypes = gene_matches['diplotype_name'].dropna().unique().tolist()

        haplotypes = []
        if not gene_matches.empty:
            if 'haplotype1_name' in gene_matches.columns:
                haplotypes.extend(gene_matches['haplotype1_name'].dropna().unique().tolist())
            if 'haplotype2_name' in gene_matches.columns:
                haplotypes.extend(gene_matches['haplotype2_name'].dropna().unique().tolist())

        # Extract variant to allele relationships
        variant_allele_map = {}
        for variant in top_variants:
            # Find rows in match_df that reference this variant
            variant_rows = gene_matches[
                gene_matches['variant_rsids'].notna() &
                gene_matches['variant_rsids'].str.contains(variant)
                ]

            if not variant_rows.empty:
                alleles = []
                if 'haplotype1_name' in variant_rows.columns:
                    alleles.extend(variant_rows['haplotype1_name'].dropna().unique().tolist())
                if 'haplotype2_name' in variant_rows.columns:
                    alleles.extend(variant_rows['haplotype2_name'].dropna().unique().tolist())

                variant_allele_map[variant] = list(set(alleles))
            else:
                variant_allele_map[variant] = []

        # Add decision pathway information to results
        results[gene]['decision_pathway'] = {
            'top_variants': top_variants,
            'diplotypes': diplotypes,
            'haplotypes': haplotypes,
            'variant_to_allele_map': variant_allele_map,
            'phenotype_function': gene_phenotypes['phenotype'].iloc[
                0] if not gene_phenotypes.empty and 'phenotype' in gene_phenotypes.columns and not pd.isna(
                gene_phenotypes['phenotype'].iloc[0]) else "Unknown"
        }

    return results


def explain_decisions(results):
    """Generate human-readable explanations for PharmCAT's decisions."""

    for gene, gene_info in results.items():
        explanation = []

        # Start with gene and phenotype
        explanation.append(f"Analysis for gene {gene}:")
        explanation.append(f"PharmCAT called phenotype: {gene_info['pharmcat_phenotype']}")
        explanation.append(f"Ground truth phenotype: {gene_info['ground_truth_phenotype']}")
        explanation.append(f"Match accuracy: {'Correct' if gene_info['match_accuracy'] == 1 else 'Incorrect'}")

        # Add diplotype information
        explanation.append(f"Diplotype: {gene_info['diplotype']}")
        explanation.append(f"Allele 1: {gene_info['allele1']}")
        explanation.append(f"Allele 2: {gene_info['allele2']}")

        # Add information about most important variants
        explanation.append("\nMost important variants in decision (ranked by SHAP importance):")
        for i, (variant, importance) in enumerate(list(gene_info['feature_importance'].items())[:5], 1):
            variant_details = gene_info['variant_details'].get(variant, {})
            genotype = variant_details.get('genotype', 'Unknown')

            explanation.append(f"{i}. {variant} (Importance: {importance:.4f}, Genotype: {genotype})")

            # Add details about this variant
            if variant in gene_info['variant_details']:
                details = gene_info['variant_details'][variant]
                explanation.append(f"   Position: {details.get('position', 'Unknown')}")
                explanation.append(f"   Reference allele: {details.get('ref', 'Unknown')}")
                explanation.append(f"   Alternate allele: {details.get('alt', 'Unknown')}")

        # Add pathway information if available
        if 'decision_pathway' in gene_info:
            pathway = gene_info['decision_pathway']

            explanation.append("\nDecision pathway:")
            for variant in pathway['top_variants']:
                alleles = pathway['variant_to_allele_map'].get(variant, [])
                alleles_str = ", ".join(alleles) if alleles else "No specific alleles"
                explanation.append(f"Variant {variant} → Contributes to alleles: {alleles_str}")

            explanation.append(f"Alleles → Diplotype: {gene_info['diplotype']}")
            explanation.append(f"Diplotype → Phenotype: {gene_info['pharmcat_phenotype']}")

        # Add explanation to the results
        results[gene]['explanation'] = "\n".join(explanation)

    return results


def main():
    # Target genes
    target_genes = ["CYP2B6", "CYP2C9", "CYP2C19", "CYP3A5", "SLCO1B1", "TPMT", "UGT1A1"]

    print("Step 1: Converting VCF to DataFrame...")
    vcf_df = vcf_to_dataframe('HG00276_freebayes.preprocessed.vcf', target_genes)

    print("Step 2: Extracting pharmcat data...")
    match_df = pd.read_csv('HG00276_freebayes.preprocessed.match.csv')
    phenotype_df = pd.read_csv('HG00276_freebayes.preprocessed.phenotype.csv')
    pharmcat_data = extract_pharmcat_data(
        'HG00276_freebayes.preprocessed.match.csv',
        'HG00276_freebayes.preprocessed.phenotype.csv',
        target_genes
    )

    print("Step 3: Getting ground truth phenotypes...")
    ground_truth = get_ground_truth('groundtruth_phenotype_filtered.csv', target_genes)

    print("Step 4: Generating synthetic samples...")
    synthetic_samples = generate_synthetic_samples(vcf_df, n_samples=100)

    print("Step 5: Running SHAP analysis...")
    results = run_shap_analysis(synthetic_samples, pharmcat_data, ground_truth, vcf_df)

    print("Step 6: Analyzing decision pathways...")
    results = analyze_decision_pathways(results, vcf_df, match_df, phenotype_df)

    print("Step 7: Generating explanations...")
    results = explain_decisions(results)

    print("Step 8: Saving results...")
    with open('pharmcat_explanation.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("Analysis complete! Results saved to pharmcat_explanation.json")


if __name__ == "__main__":
    main()
