import glob
import json
import os

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class PharmcatExplainer:
    def __init__(self, base_dir='pharmcat_processed', output_dir='xai_results'):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.focus_genes = ["CYP2B6", "CYP2C9", "CYP2C19", "CYP3A5", "SLCO1B1", "TPMT", "UGT1A1"]
        self.samples = self.get_samples()
        self.ground_truth = self.load_ground_truth('Groundtruth/groundtruth_phenotype_filtered.csv')

        os.makedirs(output_dir, exist_ok=True)

    def get_samples(self):
        samples = []
        if os.path.exists(self.base_dir):
            for sample_dir in os.listdir(self.base_dir):
                sample_path = os.path.join(self.base_dir, sample_dir)
                if os.path.isdir(sample_path):
                    samples.append(sample_dir)
        return samples

    def load_ground_truth(self, ground_truth_file):
        try:
            gt_df = pd.read_csv(ground_truth_file)
            ground_truth = {}
            for _, row in gt_df.iterrows():
                sample = row['Sample']
                ground_truth[sample] = {}
                for gene in self.focus_genes:
                    if gene in row and not pd.isna(row[gene]):
                        ground_truth[sample][gene] = row[gene]
            return ground_truth
        except Exception as e:
            print(f"Error loading ground truth: {e}")
            return {}

    def load_vcf_csv(self, sample):
        vcf_csv_pattern = os.path.join(self.base_dir, sample, f"{sample}_*.preprocessed.csv")
        vcf_csv_files = glob.glob(vcf_csv_pattern)

        if not vcf_csv_files:
            vcf_csv_pattern = os.path.join(self.base_dir, sample, f"*.csv")
            vcf_csv_files = glob.glob(vcf_csv_pattern)
            vcf_csv_files = [f for f in vcf_csv_files if not (f.endswith('.match.csv') or f.endswith('.phenotype.csv'))]

        if not vcf_csv_files:
            print(f"No CSV files found for sample {sample}")
            return None

        vcf_csv = vcf_csv_files[0]
        try:
            df = pd.read_csv(vcf_csv)
            return df
        except Exception as e:
            print(f"Error reading VCF CSV for {sample}: {e}")
            return None

    def load_match_csv(self, sample):
        match_csv = os.path.join(self.base_dir, sample, f"{sample}_*.match.csv")
        match_files = glob.glob(match_csv)

        if not match_files:
            match_csv = os.path.join(self.base_dir, sample, f"*.match.csv")
            match_files = glob.glob(match_csv)

        if not match_files:
            print(f"No match CSV found for sample {sample}")
            return None

        try:
            df = pd.read_csv(match_files[0])
            return df
        except Exception as e:
            print(f"Error reading match CSV for {sample}: {e}")
            return None

    def load_phenotype_csv(self, sample):
        phenotype_csv = os.path.join(self.base_dir, sample, f"{sample}_*.phenotype.csv")
        phenotype_files = glob.glob(phenotype_csv)

        if not phenotype_files:
            phenotype_csv = os.path.join(self.base_dir, sample, f"*.phenotype.csv")
            phenotype_files = glob.glob(phenotype_csv)

        if not phenotype_files:
            print(f"No phenotype CSV found for sample {sample}")
            return None

        try:
            df = pd.read_csv(phenotype_files[0])
            return df
        except Exception as e:
            print(f"Error reading phenotype CSV for {sample}: {e}")
            return None

    def generate_synthetic_samples(self, vcf_df, n_samples=100):
        if vcf_df is None or vcf_df.empty:
            return pd.DataFrame()

        features = {}
        for _, row in vcf_df.iterrows():
            if 'Gene' in row and 'ID' in row and pd.notnull(row['Gene']) and row['Gene'] in self.focus_genes:
                key = f"{row['Gene']}_{row['ID']}"

                # Try to extract genotype value from common formats
                genotype = None
                for col in row.index:
                    if '_GT' in col or col == 'GT':
                        genotype = row[col]
                        break

                if genotype is None:
                    continue

                # Convert genotype to numeric
                if genotype == '0/0':
                    features[key] = 0  # Reference homozygous
                elif genotype in ['0/1', '1/0', '0/2', '2/0']:
                    features[key] = 1  # Heterozygous
                elif genotype in ['1/1', '2/2', '1/2', '2/1']:
                    features[key] = 2  # Alternate homozygous or compound heterozygous
                else:
                    features[key] = np.nan

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
                        options = [g for g in [0, 1, 2] if g != current]
                        sample[key] = np.random.choice(options)

            all_samples.append(sample)

        return pd.DataFrame(all_samples)

    def extract_pharmcat_data(self, sample, match_df, phenotype_df):
        if match_df is None or phenotype_df is None:
            return {}

        gene_data = {}

        for gene in self.focus_genes:
            gene_phenotypes = phenotype_df[phenotype_df['gene'] == gene]
            gene_matches = match_df[match_df['gene'] == gene]

            if gene_phenotypes.empty:
                continue

            phenotype = "Unknown"
            for col in ['phenotype', 'activity_score']:
                if col in gene_phenotypes.columns and not pd.isna(gene_phenotypes[col].iloc[0]):
                    phenotype = gene_phenotypes[col].iloc[0]
                    break

            variants = []
            if not gene_matches.empty and 'variant_rsids' in gene_matches.columns:
                for _, row in gene_matches.iterrows():
                    if pd.notna(row['variant_rsids']) and row['variant_rsids']:
                        variants.extend(row['variant_rsids'].split(';'))

            diplotype = "Unknown"
            allele1 = "Unknown"
            allele2 = "Unknown"

            if not gene_phenotypes.empty:
                for col in ['diplotype_label', 'diplotype_name']:
                    if col in gene_phenotypes.columns and not pd.isna(gene_phenotypes[col].iloc[0]):
                        diplotype = gene_phenotypes[col].iloc[0]
                        break

                for col in ['allele1_name', 'haplotype1_name']:
                    if col in gene_phenotypes.columns and not pd.isna(gene_phenotypes[col].iloc[0]):
                        allele1 = gene_phenotypes[col].iloc[0]
                        break

                for col in ['allele2_name', 'haplotype2_name']:
                    if col in gene_phenotypes.columns and not pd.isna(gene_phenotypes[col].iloc[0]):
                        allele2 = gene_phenotypes[col].iloc[0]
                        break

            gene_data[gene] = {
                'phenotype': str(phenotype),
                'diplotype': diplotype,
                'allele1': allele1,
                'allele2': allele2,
                'variants': list(set(variants))
            }

        return gene_data

    def get_ground_truth_for_sample(self, sample):
        sample_ground_truth = {}
        sample_id = sample.split('_')[0]

        # Try exact match
        if sample in self.ground_truth:
            sample_ground_truth = self.ground_truth[sample]
        # Try partial match
        elif sample_id in self.ground_truth:
            sample_ground_truth = self.ground_truth[sample_id]
        else:
            # Try to find a match with sample ID
            for gt_sample in self.ground_truth:
                if gt_sample.startswith(sample_id):
                    sample_ground_truth = self.ground_truth[gt_sample]
                    break

        # Set defaults for missing genes
        for gene in self.focus_genes:
            if gene not in sample_ground_truth:
                sample_ground_truth[gene] = "Unknown"

        return sample_ground_truth

    def run_shap_analysis(self, sample, vcf_df, pharmcat_data, ground_truth):
        if vcf_df is None or not pharmcat_data:
            return {}

        results = {}
        synthetic_samples = self.generate_synthetic_samples(vcf_df)

        if synthetic_samples.empty:
            print(f"No synthetic samples could be generated for {sample}")
            return {}

        for gene, gene_info in pharmcat_data.items():
            print(f"Analyzing gene: {gene}")

            gene_variants = None
            if 'Gene' in vcf_df.columns:
                gene_variants = vcf_df[vcf_df['Gene'] == gene]

            # Get columns for this gene
            gene_columns = [col for col in synthetic_samples.columns if col.startswith(f"{gene}_")]

            if not gene_columns:
                print(f"  No features found for {gene}, skipping...")
                continue

            # Prepare X data
            X = synthetic_samples[gene_columns].copy()
            X.fillna(0, inplace=True)  # Replace NaN with 0

            # For each sample, simulate phenotype match or mismatch
            y = []
            original_sample = synthetic_samples.iloc[0]

            for _, sample_row in synthetic_samples.iterrows():
                # Count how many variants match the original sample
                matching_variants = sum(
                    1 for col in gene_columns
                    if sample_row[col] == original_sample[col] and not np.isnan(sample_row[col])
                )
                total_variants = sum(
                    1 for col in gene_columns
                    if not np.isnan(original_sample[col])
                )

                similarity = matching_variants / total_variants if total_variants > 0 else 0

                # If very similar to original, assign same phenotype outcome
                if similarity > 0.7:
                    y.append(1 if gene_info['phenotype'] == ground_truth.get(gene, "Unknown") else 0)
                else:
                    # Otherwise flip the outcome
                    y.append(0 if gene_info['phenotype'] == ground_truth.get(gene, "Unknown") else 1)

            y = np.array(y)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train a surrogate model
            model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                print(f"  Error training model for {gene}: {e}")
                continue

            # Get SHAP values
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
            except Exception as e:
                print(f"  Error computing SHAP values for {gene}: {e}")
                continue

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

                # Fix the error by ensuring we handle both scalar and array cases
                if isinstance(importance[i], np.ndarray):
                    # If it's an array, take the mean
                    imp_value = float(np.mean(importance[i]))
                else:
                    # If it's a scalar, convert directly
                    imp_value = float(importance[i])

                feature_importance[rsid] = imp_value

            # Sort features by importance
            sorted_importance = dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))

            # Get variant details
            variant_details = {}
            for rsid in feature_importance.keys():
                if gene_variants is not None and not gene_variants.empty:
                    variant_row = gene_variants[gene_variants['ID'] == rsid]
                    if not variant_row.empty:
                        genotype = "Unknown"
                        for col in variant_row.columns:
                            if '_GT' in col or col == 'GT':
                                genotype = variant_row[col].iloc[0]
                                break

                        variant_details[rsid] = {
                            'position': str(variant_row['POS'].iloc[0]) if 'POS' in variant_row.columns else "Unknown",
                            'ref': str(variant_row['REF'].iloc[0]) if 'REF' in variant_row.columns else "Unknown",
                            'alt': str(variant_row['ALT'].iloc[0]) if 'ALT' in variant_row.columns else "Unknown",
                            'genotype': str(genotype)
                        }

            # Store results
            results[gene] = {
                'feature_importance': sorted_importance,
                'variant_details': variant_details,
                'pharmcat_phenotype': gene_info['phenotype'],
                'ground_truth_phenotype': ground_truth.get(gene, "Unknown"),
                'match_accuracy': 1 if gene_info['phenotype'] == ground_truth.get(gene, "Unknown") else 0,
                'diplotype': gene_info['diplotype'],
                'allele1': gene_info['allele1'],
                'allele2': gene_info['allele2'],
                'pharmcat_variants': gene_info['variants']
            }

        return results

    def analyze_decision_pathways(self, sample, results, vcf_df, match_df, phenotype_df):
        if not results or match_df is None or phenotype_df is None:
            return results

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
                for col in ['haplotype1_name', 'haplotype2_name', 'allele1_name', 'allele2_name']:
                    if col in gene_matches.columns:
                        haplotypes.extend(gene_matches[col].dropna().unique().tolist())

            # Extract variant to allele relationships
            variant_allele_map = {}
            for variant in top_variants:
                variant_rows = pd.DataFrame()
                if 'variant_rsids' in gene_matches.columns:
                    variant_rows = gene_matches[
                        gene_matches['variant_rsids'].notna() &
                        gene_matches['variant_rsids'].str.contains(variant)
                        ]

                if not variant_rows.empty:
                    alleles = []
                    for col in ['haplotype1_name', 'haplotype2_name', 'allele1_name', 'allele2_name']:
                        if col in variant_rows.columns:
                            alleles.extend(variant_rows[col].dropna().unique().tolist())

                    variant_allele_map[variant] = list(set(alleles))
                else:
                    variant_allele_map[variant] = []

            # Add decision pathway information to results
            results[gene]['decision_pathway'] = {
                'top_variants': top_variants,
                'diplotypes': diplotypes,
                'haplotypes': haplotypes,
                'variant_to_allele_map': variant_allele_map,
                'phenotype_function': next((
                    gene_phenotypes[col].iloc[0]
                    for col in ['phenotype', 'activity_score']
                    if col in gene_phenotypes.columns and not gene_phenotypes.empty and not pd.isna(
                    gene_phenotypes[col].iloc[0])
                ), "Unknown")
            }

        return results

    def explain_decisions(self, sample, results):
        if not results:
            return results

        for gene, gene_info in results.items():
            explanation = []

            explanation.append(f"Analysis for gene {gene}:")
            explanation.append(f"PharmCAT called phenotype: {gene_info['pharmcat_phenotype']}")
            explanation.append(f"Ground truth phenotype: {gene_info['ground_truth_phenotype']}")
            explanation.append(f"Match accuracy: {'Correct' if gene_info['match_accuracy'] == 1 else 'Incorrect'}")

            explanation.append(f"Diplotype: {gene_info['diplotype']}")
            explanation.append(f"Allele 1: {gene_info['allele1']}")
            explanation.append(f"Allele 2: {gene_info['allele2']}")

            explanation.append("\nMost important variants in decision (ranked by SHAP importance):")
            for i, (variant, importance) in enumerate(list(gene_info['feature_importance'].items())[:5], 1):
                variant_details = gene_info['variant_details'].get(variant, {})
                genotype = variant_details.get('genotype', 'Unknown')

                explanation.append(f"{i}. {variant} (Importance: {importance:.4f}, Genotype: {genotype})")

                if variant in gene_info['variant_details']:
                    details = gene_info['variant_details'][variant]
                    explanation.append(f"   Position: {details.get('position', 'Unknown')}")
                    explanation.append(f"   Reference allele: {details.get('ref', 'Unknown')}")
                    explanation.append(f"   Alternate allele: {details.get('alt', 'Unknown')}")

            if 'decision_pathway' in gene_info:
                pathway = gene_info['decision_pathway']

                explanation.append("\nDecision pathway:")
                for variant in pathway['top_variants']:
                    alleles = pathway['variant_to_allele_map'].get(variant, [])
                    alleles_str = ", ".join(alleles) if alleles else "No specific alleles"
                    explanation.append(f"Variant {variant} → Contributes to alleles: {alleles_str}")

                explanation.append(f"Alleles → Diplotype: {gene_info['diplotype']}")
                explanation.append(f"Diplotype → Phenotype: {gene_info['pharmcat_phenotype']}")

            results[gene]['explanation'] = "\n".join(explanation)

        return results

    def save_results(self, sample, results):
        if not results:
            return

        os.makedirs(self.output_dir, exist_ok=True)
        output_file = os.path.join(self.output_dir, f"{sample}_results.json")

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"Results saved to {output_file}")

    def run(self):
        all_results = {}

        for sample in self.samples:
            print(f"\nProcessing sample: {sample}")

            vcf_df = self.load_vcf_csv(sample)
            match_df = self.load_match_csv(sample)
            phenotype_df = self.load_phenotype_csv(sample)

            if vcf_df is None or match_df is None or phenotype_df is None:
                print(f"Skipping sample {sample} due to missing data")
                continue

            sample_ground_truth = self.get_ground_truth_for_sample(sample)

            pharmcat_data = self.extract_pharmcat_data(sample, match_df, phenotype_df)

            results = self.run_shap_analysis(sample, vcf_df, pharmcat_data, sample_ground_truth)

            results = self.analyze_decision_pathways(sample, results, vcf_df, match_df, phenotype_df)

            results = self.explain_decisions(sample, results)

            self.save_results(sample, results)

            all_results[sample] = results

        # Save combined results
        combined_output_file = os.path.join(self.output_dir, "all_samples_results.json")
        with open(combined_output_file, 'w') as f:
            json.dump(all_results, f, indent=4)

        print(f"\nCombined results saved to {combined_output_file}")

        return all_results


def main():
    explainer = PharmcatExplainer()
    results = explainer.run()

    print(f"\nAnalysis complete! Results saved to {explainer.output_dir}")


if __name__ == "__main__":
    main()
