import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import shap

# Target genes for analysis
TARGET_GENES = ["CYP2B6", "CYP2C9", "CYP2C19", "CYP3A5", "SLCO1B1", "TPMT", "DPYD"]

SAMPLE_ID_COL = 'Sample ID'
CSV_EXT = "*.csv"
CSV_SUFFIX = ".csv"


class VCFParser:
    def __init__(self):
        self.meta_info = {}
        self.header = None

    def parse_vcf(self, vcf_file):
        try:
            with open(vcf_file, 'r') as f:
                lines = f.readlines()

            data_lines = []
            for line in lines:
                line = line.strip()
                if line.startswith('##'):
                    self._parse_meta_line(line)
                elif line.startswith('#CHROM'):
                    self.header = line[1:].split('\t')
                else:
                    data_lines.append(line.split('\t'))

            if not self.header or not data_lines:
                return None

            df = pd.DataFrame(data_lines, columns=self.header)
            df['Gene'] = df['INFO'].apply(self._extract_gene)

            sample_columns = self.header[9:]
            if sample_columns:
                for sample in sample_columns:
                    df[f'{sample}_GT'] = df.apply(
                        lambda row: self._extract_genotype(row['FORMAT'], row[sample]),
                        axis=1
                    )

            return df

        except Exception as e:
            print(f"Error parsing VCF file: {e}")
            return None

    def _parse_meta_line(self, line):
        if line.startswith('##'):
            line = line[2:]
            if '=' in line:
                key, value = line.split('=', 1)
                self.meta_info[key] = value

    def _extract_gene(self, info_field):
        gene_match = re.search(r'PX=([^;]+)', info_field)
        return gene_match.group(1) if gene_match else ''

    def _extract_genotype(self, format_field, sample_field):
        format_parts = format_field.split(':')
        sample_parts = sample_field.split(':')

        if 'GT' in format_parts:
            gt_index = format_parts.index('GT')
            if gt_index < len(sample_parts):
                return sample_parts[gt_index]

        return ''

    def vcf_to_csv(self, vcf_file, output_csv=None):
        if output_csv is None:
            output_csv = os.path.splitext(vcf_file)[0] + CSV_SUFFIX

        df = self.parse_vcf(vcf_file)
        if df is not None:
            df.to_csv(output_csv, index=False)
            return output_csv
        return None


def preprocess_input_data(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(input_dir, "preprocessed")

    os.makedirs(output_dir, exist_ok=True)

    vcf_files = glob.glob(os.path.join(input_dir, "*.vcf"))
    sample_to_file = {}

    for vcf_file in vcf_files:
        filename = os.path.basename(vcf_file)
        sample_id = filename.split('_')[0]

        parser = VCFParser()
        output_csv = os.path.join(output_dir, f"{sample_id}_preprocessed.csv")

        if parser.vcf_to_csv(vcf_file, output_csv):
            sample_to_file[sample_id] = output_csv
            print(f"Converted {vcf_file} to {output_csv}")

    return sample_to_file, output_dir


def load_csv_data(csv_files):
    all_data = []

    for csv_file in csv_files:
        sample_id = os.path.basename(csv_file).split('_')[0]
        try:
            df = pd.read_csv(csv_file)
            df['Sample'] = sample_id
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {str(e)}")

    if not all_data:
        raise ValueError("No valid CSV files found")

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


def extract_features(variant_data):
    features = defaultdict(dict)

    for _, row in variant_data.iterrows():
        sample = row['Sample']
        gene = row.get('Gene', None)
        if not gene or gene not in TARGET_GENES:
            continue

        # Extract variant ID and genotype
        variant_id = row.get('ID', f"{row['CHROM']}_{row['POS']}")
        if variant_id == '.' or pd.isna(variant_id):
            variant_id = f"{row['CHROM']}_{row['POS']}"

        # Try to find GT column
        gt_col = None
        for col in row.index:
            if '_GT' in col:
                gt_col = col
                break

        genotype = row[gt_col] if gt_col else "0/0"

        # Add basic features
        feature_name = f"{gene}_{variant_id}"
        features[sample][feature_name] = 1  # This variant IS present in this sample

        # Add genotype-specific features
        if genotype in ['0/1', '1/0', '0/2', '2/0']:  # Heterozygous
            features[sample][f"{feature_name}_het"] = 1
        elif genotype in ['1/1', '2/2']:  # Homozygous alt
            features[sample][f"{feature_name}_hom"] = 1

    # Convert to dataframe
    samples = list(features.keys())
    all_features = set()
    for sample_features in features.values():
        all_features.update(sample_features.keys())

    X = pd.DataFrame(0, index=samples, columns=sorted(all_features))
    for sample, sample_features in features.items():
        for feature, value in sample_features.items():
            X.loc[sample, feature] = value

    return X


def prepare_targets(phenotypes_file, feature_samples):
    phenotypes_df = pd.read_csv(phenotypes_file)

    # Ensure Sample ID column exists
    if SAMPLE_ID_COL not in phenotypes_df.columns:
        raise ValueError(f"No '{SAMPLE_ID_COL}' column found in {phenotypes_file}")

    # Filter to samples in feature matrix
    phenotypes_df = phenotypes_df[phenotypes_df[SAMPLE_ID_COL].isin(feature_samples)]

    # Set index to Sample ID
    phenotypes_df = phenotypes_df.set_index(SAMPLE_ID_COL)

    # Create phenotype mapping on-the-fly
    Y = pd.DataFrame(index=phenotypes_df.index)
    phenotype_mappings = {}

    for gene in TARGET_GENES:
        if gene in phenotypes_df.columns:
            # Get unique phenotypes and create mapping
            unique_phenotypes = sorted(phenotypes_df[gene].dropna().unique())
            phenotype_to_num = {phenotype: i for i, phenotype in enumerate(unique_phenotypes)}
            phenotype_mappings[gene] = phenotype_to_num

            # Apply mapping
            Y[gene] = phenotypes_df[gene].map(lambda x: phenotype_to_num.get(x, -1) if pd.notna(x) else -1)

    return Y, phenotypes_df, phenotype_mappings


def create_prediction_function(y_sample, gene):
    def predict_gene(x, y_sample=y_sample):
        if len(x) == len(y_sample):
            return y_sample[gene].values

        # For other shapes, sample from distribution
        rng = np.random.Generator(np.random.PCG64())
        vals = y_sample[gene].values
        return rng.choice(vals, size=len(x))

    return predict_gene


def run_shap_analysis(X, Y, phenotype_mappings, max_samples=100):
    rng = np.random.Generator(np.random.PCG64(seed=42))
    results = {}

    # Limit samples for performance
    if len(X) > max_samples:
        sample_indices = rng.choice(len(X), max_samples, replace=False)
        X_sample = X.iloc[sample_indices]
        y_sample = Y.iloc[sample_indices]
    else:
        X_sample = X
        y_sample = Y

    # Create background data
    n_background = min(50, len(X_sample))
    if len(X_sample) > n_background:
        background = shap.sample(X_sample.values, n_background)
    else:
        background = shap.sample(X_sample.values, min(len(X_sample), 20))

    feature_names = X_sample.columns.tolist()

    for gene in TARGET_GENES:
        if gene not in y_sample.columns:
            print(f"Skipping {gene} - not found in phenotypes data")
            continue

        print(f"Running SHAP analysis for {gene}...")

        # Create reverse mapping (numeric to phenotype)
        num_to_phenotype = {v: k for k, v in phenotype_mappings[gene].items()}

        # Create prediction function
        predict_func = create_prediction_function(y_sample, gene)

        # Create SHAP explainer
        explainer = shap.KernelExplainer(
            predict_func,
            background,
            link="identity",
            feature_names=feature_names
        )

        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample.values, nsamples=100)

        # Store results
        results[gene] = {
            "shap_values": shap_values,
            "feature_names": feature_names,
            "sample_indices": X_sample.index.tolist(),
            "predictions": y_sample[gene].tolist(),
            "num_to_phenotype": num_to_phenotype
        }

    return results


def generate_variant_explanation(feature, effect):
    feature_parts = feature.split('_')
    variant_text = f"{feature}"

    if len(feature_parts) > 1:
        gene = feature_parts[0]
        variant = feature_parts[1]
        if variant and gene:
            variant_text = f"{variant} in {gene}"

    return f"{variant_text} with {effect} contribution"


def create_enriched_results(shap_results, X, output_file):
    json_results = {
        "gene_explanations": {},
        "feature_importance": {},
        "sample_explanations": []
    }

    # Process each gene's results
    for gene, result in shap_results.items():
        shap_values = result["shap_values"]
        feature_names = result["feature_names"]
        sample_indices = result["sample_indices"]
        predictions = result["predictions"]
        num_to_phenotype = result["num_to_phenotype"]

        # Calculate feature importance
        abs_shap = np.abs(shap_values)
        importance = abs_shap.mean(axis=0)
        top_indices = np.argsort(-importance)[:20]

        # Store feature importance
        gene_importance = {}
        for idx in top_indices:
            if idx < len(feature_names):
                feature = feature_names[idx]
                gene_importance[feature] = float(importance[idx])

        json_results["feature_importance"][gene] = gene_importance

        # Group features by gene
        features_by_gene = defaultdict(list)
        for idx in top_indices:
            if idx < len(feature_names):
                feature = feature_names[idx]
                feature_gene = feature.split('_')[0] if '_' in feature else "Unknown"

                if feature_gene in TARGET_GENES:
                    features_by_gene[feature_gene].append({
                        "feature": feature,
                        "importance": float(importance[idx])
                    })

        # Count phenotype distribution
        phenotype_counts = {}
        for pred in predictions:
            phenotype = num_to_phenotype.get(pred, "Unknown")
            phenotype_counts[phenotype] = phenotype_counts.get(phenotype, 0) + 1

        # Store gene explanation
        json_results["gene_explanations"][gene] = {
            "prediction_distribution": phenotype_counts,
            "top_features_by_gene": {
                g: features for g, features in features_by_gene.items()
            }
        }

        # Add sample explanations (limited to 10)
        for i in range(min(10, len(sample_indices))):
            sample = sample_indices[i]
            pred = predictions[i]
            phenotype = num_to_phenotype.get(pred, "Unknown")

            # Get SHAP values for this sample
            sample_shap = shap_values[i]
            top_contrib_indices = np.argsort(-np.abs(sample_shap))[:10]

            # Collect top contributions
            top_contributions = []
            for j in top_contrib_indices:
                if j < len(feature_names):
                    feature = feature_names[j]
                    contrib = {
                        "feature": feature,
                        "value": float(X.loc[sample, feature]),
                        "shap_value": float(sample_shap[j])
                    }
                    top_contributions.append(contrib)

            # Generate explanation text
            explanation = f"The {gene} phenotype is {phenotype}. "

            if top_contributions:
                explanation += "Key contributing variants include: "
                variant_descriptions = []

                for contrib in top_contributions[:3]:
                    feature = contrib["feature"]
                    shap_val = contrib["shap_value"]
                    effect = "positive" if shap_val > 0 else "negative"
                    variant_descriptions.append(
                        generate_variant_explanation(feature, effect)
                    )

                explanation += ", ".join(variant_descriptions)

            # Add sample explanation
            json_results["sample_explanations"].append({
                "sample_id": str(sample),
                "gene": gene,
                "predicted_phenotype": phenotype,
                "top_contributions": top_contributions,
                "explanation": explanation
            })

    # Write to file
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"Results saved to {output_file}")
    return json_results


def generate_summary_report(json_results, output_dir):
    summary_file = os.path.join(output_dir, "pgx_shap_summary.txt")

    with open(summary_file, 'w') as f:
        f.write("# PharmCAT SHAP Analysis Summary Report\n\n")

        # Overall stats
        f.write("## Overview\n")
        f.write(f"Analysis performed on {len(json_results['gene_explanations'])} genes\n")
        num_samples = len(set([exp['sample_id'] for exp in json_results['sample_explanations']]))
        f.write(f"Total samples analyzed: {num_samples}\n\n")

        f.write("## Gene Summaries\n")

        for gene, gene_data in json_results['gene_explanations'].items():
            f.write(f"\n### {gene}\n")

            f.write("Phenotype distribution:\n")
            for phenotype, count in gene_data['prediction_distribution'].items():
                f.write(f"- {phenotype}: {count}\n")

            f.write("\nTop contributing features:\n")
            all_features = []
            for _, features in gene_data['top_features_by_gene'].items():
                all_features.extend(features)

            # Sort by importance
            all_features.sort(key=lambda x: x['importance'], reverse=True)

            for i, feature in enumerate(all_features[:10]):
                f.write(f"{i + 1}. {feature['feature']} - Importance: {feature['importance']:.4f}\n")

            f.write("\n")

        f.write("## Sample Explanations\n")

        for i, sample in enumerate(json_results['sample_explanations'][:5]):
            f.write(f"\n### Sample {sample['sample_id']} - {sample['gene']}\n")
            f.write(f"Phenotype: {sample['predicted_phenotype']}\n")
            f.write(f"Explanation: {sample['explanation']}\n")

            f.write("Top contributions:\n")
            for j, contrib in enumerate(sample['top_contributions'][:5]):
                f.write(f"{j + 1}. {contrib['feature']} - SHAP value: {contrib['shap_value']:.4f}\n")

    print(f"Summary report saved to {summary_file}")
    return summary_file


def find_csv_files(input_dir):
    # Direct CSV files
    csv_files = glob.glob(os.path.join(input_dir, CSV_EXT))
    if csv_files:
        return csv_files

    # Check preprocessed directory
    preprocessed_dir = os.path.join(input_dir, "preprocessed")
    if os.path.exists(preprocessed_dir):
        return glob.glob(os.path.join(preprocessed_dir, CSV_EXT))

    return []


def main():
    parser = argparse.ArgumentParser(description='PharmCAT SHAP-based explainer')
    parser.add_argument('--input_dir', required=True, help='Directory containing VCF or CSV files')
    parser.add_argument('--phenotypes_file', required=True, help='Path to phenotypes.csv file')
    parser.add_argument('--output_dir', default='pgx_shap_results', help='Output directory for results')
    parser.add_argument('--convert_vcf', action='store_true', help='Convert VCF files to CSV format')
    parser.add_argument('--max_samples', type=int, default=100, help='Maximum number of samples for SHAP analysis')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Processing input data from {args.input_dir}...")

    input_files = []
    if args.convert_vcf:
        _, csv_dir = preprocess_input_data(args.input_dir, os.path.join(args.output_dir, "preprocessed"))
        input_files = glob.glob(os.path.join(csv_dir, CSV_EXT))
        print(f"Converted VCF files to CSV format in {csv_dir}")
    else:
        input_files = find_csv_files(args.input_dir)
        if not input_files:
            raise ValueError(f"No CSV files found in {args.input_dir}. Use --convert_vcf to convert VCF files.")

    print(f"Found {len(input_files)} input files")

    print("Loading CSV data...")
    variant_data = load_csv_data(input_files)
    print(f"Loaded data for {variant_data['Sample'].nunique()} samples")

    print("Extracting features...")
    X = extract_features(variant_data)
    print(f"Created feature matrix with {X.shape[0]} samples and {X.shape[1]} features")

    print(f"Loading phenotypes from {args.phenotypes_file}...")
    Y, phenotypes_df, phenotype_mappings = prepare_targets(args.phenotypes_file, X.index)
    print(f"Prepared target matrix with {Y.shape[0]} samples")

    # Ensure same samples in X and Y
    common_samples = X.index.intersection(Y.index)
    X = X.loc[common_samples]
    Y = Y.loc[common_samples]
    print(f"Using {len(common_samples)} samples common to both matrices")

    print("Running SHAP analysis...")
    shap_results = run_shap_analysis(X, Y, phenotype_mappings, max_samples=args.max_samples)

    print("Preparing results...")
    json_output_file = os.path.join(args.output_dir, "pgx_shap_results.json")
    json_results = create_enriched_results(shap_results, X, json_output_file)

    print("Generating summary report...")
    summary_file = generate_summary_report(json_results, args.output_dir)

    print(f"Analysis complete! Results saved to {args.output_dir}")
    print(f"JSON results: {json_output_file}")
    print(f"Summary report: {summary_file}")


if __name__ == "__main__":
    main()
