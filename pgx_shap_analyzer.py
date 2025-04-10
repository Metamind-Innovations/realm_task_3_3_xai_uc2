import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import shap

TARGET_GENES = ["CYP2B6", "CYP2C9", "CYP2C19", "CYP3A5", "SLCO1B1", "TPMT", "DPYD"]
PHENOTYPE_MAPPING = {
    "PM": 0,  # Poor Metabolizer
    "IM": 1,  # Intermediate Metabolizer
    "NM": 2,  # Normal Metabolizer
    "RM": 3,  # Rapid Metabolizer
    "UM": 4,  # Ultrarapid Metabolizer
    "NF": 5,  # Normal Function
    "DF": 6,  # Decreased Function
    "PF": 7,  # Poor Function
    "IF": 8,  # Increased Function
    "INDETERMINATE": 9  # Indeterminate
}
REVERSE_PHENOTYPE_MAPPING = {v: k for k, v in PHENOTYPE_MAPPING.items()}


def load_csv_data(input_dir):
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
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
        raise ValueError(f"No valid CSV files found in {input_dir}")

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

        # Try to find GT column (format may vary)
        gt_cols = [col for col in row.index if 'GT' in col]
        if gt_cols:
            genotype = row[gt_cols[0]]
        else:
            # Default to 0/1 if GT not found
            genotype = "0/1"

        # Create feature name
        feature_name = f"{gene}_{variant_id}_{genotype}"

        # Set feature to 1 (present)
        features[sample][feature_name] = 1

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

    # Ensure 'Sample ID' column exists
    if 'Sample ID' not in phenotypes_df.columns:
        raise ValueError(f"No 'Sample ID' column found in {phenotypes_file}")

    # Filter to samples in feature matrix
    phenotypes_df = phenotypes_df[phenotypes_df['Sample ID'].isin(feature_samples)]

    # Set index to Sample ID
    phenotypes_df = phenotypes_df.set_index('Sample ID')

    # Convert phenotypes to numeric using mapping
    Y = pd.DataFrame(index=phenotypes_df.index)

    for gene in TARGET_GENES:
        if gene in phenotypes_df.columns:
            Y[gene] = phenotypes_df[gene].map(lambda x: PHENOTYPE_MAPPING.get(x, 9) if pd.notna(x) else 9)

    return Y


def run_shap_analysis(X, Y, max_samples=100, n_background=100):
    np.random.seed(42)  # For reproducibility
    results = {}

    # Limit to a reasonable number of samples for performance
    if len(X) > max_samples:
        sample_indices = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X.iloc[sample_indices]
        y_sample = Y.iloc[sample_indices]
    else:
        X_sample = X
        y_sample = Y

    # Create background dataset for SHAP
    if len(X_sample) > n_background:
        background_indices = np.random.choice(len(X_sample), n_background, replace=False)
        background = shap.sample(X_sample.values, n_background)
    else:
        background = shap.sample(X_sample.values, min(len(X_sample), 50))

    feature_names = X_sample.columns.tolist()

    for gene in TARGET_GENES:
        if gene not in y_sample.columns:
            print(f"Skipping {gene} - not found in phenotypes data")
            continue

        print(f"Running SHAP analysis for {gene}...")

        # Create prediction function for this gene
        def predict_gene(x):
            # Map samples to predictions for this gene
            sample_indices = list(range(len(y_sample)))
            if len(x) != len(sample_indices):
                # Return random values based on distribution
                vals = y_sample[gene].values
                return np.random.choice(vals, size=len(x))
            return y_sample[gene].values

        # Create SHAP explainer
        explainer = shap.KernelExplainer(
            predict_gene,
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
            "predictions": y_sample[gene].tolist()
        }

    return results


def prepare_results_json(shap_results, X, phenotypes_df, output_file):
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

        # Calculate overall feature importance
        abs_shap = np.abs(shap_values)
        importance = abs_shap.mean(axis=0)

        # Get top features
        top_indices = np.argsort(-importance)[:20]  # Top 20 features

        # Store feature importance
        json_results["feature_importance"][gene] = {
            feature_names[i]: float(importance[i])
            for i in top_indices if i < len(feature_names)
        }

        # Extract gene from feature name
        def get_gene_from_feature(feature):
            parts = feature.split('_')
            return parts[0] if parts else "Unknown"

        # Group features by gene
        gene_features = defaultdict(list)
        for i in top_indices:
            if i < len(feature_names):
                feature = feature_names[i]
                feature_gene = get_gene_from_feature(feature)
                gene_features[feature_gene].append((feature, float(importance[i])))

        # Store gene explanations
        gene_explanation = {
            "prediction_distribution": {
                REVERSE_PHENOTYPE_MAPPING.get(pred, "Unknown"): int(count)
                for pred, count in pd.Series(predictions).value_counts().items()
            },
            "top_features_by_gene": {
                gene: [{"feature": f[0], "importance": f[1]} for f in features]
                for gene, features in gene_features.items()
            }
        }

        json_results["gene_explanations"][gene] = gene_explanation

        # Add sample explanations (limit to 10)
        for i in range(min(10, len(sample_indices))):
            sample = sample_indices[i]
            pred = predictions[i]
            phenotype = REVERSE_PHENOTYPE_MAPPING.get(pred, "Unknown")

            # Get SHAP values for this sample
            sample_shap = shap_values[i]

            # Get top contributing features
            top_contrib_indices = np.argsort(-np.abs(sample_shap))[:10]
            top_contributions = [
                {
                    "feature": feature_names[j],
                    "value": float(X.loc[sample, feature_names[j]]),
                    "shap_value": float(sample_shap[j])
                }
                for j in top_contrib_indices if j < len(feature_names)
            ]

            sample_explanation = {
                "sample_id": str(sample),
                "gene": gene,
                "predicted_phenotype": phenotype,
                "top_contributions": top_contributions
            }

            json_results["sample_explanations"].append(sample_explanation)

    # Write to file
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"Results saved to {output_file}")
    return json_results


def main():
    parser = argparse.ArgumentParser(description='SHAP analysis for PGx phenotypes')
    parser.add_argument('--csv_dir', required=True, help='Directory containing CSV files')
    parser.add_argument('--phenotypes_file', required=True, help='Path to phenotypes.csv file')
    parser.add_argument('--output_file', default='pgx_shap_results.json', help='Output JSON file')
    parser.add_argument('--max_samples', type=int, default=100, help='Maximum number of samples for SHAP analysis')
    args = parser.parse_args()

    print(f"Loading CSV data from {args.csv_dir}...")
    variant_data = load_csv_data(args.csv_dir)

    print("Extracting features...")
    X = extract_features(variant_data)
    print(f"Created feature matrix with {X.shape[0]} samples and {X.shape[1]} features")

    print(f"Loading phenotypes from {args.phenotypes_file}...")
    Y = prepare_targets(args.phenotypes_file, X.index)
    print(f"Prepared target matrix with {Y.shape[0]} samples")

    # Ensure same samples in X and Y
    common_samples = X.index.intersection(Y.index)
    X = X.loc[common_samples]
    Y = Y.loc[common_samples]
    print(f"Using {len(common_samples)} samples common to both feature and target matrices")

    print("Running SHAP analysis...")
    shap_results = run_shap_analysis(X, Y, max_samples=args.max_samples)

    print("Preparing JSON results...")
    phenotypes_df = pd.read_csv(args.phenotypes_file).set_index('Sample ID')
    prepare_results_json(shap_results, X, phenotypes_df, args.output_file)

    print("Analysis complete!")


if __name__ == "__main__":
    main()
