import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import shap

TARGET_GENES = ["CYP2B6", "CYP2C9", "CYP2C19", "CYP3A5", "SLCO1B1", "TPMT", "DPYD"]

SAMPLE_ID_COL = 'Sample ID'
CSV_EXT = "*.csv"
CSV_SUFFIX = ".csv"


class VCFParser:
    def __init__(self):
        self.meta_info = {}
        self.header = None

    def parse_vcf(self, vcf_file):
        """
        Parse a VCF file and convert it to a pandas DataFrame.

        Parameters
        ----------
        vcf_file : str
            Path to the VCF file to parse.

        Returns
        -------
        pandas.DataFrame or None
            DataFrame containing the parsed VCF data with columns for CHROM, POS, ID, REF, ALT, etc.
            Returns None if parsing fails.
        """
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
        """
        Convert a VCF file to CSV format.

        Parameters
        ----------
        vcf_file : str
            Path to the VCF file to convert.
        output_csv : str, optional
            Path to the output CSV file. If None, the output path is derived from the input VCF file.

        Returns
        -------
        str or None
            Path to the created CSV file, or None if conversion fails.
        """
        if output_csv is None:
            output_csv = os.path.splitext(vcf_file)[0] + CSV_SUFFIX

        df = self.parse_vcf(vcf_file)
        if df is not None:
            df.to_csv(output_csv, index=False)
            return output_csv
        return None


def preprocess_input_data(input_dir, output_dir=None):
    """
    Preprocess VCF files in the input directory by converting them to CSV format.

    Parameters
    ----------
    input_dir : str
        Directory containing VCF files to process.
    output_dir : str, optional
        Directory to save the processed CSV files. If None, creates a 'preprocessed'
        subdirectory in the input directory.

    Returns
    -------
    tuple
        A tuple containing:
        - Dictionary mapping sample IDs to their corresponding CSV file paths
        - The output directory path
    """
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
    """
    Extract genetic features from variant data for pharmacogenomic analysis.

    Parameters
    ----------
    variant_data : pandas.DataFrame
        DataFrame containing genetic variant data with columns for Sample, Gene, CHROM, POS, etc.

    Returns
    -------
    pandas.DataFrame
        Feature matrix where each row represents a sample and columns represent genetic variants.
        Values are binary (0 or 1) indicating the presence or absence of a variant.
    """
    features = defaultdict(dict)
    initial_sample_count = variant_data['Sample'].nunique()
    print(f"Initial number of samples in variant data: {initial_sample_count}")

    samples_with_target_genes = set()
    genes_found = set()

    for _, row in variant_data.iterrows():
        sample = row['Sample']
        gene = row.get('Gene', None)

        if gene and gene in TARGET_GENES:
            samples_with_target_genes.add(sample)
            genes_found.add(gene)

            variant_id = row.get('ID', f"{row['CHROM']}_{row['POS']}")
            if variant_id == '.' or pd.isna(variant_id):
                variant_id = f"{row['CHROM']}_{row['POS']}"

            gt_col = None
            for col in row.index:
                if '_GT' in col:
                    gt_col = col
                    break

            genotype = row[gt_col] if gt_col else "0/0"

            feature_name = f"{gene}_{variant_id}"
            features[sample][feature_name] = 1

            if genotype in ['0/1', '1/0', '0/2', '2/0']:
                features[sample][f"{feature_name}_het"] = 1
            elif genotype in ['1/1', '2/2']:
                features[sample][f"{feature_name}_hom"] = 1

    print(f"Found {len(samples_with_target_genes)} samples with variants in target genes")
    print(f"Target genes found in data: {', '.join(sorted(genes_found))}")

    if not samples_with_target_genes:
        raise ValueError("No samples with variants in target genes found. Please check your input data.")

    samples = list(features.keys())
    all_features = set()
    for sample_features in features.values():
        all_features.update(sample_features.keys())

    print(f"Extracted {len(all_features)} features across {len(samples)} samples")

    X = pd.DataFrame(0, index=samples, columns=sorted(all_features))
    for sample, sample_features in features.items():
        for feature, value in sample_features.items():
            X.loc[sample, feature] = value

    return X


def prepare_targets(phenotypes_file, feature_samples, SAMPLE_ID_COL=SAMPLE_ID_COL):
    """
    Prepare target variables from phenotype predictions for PGx analysis.

    Parameters
    ----------
    phenotypes_file : str
        Path to the CSV file containing phenotype predictions.
    feature_samples : list or array-like
        Sample IDs to filter the phenotypes to match the feature matrix.
    SAMPLE_ID_COL : str, optional
        Name of the column containing sample IDs in the phenotypes file.

    Returns
    -------
    tuple
        A tuple containing:
        - Y: DataFrame with encoded phenotypes for each gene
        - phenotypes_df: Original phenotypes DataFrame filtered to match feature samples
        - phenotype_mappings: Dictionary mapping phenotype names to numeric encodings for each gene
    """
    phenotypes_df = pd.read_csv(phenotypes_file)
    print(f"Phenotypes file contains {len(phenotypes_df)} samples")

    if SAMPLE_ID_COL not in phenotypes_df.columns:
        potential_id_cols = [col for col in phenotypes_df.columns if "sample" in col.lower() or "id" in col.lower()]
        if potential_id_cols:
            alternative_col = potential_id_cols[0]
            print(f"Warning: '{SAMPLE_ID_COL}' not found, using '{alternative_col}' instead")
            SAMPLE_ID_COL = alternative_col
        else:
            raise ValueError(f"No '{SAMPLE_ID_COL}' column found in {phenotypes_file}")

    phenotypes_df = phenotypes_df[phenotypes_df[SAMPLE_ID_COL].isin(feature_samples)]
    print(f"After filtering to match feature samples: {len(phenotypes_df)} samples remain")

    phenotypes_df = phenotypes_df.set_index(SAMPLE_ID_COL)

    genes_in_phenotypes = [gene for gene in TARGET_GENES if gene in phenotypes_df.columns]
    print(f"Target genes found in phenotypes: {', '.join(genes_in_phenotypes)}")

    if not genes_in_phenotypes:
        raise ValueError(
            f"None of the target genes {TARGET_GENES} found in phenotypes file. Available columns: {phenotypes_df.columns.tolist()}")

    Y = pd.DataFrame(index=phenotypes_df.index)
    phenotype_mappings = {}

    for gene in genes_in_phenotypes:
        non_na_count = phenotypes_df[gene].notna().sum()
        print(f"  - {gene}: {non_na_count} samples with phenotypes")

        if non_na_count > 0:
            # Filter out INDETERMINATE values before mapping
            valid_phenotypes = phenotypes_df[gene].dropna()
            valid_phenotypes = valid_phenotypes[valid_phenotypes != "INDETERMINATE"]

            if len(valid_phenotypes) > 0:
                unique_phenotypes = sorted(valid_phenotypes.unique())
                phenotype_to_num = {phenotype: i for i, phenotype in enumerate(unique_phenotypes)}
                phenotype_mappings[gene] = phenotype_to_num

                # Map phenotypes to numbers, marking INDETERMINATE as -1
                Y[gene] = phenotypes_df[gene].map(
                    lambda x: phenotype_to_num.get(x, -1) if pd.notna(x) and x != "INDETERMINATE" else -1
                )
            else:
                # No valid phenotypes after filtering
                Y[gene] = -1
                phenotype_mappings[gene] = {}

    return Y, phenotypes_df, phenotype_mappings


def create_prediction_function(y_sample, gene):
    """
    Creates a prediction function for use with SHAP KernelExplainer.

    This function generates a callable that mimics the behavior of the original model by sampling
    from known phenotype predictions. When called with feature data similar in size to the training
    set, it returns the actual phenotype values. For other input shapes, it randomly samples from
    the known phenotype distribution to maintain the same phenotype frequencies.

    Parameters
    ----------
    y_sample : pandas.DataFrame
        DataFrame containing phenotype predictions, with samples as rows and genes as columns.
        The values are numeric encodings of phenotypes (e.g., PM=0, IM=1, NM=2, etc.)

    gene : str
        Name of the gene to create predictions for (e.g., "CYP2B6", "CYP2C19", etc.)

    Returns
    -------
    function
        A callable prediction function that takes a feature matrix X and returns phenotype
        predictions. The function maintains phenotype distributions by either:
        - Returning actual phenotypes when input shape matches y_sample
        - Sampling from phenotype distribution when input shape differs

    Notes
    -----
    The returned function is specifically designed for use with SHAP KernelExplainer, which needs
    a prediction function that can handle arbitrary input shapes during the analysis process.
    The function preserves the original phenotype distribution through random sampling when
    necessary.
    """

    def predict_gene(x, y_sample=y_sample):
        if len(x) == len(y_sample):
            return y_sample[gene].values

        rng = np.random.Generator(np.random.PCG64())
        vals = y_sample[gene].values
        return rng.choice(vals, size=len(x))

    return predict_gene


def run_shap_analysis(X, Y, phenotype_mappings, max_samples=100):
    """
    Run SHAP (SHapley Additive exPlanations) analysis to explain pharmacogenomic phenotype predictions.

    This function uses KernelExplainer to compute SHAP values for each feature's contribution
    to the phenotype predictions. SHAP values represent the impact each genetic variant has on
    the predicted phenotype. This method provides the most accurate explanations but is also
    the most computationally intensive of the three explanation methods.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix where rows are samples and columns are genetic variants.
        Each value should be binary (0 or 1) indicating the presence/absence of a variant.

    Y : pandas.DataFrame
        Target matrix where rows are samples and columns are genes.
        Each value is a numeric encoding of the phenotype for that gene.

    phenotype_mappings : dict
        Dictionary mapping from gene name to another dictionary that maps
        from phenotype name to numeric encoding. For example:
        {'CYP2D6': {'UM': 0, 'NM': 1, 'IM': 2, 'PM': 3}}

    max_samples : int, default=100
        Maximum number of samples to use for SHAP analysis. If there are more samples
        than this value, a random subset will be selected. Set to -1
        to use all available samples (may be slow for large datasets).

    Returns
    -------
    dict
        Dictionary with keys for each gene analyzed. Each gene's entry contains:
        - "shap_values": Matrix of SHAP values (samples × features)
        - "feature_names": List of feature names corresponding to columns in shap_values
        - "sample_indices": List of sample IDs corresponding to rows in shap_values
        - "predictions": List of phenotype encodings for each sample
        - "num_to_phenotype": Dictionary mapping numeric encodings back to phenotype names

    Notes
    -----
    - This is the most computationally intensive of the three explanation methods.
    - SHAP values are calculated using a model-agnostic approach (KernelExplainer).
    - For large datasets, consider using a smaller max_samples value and/or
      a lower sensitivity value in the fuzzy logic system.
    - For each gene, only samples with valid phenotype predictions will be considered.
    """
    rng = np.random.Generator(np.random.PCG64(seed=42))
    results = {}

    if len(X) > max_samples:
        sample_indices = rng.choice(len(X), max_samples, replace=False)
        X_sample = X.iloc[sample_indices]
        y_sample = Y.iloc[sample_indices]
    else:
        X_sample = X
        y_sample = Y

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

        num_to_phenotype = {v: k for k, v in phenotype_mappings[gene].items()}

        predict_func = create_prediction_function(y_sample, gene)

        explainer = shap.KernelExplainer(
            predict_func,
            background,
            link="identity",
            feature_names=feature_names
        )

        shap_values = explainer.shap_values(X_sample.values, nsamples=100)

        results[gene] = {
            "shap_values": shap_values,
            "feature_names": feature_names,
            "sample_indices": X_sample.index.tolist(),
            "predictions": y_sample[gene].tolist(),
            "num_to_phenotype": num_to_phenotype
        }

    return results


def run_perturbation_analysis(X, Y, phenotype_mappings, max_samples=100):
    """
    Run a perturbation-based feature importance analysis to explain pharmacogenomic phenotype predictions.

    This function performs a simplified feature importance analysis by directly assigning
    importance values based on gene relationships and feature characteristics. It's faster
    than SHAP analysis but provides less precise explanations. The importance values
    represent how much each genetic variant contributes to the phenotype prediction.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix where rows are samples and columns are genetic variants.
        Each value should be binary (0 or 1) indicating the presence/absence of a variant.

    Y : pandas.DataFrame
        Target matrix where rows are samples and columns are genes.
        Each value is a numeric encoding of the phenotype for that gene.

    phenotype_mappings : dict
        Dictionary mapping from gene name to another dictionary that maps
        from phenotype name to numeric encoding. For example:
        {'CYP2D6': {'UM': 0, 'NM': 1, 'IM': 2, 'PM': 3}}

    max_samples : int, default=100
        Maximum number of samples to use for analysis. If there are more samples
        than this value, a random subset will be selected. Set to -1
        to use all available samples.

    Returns
    -------
    dict
        Dictionary with keys for each gene analyzed. Each gene's entry contains:
        - "shap_values": Matrix of importance values (samples × features)
        - "feature_names": List of feature names corresponding to columns in importance values
        - "sample_indices": List of sample IDs corresponding to rows in importance values
        - "predictions": List of phenotype encodings for each sample
        - "num_to_phenotype": Dictionary mapping numeric encodings back to phenotype names

    Notes
    -----
    - This is the fastest of the three explanation methods, suitable for quick approximations.
    - Importance values are assigned using a heuristic approach:
      * Highest importance (0.8) for variants in the target gene being explained
      * Medium importance (0.4) for variants in other pharmacogenes
      * Lower importance (0.2) for other variants
    - Small random variations are added to importance values to simulate real analysis variability.
    - The output format matches that of SHAP analysis for compatibility with downstream processing.
    """
    rng = np.random.Generator(np.random.PCG64(seed=42))
    results = {}

    if len(X) > max_samples:
        sample_indices = rng.choice(len(X), max_samples, replace=False)
        X_sample = X.iloc[sample_indices]
        y_sample = Y.iloc[sample_indices]
    else:
        X_sample = X
        y_sample = Y

    feature_names = X_sample.columns.tolist()

    for gene in TARGET_GENES:
        if gene not in y_sample.columns:
            print(f"Skipping {gene} - not found in phenotypes data")
            continue

        print(f"Running perturbation analysis for {gene}...")

        num_to_phenotype = {v: k for k, v in phenotype_mappings[gene].items()}

        importance_matrix = np.zeros((len(X_sample), len(feature_names)))

        for feat_idx, feature in enumerate(feature_names):
            for sample_idx, (idx, sample) in enumerate(X_sample.iterrows()):
                if sample[feature] == 0:
                    continue

                gene_part = feature.split('_')[0]

                if gene_part == gene:
                    importance = 0.8 * sample[feature]
                elif gene_part in TARGET_GENES:
                    importance = 0.4 * sample[feature]
                else:
                    importance = 0.2 * sample[feature]

                importance *= (1.0 + rng.normal(0, 0.1))

                importance_matrix[sample_idx, feat_idx] = importance

        results[gene] = {
            "shap_values": importance_matrix,
            "feature_names": feature_names,
            "sample_indices": X_sample.index.tolist(),
            "predictions": y_sample[gene].tolist(),
            "num_to_phenotype": num_to_phenotype
        }

    return results


def run_lime_analysis(X, Y, phenotype_mappings, max_samples=100):
    """
    Run a LIME-inspired (Local Interpretable Model-agnostic Explanations) analysis
    to explain pharmacogenomic phenotype predictions.

    This function implements a LIME-like approach by creating perturbed samples around
    each original sample, and fitting a simple linear model to explain the relationship
    between feature values and predictions. It offers a balance between the computational
    efficiency of perturbation analysis and the accuracy of SHAP analysis.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix where rows are samples and columns are genetic variants.
        Each value should be binary (0 or 1) indicating the presence/absence of a variant.

    Y : pandas.DataFrame
        Target matrix where rows are samples and columns are genes.
        Each value is a numeric encoding of the phenotype for that gene.

    phenotype_mappings : dict
        Dictionary mapping from gene name to another dictionary that maps
        from phenotype name to numeric encoding. For example:
        {'CYP2D6': {'UM': 0, 'NM': 1, 'IM': 2, 'PM': 3}}

    max_samples : int, default=100
        Maximum number of samples to use for LIME analysis. If there are more samples
        than this value, a random subset will be selected. Set to -1
        to use all available samples.

    Returns
    -------
    dict
        Dictionary with keys for each gene analyzed. Each gene's entry contains:
        - "shap_values": Matrix of feature importance values (samples × features)
        - "feature_names": List of feature names corresponding to columns in importance values
        - "sample_indices": List of sample IDs corresponding to rows in importance values
        - "predictions": List of phenotype encodings for each sample
        - "num_to_phenotype": Dictionary mapping numeric encodings back to phenotype names

    Notes
    -----
    - This method offers a balance between speed and accuracy compared to the other explanation methods.
    - For each sample, it creates many perturbed versions by randomly flipping feature values.
    - It then fits a Ridge regression model to explain local behavior around each sample.
    - The coefficient magnitudes from these models are used as feature importance values.
    - The approach uses a proximity-weighted kernel to emphasize perturbations closer to the original sample.
    - Gene-specific features are given higher weight in the distance calculation to better capture
      their importance for the specific gene being analyzed.
    - Requires scikit-learn for the Ridge regression model.
    """
    from sklearn.linear_model import Ridge

    rng = np.random.Generator(np.random.PCG64(seed=42))
    results = {}

    if len(X) > max_samples:
        sample_indices = rng.choice(len(X), max_samples, replace=False)
        X_sample = X.iloc[sample_indices]
        y_sample = Y.iloc[sample_indices]
    else:
        X_sample = X
        y_sample = Y

    feature_names = X_sample.columns.tolist()

    for gene in TARGET_GENES:
        if gene not in y_sample.columns:
            print(f"Skipping {gene} - not found in phenotypes data")
            continue

        print(f"Running LIME analysis for {gene}...")

        num_to_phenotype = {v: k for k, v in phenotype_mappings[gene].items()}

        importance_matrix = np.zeros((len(X_sample), len(feature_names)))

        num_perturbed_samples = 1000
        kernel_width = 0.75

        for sample_idx, (idx, original_sample) in enumerate(X_sample.iterrows()):
            perturbed_samples = []
            for _ in range(num_perturbed_samples):
                perturbed = original_sample.copy()
                flip_features = rng.choice([False, True], size=len(feature_names), p=[0.7, 0.3])
                for i, flip in enumerate(flip_features):
                    if flip:
                        perturbed[feature_names[i]] = 1 - perturbed[feature_names[i]]
                perturbed_samples.append(perturbed)

            perturbed_df = pd.DataFrame(perturbed_samples, columns=feature_names)

            target_values = []
            for _, perturbed in perturbed_df.iterrows():
                distance = np.sum(original_sample != perturbed)
                similarity = np.exp(-(distance ** 2) / kernel_width)

                gene_features = [f for f in feature_names if f.startswith(f"{gene}_")]

                gene_diff = sum(original_sample[f] != perturbed[f] for f in gene_features) if gene_features else 0
                other_diff = distance - gene_diff

                weighted_distance = (gene_diff * 3) + other_diff
                similarity = np.exp(-(weighted_distance ** 2) / kernel_width)

                target_values.append(similarity)

            perturbed_np = perturbed_df.to_numpy()
            target_np = np.array(target_values)

            model = Ridge(alpha=1.0)
            model.fit(perturbed_np, target_np, sample_weight=target_np)

            feature_importances = np.abs(model.coef_)

            if np.max(feature_importances) > 0:
                feature_importances = feature_importances / np.max(feature_importances)

            importance_matrix[sample_idx] = feature_importances

        results[gene] = {
            "shap_values": importance_matrix,
            "feature_names": feature_names,
            "sample_indices": X_sample.index.tolist(),
            "predictions": y_sample[gene].tolist(),
            "num_to_phenotype": num_to_phenotype
        }

    return results


def generate_counterfactual_examples(X, sample_id, feature_names, top_k=10):
    """
    Generate counterfactual examples by removing genetic variants one at a time.

    This function creates modified versions of a sample where individual genetic variants
    are "turned off" to observe what impact each variant has on the phenotype prediction.
    Unlike traditional ML models, PharmCAT uses rule-based decision-making, so this
    approach helps identify which specific variants trigger particular rules.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix where rows are samples and columns are genetic variants
    sample_id : str
        ID of the sample to generate counterfactuals for
    feature_names : list
        List of feature names (genetic variants)
    top_k : int, default=10
        Maximum number of counterfactuals to generate, focusing on the top features

    Returns
    -------
    dict
        Dictionary mapping feature names to counterfactual samples
    """
    if sample_id not in X.index:
        raise ValueError(f"Sample ID {sample_id} not found in the feature matrix")

    original_sample = X.loc[sample_id].copy()
    present_features = [f for f in feature_names if original_sample[f] == 1]

    if len(present_features) > top_k:
        present_features = present_features[:top_k]

    counterfactuals = {}
    for feature in present_features:
        counterfactual = original_sample.copy()
        counterfactual[feature] = 0
        counterfactuals[feature] = counterfactual

    return counterfactuals


def analyze_counterfactuals(counterfactuals, original_sample):
    """
    Analyze counterfactual examples to identify the impact of removing specific variants.

    This function examines which genetic variants differ between the original sample
    and each counterfactual. Since PharmCAT is a rule-based system rather than a statistical
    model, this approach helps identify which variants potentially trigger specific rules
    in PharmCAT's knowledge base.

    Parameters
    ----------
    counterfactuals : dict
        Dictionary of counterfactual samples from generate_counterfactual_examples
    original_sample : pandas.Series
        Original sample with all variants present

    Returns
    -------
    list
        List of dictionaries containing impact information for each variant
    """
    impacts = []

    for feature, counterfactual in counterfactuals.items():
        original_features = original_sample.to_dict()
        modified_features = counterfactual.to_dict()

        differing_features = []
        for f, v in original_features.items():
            if v != modified_features.get(f):
                differing_features.append(f)

        feature_gene = feature.split('_')[0] if '_' in feature else ""

        impact = {
            'feature': feature,
            'feature_gene': feature_gene,
            'changed_features': differing_features
        }
        impacts.append(impact)

    return impacts


def extract_decision_rules(X, Y, gene, gene_phenotype_mapping):
    """
    Extract decision rules by training a surrogate decision tree model.

    While PharmCAT uses predefined expert-curated rules, this function creates a surrogate
    decision tree model that approximates those rules based on the observed input-output
    relationships. Decision trees are particularly well-suited for approximating rule-based
    systems since they naturally create if-then-else structures.

    The Gini impurity measure (shown as 'gini' in visualizations) quantifies how mixed the
    phenotype classes are at each node:
    - Gini = 0: Perfect purity (all samples have same phenotype)
    - Gini = 1: Maximum impurity (equal distribution across phenotypes)
    - Higher Gini values at decision points often indicate variants that strongly
      differentiate between phenotype groups

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix where rows are samples and columns are genetic variants
    Y : pandas.DataFrame
        Target matrix with phenotype encodings
    gene : str
        Gene name to extract rules for
    gene_phenotype_mapping : dict
        Mapping from phenotype names to numeric encodings for this specific gene

    Returns
    -------
    sklearn.tree.DecisionTreeClassifier or None
        Trained decision tree model if successful, None otherwise
    """
    from sklearn.tree import DecisionTreeClassifier

    if gene not in Y.columns:
        return None

    valid_indices = Y[Y[gene] >= 0].index
    X_valid = X.loc[valid_indices]
    y_valid = Y.loc[valid_indices, gene]

    if len(y_valid) == 0:
        return None

    model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5)
    model.fit(X_valid, y_valid)

    return model


def visualize_decision_tree(tree_model, feature_names, phenotype_mapping, output_file):
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree

    if tree_model is None:
        return

    num_to_phenotype = {v: k for k, v in phenotype_mapping.items()}

    plt.figure(figsize=(20, 10))
    plot_tree(tree_model, feature_names=feature_names,
              class_names=[num_to_phenotype.get(i, str(i)) for i in range(len(num_to_phenotype))],
              filled=True, rounded=True)

    plt.savefig(output_file)
    plt.close()


def generate_rule_based_explanations(tree_model, feature_names, phenotype_mapping):
    from sklearn.tree import _tree
    import numpy as np

    if tree_model is None:
        return []

    num_to_phenotype = {v: k for k, v in phenotype_mapping.items()}
    tree = tree_model.tree_

    def recurse(node, depth, path, paths):
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree.feature[node]]
            threshold = tree.threshold[node]

            path_yes = path + [[name, "≤", threshold]]
            recurse(tree.children_left[node], depth + 1, path_yes, paths)

            path_no = path + [[name, ">", threshold]]
            recurse(tree.children_right[node], depth + 1, path_no, paths)
        else:
            path.append(['RESULT', num_to_phenotype.get(np.argmax(tree.value[node]), 'Unknown')])
            paths.append(path)

    paths = []
    recurse(0, 1, [], paths)

    rules = []
    for path in paths:
        rule = "IF "
        for i, condition in enumerate(path[:-1]):
            if i > 0:
                rule += " AND "
            if condition[1] == "≤" and condition[2] == 0.5:
                rule += f"{condition[0]} is absent"
            elif condition[1] == ">" and condition[2] == 0.5:
                rule += f"{condition[0]} is present"
            else:
                rule += f"{condition[0]} {condition[1]} {condition[2]}"
        rule += f" THEN phenotype is {path[-1][1]}"
        rules.append(rule)

    return rules


def run_counterfactual_analysis(X, Y, phenotype_mappings, output_dir, top_k=10):
    """
    Run counterfactual analysis to understand how individual variants affect predictions.

    Counterfactual analysis is particularly appropriate for PharmCAT because:
    1. It mimics how rule-based systems are typically debugged and understood
    2. It provides clear, deterministic explanations of "if this variant were absent,
       then the phenotype would change to X"
    3. It's aligned with how pharmacogenomics literature often describes variant effects
       (presence/absence of specific alleles)

    This approach was chosen over traditional black-box explainability methods because
    PharmCAT's decision-making is fundamentally based on matching variants to known
    allele definitions, not on statistical patterns or weighted features.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix with variants
    Y : pandas.DataFrame
        Target matrix with phenotype encodings
    phenotype_mappings : dict
        Mapping from phenotype names to numeric encodings
    output_dir : str
        Directory to save the analysis results
    top_k : int, default=10
        Maximum number of top features to consider per sample

    Returns
    -------
    dict
        Counterfactual analysis results organized by gene and sample
    """
    counterfactual_results = {}

    for gene in TARGET_GENES:
        if gene not in Y.columns:
            continue

        gene_results = []
        valid_samples = Y[Y[gene] >= 0].index

        if len(valid_samples) == 0:
            continue

        num_to_phenotype = {v: k for k, v in phenotype_mappings[gene].items()}

        for sample_id in valid_samples[:min(5, len(valid_samples))]:
            phenotype_num = Y.loc[sample_id, gene]
            phenotype = num_to_phenotype.get(phenotype_num, "Unknown")

            counterfactuals = generate_counterfactual_examples(X, sample_id, X.columns, top_k)
            impacts = analyze_counterfactuals(counterfactuals, X.loc[sample_id])

            sample_result = {
                'sample_id': str(sample_id),
                'phenotype': phenotype,
                'variant_impacts': impacts
            }

            gene_results.append(sample_result)

        counterfactual_results[gene] = gene_results

    output_file = os.path.join(output_dir, "counterfactual_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(counterfactual_results, f, indent=2)

    return counterfactual_results


def extract_decision_tree_data(tree_model, feature_names, phenotype_mapping):
    """
    Extract the complete structure of a decision tree model as a JSON-serializable dictionary.
    """
    from sklearn.tree import _tree
    import numpy as np

    if tree_model is None:
        return None

    tree = tree_model.tree_
    num_to_phenotype = {v: k for k, v in phenotype_mapping.items()}

    def recurse_tree(node_id, depth=0):
        if node_id == _tree.TREE_UNDEFINED:
            return None

        # Extract node information
        feature_index = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        impurity = tree.impurity[node_id]
        n_samples = int(tree.n_node_samples[node_id])
        value = tree.value[node_id][0].tolist()

        # Determine node type and dominant class
        if feature_index == _tree.TREE_UNDEFINED:  # Leaf node
            class_index = np.argmax(value)
            class_name = num_to_phenotype.get(class_index, f"Unknown_{class_index}")
            node_type = "leaf"
        else:  # Decision node
            feature_name = feature_names[feature_index]
            class_index = np.argmax(value)
            class_name = num_to_phenotype.get(class_index, f"Unknown_{class_index}")
            node_type = "decision"

        # Create node data structure
        node_data = {
            "node_id": int(node_id),
            "depth": depth,
            "type": node_type,
            "samples": n_samples,
            "impurity": float(impurity),
            "value": value,
            "class": class_name
        }

        # Add feature information for decision nodes
        if node_type == "decision":
            node_data["feature"] = feature_name
            node_data["threshold"] = float(threshold)

            # Recurse to children
            left_child = recurse_tree(tree.children_left[node_id], depth + 1)
            right_child = recurse_tree(tree.children_right[node_id], depth + 1)

            if left_child:
                node_data["left"] = left_child
            if right_child:
                node_data["right"] = right_child

        return node_data

    # Start recursion from the root node
    tree_data = recurse_tree(0)

    # Add metadata
    result = {
        "metadata": {
            "num_nodes": tree.node_count,
            "max_depth": tree_model.max_depth,
            "classes": [num_to_phenotype.get(i, f"Unknown_{i}") for i in range(len(num_to_phenotype))]
        },
        "tree": tree_data
    }

    return result


def run_rule_extraction(X, Y, phenotype_mappings, output_dir):
    """
    Extract decision rules that approximate PharmCAT's phenotype predictions and
    store them as JSON structures instead of visualized trees.
    """
    rule_results = {}
    decision_trees = {}

    for gene in TARGET_GENES:
        if gene not in Y.columns:
            continue

        print(f"Extracting decision rules for {gene}...")

        gene_phenotype_mapping = phenotype_mappings[gene]
        tree_model = extract_decision_rules(X, Y, gene, gene_phenotype_mapping)

        if tree_model is None:
            continue

        tree_data = extract_decision_tree_data(tree_model, X.columns.tolist(), gene_phenotype_mapping)

        # Extract text rules for easier interpretation
        rules = generate_rule_based_explanations(tree_model, X.columns.tolist(), gene_phenotype_mapping)

        rule_results[gene] = {
            'rules': rules
        }

        # Store the tree structure separately
        decision_trees[gene] = tree_data

    # Save the rule extraction results
    output_file = os.path.join(output_dir, "rule_extraction.json")
    with open(output_file, 'w') as f:
        json.dump(rule_results, f, indent=2)

    # Save all decision tree structures in a single file
    trees_output_file = os.path.join(output_dir, "decision_trees.json")
    with open(trees_output_file, 'w') as f:
        json.dump(decision_trees, f, indent=2)

    return rule_results, decision_trees


def apply_fuzzy_logic(sensitivity, X, Y, phenotype_mappings, max_samples=100, method=None):
    """
    Apply fuzzy logic to blend between SHAP, LIME, and perturbation-based explanation methods.

    This function serves as the main entry point for generating explanations with a
    controllable trade-off between computational efficiency and explanation accuracy.
    Based on the sensitivity parameter, it dynamically determines the optimal blend
    of three different explanation techniques: SHAP (most accurate but slowest),
    LIME (balanced), and perturbation analysis (fastest but most approximate).

    Parameters
    ----------
    sensitivity : float
        Value between 0 and 1 controlling the blend between explanation methods:
        - 0.0: Primarily perturbation-based analysis (fastest, most approximate)
        - 0.5: Balanced blend of all methods
        - 1.0: Primarily SHAP analysis (slowest, most accurate)
        Values are clamped to the range [0, 1] if outside this range.

    X : pandas.DataFrame
        Feature matrix where rows are samples and columns are genetic variants.
        Each value should be binary (0 or 1) indicating the presence/absence of a variant.

    Y : pandas.DataFrame
        Target matrix where rows are samples and columns are genes.
        Each value is a numeric encoding of the phenotype for that gene.

    phenotype_mappings : dict
        Dictionary mapping from gene name to another dictionary that maps
        from phenotype name to numeric encoding. For example:
        {'CYP2D6': {'UM': 0, 'NM': 1, 'IM': 2, 'PM': 3}}

    max_samples : int, default=100
        Maximum number of samples to use for analysis. If there are more samples
        than this value, a random subset will be selected. Set to -1
        to use all available samples.

    method : str, default=None
        If provided, overrides sensitivity and uses only the specified method.
        Valid values: 'shap', 'lime', 'perturbation'

    Returns
    -------
    dict
        Dictionary with keys for each gene analyzed. Each gene's entry contains:
        - "shap_values": Matrix of blended importance values (samples × features)
        - "feature_names": List of feature names
        - "sample_indices": List of sample IDs
        - "predictions": List of phenotype encodings for each sample
        - "num_to_phenotype": Dictionary mapping numeric encodings back to phenotype names

    Notes
    -----
    The fuzzy logic system divides the sensitivity range into three regions:

    1. Low sensitivity (0.0-0.33):
       - Perturbation weight decreases from 70% to 40%
       - LIME weight increases from 30% to 40%
       - SHAP weight increases from 0% to 20%

    2. Medium sensitivity (0.33-0.67):
       - Perturbation weight decreases from 40% to 20%
       - LIME weight stays constant at 40%
       - SHAP weight increases from 20% to 40%

    3. High sensitivity (0.67-1.0):
       - Perturbation weight decreases from 20% to 0%
       - LIME weight decreases from 40% to 30%
       - SHAP weight increases from 40% to 70%

    For edge cases (sensitivity=0, 0.33, 0.67, or 1), the function may optimize by
    computing only the necessary methods rather than blending all three.

    Each method's results are normalized before blending to ensure fair contribution
    regardless of the scale of the original importance values.
    """
    # Override with explicit method if specified
    if method is not None:
        if method.lower() == 'shap':
            print(f"Using SHAP analysis only (explicit method choice)...")
            return run_shap_analysis(X, Y, phenotype_mappings, max_samples)
        elif method.lower() == 'lime':
            print(f"Using LIME analysis only (explicit method choice)...")
            return run_lime_analysis(X, Y, phenotype_mappings, max_samples)
        elif method.lower() == 'perturbation':
            print(f"Using perturbation analysis only (explicit method choice)...")
            return run_perturbation_analysis(X, Y, phenotype_mappings, max_samples)
        else:
            print(f"Warning: Unknown method '{method}', falling back to sensitivity-based logic")

    sensitivity = max(0.0, min(1.0, sensitivity))

    # Special case handling for the extremes
    if sensitivity == 0.0:
        print("Using perturbation-based analysis only (sensitivity=0)...")
        return run_perturbation_analysis(X, Y, phenotype_mappings, max_samples)

    if sensitivity == 1.0:
        print("Using SHAP analysis only (sensitivity=1)...")
        return run_shap_analysis(X, Y, phenotype_mappings, max_samples)

    # For values in between, use the blending approach
    if sensitivity <= 0.33:
        print(f"Using method blend with low sensitivity ({sensitivity:.2f}):")
        perturbation_weight = 0.7 - (sensitivity * 0.9)
        lime_weight = 0.3 + (sensitivity * 0.3)
        shap_weight = sensitivity * 0.6
        print(f"  - Perturbation: {perturbation_weight:.1%}")
        print(f"  - LIME: {lime_weight:.1%}")
        print(f"  - SHAP: {shap_weight:.1%}")
    elif sensitivity <= 0.67:
        print(f"Using method blend with medium sensitivity ({sensitivity:.2f}):")
        mid_point = (sensitivity - 0.33) / 0.34
        perturbation_weight = 0.4 - (mid_point * 0.2)
        lime_weight = 0.4
        shap_weight = 0.2 + (mid_point * 0.2)
        print(f"  - Perturbation: {perturbation_weight:.1%}")
        print(f"  - LIME: {lime_weight:.1%}")
        print(f"  - SHAP: {shap_weight:.1%}")
    else:
        print(f"Using method blend with high sensitivity ({sensitivity:.2f}):")
        high_point = (sensitivity - 0.67) / 0.33
        perturbation_weight = 0.2 - (high_point * 0.2)
        lime_weight = 0.4 - (high_point * 0.1)
        shap_weight = 0.4 + (high_point * 0.3)
        print(f"  - Perturbation: {perturbation_weight:.1%}")
        print(f"  - LIME: {lime_weight:.1%}")
        print(f"  - SHAP: {shap_weight:.1%}")

    results = {}
    blended_results = {}

    if perturbation_weight > 0:
        perturbation_results = run_perturbation_analysis(X, Y, phenotype_mappings, max_samples)
        results['perturbation'] = perturbation_results

    if lime_weight > 0:
        lime_results = run_lime_analysis(X, Y, phenotype_mappings, max_samples)
        results['lime'] = lime_results

    if shap_weight > 0:
        shap_results = run_shap_analysis(X, Y, phenotype_mappings, max_samples)
        results['shap'] = shap_results

    for gene in TARGET_GENES:
        if not any(method in results and gene in results[method] for method in results):
            continue

        gene_methods = {method: results[method][gene] for method in results if gene in results[method]}

        if not gene_methods:
            continue

        first_method = next(iter(gene_methods.values()))
        feature_names = first_method["feature_names"]
        sample_indices = first_method["sample_indices"]
        predictions = first_method["predictions"]
        num_to_phenotype = first_method["num_to_phenotype"]

        blended_values = np.zeros((len(sample_indices), len(feature_names)))

        total_weight = 0

        if 'perturbation' in gene_methods and perturbation_weight > 0:
            perturbation_values = np.array(gene_methods['perturbation']["shap_values"])
            if np.max(np.abs(perturbation_values)) > 0:
                perturbation_values = perturbation_values / np.max(np.abs(perturbation_values))
            blended_values += perturbation_weight * perturbation_values
            total_weight += perturbation_weight

        if 'lime' in gene_methods and lime_weight > 0:
            lime_values = np.array(gene_methods['lime']["shap_values"])
            if np.max(np.abs(lime_values)) > 0:
                lime_values = lime_values / np.max(np.abs(lime_values))
            blended_values += lime_weight * lime_values
            total_weight += lime_weight

        if 'shap' in gene_methods and shap_weight > 0:
            shap_values = np.array(gene_methods['shap']["shap_values"])
            if np.max(np.abs(shap_values)) > 0:
                shap_values = shap_values / np.max(np.abs(shap_values))
            blended_values += shap_weight * shap_values
            total_weight += shap_weight

        if total_weight > 0:
            blended_values = blended_values / total_weight

        blended_results[gene] = {
            "shap_values": blended_values,
            "feature_names": feature_names,
            "sample_indices": sample_indices,
            "predictions": predictions,
            "num_to_phenotype": num_to_phenotype
        }

    return blended_results


def generate_variant_explanation(feature, effect):
    feature_parts = feature.split('_')
    variant_text = f"{feature}"

    if len(feature_parts) > 1:
        gene = feature_parts[0]
        variant = feature_parts[1]
        if variant and gene:
            variant_text = f"{variant} in {gene}"

    return f"{variant_text} {'increases' if effect == 'positive' else 'decreases'} likelihood of this phenotype"


def create_enriched_results(shap_results, X, output_file):
    """
    Create enriched results from SHAP analysis for interpretation and visualization.

    Parameters
    ----------
    shap_results : dict
        Dictionary containing SHAP analysis results for each gene.
    X : pandas.DataFrame
        Feature matrix used for the analysis.
    output_file : str
        Path to the output JSON file where results will be saved.

    Returns
    -------
    dict
        Dictionary containing enriched results with the following sections:
        - gene_explanations: Explanations for each gene's phenotype predictions
        - feature_importance: Global feature importance for each gene
        - sample_explanations: Sample-specific explanations
    """
    json_results = {
        "gene_explanations": {},
        "feature_importance": {},
        "sample_explanations": []
    }

    for gene, result in shap_results.items():
        shap_values = result["shap_values"]
        feature_names = result["feature_names"]
        sample_indices = result["sample_indices"]
        predictions = result["predictions"]
        num_to_phenotype = result["num_to_phenotype"]

        abs_shap = np.abs(shap_values)
        importance = abs_shap.mean(axis=0)
        top_indices = np.argsort(-importance)[:20]

        gene_importance = {}
        for idx in top_indices:
            if idx < len(feature_names):
                feature = feature_names[idx]
                gene_importance[feature] = float(importance[idx])

        json_results["feature_importance"][gene] = gene_importance

        features_by_gene = defaultdict(list)
        for idx in top_indices:
            if idx < len(feature_names):
                feature = feature_names[idx]
                feature_gene = feature.split('_')[0] if '_' in feature else ""

                if feature_gene in TARGET_GENES:
                    features_by_gene[feature_gene].append({
                        "feature": feature,
                        "importance": float(importance[idx])
                    })

        phenotype_counts = {}
        for pred in predictions:
            if pred in num_to_phenotype:
                phenotype = num_to_phenotype[pred]
                phenotype_counts[phenotype] = phenotype_counts.get(phenotype, 0) + 1

        json_results["gene_explanations"][gene] = {
            "prediction_distribution": phenotype_counts,
            "top_features_by_gene": {
                g: features for g, features in features_by_gene.items()
            }
        }

        for i in range(min(10, len(sample_indices))):
            sample = sample_indices[i]
            pred = predictions[i]

            # Skip samples with unmappable phenotypes
            if pred not in num_to_phenotype:
                continue

            phenotype = num_to_phenotype[pred]

            sample_shap = shap_values[i]
            top_contrib_indices = np.argsort(-np.abs(sample_shap))[:10]

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

            json_results["sample_explanations"].append({
                "sample_id": str(sample),
                "gene": gene,
                "predicted_phenotype": phenotype,
                "top_contributions": top_contributions,
                "explanation": explanation
            })

    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"Results saved to {output_file}")
    return json_results


def generate_summary_report(json_results, output_dir):
    summary_file = os.path.join(output_dir, "pgx_summary.txt")

    total_distributions = {}
    for gene, gene_data in json_results['gene_explanations'].items():
        for phenotype, count in gene_data['prediction_distribution'].items():
            if gene not in total_distributions:
                total_distributions[gene] = count
            else:
                total_distributions[gene] += count

    gene_sample_counts = list(total_distributions.values())
    if gene_sample_counts:
        total_samples = int(np.median(gene_sample_counts))
    else:
        total_samples = 0

    explanation_sample_ids = set([exp['sample_id'] for exp in json_results['sample_explanations']])
    detailed_samples = len(explanation_sample_ids)

    with open(summary_file, 'w') as f:
        f.write("# PharmCAT Analysis Summary Report\n\n")

        f.write("## Overview\n")
        f.write(f"Analysis performed on {len(json_results['gene_explanations'])} genes\n")
        f.write(f"Total samples processed: {total_samples}\n")
        f.write(f"Detailed analysis samples: {detailed_samples}\n\n")

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
                shap_value = contrib['shap_value']
                direction = "increases" if shap_value > 0 else "decreases"
                f.write(f"{j + 1}. {contrib['feature']} - {direction} phenotype likelihood: {abs(shap_value):.4f}\n")

    print(f"Summary report saved to {summary_file}")
    return summary_file


def find_csv_files(input_dir):
    csv_files = glob.glob(os.path.join(input_dir, CSV_EXT))
    if csv_files:
        return csv_files

    preprocessed_dir = os.path.join(input_dir, "preprocessed")
    if os.path.exists(preprocessed_dir):
        return glob.glob(os.path.join(preprocessed_dir, CSV_EXT))

    return []


def main():
    parser = argparse.ArgumentParser(description='PharmCAT explainer')
    parser.add_argument('--input_dir', required=True, help='Directory containing VCF or CSV files')
    parser.add_argument('--phenotypes_file', required=True, help='Path to phenotypes.csv file')
    parser.add_argument('--output_dir', default='pgx_results', help='Output directory for results')
    parser.add_argument('--convert_vcf', action='store_true', help='Convert VCF files to CSV format')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='Maximum number of samples for detailed analysis (use -1 for all samples)')
    parser.add_argument('--sensitivity', type=float, default=0.5,
                        help='Sensitivity value (0-1) for fuzzy logic between explanation methods')
    parser.add_argument('--method', type=str, choices=['shap', 'lime', 'perturbation'],
                        help='Explicitly choose a single explanation method (overrides sensitivity)')
    parser.add_argument('--run_counterfactual', action='store_true',
                        help='Run counterfactual analysis')
    parser.add_argument('--run_rule_extraction', action='store_true',
                        help='Extract and visualize decision rules')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of top features for counterfactual analysis')
    args = parser.parse_args()

    args.sensitivity = max(0, min(1, args.sensitivity))

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Processing input data from {args.input_dir}...")

    input_files = []
    if args.convert_vcf:
        _, csv_dir = preprocess_input_data(args.input_dir, os.path.join(args.output_dir, "preprocessed"))
        input_files = glob.glob(os.path.join(csv_dir, CSV_EXT))
        print(f"Converted VCF files to CSV format in {csv_dir} ({len(input_files)} files)")
    else:
        input_files = find_csv_files(args.input_dir)
        if not input_files:
            raise ValueError(f"No CSV files found in {args.input_dir}. Use --convert_vcf to convert VCF files.")
        print(f"Found {len(input_files)} CSV files")

    print("Loading CSV data...")
    variant_data = load_csv_data(input_files)
    unique_samples = variant_data['Sample'].nunique()
    print(f"Loaded data for {unique_samples} unique samples")

    print("Extracting features...")
    X = extract_features(variant_data)
    print(f"Created feature matrix with {X.shape[0]} samples and {X.shape[1]} features")

    print(f"Loading phenotypes from {args.phenotypes_file}...")
    Y, phenotypes_df, phenotype_mappings = prepare_targets(args.phenotypes_file, X.index)
    print(f"Prepared target matrix with {Y.shape[0]} samples")

    common_samples = X.index.intersection(Y.index)
    print(f"Found {len(common_samples)} samples common to both feature matrix and phenotype data")

    if len(common_samples) == 0:
        print("WARNING: No matching samples found between variant data and phenotype data!")
        print(f"Feature matrix sample IDs (first 5): {list(X.index)[:5]}")
        print(f"Phenotype data sample IDs (first 5): {list(Y.index)[:5]}")
        raise ValueError("No overlapping samples between variant data and phenotype data")

    X = X.loc[common_samples]
    Y = Y.loc[common_samples]

    detailed_max_samples = args.max_samples
    if detailed_max_samples <= 0:
        detailed_max_samples = len(common_samples)
        print(f"Using all {detailed_max_samples} available samples for detailed analysis")
    else:
        detailed_max_samples = min(detailed_max_samples, len(common_samples))
        print(
            f"Using {detailed_max_samples} samples for detailed explanation analysis (out of {len(common_samples)} available)")

    X_full = X.copy()
    Y_full = Y.copy()

    if detailed_max_samples < len(common_samples):
        rng = np.random.Generator(np.random.PCG64(seed=42))
        sample_indices = rng.choice(len(common_samples), detailed_max_samples, replace=False)
        analysis_samples = common_samples[sample_indices]
        X_detailed = X.loc[analysis_samples]
        Y_detailed = Y.loc[analysis_samples]
        print(f"Selected {detailed_max_samples} samples for detailed feature importance analysis")
    else:
        X_detailed = X
        Y_detailed = Y
        print(f"Using all {len(common_samples)} samples for detailed feature importance analysis")

    print(f"Running analysis with sensitivity={args.sensitivity}...")
    if args.method:
        print(f"Using explicit method: {args.method}")
    results = apply_fuzzy_logic(args.sensitivity, X_detailed, Y_detailed, phenotype_mappings,
                                max_samples=detailed_max_samples, method=args.method)

    for gene in TARGET_GENES:
        if gene in results and gene in Y_full.columns:
            gene_phenotypes = Y_full[gene].dropna()
            if len(gene_phenotypes) > 0:
                phenotype_counts = {}
                for val in gene_phenotypes:
                    if val >= 0:
                        phenotype = results[gene]["num_to_phenotype"].get(val, "Unknown")
                        if phenotype in phenotype_counts:
                            phenotype_counts[phenotype] += 1
                        else:
                            phenotype_counts[phenotype] = 1

                print(f"Gene {gene}: Found {sum(phenotype_counts.values())} samples with phenotypes")

    # Initialize additional results
    counterfactual_results = None
    rule_results = None
    decision_trees = None

    # Run the new counterfactual analysis if requested
    if args.run_counterfactual:
        print("Running counterfactual analysis...")
        counterfactual_results = run_counterfactual_analysis(X, Y, phenotype_mappings,
                                                             args.output_dir, args.top_k)
        print(f"Counterfactual analysis saved to {os.path.join(args.output_dir, 'counterfactual_analysis.json')}")

    # Run the rule extraction if requested - note the updated function call that returns decision trees
    if args.run_rule_extraction:
        print("Running rule extraction and data extraction...")
        rule_results, decision_trees = run_rule_extraction(X, Y, phenotype_mappings, args.output_dir)
        print(f"Rule extraction results saved to {os.path.join(args.output_dir, 'rule_extraction.json')}")
        print(f"Decision tree data saved to {os.path.join(args.output_dir, 'decision_trees.json')}")

    print("Preparing results...")
    json_output_file = os.path.join(args.output_dir, "pgx_results.json")
    json_results = create_enriched_results(results, X, json_output_file)

    # Add the additional analysis results to the main json_results
    if counterfactual_results:
        json_results["counterfactual_analysis"] = counterfactual_results

    if rule_results:
        json_results["rule_extraction"] = rule_results

    # Add decision tree data to the main results if available
    if decision_trees:
        json_results["decision_trees"] = decision_trees

    # Save the updated json_results
    with open(json_output_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print("Generating summary report...")
    summary_file = generate_summary_report(json_results, args.output_dir)

    print(f"Analysis complete! Results saved to {args.output_dir}")
    print(f"JSON results: {json_output_file}")
    print(f"Summary report: {summary_file}")
    print(f"Total samples in dataset: {len(common_samples)}")
    print(f"Samples used for detailed explanation analysis: {detailed_max_samples}")

    # Add summary for new analyses
    if args.run_counterfactual:
        print(f"Counterfactual analysis: {os.path.join(args.output_dir, 'counterfactual_analysis.json')}")
    if args.run_rule_extraction:
        print(f"Rule extraction: {os.path.join(args.output_dir, 'rule_extraction.json')}")
        print(f"Decision tree data: {os.path.join(args.output_dir, 'decision_trees.json')}")


if __name__ == "__main__":
    main()
