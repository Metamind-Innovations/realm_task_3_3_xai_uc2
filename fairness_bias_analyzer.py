import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


def parse_population_codes(population_codes_file):
    with open(population_codes_file, 'r') as f:
        content = f.read()

    population_info = {}

    table_pattern = r'\|\s+(\w+)\s+\|\s+([^|]+)\|\s+(\w+)\s+\|'
    matches = re.findall(table_pattern, content)

    for match in matches:
        pop_code = match[0].strip()
        description = match[1].strip()
        super_pop = match[2].strip()

        population_info[pop_code] = {
            'description': description,
            'super_population': super_pop
        }

    return population_info


def load_cohort_data(cohort_file):
    df = pd.read_csv(cohort_file)
    if 'Sample ID' not in df.columns:
        if len(df.columns) > 0:
            df = df.rename(columns={df.columns[0]: 'Sample ID'})
    return df


def load_phenotype_data(phenotypes_file):
    df = pd.read_csv(phenotypes_file)
    if 'Sample ID' not in df.columns and 'Sample' in df.columns:
        df = df.rename(columns={'Sample': 'Sample ID'})
    elif 'Sample ID' not in df.columns and len(df.columns) > 0:
        df = df.rename(columns={df.columns[0]: 'Sample ID'})
    return df


def load_groundtruth_data(groundtruth_file):
    df = pd.read_csv(groundtruth_file)
    if 'Sample ID' not in df.columns and 'Sample' in df.columns:
        df = df.rename(columns={'Sample': 'Sample ID'})
    elif 'Sample ID' not in df.columns and len(df.columns) > 0:
        df = df.rename(columns={df.columns[0]: 'Sample ID'})
    return df


def merge_data(cohort_df, phenotype_df, groundtruth_df=None):
    merged_df = pd.merge(
        phenotype_df,
        cohort_df,
        on='Sample ID',
        how='left'
    )

    if groundtruth_df is not None:
        merged_df = pd.merge(
            merged_df,
            groundtruth_df,
            on='Sample ID',
            how='left',
            suffixes=('', '_groundtruth')
        )

    return merged_df


def calculate_phenotype_distributions(data, target_genes):
    distributions = {}

    for gene in target_genes:
        if gene in data.columns:
            overall_dist = data[gene].value_counts(normalize=True).to_dict()
            distributions[gene] = {
                'overall': overall_dist
            }

            for demographic in ['Sex', 'Population', 'Superpopulation']:
                if demographic in data.columns:
                    distributions[gene][demographic] = {}
                    for group in data[demographic].dropna().unique():
                        group_data = data[data[demographic] == group]
                        if len(group_data) > 0:
                            group_dist = group_data[gene].value_counts(normalize=True).to_dict()
                            distributions[gene][demographic][group] = {
                                'distribution': group_dist,
                                'sample_size': len(group_data)
                            }

    return distributions


def calculate_equalized_odds(data, target_genes, demographics):
    """
    Calculate a simplified Equalized Odds metric for each gene and demographic group.
    Equalized Odds measures whether error rates are similar across different demographic groups.
    """
    metrics = {}

    for gene in target_genes:
        gene_groundtruth = f"{gene}_groundtruth"

        if gene not in data.columns or gene_groundtruth not in data.columns:
            continue

        gene_metrics = {}

        for demographic in demographics:
            if demographic not in data.columns:
                continue

            phenotype_metrics = {}
            all_phenotypes = set(data[gene].dropna().unique()) | set(data[gene_groundtruth].dropna().unique())

            for phenotype in all_phenotypes:
                group_error_rates = {}
                groups = data[demographic].dropna().unique()

                for group in groups:
                    group_data = data[data[demographic] == group]

                    if len(group_data) < 5:  # Skip groups with too few samples
                        continue

                    # Calculate true positive rate (TPR) and false positive rate (FPR)
                    true_pos = sum((group_data[gene] == phenotype) & (group_data[gene_groundtruth] == phenotype))
                    false_pos = sum((group_data[gene] == phenotype) & (group_data[gene_groundtruth] != phenotype))

                    actual_pos = sum(group_data[gene_groundtruth] == phenotype)
                    actual_neg = sum(group_data[gene_groundtruth] != phenotype)

                    tpr = true_pos / actual_pos if actual_pos > 0 else None
                    fpr = false_pos / actual_neg if actual_neg > 0 else None

                    group_error_rates[str(group)] = {
                        "true_positive_rate": float(tpr) if tpr is not None else None,
                        "false_positive_rate": float(fpr) if fpr is not None else None
                    }

                # Only include phenotypes with valid metrics for multiple groups
                if len(group_error_rates) >= 2:
                    # Extract valid rates
                    valid_tpr = [rates["true_positive_rate"] for rates in group_error_rates.values()
                                 if rates["true_positive_rate"] is not None]
                    valid_fpr = [rates["false_positive_rate"] for rates in group_error_rates.values()
                                 if rates["false_positive_rate"] is not None]

                    # Calculate disparities
                    tpr_disparity = max(valid_tpr) - min(valid_tpr) if len(valid_tpr) >= 2 else None
                    fpr_disparity = max(valid_fpr) - min(valid_fpr) if len(valid_fpr) >= 2 else None

                    phenotype_metrics[str(phenotype)] = {
                        "error_rates_by_group": group_error_rates,
                        "disparity": {
                            "true_positive_rate": float(tpr_disparity) if tpr_disparity is not None else None,
                            "false_positive_rate": float(fpr_disparity) if fpr_disparity is not None else None
                        }
                    }

            if phenotype_metrics:
                gene_metrics[demographic] = phenotype_metrics

        if gene_metrics:
            metrics[gene] = gene_metrics

    return metrics


def calculate_demographic_parity(data, target_genes, demographics):
    """
    Calculate a simplified Demographic Parity metric for each gene and demographic group.
    Demographic Parity ensures that prediction rates are similar across different demographic groups.
    """
    metrics = {}

    for gene in target_genes:
        if gene not in data.columns:
            continue

        gene_metrics = {}

        for demographic in demographics:
            if demographic not in data.columns:
                continue

            phenotype_metrics = {}
            phenotypes = data[gene].dropna().unique()

            for phenotype in phenotypes:
                group_rates = {}
                overall_rate = (data[gene] == phenotype).mean()
                groups = data[demographic].dropna().unique()

                for group in groups:
                    group_data = data[data[demographic] == group]

                    if len(group_data) < 5:  # Skip groups with too few samples
                        continue

                    rate = (group_data[gene] == phenotype).mean()
                    group_rates[str(group)] = float(rate)

                # Only include phenotypes with valid rates for multiple groups
                if len(group_rates) >= 2:
                    max_rate = max(group_rates.values())
                    min_rate = min(group_rates.values())

                    phenotype_metrics[str(phenotype)] = {
                        "overall_rate": float(overall_rate),
                        "prediction_rates_by_group": group_rates,
                        "disparity": {
                            "maximum_difference": float(max_rate - min_rate),
                            "min_to_max_ratio": float(min_rate / max_rate) if max_rate > 0 else 1.0
                        }
                    }

            if phenotype_metrics:
                gene_metrics[demographic] = phenotype_metrics

        if gene_metrics:
            metrics[gene] = gene_metrics

    return metrics


def analyze_fairness_bias(cohort_file, phenotypes_file, population_codes_file, groundtruth_file, output_file):
    target_genes = ["CYP2B6", "CYP2C9", "CYP2C19", "CYP3A5", "SLCO1B1", "TPMT", "DPYD"]
    demographics = ['Sex', 'Population', 'Superpopulation']

    print(f"Loading population codes from {population_codes_file}")
    population_info = parse_population_codes(population_codes_file)

    print(f"Loading cohort data from {cohort_file}")
    cohort_df = load_cohort_data(cohort_file)

    print(f"Loading phenotype data from {phenotypes_file}")
    phenotype_df = load_phenotype_data(phenotypes_file)

    print(f"Loading ground truth data from {groundtruth_file}")
    groundtruth_df = load_groundtruth_data(groundtruth_file)

    print("Merging datasets")
    merged_df = merge_data(cohort_df, phenotype_df, groundtruth_df)

    print("Calculating phenotype distributions")
    distributions = calculate_phenotype_distributions(merged_df, target_genes)

    print("Calculating Equalized Odds metrics")
    equalized_odds_metrics = calculate_equalized_odds(merged_df, target_genes, demographics)

    print("Calculating Demographic Parity metrics")
    demographic_parity_metrics = calculate_demographic_parity(merged_df, target_genes, demographics)

    # Prepare final results
    results = {
        'metadata': {
            'sample_count': len(merged_df),
            'genes_analyzed': target_genes,
            'demographics': {
                'sex_distribution': merged_df['Sex'].value_counts().to_dict() if 'Sex' in merged_df.columns else {},
                'population_distribution': merged_df[
                    'Population'].value_counts().to_dict() if 'Population' in merged_df.columns else {},
                'superpopulation_distribution': merged_df[
                    'Superpopulation'].value_counts().to_dict() if 'Superpopulation' in merged_df.columns else {}
            }
        },
        'phenotype_distributions': distributions,
        'equalized_odds_metrics': equalized_odds_metrics,
        'demographic_parity_metrics': demographic_parity_metrics
    }

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"Fairness analysis completed. Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze fairness and bias in PharmCAT phenotype predictions')
    parser.add_argument('--population-codes', required=True, help='Path to population codes markdown file')
    parser.add_argument('--cohort', required=True, help='Path to cohort CSV file')
    parser.add_argument('--phenotypes', required=True, help='Path to phenotypes CSV file')
    parser.add_argument('--groundtruth', required=True, help='Path to ground truth phenotypes CSV file')
    parser.add_argument('--output', default='fairness_analysis.json', help='Output JSON file path')

    args = parser.parse_args()

    analyze_fairness_bias(args.cohort, args.phenotypes, args.population_codes, args.groundtruth, args.output)


if __name__ == "__main__":
    main()
