import argparse
import json
import pandas as pd
import numpy as np
import re
from pathlib import Path

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
    if 'Sample ID' not in df.columns and len(df.columns) > 0:
        df = df.rename(columns={df.columns[0]: 'Sample ID'})
    return df


def merge_data(cohort_df, phenotype_df):
    merged_df = pd.merge(
        phenotype_df,
        cohort_df,
        on='Sample ID',
        how='left'
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


def calculate_fairness_metrics(data, target_genes, population_info):
    metrics = {}

    for gene in target_genes:
        if gene not in data.columns:
            continue

        gene_metrics = {
            'overall': {
                'sample_count': int(len(data)),  # Convert numpy.int64 to Python int
                'phenotype_distribution': {k: int(v) for k, v in data[gene].value_counts().to_dict().items()}  # Convert values to Python int
            },
            'demographic_disparity': {}
        }

        for demographic in ['Sex', 'Population', 'Superpopulation']:
            if demographic not in data.columns:
                continue

            groups = data[demographic].dropna().unique()
            if len(groups) < 2:
                continue

            group_metrics = {}
            phenotypes = data[gene].dropna().unique()

            for phenotype in phenotypes:
                overall_rate = float((data[gene] == phenotype).mean())  # Convert numpy.float64 to Python float
                phenotype_metrics = {
                    'overall_rate': overall_rate,
                    'group_rates': {},
                    'disparities': []
                }

                max_diff = 0.0
                min_rate = 1.0
                max_rate = 0.0

                for group in groups:
                    group_data = data[data[demographic] == group]
                    group_size = len(group_data)

                    if group_size < 5:
                        continue

                    group_rate = float((group_data[gene] == phenotype).mean())  # Convert numpy.float64 to Python float
                    phenotype_metrics['group_rates'][str(group)] = {
                        'rate': group_rate,
                        'sample_size': int(group_size)  # Convert numpy.int64 to Python int
                    }

                    diff = abs(group_rate - overall_rate)
                    if diff > max_diff:
                        max_diff = diff

                    if group_rate < min_rate:
                        min_rate = group_rate
                    if group_rate > max_rate:
                        max_rate = group_rate

                if len(phenotype_metrics['group_rates']) < 2:
                    continue

                disparity_ratio = 0.0
                if max_rate > 0:
                    disparity_ratio = min_rate / max_rate

                # Calculate disparities between groups
                groups_list = list(phenotype_metrics['group_rates'].keys())
                for i in range(len(groups_list)):
                    for j in range(i + 1, len(groups_list)):
                        group1 = groups_list[i]
                        group2 = groups_list[j]
                        rate1 = phenotype_metrics['group_rates'][group1]['rate']
                        rate2 = phenotype_metrics['group_rates'][group2]['rate']

                        diff = abs(rate1 - rate2)
                        ratio = min(rate1, rate2) / max(rate1, rate2) if max(rate1, rate2) > 0 else 1.0

                        if diff >= 0.1 or ratio <= 0.8:
                            disparity = {
                                'group1': group1,
                                'group2': group2,
                                'rate1': rate1,
                                'rate2': rate2,
                                'difference': diff,
                                'ratio': ratio,
                                'is_significant': bool(diff >= 0.2 or ratio <= 0.5)  # Convert numpy.bool_ to Python bool
                            }

                            phenotype_metrics['disparities'].append(disparity)

                phenotype_metrics['max_disparity'] = float(max_diff)  # Convert numpy.float64 to Python float
                phenotype_metrics['disparity_ratio'] = float(disparity_ratio)  # Convert numpy.float64 to Python float
                phenotype_metrics['has_disparity'] = bool(max_diff >= 0.2 or disparity_ratio <= 0.5)  # Convert numpy.bool_ to Python bool

                group_metrics[str(phenotype)] = phenotype_metrics

            gene_metrics['demographic_disparity'][demographic] = group_metrics

        # Add genetic context for population-based differences
        if 'demographic_disparity' in gene_metrics and 'Population' in gene_metrics['demographic_disparity']:
            gene_metrics['genetic_context'] = {
                'note': "Population differences in pharmacogenes often reflect natural genetic variation rather than bias",
                'expected_variation': True
            }

        metrics[gene] = gene_metrics

    return metrics


def generate_fairness_summary(fairness_metrics, population_info):
    summary = {
        'overall_assessment': {},
        'demographic_findings': {},
        'recommendations': []
    }

    # Overall assessment
    total_disparities = 0
    significant_disparities = 0
    genes_with_disparities = set()

    for gene, metrics in fairness_metrics.items():
        for demographic, demo_metrics in metrics.get('demographic_disparity', {}).items():
            for phenotype, pheno_metrics in demo_metrics.items():
                if pheno_metrics['has_disparity']:
                    total_disparities += 1

                    if any(disp['is_significant'] for disp in pheno_metrics['disparities']):
                        significant_disparities += 1
                        genes_with_disparities.add(gene)

    if significant_disparities == 0:
        fairness_score = 100
        fairness_rating = "Excellent"
    elif significant_disparities <= 2:
        fairness_score = 90
        fairness_rating = "Good"
    elif significant_disparities <= 5:
        fairness_score = 75
        fairness_rating = "Fair"
    else:
        fairness_score = 60
        fairness_rating = "Concerning"

    summary['overall_assessment'] = {
        'fairness_score': fairness_score,
        'fairness_rating': fairness_rating,
        'total_disparities': total_disparities,
        'significant_disparities': significant_disparities,
        'genes_with_disparities': list(genes_with_disparities)
    }

    # Demographic findings
    for demographic in ['Sex', 'Population', 'Superpopulation']:
        demo_findings = []

        for gene, metrics in fairness_metrics.items():
            if demographic in metrics.get('demographic_disparity', {}):
                for phenotype, pheno_metrics in metrics['demographic_disparity'][demographic].items():
                    if pheno_metrics['has_disparity']:
                        finding = {
                            'gene': gene,
                            'phenotype': phenotype,
                            'max_disparity': pheno_metrics['max_disparity'],
                            'disparity_ratio': pheno_metrics['disparity_ratio'],
                            'group_disparities': []
                        }

                        for disparity in pheno_metrics['disparities']:
                            if disparity['is_significant']:
                                finding['group_disparities'].append(disparity)

                        if finding['group_disparities']:
                            demo_findings.append(finding)

        if demo_findings:
            summary['demographic_findings'][demographic] = demo_findings

    # Generate recommendations
    if significant_disparities == 0:
        summary['recommendations'].append(
            "No significant fairness issues detected. Continue monitoring with larger datasets."
        )
    else:
        if 'Population' in summary['demographic_findings'] or 'Superpopulation' in summary['demographic_findings']:
            summary['recommendations'].append(
                "Some population differences were detected. Consider whether these represent bias or expected genetic variation in pharmacogenes."
            )

        if 'Sex' in summary['demographic_findings']:
            summary['recommendations'].append(
                "Investigate sex-based disparities in phenotype predictions, which may indicate potential bias."
            )

        if significant_disparities > 5:
            summary['recommendations'].append(
                "Consider rebalancing training data or implementing fairness constraints in the model."
            )

    # Add genetic context
    summary['genetic_context'] = {
        'note': "Pharmacogenomic differences between populations often reflect natural genetic variation rather than algorithmic bias.",
        'implications': "Some disparities identified may represent actual biological differences rather than model bias."
    }

    return summary


def analyze_fairness_bias(cohort_file, phenotypes_file, population_codes_file, output_file):
    target_genes = ["CYP2B6", "CYP2C9", "CYP2C19", "CYP3A5", "SLCO1B1", "TPMT", "DPYD"]

    print(f"Loading population codes from {population_codes_file}")
    population_info = parse_population_codes(population_codes_file)

    print(f"Loading cohort data from {cohort_file}")
    cohort_df = load_cohort_data(cohort_file)

    print(f"Loading phenotype data from {phenotypes_file}")
    phenotype_df = load_phenotype_data(phenotypes_file)

    print("Merging datasets")
    merged_df = merge_data(cohort_df, phenotype_df)

    print("Calculating phenotype distributions")
    distributions = calculate_phenotype_distributions(merged_df, target_genes)

    print("Calculating fairness metrics")
    fairness_metrics = calculate_fairness_metrics(merged_df, target_genes, population_info)

    print("Generating fairness summary")
    fairness_summary = generate_fairness_summary(fairness_metrics, population_info)

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
        'fairness_metrics': fairness_metrics,
        'fairness_summary': fairness_summary
    }

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Fairness analysis completed. Results saved to {output_file}")

    # Print summary
    print("\nFairness Analysis Summary:")
    print(
        f"Overall Fairness Score: {fairness_summary['overall_assessment']['fairness_score']}/100 ({fairness_summary['overall_assessment']['fairness_rating']})")
    print(f"Total disparities detected: {fairness_summary['overall_assessment']['total_disparities']}")
    print(f"Significant disparities: {fairness_summary['overall_assessment']['significant_disparities']}")

    if fairness_summary['recommendations']:
        print("\nRecommendations:")
        for i, rec in enumerate(fairness_summary['recommendations'], 1):
            print(f"{i}. {rec}")


def main():
    parser = argparse.ArgumentParser(description='Analyze fairness and bias in PharmCAT phenotype predictions')
    parser.add_argument('--population-codes', required=True, help='Path to population codes markdown file')
    parser.add_argument('--cohort', required=True, help='Path to cohort CSV file')
    parser.add_argument('--phenotypes', required=True, help='Path to phenotypes CSV file')
    parser.add_argument('--output', default='fairness_analysis.json', help='Output JSON file path')

    args = parser.parse_args()

    analyze_fairness_bias(args.cohort, args.phenotypes, args.population_codes, args.output)


if __name__ == "__main__":
    main()
