import argparse
import json
import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import defaultdict


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
    Calculate Equalized Odds metric for each gene and demographic group.
    Equalized Odds measures if error rates are similar across different demographic groups.
    """
    equalized_odds_metrics = {}

    for gene in target_genes:
        gene_groundtruth = f"{gene}_groundtruth"

        if gene not in data.columns or gene_groundtruth not in data.columns:
            continue

        gene_metrics = {}

        # Get all unique phenotype values
        all_phenotypes = set(data[gene].dropna().unique()) | set(data[gene_groundtruth].dropna().unique())

        for demographic in demographics:
            if demographic not in data.columns:
                continue

            demo_metrics = {}
            groups = data[demographic].dropna().unique()

            if len(groups) < 2:
                continue

            # Calculate TPR (true positive rate) and FPR (false positive rate) for each phenotype class
            for phenotype in all_phenotypes:
                tpr_by_group = {}
                fpr_by_group = {}

                for group in groups:
                    group_data = data[data[demographic] == group]

                    if len(group_data) < 5:  # Skip groups with too few samples
                        continue

                    # True positive rate (recall)
                    true_positives = sum((group_data[gene_groundtruth] == phenotype) &
                                         (group_data[gene] == phenotype))
                    actual_positives = sum(group_data[gene_groundtruth] == phenotype)

                    tpr = true_positives / actual_positives if actual_positives > 0 else np.nan

                    # False positive rate
                    false_positives = sum((group_data[gene_groundtruth] != phenotype) &
                                          (group_data[gene] == phenotype))
                    actual_negatives = sum(group_data[gene_groundtruth] != phenotype)

                    fpr = false_positives / actual_negatives if actual_negatives > 0 else np.nan

                    tpr_by_group[str(group)] = float(tpr) if not np.isnan(tpr) else None
                    fpr_by_group[str(group)] = float(fpr) if not np.isnan(fpr) else None

                # Calculate disparities between groups
                tpr_values = [v for v in tpr_by_group.values() if v is not None]
                fpr_values = [v for v in fpr_by_group.values() if v is not None]

                tpr_disparity = max(tpr_values) - min(tpr_values) if len(tpr_values) >= 2 else np.nan
                fpr_disparity = max(fpr_values) - min(fpr_values) if len(fpr_values) >= 2 else np.nan

                demo_metrics[phenotype] = {
                    'true_positive_rates': tpr_by_group,
                    'false_positive_rates': fpr_by_group,
                    'tpr_disparity': float(tpr_disparity) if not np.isnan(tpr_disparity) else None,
                    'fpr_disparity': float(fpr_disparity) if not np.isnan(fpr_disparity) else None,
                    'equalized_odds_satisfied': (
                        float(tpr_disparity) < 0.2 and float(fpr_disparity) < 0.2
                        if not np.isnan(tpr_disparity) and not np.isnan(fpr_disparity)
                        else None
                    )
                }

            gene_metrics[demographic] = demo_metrics

        equalized_odds_metrics[gene] = gene_metrics

    return equalized_odds_metrics


def calculate_demographic_parity(data, target_genes, demographics):
    """
    Calculate Demographic Parity metric for each gene and demographic group.
    Demographic Parity measures if prediction rates are similar across different demographic groups.
    """
    demographic_parity_metrics = {}

    for gene in target_genes:
        if gene not in data.columns:
            continue

        gene_metrics = {}
        phenotypes = data[gene].dropna().unique()

        for demographic in demographics:
            if demographic not in data.columns:
                continue

            demo_metrics = {}
            groups = data[demographic].dropna().unique()

            if len(groups) < 2:
                continue

            # Calculate prediction rates for each phenotype class
            for phenotype in phenotypes:
                prediction_rates = {}
                overall_rate = (data[gene] == phenotype).mean()

                for group in groups:
                    group_data = data[data[demographic] == group]

                    if len(group_data) < 5:  # Skip groups with too few samples
                        continue

                    # Prediction rate for this group
                    prediction_rate = (group_data[gene] == phenotype).mean()
                    prediction_rates[str(group)] = float(prediction_rate)

                # Calculate disparity between groups
                if prediction_rates:
                    max_rate = max(prediction_rates.values())
                    min_rate = min(prediction_rates.values())
                    disparity = max_rate - min_rate

                    # Calculate disparity ratio (min/max)
                    disparity_ratio = min_rate / max_rate if max_rate > 0 else 1.0

                    demo_metrics[phenotype] = {
                        'prediction_rates': prediction_rates,
                        'overall_rate': float(overall_rate),
                        'max_disparity': float(disparity),
                        'disparity_ratio': float(disparity_ratio),
                        'demographic_parity_satisfied': float(disparity) < 0.2
                    }

            gene_metrics[demographic] = demo_metrics

        demographic_parity_metrics[gene] = gene_metrics

    return demographic_parity_metrics


def calculate_fairness_metrics(data, target_genes, population_info):
    metrics = {}

    for gene in target_genes:
        if gene not in data.columns:
            continue

        gene_metrics = {
            'overall': {
                'sample_count': int(len(data)),
                'phenotype_distribution': {k: int(v) for k, v in data[gene].value_counts().to_dict().items()}
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
                overall_rate = float((data[gene] == phenotype).mean())
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

                    group_rate = float((group_data[gene] == phenotype).mean())
                    phenotype_metrics['group_rates'][str(group)] = {
                        'rate': group_rate,
                        'sample_size': int(group_size)
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
                                'is_significant': bool(diff >= 0.2 or ratio <= 0.5)
                            }

                            phenotype_metrics['disparities'].append(disparity)

                phenotype_metrics['max_disparity'] = float(max_diff)
                phenotype_metrics['disparity_ratio'] = float(disparity_ratio)
                phenotype_metrics['has_disparity'] = bool(max_diff >= 0.2 or disparity_ratio <= 0.5)

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


def calculate_fairness_scores(equalized_odds_metrics, demographic_parity_metrics):
    """
    Calculate fairness scores based on Equalized Odds and Demographic Parity metrics
    """
    fairness_scores = {}

    # Calculate demographic-based fairness scores
    for demographic in ['Sex', 'Population', 'Superpopulation']:
        demographic_scores = {
            'equalized_odds': {},
            'demographic_parity': {},
            'overall': {}
        }

        # Evaluate Equalized Odds
        eo_violation_count = 0
        eo_total_checks = 0

        for gene, gene_metrics in equalized_odds_metrics.items():
            if demographic not in gene_metrics:
                continue

            for phenotype, phenotype_metrics in gene_metrics[demographic].items():
                if phenotype_metrics['equalized_odds_satisfied'] is not None:
                    eo_total_checks += 1
                    if not phenotype_metrics['equalized_odds_satisfied']:
                        eo_violation_count += 1

        eo_score = 100 - (eo_violation_count / eo_total_checks * 100) if eo_total_checks > 0 else None

        # Evaluate Demographic Parity
        dp_violation_count = 0
        dp_total_checks = 0

        for gene, gene_metrics in demographic_parity_metrics.items():
            if demographic not in gene_metrics:
                continue

            for phenotype, phenotype_metrics in gene_metrics[demographic].items():
                dp_total_checks += 1
                if not phenotype_metrics['demographic_parity_satisfied']:
                    dp_violation_count += 1

        dp_score = 100 - (dp_violation_count / dp_total_checks * 100) if dp_total_checks > 0 else None

        # Calculate overall fairness score (average of both metrics)
        overall_score = None
        if eo_score is not None and dp_score is not None:
            overall_score = (eo_score + dp_score) / 2
        elif eo_score is not None:
            overall_score = eo_score
        elif dp_score is not None:
            overall_score = dp_score

        # Assign fairness rating
        if overall_score is not None:
            if overall_score >= 90:
                fairness_rating = "Excellent"
            elif overall_score >= 80:
                fairness_rating = "Good"
            elif overall_score >= 70:
                fairness_rating = "Fair"
            elif overall_score >= 60:
                fairness_rating = "Concerning"
            else:
                fairness_rating = "Poor"
        else:
            fairness_rating = "Insufficient Data"

        demographic_scores['equalized_odds'] = {
            'score': eo_score,
            'violations': eo_violation_count,
            'total_checks': eo_total_checks
        }

        demographic_scores['demographic_parity'] = {
            'score': dp_score,
            'violations': dp_violation_count,
            'total_checks': dp_total_checks
        }

        demographic_scores['overall'] = {
            'score': overall_score,
            'rating': fairness_rating
        }

        fairness_scores[demographic] = demographic_scores

    return fairness_scores


def generate_fairness_summary(fairness_metrics, equalized_odds_metrics, demographic_parity_metrics, fairness_scores):
    summary = {
        'overall_assessment': {},
        'demographic_findings': {},
        'equalized_odds_findings': {},
        'demographic_parity_findings': {},
        'recommendations': []
    }

    # Process fairness scores for overall assessment
    overall_scores = []
    for demographic, scores in fairness_scores.items():
        if scores['overall']['score'] is not None:
            overall_scores.append(scores['overall']['score'])

    if overall_scores:
        avg_score = sum(overall_scores) / len(overall_scores)

        if avg_score >= 90:
            fairness_rating = "Excellent"
        elif avg_score >= 80:
            fairness_rating = "Good"
        elif avg_score >= 70:
            fairness_rating = "Fair"
        elif avg_score >= 60:
            fairness_rating = "Concerning"
        else:
            fairness_rating = "Poor"
    else:
        avg_score = None
        fairness_rating = "Insufficient Data"

    # Count significant disparities from traditional fairness metrics
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

    # Compile Equalized Odds findings
    eo_findings = []
    for gene, gene_metrics in equalized_odds_metrics.items():
        for demographic, demo_metrics in gene_metrics.items():
            for phenotype, phenotype_metrics in demo_metrics.items():
                if phenotype_metrics['equalized_odds_satisfied'] is False:
                    finding = {
                        'gene': gene,
                        'demographic': demographic,
                        'phenotype': phenotype,
                        'tpr_disparity': phenotype_metrics['tpr_disparity'],
                        'fpr_disparity': phenotype_metrics['fpr_disparity'],
                        'true_positive_rates': phenotype_metrics['true_positive_rates'],
                        'false_positive_rates': phenotype_metrics['false_positive_rates']
                    }
                    eo_findings.append(finding)

    # Compile Demographic Parity findings
    dp_findings = []
    for gene, gene_metrics in demographic_parity_metrics.items():
        for demographic, demo_metrics in gene_metrics.items():
            for phenotype, phenotype_metrics in demo_metrics.items():
                if not phenotype_metrics['demographic_parity_satisfied']:
                    finding = {
                        'gene': gene,
                        'demographic': demographic,
                        'phenotype': phenotype,
                        'max_disparity': phenotype_metrics['max_disparity'],
                        'disparity_ratio': phenotype_metrics['disparity_ratio'],
                        'prediction_rates': phenotype_metrics['prediction_rates']
                    }
                    dp_findings.append(finding)

    summary['overall_assessment'] = {
        'fairness_score': avg_score,
        'fairness_rating': fairness_rating,
        'total_disparities': total_disparities,
        'significant_disparities': significant_disparities,
        'genes_with_disparities': list(genes_with_disparities),
        'fairness_scores_by_demographic': fairness_scores
    }

    summary['equalized_odds_findings'] = eo_findings
    summary['demographic_parity_findings'] = dp_findings

    # Generate recommendations based on findings
    if not eo_findings and not dp_findings:
        summary['recommendations'].append(
            "No significant fairness issues detected. Continue monitoring with larger datasets."
        )
    else:
        if len(eo_findings) > 0:
            summary['recommendations'].append(
                f"Equalized Odds violations detected in {len(eo_findings)} cases. Address disparities in error rates across demographic groups."
            )

        if len(dp_findings) > 0:
            summary['recommendations'].append(
                f"Demographic Parity violations detected in {len(dp_findings)} cases. Address disparities in prediction rates across demographic groups."
            )

        if any(finding['demographic'] == 'Population' for finding in eo_findings + dp_findings):
            summary['recommendations'].append(
                "Some population differences were detected. Consider whether these represent bias or expected genetic variation in pharmacogenes."
            )

        if any(finding['demographic'] == 'Sex' for finding in eo_findings + dp_findings):
            summary['recommendations'].append(
                "Investigate sex-based disparities in phenotype predictions, which may indicate potential bias."
            )

    # Add genetic context
    summary['genetic_context'] = {
        'note': "Pharmacogenomic differences between populations often reflect natural genetic variation rather than algorithmic bias.",
        'implications': "Some disparities identified may represent actual biological differences rather than model bias."
    }

    return summary


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

    print("Calculating fairness metrics")
    fairness_metrics = calculate_fairness_metrics(merged_df, target_genes, population_info)

    print("Calculating Equalized Odds metrics")
    equalized_odds_metrics = calculate_equalized_odds(merged_df, target_genes, demographics)

    print("Calculating Demographic Parity metrics")
    demographic_parity_metrics = calculate_demographic_parity(merged_df, target_genes, demographics)

    print("Calculating fairness scores")
    fairness_scores = calculate_fairness_scores(equalized_odds_metrics, demographic_parity_metrics)

    print("Generating fairness summary")
    fairness_summary = generate_fairness_summary(fairness_metrics, equalized_odds_metrics, demographic_parity_metrics,
                                                 fairness_scores)

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
        'equalized_odds_metrics': equalized_odds_metrics,
        'demographic_parity_metrics': demographic_parity_metrics,
        'fairness_scores': fairness_scores,
        'fairness_summary': fairness_summary
    }

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"Fairness analysis completed. Results saved to {output_file}")

    # Print summary
    print("\nFairness Analysis Summary:")
    print(
        f"Overall Fairness Score: {fairness_summary['overall_assessment'].get('fairness_score', 'N/A')}/100 ({fairness_summary['overall_assessment'].get('fairness_rating', 'N/A')})")
    print(f"Total disparities detected: {fairness_summary['overall_assessment']['total_disparities']}")
    print(f"Significant disparities: {fairness_summary['overall_assessment']['significant_disparities']}")

    print("\nEqualized Odds Metrics:")
    for demographic, scores in fairness_scores.items():
        if scores['equalized_odds']['score'] is not None:
            print(
                f"  {demographic}: {scores['equalized_odds']['score']:.1f}/100 ({scores['equalized_odds']['violations']} violations in {scores['equalized_odds']['total_checks']} checks)")

    print("\nDemographic Parity Metrics:")
    for demographic, scores in fairness_scores.items():
        if scores['demographic_parity']['score'] is not None:
            print(
                f"  {demographic}: {scores['demographic_parity']['score']:.1f}/100 ({scores['demographic_parity']['violations']} violations in {scores['demographic_parity']['total_checks']} checks)")

    if fairness_summary['recommendations']:
        print("\nRecommendations:")
        for i, rec in enumerate(fairness_summary['recommendations'], 1):
            print(f"{i}. {rec}")


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
