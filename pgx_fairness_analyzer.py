import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

# Target genes for analysis
TARGET_GENES = ["CYP2B6", "CYP2C9", "CYP2C19", "CYP3A5", "SLCO1B1", "TPMT", "DPYD"]

# Phenotype mappings and descriptions
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

PHENOTYPE_DESCRIPTIONS = {
    'NM': 'Normal Metabolizer',
    'IM': 'Intermediate Metabolizer',
    'PM': 'Poor Metabolizer',
    'UM': 'Ultra Rapid Metabolizer',
    'RM': 'Rapid Metabolizer',
    'NF': 'Normal Function',
    'DF': 'Decreased Function',
    'IF': 'Increased Function',
    'PF': 'Poor Function',
    'INDETERMINATE': 'Indeterminate'
}

# Phenotype categories
PHENOTYPE_CATEGORIES = {
    'decreased': ['PM', 'IM', 'DF', 'PF'],
    'normal': ['NM', 'NF'],
    'increased': ['UM', 'RM', 'IF'],
    'indeterminate': ['INDETERMINATE']
}


def json_serializable(obj):
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    else:
        return obj


def ensure_serializable(data):
    if isinstance(data, dict):
        return {k: ensure_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [ensure_serializable(i) for i in data]
    else:
        return json_serializable(data)


def load_demographic_data(demographic_file):
    try:
        df = pd.read_csv(demographic_file)
        print(f"Demographic file columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        raise ValueError(f"Error loading demographic data: {str(e)}")


def load_phenotype_data(phenotypes_file):
    try:
        df = pd.read_csv(phenotypes_file)
        print(f"Phenotype file columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        raise ValueError(f"Error loading phenotype data: {str(e)}")


def merge_data(demographic_df, phenotype_df):
    # Find sample ID columns in both dataframes
    demo_id_col = 'Sample ID' if 'Sample ID' in demographic_df.columns else demographic_df.columns[0]
    pheno_id_col = 'Sample ID' if 'Sample ID' in phenotype_df.columns else phenotype_df.columns[0]

    print(f"Merging on {pheno_id_col} from phenotype_df and {demo_id_col} from demographic_df")

    # Merge datasets
    merged_df = pd.merge(
        phenotype_df,
        demographic_df,
        left_on=pheno_id_col,
        right_on=demo_id_col,
        how='left'
    )

    # Add standardized sample ID column if it doesn't exist
    if 'SampleID' not in merged_df.columns:
        merged_df['SampleID'] = merged_df[pheno_id_col]

    return merged_df


def calculate_phenotype_distributions(data):
    distributions = {}
    demographic_fields = ['Sex', 'Population', 'Superpopulation']

    # Overall distribution
    overall_dist = {}
    for gene in TARGET_GENES:
        if gene in data.columns:
            value_counts = data[gene].value_counts(normalize=True).to_dict()
            overall_dist[gene] = value_counts
    distributions['overall'] = overall_dist

    # Distribution by demographic groups
    for field in demographic_fields:
        if field in data.columns:
            field_dist = {}
            for group in data[field].unique():
                if pd.notna(group):
                    group_data = data[data[field] == group]
                    group_dist = {}
                    for gene in TARGET_GENES:
                        if gene in data.columns:
                            value_counts = group_data[gene].value_counts(normalize=True).to_dict()
                            group_dist[gene] = value_counts
                    field_dist[str(group)] = group_dist
            distributions[field] = field_dist

    return distributions


def calculate_fairness_metrics(data):
    fairness_metrics = {}
    demographic_fields = ['Sex', 'Population', 'Superpopulation']

    for field in demographic_fields:
        if field in data.columns:
            field_metrics = {}
            for gene in TARGET_GENES:
                if gene in data.columns:
                    # Convert phenotypes to numeric for statistical testing
                    data[f'{gene}_numeric'] = data[gene].map(lambda x: PHENOTYPE_MAPPING.get(x, 9))

                    # Statistical tests across demographic groups
                    groups = data[field].unique()
                    groups = [g for g in groups if pd.notna(g)]

                    if len(groups) >= 2:
                        # For Sex (binary), use t-test
                        if field == 'Sex' and len(groups) == 2:
                            group_data = [data[data[field] == g][f'{gene}_numeric'] for g in groups]
                            # Ensure we have data in both groups
                            if all(len(gd) > 0 for gd in group_data):
                                try:
                                    test_result = stats.ttest_ind(*group_data, equal_var=False, nan_policy='omit')
                                    field_metrics[gene] = {
                                        'test': 't-test',
                                        'statistic': float(test_result.statistic),
                                        'p_value': float(test_result.pvalue),
                                        'significant': bool(test_result.pvalue < 0.05)
                                    }
                                except Exception as e:
                                    print(f"Error in t-test for {gene} by {field}: {e}")
                        # For multiple groups, use ANOVA
                        else:
                            group_data = [data[data[field] == g][f'{gene}_numeric'] for g in groups]
                            # Filter out empty groups
                            group_data = [g for g in group_data if len(g) > 0]
                            if len(group_data) >= 2:
                                try:
                                    test_result = stats.f_oneway(*group_data)
                                    field_metrics[gene] = {
                                        'test': 'ANOVA',
                                        'statistic': float(test_result.statistic),
                                        'p_value': float(test_result.pvalue),
                                        'significant': bool(test_result.pvalue < 0.05)
                                    }
                                except Exception as e:
                                    print(f"Error in ANOVA for {gene} by {field}: {e}")

            fairness_metrics[field] = field_metrics

    return fairness_metrics


def calculate_disparate_impact(data):
    impact_metrics = {}
    demographic_fields = ['Sex', 'Population', 'Superpopulation']

    for field in demographic_fields:
        if field in data.columns:
            field_impact = {}
            for gene in TARGET_GENES:
                if gene in data.columns:
                    gene_impact = {}
                    # Group phenotypes into categories
                    data['phenotype_category'] = data[gene].apply(
                        lambda x: next((cat for cat, vals in PHENOTYPE_CATEGORIES.items()
                                        if x in vals), 'indeterminate')
                    )

                    # Calculate disparate impact for each phenotype category
                    for category in ['decreased', 'normal', 'increased']:
                        groups = data[field].unique()
                        groups = [g for g in groups if pd.notna(g)]

                        if len(groups) >= 2:
                            category_rates = {}
                            for group in groups:
                                group_data = data[data[field] == group]
                                if len(group_data) > 0:
                                    rate = (group_data['phenotype_category'] == category).mean()
                                    category_rates[str(group)] = float(rate)

                            if category_rates:
                                reference_group = max(category_rates.items(), key=lambda x: x[1])[0]
                                reference_rate = category_rates[reference_group]

                                impact_ratios = {}
                                for group, rate in category_rates.items():
                                    if group != reference_group and reference_rate > 0:
                                        ratio = rate / reference_rate
                                        impact_ratios[group] = float(ratio)

                                gene_impact[category] = {
                                    'reference_group': reference_group,
                                    'reference_rate': float(reference_rate),
                                    'impact_ratios': impact_ratios,
                                    'disparate_impact': bool(any(ratio < 0.8 for ratio in impact_ratios.values()))
                                }

                    field_impact[gene] = gene_impact

            impact_metrics[field] = field_impact

    return impact_metrics


def generate_individual_reports(merged_data, distributions, fairness_metrics, impact_metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for _, sample in merged_data.iterrows():
        sample_id = sample['SampleID']
        report = {
            'sample_id': str(sample_id),
            'demographics': {},
            'phenotypes': {},
            'fairness_analysis': {
                'demographic_comparison': {},
                'statistical_significance': {},
                'disparate_impact': {}
            }
        }

        # Add demographic information
        for field in ['Sex', 'Population', 'Superpopulation']:
            if field in sample and pd.notna(sample[field]):
                report['demographics'][field] = str(sample[field])

        # Add phenotype predictions
        for gene in TARGET_GENES:
            if gene in sample and pd.notna(sample[gene]):
                phenotype = sample[gene]
                report['phenotypes'][gene] = {
                    'phenotype': str(phenotype),
                    'description': PHENOTYPE_DESCRIPTIONS.get(phenotype, 'Unknown'),
                    'category': next((cat for cat, vals in PHENOTYPE_CATEGORIES.items()
                                      if phenotype in vals), 'indeterminate')
                }

        # Add demographic comparison
        for field in ['Sex', 'Population', 'Superpopulation']:
            if field in sample and pd.notna(sample[field]):
                group_value = str(sample[field])
                if field in distributions and group_value in distributions[field]:
                    comparison = {}
                    for gene in TARGET_GENES:
                        if gene in sample and pd.notna(sample[gene]):
                            phenotype = str(sample[gene])

                            # Group distribution
                            group_dist = distributions[field][group_value].get(gene, {})
                            group_rate = group_dist.get(phenotype, 0)

                            # Overall distribution
                            overall_dist = distributions['overall'].get(gene, {})
                            overall_rate = overall_dist.get(phenotype, 0)

                            comparison[gene] = {
                                'group_rate': float(group_rate),
                                'overall_rate': float(overall_rate),
                                'difference': float(group_rate - overall_rate),
                                'ratio': float(group_rate / overall_rate) if overall_rate > 0 else None
                            }

                    report['fairness_analysis']['demographic_comparison'][field] = comparison

        # Add statistical significance
        for field in ['Sex', 'Population', 'Superpopulation']:
            if field in fairness_metrics:
                field_metrics = {}
                for gene in TARGET_GENES:
                    if gene in fairness_metrics[field]:
                        field_metrics[gene] = fairness_metrics[field][gene]
                report['fairness_analysis']['statistical_significance'][field] = field_metrics

        # Add disparate impact
        for field in ['Sex', 'Population', 'Superpopulation']:
            if field in impact_metrics:
                field_impact = {}
                for gene in TARGET_GENES:
                    if gene in impact_metrics[field]:
                        field_impact[gene] = impact_metrics[field][gene]
                report['fairness_analysis']['disparate_impact'][field] = field_impact

        # Generate summary
        report['summary'] = generate_bias_summary(report)

        # Save report - directly use simple dict without complex serialization
        output_file = os.path.join(output_dir, f"{sample_id}_fairness_report.json")
        try:
            # Manual conversion to basic Python types
            simple_report = {}
            for k, v in report.items():
                if isinstance(v, dict):
                    simple_report[k] = {}
                    for k2, v2 in v.items():
                        if isinstance(v2, dict):
                            simple_report[k][k2] = {}
                            for k3, v3 in v2.items():
                                if isinstance(v3, dict):
                                    simple_report[k][k2][k3] = {}
                                    for k4, v4 in v3.items():
                                        simple_report[k][k2][k3][k4] = str(v4) if not isinstance(v4, (
                                            bool, int, float, str, list, dict)) else v4
                                else:
                                    simple_report[k][k2][k3] = str(v3) if not isinstance(v3, (
                                        bool, int, float, str, list, dict)) else v3
                        else:
                            simple_report[k][k2] = str(v2) if not isinstance(v2, (
                                bool, int, float, str, list, dict)) else v2
                else:
                    simple_report[k] = str(v) if not isinstance(v, (bool, int, float, str, list, dict)) else v

            with open(output_file, 'w') as f:
                json.dump(simple_report, f, indent=2)
        except Exception as e:
            print(f"Error saving report for {sample_id}: {e}")


def generate_bias_summary(report):
    summary = []

    # Check for demographic bias by comparing individual phenotypes to their group rates
    for field, comparisons in report['fairness_analysis']['demographic_comparison'].items():
        group_value = report['demographics'].get(field)
        if group_value:
            for gene, comparison in comparisons.items():
                phenotype = report['phenotypes'].get(gene, {}).get('phenotype')
                if phenotype:
                    group_rate = comparison.get('group_rate', 0)
                    overall_rate = comparison.get('overall_rate', 0)
                    difference = comparison.get('difference', 0)

                    if abs(difference) > 0.2:  # Threshold for notable difference
                        direction = "higher" if difference > 0 else "lower"
                        summary.append(
                            f"The {phenotype} phenotype for {gene} is notably {direction} in {group_value} "
                            f"({field}) compared to the overall population "
                            f"({group_rate:.1%} vs {overall_rate:.1%})."
                        )

    # Check for statistically significant demographic bias
    for field, significances in report['fairness_analysis']['statistical_significance'].items():
        group_value = report['demographics'].get(field)
        if group_value:
            for gene, significance in significances.items():
                if significance.get('significant', False):
                    test_name = significance.get('test', 'statistical test')
                    p_value = significance.get('p_value', 1.0)
                    summary.append(
                        f"There is a statistically significant difference in {gene} phenotypes across {field} "
                        f"groups ({test_name}, p={p_value:.4f})."
                    )

    # Check for disparate impact
    for field, impacts in report['fairness_analysis']['disparate_impact'].items():
        group_value = report['demographics'].get(field)
        if group_value:
            for gene, categories in impacts.items():
                for category, impact in categories.items():
                    if impact.get('disparate_impact', False):
                        reference_group = impact.get('reference_group', 'unknown')
                        if group_value in impact.get('impact_ratios', {}):
                            ratio = impact['impact_ratios'][group_value]
                            if ratio < 0.8:
                                summary.append(
                                    f"The {group_value} group shows potential disparate impact for {gene} "
                                    f"in the {category} category compared to the {reference_group} group "
                                    f"(impact ratio: {ratio:.2f})."
                                )

    if not summary:
        summary.append("No significant demographic bias detected in the phenotype predictions.")

    return summary


def generate_overall_report(merged_data, distributions, fairness_metrics, impact_metrics, output_dir):
    report = {
        'overall_metrics': {
            'total_samples': len(merged_data),
            'demographic_distribution': {},
            'phenotype_distribution': {}
        },
        'bias_analysis': {
            'significant_demographic_effects': [],
            'disparate_impact_summary': []
        }
    }

    # Demographic distribution
    for field in ['Sex', 'Population', 'Superpopulation']:
        if field in merged_data.columns:
            report['overall_metrics']['demographic_distribution'][field] = {
                str(k): int(v) for k, v in merged_data[field].value_counts().to_dict().items()
            }

    # Phenotype distribution
    for gene in TARGET_GENES:
        if gene in merged_data.columns:
            report['overall_metrics']['phenotype_distribution'][gene] = {
                str(k): int(v) for k, v in merged_data[gene].value_counts().to_dict().items()
            }

    # Significant demographic effects
    for field in fairness_metrics:
        for gene, metrics in fairness_metrics[field].items():
            if metrics.get('significant', False):
                report['bias_analysis']['significant_demographic_effects'].append({
                    'demographic_factor': field,
                    'gene': gene,
                    'test': metrics.get('test', 'statistical test'),
                    'p_value': float(metrics.get('p_value', 1.0)),
                    'statistic': float(metrics.get('statistic', 0))
                })

    # Disparate impact summary
    for field in impact_metrics:
        for gene, categories in impact_metrics[field].items():
            for category, impact in categories.items():
                if impact.get('disparate_impact', False):
                    impact_summary = {
                        'demographic_factor': field,
                        'gene': gene,
                        'phenotype_category': category,
                        'reference_group': str(impact.get('reference_group', 'unknown')),
                        'impact_ratios': {}
                    }

                    # Convert impact ratios to simple types
                    for k, v in impact.get('impact_ratios', {}).items():
                        impact_summary['impact_ratios'][str(k)] = float(v)

                    report['bias_analysis']['disparate_impact_summary'].append(impact_summary)

    # Generate summary statements
    report['summary'] = []

    # Summary of significant demographic effects
    if report['bias_analysis']['significant_demographic_effects']:
        effects = report['bias_analysis']['significant_demographic_effects']
        report['summary'].append(
            f"Found {len(effects)} statistically significant demographic effects on phenotype predictions."
        )
        for field in ['Sex', 'Population', 'Superpopulation']:
            field_effects = [e for e in effects if e['demographic_factor'] == field]
            if field_effects:
                affected_genes = [e['gene'] for e in field_effects]
                report['summary'].append(
                    f"{field} has significant effects on {len(field_effects)} genes: {', '.join(affected_genes)}."
                )
    else:
        report['summary'].append(
            "No statistically significant demographic effects on phenotype predictions were detected."
        )

    # Summary of disparate impact
    if report['bias_analysis']['disparate_impact_summary']:
        impacts = report['bias_analysis']['disparate_impact_summary']
        report['summary'].append(
            f"Found {len(impacts)} instances of potential disparate impact across demographic groups."
        )
        for field in ['Sex', 'Population', 'Superpopulation']:
            field_impacts = [i for i in impacts if i['demographic_factor'] == field]
            if field_impacts:
                report['summary'].append(
                    f"{field} shows potential disparate impact in {len(field_impacts)} cases."
                )
    else:
        report['summary'].append(
            "No instances of disparate impact were detected across demographic groups."
        )

    # Save overall report
    output_file = os.path.join(output_dir, "overall_fairness_report.json")
    try:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
    except Exception as e:
        print(f"Error saving overall report: {e}")


def main():
    parser = argparse.ArgumentParser(description='PharmCAT Fairness/Bias Analyzer')
    parser.add_argument('--demographic_file', required=True, help='Path to demographic data CSV file')
    parser.add_argument('--phenotypes_file', required=True, help='Path to phenotype predictions CSV file')
    parser.add_argument('--output_dir', default='pgx_fairness_results', help='Output directory for results')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading demographic data from {args.demographic_file}...")
    demographic_df = load_demographic_data(args.demographic_file)

    print(f"Loading phenotype data from {args.phenotypes_file}...")
    phenotype_df = load_phenotype_data(args.phenotypes_file)

    # Merge data
    print("Merging demographic and phenotype data...")
    merged_data = merge_data(demographic_df, phenotype_df)

    # Calculate phenotype distributions by demographic group
    print("Calculating phenotype distributions...")
    distributions = calculate_phenotype_distributions(merged_data)

    # Calculate fairness metrics
    print("Calculating fairness metrics...")
    fairness_metrics = calculate_fairness_metrics(merged_data)

    # Calculate disparate impact
    print("Calculating disparate impact metrics...")
    impact_metrics = calculate_disparate_impact(merged_data)

    # Generate individual reports
    print("Generating individual fairness reports...")
    generate_individual_reports(merged_data, distributions, fairness_metrics, impact_metrics, args.output_dir)

    # Generate overall report
    print("Generating overall fairness report...")
    generate_overall_report(merged_data, distributions, fairness_metrics, impact_metrics, args.output_dir)

    print(f"Analysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
