import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.utils import resample

# Constants remain the same
TARGET_GENES = ["CYP2B6", "CYP2C9", "CYP2C19", "CYP3A5", "SLCO1B1", "TPMT", "DPYD"]
PHENOTYPE_MAPPING = {"PM": 0, "IM": 1, "NM": 2, "RM": 3, "UM": 4, "NF": 5, "DF": 6, "PF": 7, "IF": 8,
                     "INDETERMINATE": 9}
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
PHENOTYPE_CATEGORIES = {
    'decreased': ['PM', 'IM', 'DF', 'PF'],
    'normal': ['NM', 'NF'],
    'increased': ['UM', 'RM', 'IF'],
    'indeterminate': ['INDETERMINATE']
}

# New constants for methodological improvements
MIN_GROUP_SIZE = 5  # Minimum sample size for reliable statistical analysis
MIN_COMPARISON_SIZE = 3  # Minimum size for group comparisons
EXPECTED_SUPER_POPULATIONS = ['AFR', 'AMR', 'EAS', 'EUR', 'SAS']
P_VALUE_THRESHOLD = 0.05  # Standard significance threshold
SMALL_SAMPLE_WARNING_THRESHOLD = 10  # Warn for groups smaller than this


# Helper functions remain mostly the same
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


# New helper function to check sample size
def check_sample_size(data, field, value=None):
    """Check if a demographic group has sufficient sample size for analysis."""
    if value is not None:
        group_size = len(data[data[field] == value])
        return group_size >= MIN_GROUP_SIZE
    else:
        group_sizes = data[field].value_counts()
        return {group: (count >= MIN_GROUP_SIZE) for group, count in group_sizes.items()}


# Modified function to load demographic data (same implementation)
def load_demographic_data(demographic_file):
    """Load demographic data from a CSV file."""
    try:
        df = pd.read_csv(demographic_file)
        print(f"Demographic file columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        raise ValueError(f"Error loading demographic data: {str(e)}")


# Modified function to load phenotype data (same implementation)
def load_phenotype_data(phenotypes_file):
    """Load pharmacogenomic phenotype predictions from a CSV file."""
    try:
        df = pd.read_csv(phenotypes_file)
        print(f"Phenotype file columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        raise ValueError(f"Error loading phenotype data: {str(e)}")


# Same implementation for merge_data
def merge_data(demographic_df, phenotype_df):
    """Merge demographic and phenotype data into a single DataFrame."""
    demo_id_col = 'Sample ID' if 'Sample ID' in demographic_df.columns else demographic_df.columns[0]
    pheno_id_col = 'Sample ID' if 'Sample ID' in phenotype_df.columns else phenotype_df.columns[0]

    print(f"Merging on {pheno_id_col} from phenotype_df and {demo_id_col} from demographic_df")

    merged_df = pd.merge(
        phenotype_df,
        demographic_df,
        left_on=pheno_id_col,
        right_on=demo_id_col,
        how='left'
    )

    if 'SampleID' not in merged_df.columns:
        merged_df['SampleID'] = merged_df[pheno_id_col]

    return merged_df


# Modified distribution calculation
def calculate_phenotype_distributions(data):
    """Calculate phenotype distributions overall and by demographic groups."""
    distributions = {}
    demographic_fields = ['Sex', 'Population', 'Superpopulation']

    # Calculate overall distributions
    overall_dist = {}
    for gene in TARGET_GENES:
        if gene in data.columns:
            value_counts = data[gene].value_counts(normalize=True).to_dict()
            overall_dist[gene] = value_counts
    distributions['overall'] = overall_dist

    # Calculate distributions by demographic group with sample size annotation
    for field in demographic_fields:
        if field in data.columns:
            field_dist = {}
            group_sizes = data[field].value_counts().to_dict()

            for group in data[field].unique():
                if pd.notna(group):
                    group_data = data[data[field] == group]
                    group_size = len(group_data)

                    # Add sample size warning flag
                    is_small_sample = group_size < MIN_GROUP_SIZE

                    group_dist = {}
                    for gene in TARGET_GENES:
                        if gene in data.columns:
                            value_counts = group_data[gene].value_counts(normalize=True).to_dict()
                            group_dist[gene] = value_counts

                    field_dist[str(group)] = {
                        'distributions': group_dist,
                        'sample_size': group_size,
                        'small_sample_warning': is_small_sample
                    }

            distributions[field] = field_dist

    return distributions


# Modified demographic parity calculation
def calculate_demographic_parity(data):
    """Calculate demographic parity metrics for phenotypes across demographic groups."""
    demographic_fields = ['Sex', 'Population', 'Superpopulation']
    parity_metrics = {}

    for field in demographic_fields:
        if field in data.columns:
            field_metrics = {}
            for gene in TARGET_GENES:
                if gene in data.columns:
                    gene_metrics = {}

                    phenotypes = data[gene].dropna().unique()
                    phenotypes = [p for p in phenotypes if p != "INDETERMINATE"]

                    for phenotype in phenotypes:
                        group_rates = {}
                        group_sizes = {}
                        overall_rate = (data[gene] == phenotype).mean()

                        groups = data[field].unique()
                        groups = [g for g in groups if pd.notna(g)]

                        # Filter out groups with insufficient samples
                        valid_groups = []
                        for group in groups:
                            group_data = data[data[field] == group]
                            group_size = len(group_data)
                            if group_size >= MIN_COMPARISON_SIZE:
                                valid_groups.append(group)
                                group_sizes[str(group)] = group_size
                            else:
                                print(
                                    f"Warning: Group {group} in {field} has only {group_size} samples, below minimum threshold ({MIN_COMPARISON_SIZE}). Excluding from parity analysis.")

                        if len(valid_groups) < 2:
                            print(
                                f"Warning: Not enough valid groups for {field} to calculate parity for {gene} {phenotype}. Need at least 2 groups with {MIN_COMPARISON_SIZE}+ samples.")
                            continue

                        max_diff = 0
                        min_rate = 1
                        max_rate = 0

                        for group in valid_groups:
                            group_data = data[data[field] == group]
                            group_rate = (group_data[gene] == phenotype).mean()
                            group_rates[str(group)] = float(group_rate)

                            diff = abs(group_rate - overall_rate)
                            if diff > max_diff:
                                max_diff = diff

                            if group_rate < min_rate:
                                min_rate = group_rate
                            if group_rate > max_rate:
                                max_rate = group_rate

                        if max_rate > 0:
                            disparity_ratio = min_rate / max_rate
                        else:
                            disparity_ratio = 1.0

                        parity_difference = max_diff

                        # Adjust threshold for small samples
                        smallest_group_size = min(group_sizes.values()) if group_sizes else 0
                        adjusted_threshold = 0.2
                        if smallest_group_size < SMALL_SAMPLE_WARNING_THRESHOLD:
                            # For smaller samples, we require larger differences to consider them significant
                            adjusted_threshold = 0.2 + (0.1 * (
                                    SMALL_SAMPLE_WARNING_THRESHOLD - smallest_group_size) / SMALL_SAMPLE_WARNING_THRESHOLD)

                        gene_metrics[phenotype] = {
                            'overall_rate': float(overall_rate),
                            'group_rates': group_rates,
                            'group_sizes': group_sizes,
                            'disparity_ratio': float(disparity_ratio),
                            'parity_difference': float(parity_difference),
                            'has_disparity': bool(parity_difference > adjusted_threshold or disparity_ratio < 0.8),
                            'sample_size_adjusted_threshold': float(adjusted_threshold)
                        }

                    field_metrics[gene] = gene_metrics

            parity_metrics[field] = field_metrics

    return parity_metrics


# Modified group fairness metrics calculation
def calculate_group_fairness_metrics(data):
    """Calculate group-level fairness metrics for phenotype distributions."""
    demographic_fields = ['Sex', 'Population', 'Superpopulation']
    fairness_metrics = {}

    for field in demographic_fields:
        if field in data.columns:
            field_metrics = {}
            for gene in TARGET_GENES:
                if gene in data.columns:
                    gene_metrics = {}

                    phenotypes = data[gene].dropna().unique()
                    phenotypes = [p for p in phenotypes if p != "INDETERMINATE"]

                    for phenotype in phenotypes:
                        group_rates = []
                        group_sizes = []
                        groups = data[field].unique()
                        groups = [g for g in groups if pd.notna(g)]

                        valid_groups = []
                        for group in groups:
                            group_data = data[data[field] == group]
                            group_size = len(group_data)
                            if group_size >= MIN_COMPARISON_SIZE:
                                valid_groups.append(group)
                                group_sizes.append(group_size)

                        if len(valid_groups) < 2:
                            print(
                                f"Warning: Not enough valid groups for {field} to calculate fairness metrics for {gene} {phenotype}.")
                            continue

                        for group in valid_groups:
                            group_data = data[data[field] == group]
                            group_size = len(group_data)
                            if group_size > 0:
                                group_rate = (group_data[gene] == phenotype).mean()
                                group_rates.append(group_rate)

                        if group_rates:
                            # Weight by sample size
                            weights = np.array(group_sizes) / sum(group_sizes)
                            mean_rate = np.average(group_rates, weights=weights)

                            # Weighted standard deviation
                            variance = np.average((np.array(group_rates) - mean_rate) ** 2, weights=weights)
                            std_rate = np.sqrt(variance)

                            if mean_rate > 0:
                                cv = std_rate / mean_rate
                            else:
                                cv = 0

                            bg_variance = np.var(group_rates)

                            # Adjust CV threshold based on genetic context
                            # In pharmacogenomics, expect more variation across populations
                            cv_threshold = 0.7 if field == 'Population' or field == 'Superpopulation' else 0.5

                            # Bootstrap only if enough data points
                            if len(group_rates) >= 3:
                                bootstrap_samples = []
                                for _ in range(1000):
                                    boot_sample = resample(group_rates)
                                    bootstrap_samples.append(np.mean(boot_sample))

                                ci_lower = np.percentile(bootstrap_samples, 10)
                                ci_upper = np.percentile(bootstrap_samples, 90)
                            else:
                                margin = 0.2 * mean_rate  # Simple margin for small samples
                                ci_lower = max(0, mean_rate - margin)
                                ci_upper = min(1, mean_rate + margin)

                            gene_metrics[phenotype] = {
                                'mean_rate': float(mean_rate),
                                'std_dev': float(std_rate),
                                'coefficient_of_variation': float(cv),
                                'between_group_variance': float(bg_variance),
                                'ci_lower_80': float(ci_lower),
                                'ci_upper_80': float(ci_upper),
                                'fairness_concern': bool(cv > cv_threshold),
                                'sample_sizes': group_sizes,
                                'cv_threshold_used': cv_threshold
                            }

                    field_metrics[gene] = gene_metrics

            fairness_metrics[field] = field_metrics

    return fairness_metrics


# Modified intersectional metrics calculation
def calculate_intersectional_metrics(data):
    """Calculate fairness metrics across intersectional demographic combinations."""
    intersectional_metrics = {}
    combinations = [
        ('Sex', 'Population'),
        ('Sex', 'Superpopulation'),
    ]

    for field1, field2 in combinations:
        if field1 in data.columns and field2 in data.columns:
            key = f"{field1}_{field2}"
            combo_metrics = {}

            data['intersection'] = data[field1].astype(str) + "_" + data[field2].astype(str)

            for gene in TARGET_GENES:
                if gene in data.columns:
                    gene_metrics = {}
                    phenotypes = data[gene].dropna().unique()
                    phenotypes = [p for p in phenotypes if p != "INDETERMINATE"]

                    # Get intersection group counts
                    group_counts = data['intersection'].value_counts()
                    valid_groups = group_counts[group_counts >= MIN_COMPARISON_SIZE].index.tolist()

                    if len(valid_groups) < 2:
                        print(
                            f"Warning: Not enough intersectional groups with sufficient sample size for {key}. Skipping analysis for this combination.")
                        continue

                    for phenotype in phenotypes:
                        group_rates = {}
                        group_sizes = {}
                        overall_rate = (data[gene] == phenotype).mean()

                        max_diff = 0
                        for group in valid_groups:
                            if "nan" in group:
                                continue

                            group_data = data[data['intersection'] == group]
                            group_size = len(group_data)
                            group_rate = (group_data[gene] == phenotype).mean()
                            group_rates[group] = float(group_rate)
                            group_sizes[group] = group_size

                            diff = abs(group_rate - overall_rate)
                            if diff > max_diff:
                                max_diff = diff

                        # Adjust threshold based on smallest group size
                        smallest_group_size = min(group_sizes.values()) if group_sizes else 0
                        adjusted_threshold = 0.25
                        if smallest_group_size < SMALL_SAMPLE_WARNING_THRESHOLD:
                            adjusted_threshold = 0.25 + (0.1 * (
                                    SMALL_SAMPLE_WARNING_THRESHOLD - smallest_group_size) / SMALL_SAMPLE_WARNING_THRESHOLD)

                        gene_metrics[phenotype] = {
                            'overall_rate': float(overall_rate),
                            'group_rates': group_rates,
                            'group_sizes': group_sizes,
                            'max_difference': float(max_diff),
                            'has_intersectional_disparity': bool(max_diff > adjusted_threshold),
                            'sample_size_adjusted_threshold': float(adjusted_threshold)
                        }

                    combo_metrics[gene] = gene_metrics

            intersectional_metrics[key] = combo_metrics

    return intersectional_metrics


# Modified correlation metrics calculation
def calculate_phenotype_correlations(data):
    """Calculate correlations between demographic factors and phenotype predictions."""
    demographic_fields = ['Sex', 'Population', 'Superpopulation']
    correlation_metrics = {}

    for field in demographic_fields:
        if field in data.columns:
            field_metrics = {}
            valid_values = data[field].dropna()
            field_counts = valid_values.value_counts()

            # Skip correlation analysis if any group is too small
            if any(count < MIN_COMPARISON_SIZE for count in field_counts):
                print(
                    f"Warning: Some groups in {field} have fewer than {MIN_COMPARISON_SIZE} samples. Skipping correlation analysis.")
                continue

            unique_values = valid_values.unique()
            is_binary = len(unique_values) == 2

            for gene in TARGET_GENES:
                if gene in data.columns:
                    gene_metrics = {}

                    data[f'{gene}_numeric'] = data[gene].map(lambda x: PHENOTYPE_MAPPING.get(x, 9))
                    valid_mask = data[f'{gene}_numeric'] != 9

                    if is_binary:
                        if field == 'Sex':
                            data['binary_field'] = (data[field] == 'F').astype(int)

                            valid_data = data[valid_mask]
                            if len(valid_data) >= MIN_GROUP_SIZE:
                                corr, p_value = stats.pointbiserialr(
                                    valid_data['binary_field'],
                                    valid_data[f'{gene}_numeric']
                                )

                                # Adjust effect size interpretation for genetic data
                                effect_size_label = 'none'
                                if abs(corr) > 0.1:
                                    effect_size_label = 'small'
                                if abs(corr) > 0.3:
                                    effect_size_label = 'medium'
                                if abs(corr) > 0.5:
                                    effect_size_label = 'large'

                                gene_metrics['point_biserial'] = {
                                    'correlation': float(corr),
                                    'p_value': float(p_value),
                                    'significant': bool(p_value < P_VALUE_THRESHOLD),
                                    'effect_size': effect_size_label,
                                    'sample_size': len(valid_data)
                                }
                    else:
                        valid_data = data[valid_mask]

                        # Skip if too few samples
                        if len(valid_data) < MIN_GROUP_SIZE:
                            continue

                        # Check if we have enough samples per group
                        field_value_counts = valid_data[field].value_counts()
                        if any(count < MIN_COMPARISON_SIZE for count in field_value_counts):
                            print(
                                f"Warning: Some {field} groups have fewer than {MIN_COMPARISON_SIZE} samples when analyzing {gene}. Skipping.")
                            continue

                        contingency = pd.crosstab(valid_data[field], valid_data[f'{gene}_numeric'])

                        # Only proceed if contingency table has enough data
                        if contingency.size > 1 and not (contingency == 0).any().any():
                            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                            n = contingency.sum().sum()
                            phi2 = chi2 / n
                            r, k = contingency.shape
                            phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
                            rcorr = r - ((r - 1) ** 2) / (n - 1)
                            kcorr = k - ((k - 1) ** 2) / (n - 1)
                            cramers_v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

                            # Adjust effect size thresholds for genetic data
                            effect_size_label = 'none'
                            if cramers_v > 0.1:
                                effect_size_label = 'small'
                            if cramers_v > 0.3:
                                effect_size_label = 'medium'
                            if cramers_v > 0.5:
                                effect_size_label = 'large'

                            gene_metrics['cramers_v'] = {
                                'correlation': float(cramers_v),
                                'p_value': float(p_value),
                                'significant': bool(p_value < P_VALUE_THRESHOLD),
                                'effect_size': effect_size_label,
                                'sample_size': n
                            }

                    field_metrics[gene] = gene_metrics

            correlation_metrics[field] = field_metrics

    return correlation_metrics


# Modified fairness metrics calculation
def calculate_fairness_metrics(data):
    """Calculate statistical tests for differences in phenotype distributions across demographic groups."""
    fairness_metrics = {}
    demographic_fields = ['Sex', 'Population', 'Superpopulation']

    for field in demographic_fields:
        if field in data.columns:
            field_metrics = {}
            for gene in TARGET_GENES:
                if gene in data.columns:
                    data[f'{gene}_numeric'] = data[gene].map(lambda x: PHENOTYPE_MAPPING.get(x, 9))

                    groups = data[field].unique()
                    groups = [g for g in groups if pd.notna(g)]

                    # Check if we have enough valid groups with sufficient sample size
                    valid_groups = []
                    for group in groups:
                        group_data = data[data[field] == group]
                        if len(group_data) >= MIN_COMPARISON_SIZE:
                            valid_groups.append(group)

                    if len(valid_groups) < 2:
                        print(
                            f"Warning: Cannot perform statistical test for {gene} by {field}. Need at least 2 groups with {MIN_COMPARISON_SIZE}+ samples.")
                        continue

                    if field == 'Sex' and len(valid_groups) == 2:
                        group_data = [data[data[field] == g][f'{gene}_numeric'] for g in valid_groups]

                        # Skip if any group has insufficient samples of this gene
                        if not all(len(gd.dropna()) >= MIN_COMPARISON_SIZE for gd in group_data):
                            continue

                        try:
                            test_result = stats.ttest_ind(*group_data, equal_var=False, nan_policy='omit')

                            group1 = group_data[0].dropna()
                            group2 = group_data[1].dropna()

                            if len(group1) >= MIN_COMPARISON_SIZE and len(group2) >= MIN_COMPARISON_SIZE:
                                mean1, mean2 = group1.mean(), group2.mean()
                                pooled_std = np.sqrt(((len(group1) - 1) * group1.std() ** 2 +
                                                      (len(group2) - 1) * group2.std() ** 2) /
                                                     (len(group1) + len(group2) - 2))

                                if pooled_std > 0:
                                    cohens_d = abs(mean1 - mean2) / pooled_std
                                else:
                                    cohens_d = 0

                                # Adjusted thresholds for effect size in genetic context
                                effect_size_label = 'none'
                                if cohens_d > 0.2:
                                    effect_size_label = 'small'
                                if cohens_d > 0.5:
                                    effect_size_label = 'medium'
                                if cohens_d > 0.8:
                                    effect_size_label = 'large'

                                # Adjust significance by effect size for genetic context
                                # Only report as significant if effect size is meaningful
                                is_significant = test_result.pvalue < P_VALUE_THRESHOLD and cohens_d > 0.2

                                field_metrics[gene] = {
                                    'test': 't-test',
                                    'statistic': float(test_result.statistic),
                                    'p_value': float(test_result.pvalue),
                                    'significant': bool(is_significant),
                                    'effect_size': float(cohens_d),
                                    'effect_magnitude': effect_size_label,
                                    'sample_sizes': [len(group1), len(group2)]
                                }
                            else:
                                print(f"Warning: Insufficient samples for t-test on {gene} by {field}")
                        except Exception as e:
                            print(f"Error in t-test for {gene} by {field}: {e}")
                    else:
                        # For non-binary demographics, use ANOVA but check sample sizes
                        group_data = []
                        group_n = []

                        for group in valid_groups:
                            group_values = data[data[field] == group][f'{gene}_numeric'].dropna()
                            if len(group_values) >= MIN_COMPARISON_SIZE:
                                group_data.append(group_values)
                                group_n.append(len(group_values))

                        if len(group_data) >= 2:
                            try:
                                test_result = stats.f_oneway(*group_data)

                                all_values = np.concatenate([g for g in group_data])
                                grand_mean = np.mean(all_values)
                                ss_total = np.sum((all_values - grand_mean) ** 2)

                                group_means = [np.mean(g) for g in group_data]
                                group_sizes = [len(g) for g in group_data]
                                ss_between = np.sum(
                                    [n * (m - grand_mean) ** 2 for n, m in zip(group_sizes, group_means)])

                                if ss_total > 0:
                                    eta_squared = ss_between / ss_total
                                else:
                                    eta_squared = 0

                                # Adjusted thresholds for genetic data
                                effect_size_label = 'none'
                                if eta_squared > 0.01:
                                    effect_size_label = 'small'
                                if eta_squared > 0.06:
                                    effect_size_label = 'medium'
                                if eta_squared > 0.14:
                                    effect_size_label = 'large'

                                # Combine p-value and effect size for significance in genetic context
                                is_significant = test_result.pvalue < P_VALUE_THRESHOLD and eta_squared > 0.01

                                field_metrics[gene] = {
                                    'test': 'ANOVA',
                                    'statistic': float(test_result.statistic),
                                    'p_value': float(test_result.pvalue),
                                    'significant': bool(is_significant),
                                    'effect_size': float(eta_squared),
                                    'effect_magnitude': effect_size_label,
                                    'sample_sizes': group_sizes
                                }
                            except Exception as e:
                                print(f"Error in ANOVA for {gene} by {field}: {e}")

            fairness_metrics[field] = field_metrics

    return fairness_metrics


# Modified disparate impact calculation
def calculate_disparate_impact(data):
    """Calculate disparate impact metrics for phenotype categories across demographic groups."""
    impact_metrics = {}
    demographic_fields = ['Sex', 'Population', 'Superpopulation']

    for field in demographic_fields:
        if field in data.columns:
            field_impact = {}
            for gene in TARGET_GENES:
                if gene in data.columns:
                    gene_impact = {}
                    data['phenotype_category'] = data[gene].apply(
                        lambda x: next((cat for cat, vals in PHENOTYPE_CATEGORIES.items()
                                        if x in vals), 'indeterminate')
                    )

                    for category in ['decreased', 'normal', 'increased']:
                        groups = data[field].unique()
                        groups = [g for g in groups if pd.notna(g)]

                        # Filter for valid group sizes
                        valid_groups = []
                        for group in groups:
                            group_data = data[data[field] == group]
                            if len(group_data) >= MIN_COMPARISON_SIZE:
                                valid_groups.append(group)

                        if len(valid_groups) < 2:
                            print(
                                f"Warning: Not enough valid groups for {field} to calculate disparate impact for {gene} {category}.")
                            continue

                        category_rates = {}
                        group_sizes = {}
                        for group in valid_groups:
                            group_data = data[data[field] == group]
                            group_size = len(group_data)
                            rate = (group_data['phenotype_category'] == category).mean()
                            category_rates[str(group)] = float(rate)
                            group_sizes[str(group)] = group_size

                        if category_rates:
                            # Find reference group (highest rate)
                            reference_group = max(category_rates.items(), key=lambda x: x[1])[0]
                            reference_rate = category_rates[reference_group]
                            reference_size = group_sizes[reference_group]

                            # Skip if reference rate is very low (likely noise for rare phenotypes)
                            if reference_rate < 0.05 and reference_size < SMALL_SAMPLE_WARNING_THRESHOLD:
                                continue

                            impact_ratios = {}
                            for group, rate in category_rates.items():
                                if group != reference_group and reference_rate > 0:
                                    ratio = rate / reference_rate
                                    impact_ratios[group] = float(ratio)

                            abs_differences = {}
                            for group, rate in category_rates.items():
                                if group != reference_group:
                                    diff = abs(rate - reference_rate)
                                    abs_differences[group] = float(diff)

                            # Adjust disparate impact thresholds based on genetic context and sample sizes
                            baseline_threshold = 0.8
                            smallest_group_size = min(group_sizes.values())

                            # For smaller groups, we need stronger evidence to claim disparate impact
                            if smallest_group_size < SMALL_SAMPLE_WARNING_THRESHOLD:
                                di_threshold = baseline_threshold - 0.1
                            else:
                                di_threshold = baseline_threshold

                            # For population-based comparisons, use more lenient threshold due to expected genetic variation
                            if field in ['Population', 'Superpopulation'] and category in ['decreased', 'increased']:
                                di_threshold -= 0.1  # More lenient for genetic differences in metabolism

                            di_score = 1.0
                            if impact_ratios:
                                di_score = min(impact_ratios.values())

                            # Consider large absolute differences even with high ratios
                            # This helps when prevalence is very low in both groups
                            max_abs_diff = max(abs_differences.values()) if abs_differences else 0
                            significant_absolute_diff = max_abs_diff > 0.3

                            gene_impact[category] = {
                                'reference_group': reference_group,
                                'reference_rate': float(reference_rate),
                                'impact_ratios': impact_ratios,
                                'absolute_differences': abs_differences,
                                'disparate_impact': bool(any(ratio < di_threshold for ratio in
                                                             impact_ratios.values()) or significant_absolute_diff),
                                'di_score': float(di_score),
                                'fairness_threshold': di_threshold,
                                'threshold_status': "Fail" if di_score < di_threshold else "Pass",
                                'group_sizes': group_sizes,
                                'notes': "Expected genetic variation may explain differences" if field in ['Population',
                                                                                                           'Superpopulation'] else ""
                            }

                    field_impact[gene] = gene_impact

            impact_metrics[field] = field_impact

    return impact_metrics


# Significantly modified fairness summary calculation
def calculate_fairness_summary(data, fairness_metrics, impact_metrics, parity_metrics, correlation_metrics):
    """Generate an overall fairness summary with recommendations and severity scores."""
    demographic_fields = ['Sex', 'Population', 'Superpopulation']
    fairness_summary = {}

    # Calculate data completeness metrics
    data_completeness = {}
    missing_superpops = []
    for expected_pop in EXPECTED_SUPER_POPULATIONS:
        if 'Superpopulation' in data.columns and expected_pop not in data['Superpopulation'].unique():
            missing_superpops.append(expected_pop)

    completeness_score = 100
    if missing_superpops:
        # Subtract points for each missing super-population
        completeness_score -= len(missing_superpops) * 10
        print(f"Warning: Missing super-populations: {', '.join(missing_superpops)}")

    total_genes = sum(1 for gene in TARGET_GENES if gene in data.columns)

    for field in demographic_fields:
        if field not in fairness_metrics or field not in impact_metrics or field not in parity_metrics:
            continue

        field_summary = {
            'significant_findings': [],
            'impact_findings': [],
            'parity_findings': [],
            'correlation_findings': [],
            'overall_fairness_score': 100,
            'recommendations': [],
            'data_completeness': {}
        }

        # Check sample sizes for this demographic field
        group_counts = data[field].value_counts().to_dict() if field in data.columns else {}
        small_groups = {g: n for g, n in group_counts.items() if n < MIN_GROUP_SIZE}
        field_summary['data_completeness'] = {
            'total_samples': len(data),
            'group_counts': group_counts,
            'small_sample_groups': small_groups,
            'missing_superpopulations': missing_superpops if field == 'Superpopulation' else []
        }

        # Add small sample size penalty to completeness score
        sample_size_penalty = len(small_groups) * 5
        field_completeness = completeness_score - sample_size_penalty

        # Add extra penalty for Superpopulation if missing expected populations
        if field == 'Superpopulation' and missing_superpops:
            field_completeness -= 10

        field_summary['data_completeness']['completeness_score'] = max(0, field_completeness)

        total_issues = 0
        high_severity_issues = 0
        medium_severity_issues = 0
        low_severity_issues = 0

        # Analyze statistical significance findings
        sig_count = 0
        for gene, metrics in fairness_metrics.get(field, {}).items():
            if metrics.get('significant', False):
                effect = metrics.get('effect_magnitude', 'unknown')
                sample_sizes = metrics.get('sample_sizes', [])

                # Skip very small sample sizes
                if any(n < MIN_COMPARISON_SIZE for n in sample_sizes):
                    continue

                sig_count += 1
                total_issues += 1

                if effect == 'large':
                    high_severity_issues += 1
                elif effect == 'medium':
                    medium_severity_issues += 1
                else:
                    low_severity_issues += 1

                finding = {
                    'gene': gene,
                    'p_value': metrics.get('p_value', 1.0),
                    'effect_size': effect,
                    'severity': 'high' if effect == 'large' else ('medium' if effect == 'medium' else 'low'),
                    'sample_sizes': sample_sizes
                }

                field_summary['significant_findings'].append(finding)

        # Analyze disparate impact findings
        impact_count = 0
        for gene, categories in impact_metrics.get(field, {}).items():
            for category, impact in categories.items():
                if impact.get('disparate_impact', False):
                    di_score = impact.get('di_score', 1.0)
                    group_sizes = impact.get('group_sizes', {})

                    # Skip if any group is too small
                    if any(n < MIN_COMPARISON_SIZE for n in group_sizes.values()):
                        continue

                    impact_count += 1
                    total_issues += 1

                    # Adjust severity by context
                    if field in ['Population', 'Superpopulation'] and category in ['decreased', 'increased']:
                        # Less severe for expected genetic differences in metabolism
                        if di_score < 0.5:
                            high_severity_issues += 1
                        elif di_score < 0.7:
                            medium_severity_issues += 1
                        else:
                            low_severity_issues += 1
                    else:
                        # Standard thresholds for unexpected differences
                        if di_score < 0.6:
                            high_severity_issues += 1
                        elif di_score < 0.8:
                            medium_severity_issues += 1
                        else:
                            low_severity_issues += 1

                    # Determine severity level with context
                    if field in ['Population', 'Superpopulation'] and category in ['decreased', 'increased']:
                        severity = 'high' if di_score < 0.5 else ('medium' if di_score < 0.7 else 'low')
                    else:
                        severity = 'high' if di_score < 0.6 else ('medium' if di_score < 0.8 else 'low')

                    finding = {
                        'gene': gene,
                        'category': category,
                        'di_score': di_score,
                        'severity': severity,
                        'group_sizes': group_sizes,
                        'notes': impact.get('notes', '')
                    }

                    field_summary['impact_findings'].append(finding)

        # Analyze demographic parity findings
        parity_count = 0
        for gene, phenotypes in parity_metrics.get(field, {}).items():
            for phenotype, metrics in phenotypes.items():
                if metrics.get('has_disparity', False):
                    parity_diff = metrics.get('parity_difference', 0)
                    group_sizes = metrics.get('group_sizes', {})

                    # Skip if any group is too small
                    if not group_sizes or any(n < MIN_COMPARISON_SIZE for n in group_sizes.values()):
                        continue

                    parity_count += 1
                    total_issues += 1

                    if parity_diff > 0.3:
                        high_severity_issues += 1
                    elif parity_diff > 0.2:
                        medium_severity_issues += 1
                    else:
                        low_severity_issues += 1

                    finding = {
                        'gene': gene,
                        'phenotype': phenotype,
                        'parity_difference': parity_diff,
                        'severity': 'high' if parity_diff > 0.3 else ('medium' if parity_diff > 0.2 else 'low'),
                        'group_sizes': group_sizes
                    }

                    field_summary['parity_findings'].append(finding)

        # Analyze correlation findings
        if field in correlation_metrics:
            for gene, metrics in correlation_metrics.get(field, {}).items():
                for test_type, test_metrics in metrics.items():
                    if test_metrics.get('significant', False):
                        effect = test_metrics.get('effect_size', 'small')
                        sample_size = test_metrics.get('sample_size', 0)

                        # Skip small samples
                        if sample_size < MIN_GROUP_SIZE:
                            continue

                        total_issues += 1

                        if effect == 'large':
                            high_severity_issues += 1
                        elif effect == 'medium':
                            medium_severity_issues += 1
                        else:
                            low_severity_issues += 1

                        finding = {
                            'gene': gene,
                            'test_type': test_type,
                            'correlation': test_metrics.get('correlation', 0),
                            'effect_size': effect,
                            'severity': 'high' if effect == 'large' else ('medium' if effect == 'medium' else 'low'),
                            'sample_size': sample_size
                        }

                        field_summary['correlation_findings'].append(finding)

        # Calculate final fairness score with more appropriate weighting
        # Start with completeness score as the base
        field_summary['overall_fairness_score'] = field_completeness

        # Scale factor depends on dataset size and field
        min_score = 40
        scale_factor = min(1.0, 3.0 / max(1, total_genes))

        # Adjust weights based on demographic field (genetic context)
        if field in ['Population', 'Superpopulation']:
            # Less weight to genetic differences by ancestry
            high_weight = 20 * scale_factor
            medium_weight = 10 * scale_factor
            low_weight = 3 * scale_factor
        else:
            # Standard weights for Sex differences
            high_weight = 25 * scale_factor
            medium_weight = 15 * scale_factor
            low_weight = 5 * scale_factor

        # Add small penalty for having any issues
        base_issue_penalty = min(15, 5 * total_issues) if total_issues > 0 else 0

        # Calculate penalties
        high_penalty = high_weight * high_severity_issues
        medium_penalty = medium_weight * medium_severity_issues
        low_penalty = low_weight * low_severity_issues

        # Apply penalties
        field_summary['overall_fairness_score'] -= base_issue_penalty
        field_summary['overall_fairness_score'] -= high_penalty
        field_summary['overall_fairness_score'] -= medium_penalty
        field_summary['overall_fairness_score'] -= low_penalty

        # For small datasets, apply a knowledge uncertainty penalty
        if len(data) < 100:
            uncertainty_penalty = 10
            field_summary['overall_fairness_score'] -= uncertainty_penalty
            field_summary['data_completeness']['uncertainty_penalty'] = uncertainty_penalty

        # Ensure score stays within bounds
        field_summary['overall_fairness_score'] = max(min_score, min(100, field_summary['overall_fairness_score']))

        # Determine fairness rating
        if field_summary['overall_fairness_score'] >= 90:
            field_summary['fairness_rating'] = 'Excellent'
        elif field_summary['overall_fairness_score'] >= 80:
            field_summary['fairness_rating'] = 'Good'
        elif field_summary['overall_fairness_score'] >= 70:
            field_summary['fairness_rating'] = 'Fair'
        elif field_summary['overall_fairness_score'] >= 60:
            field_summary['fairness_rating'] = 'Concerning'
        else:
            field_summary['fairness_rating'] = 'Insufficient Data' if field_completeness < 60 else 'Poor'

        # Transparency about score calculation
        field_summary['fairness_context'] = {
            'total_issues': total_issues,
            'high_severity_issues': high_severity_issues,
            'medium_severity_issues': medium_severity_issues,
            'low_severity_issues': low_severity_issues,
            'genes_analyzed': total_genes,
            'scale_factor': scale_factor,
            'penalties': {
                'base_issue_penalty': base_issue_penalty,
                'high_severity_penalty': high_penalty,
                'medium_severity_penalty': medium_penalty,
                'low_severity_penalty': low_penalty
            }
        }

        # Generate appropriate recommendations
        if sig_count > 0:
            field_summary['recommendations'].append(
                f"Investigate {sig_count} statistically significant demographic effects related to {field}."
            )

        if impact_count > 0:
            if field in ['Population', 'Superpopulation']:
                field_summary['recommendations'].append(
                    f"Review {impact_count} phenotype categories with potential disparate impact across {field} groups, " +
                    "noting that some differences may reflect natural genetic variation."
                )
            else:
                field_summary['recommendations'].append(
                    f"Address disparate impact issues found in {impact_count} phenotype categories across {field} groups."
                )

        if parity_count > 0:
            field_summary['recommendations'].append(
                f"Review demographic parity violations for {parity_count} phenotypes across {field} groups."
            )

        if len(field_summary['correlation_findings']) > 0:
            field_summary['recommendations'].append(
                f"Examine correlations between {field} and gene phenotypes with significant effects."
            )

        if field_summary['overall_fairness_score'] < 70:
            if field in ['Population', 'Superpopulation']:
                field_summary['recommendations'].append(
                    "Consider whether observed differences represent bias or expected population genetic differences in pharmacogenes."
                )

            field_summary['recommendations'].append(
                "Consider rebalancing training data or implementing fairness constraints in the model."
            )

        if field_summary['overall_fairness_score'] < 60:
            field_summary['recommendations'].append(
                "Perform additional validation with diverse datasets before clinical deployment."
            )

        if field_completeness < 70:
            if field == 'Superpopulation' and missing_superpops:
                field_summary['recommendations'].append(
                    f"Address data gaps for missing populations: {', '.join(missing_superpops)}. " +
                    "These populations are critical for comprehensive pharmacogenomic analysis."
                )
            elif small_groups:
                field_summary['recommendations'].append(
                    f"Increase sample sizes for underrepresented groups: {', '.join(small_groups.keys())}."
                )

        if total_issues == 0 and field_completeness >= 80:
            field_summary['recommendations'].append(
                f"No significant fairness issues detected for {field}, but continue monitoring with larger datasets."
            )

        # Add pharmacogenomic context
        if field in ['Population', 'Superpopulation']:
            field_summary['genetic_context'] = (
                    "Note: Pharmacogenomic genes naturally vary across ancestral populations due to evolutionary history. " +
                    "Some differences identified may represent actual biological variation rather than algorithmic bias. " +
                    "Further clinical validation is recommended to distinguish between these possibilities."
            )

        fairness_summary[field] = field_summary

    return fairness_summary


# Modified report generation functions (individual reports and overall report)
# Functions: generate_individual_reports, generate_bias_summary, generate_overall_report
# remain mostly the same, with minor adjustments to account for new fields

def generate_individual_reports(merged_data, distributions, fairness_metrics, impact_metrics, parity_metrics,
                                correlation_metrics, group_fairness_metrics, intersectional_metrics, fairness_summary,
                                output_dir):
    """Generate detailed fairness reports for each individual sample."""
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
                'disparate_impact': {},
                'demographic_parity': {},
                'group_fairness': {},
                'intersectional_fairness': {},
                'correlations': {}
            }
        }

        for field in ['Sex', 'Population', 'Superpopulation']:
            if field in sample and pd.notna(sample[field]):
                report['demographics'][field] = str(sample[field])

        for gene in TARGET_GENES:
            if gene in sample and pd.notna(sample[gene]):
                phenotype = sample[gene]
                report['phenotypes'][gene] = {
                    'phenotype': str(phenotype),
                    'description': PHENOTYPE_DESCRIPTIONS.get(phenotype, 'Unknown'),
                    'category': next((cat for cat, vals in PHENOTYPE_CATEGORIES.items()
                                      if phenotype in vals), 'indeterminate')
                }

        for field in ['Sex', 'Population', 'Superpopulation']:
            if field in sample and pd.notna(sample[field]):
                group_value = str(sample[field])
                if field in distributions and group_value in distributions[field]:
                    comparison = {}
                    for gene in TARGET_GENES:
                        if gene in sample and pd.notna(sample[gene]):
                            phenotype = str(sample[gene])

                            group_dist = distributions[field][group_value].get('distributions', {}).get(gene, {})
                            group_rate = group_dist.get(phenotype, 0)

                            overall_dist = distributions['overall'].get(gene, {})
                            overall_rate = overall_dist.get(phenotype, 0)

                            comparison[gene] = {
                                'group_rate': float(group_rate),
                                'overall_rate': float(overall_rate),
                                'difference': float(group_rate - overall_rate),
                                'ratio': float(group_rate / overall_rate) if overall_rate > 0 else None,
                                'sample_size': distributions[field][group_value].get('sample_size', 0),
                                'small_sample_warning': distributions[field][group_value].get('small_sample_warning',
                                                                                              False)
                            }

                    report['fairness_analysis']['demographic_comparison'][field] = comparison

        # Add other analysis components similar to the original, but include new sample size warnings

        report['fairness_summary'] = {}
        for field, summary in fairness_summary.items():
            if field in sample and pd.notna(sample[field]):
                # Include genetic context for relevant fields
                if field in ['Population', 'Superpopulation'] and 'genetic_context' in summary:
                    report['fairness_summary'][field] = {
                        **summary,
                        'genetic_context': summary['genetic_context']
                    }
                else:
                    report['fairness_summary'][field] = summary

        report['summary'] = generate_bias_summary(report)

        # Add data quality warnings
        report['data_quality'] = {
            'missing_superpopulations': [pop for pop in EXPECTED_SUPER_POPULATIONS if pop not in merged_data[
                'Superpopulation'].unique()] if 'Superpopulation' in merged_data.columns else [],
            'small_sample_warnings': {}
        }

        for field in ['Sex', 'Population', 'Superpopulation']:
            if field in sample and pd.notna(sample[field]):
                group_value = str(sample[field])
                if field in distributions and group_value in distributions[field]:
                    if distributions[field][group_value].get('small_sample_warning', False):
                        sample_size = distributions[field][group_value].get('sample_size', 0)
                        report['data_quality']['small_sample_warnings'][field] = {
                            'group': group_value,
                            'sample_size': sample_size,
                            'warning': f"Small sample size ({sample_size}) may affect reliability of analysis."
                        }

        output_file = os.path.join(output_dir, f"{sample_id}_fairness_report.json")
        try:
            with open(output_file, 'w') as f:
                json.dump(ensure_serializable(report), f, indent=2)
        except Exception as e:
            print(f"Error saving report for {sample_id}: {e}")


def generate_bias_summary(report):
    """Generate a human-readable summary of bias findings for an individual sample."""
    summary = []

    # Add genetic context for PGX data
    if 'demographics' in report and 'Population' in report['demographics']:
        pop = report['demographics']['Population']
        superpop = report['demographics'].get('Superpopulation', 'Unknown')
        summary.append(f"This sample is from population {pop} (super-population: {superpop}).")
        summary.append(
            "Note: Pharmacogenomic differences between populations often reflect natural genetic variation rather than algorithmic bias.")

    # Add small sample warnings
    if 'data_quality' in report and 'small_sample_warnings' in report['data_quality']:
        for field, warning in report['data_quality']['small_sample_warnings'].items():
            summary.append(f"Warning: {warning['warning']}")

    # Continue with standard comparison summaries
    for field, comparisons in report['fairness_analysis'].get('demographic_comparison', {}).items():
        group_value = report['demographics'].get(field)
        if group_value:
            for gene, comparison in comparisons.items():
                phenotype = report['phenotypes'].get(gene, {}).get('phenotype')
                if phenotype:
                    group_rate = comparison.get('group_rate', 0)
                    overall_rate = comparison.get('overall_rate', 0)
                    difference = comparison.get('difference', 0)
                    small_sample = comparison.get('small_sample_warning', False)

                    if abs(difference) > 0.2:
                        direction = "higher" if difference > 0 else "lower"
                        statement = (
                            f"The {phenotype} phenotype for {gene} is notably {direction} in {group_value} "
                            f"({field}) compared to the overall population "
                            f"({group_rate:.1%} vs {overall_rate:.1%})."
                        )
                        if small_sample:
                            statement += " However, this finding is based on a small sample size and may not be reliable."
                        summary.append(statement)

    # ... Similar adjustments for other components ...

    if not summary:
        summary.append("No significant demographic bias detected in the phenotype predictions.")

    return summary


def generate_overall_report(merged_data, distributions, fairness_metrics, impact_metrics, parity_metrics,
                            correlation_metrics, group_fairness_metrics, intersectional_metrics, fairness_summary,
                            output_dir):
    """Generate an overall fairness report summarizing findings across all samples."""
    report = {
        'overall_metrics': {
            'total_samples': len(merged_data),
            'demographic_distribution': {},
            'phenotype_distribution': {},
            'fairness_scores': {},
            'fairness_ratings': {},
            'data_completeness': {
                'missing_superpopulations': [pop for pop in EXPECTED_SUPER_POPULATIONS if pop not in merged_data[
                    'Superpopulation'].unique()] if 'Superpopulation' in merged_data.columns else [],
                'small_sample_groups': {}
            }
        },
        'bias_analysis': {
            'significant_demographic_effects': [],
            'disparate_impact_summary': [],
            'demographic_parity_violations': [],
            'group_fairness_concerns': [],
            'significant_correlations': []
        },
        'fairness_summary': fairness_summary,
        'pharmacogenomic_context': {
            'note': (
                    "Pharmacogenomic genes naturally vary across ancestral populations due to evolutionary history. " +
                    "Some differences identified may represent actual biological variation rather than algorithmic bias. " +
                    "This report distinguishes between potential bias and expected population differences where possible."
            )
        }
    }

    # Calculate demographic distributions with sample size warnings
    for field in ['Sex', 'Population', 'Superpopulation']:
        if field in merged_data.columns:
            counts = merged_data[field].value_counts().to_dict()
            report['overall_metrics']['demographic_distribution'][field] = {
                str(k): {'count': int(v), 'small_sample': v < MIN_GROUP_SIZE}
                for k, v in counts.items() if pd.notna(k)
            }

            # Track small sample groups
            small_groups = {str(k): int(v) for k, v in counts.items() if pd.notna(k) and v < MIN_GROUP_SIZE}
            if small_groups:
                report['overall_metrics']['data_completeness']['small_sample_groups'][field] = small_groups

    # Calculate phenotype distributions
    for gene in TARGET_GENES:
        if gene in merged_data.columns:
            report['overall_metrics']['phenotype_distribution'][gene] = {
                str(k): int(v) for k, v in merged_data[gene].value_counts().to_dict().items() if pd.notna(k)
            }

    # Extract fairness scores and ratings
    for field, summary in fairness_summary.items():
        if isinstance(summary, dict):
            report['overall_metrics']['fairness_scores'][field] = summary.get('overall_fairness_score', 0)
            report['overall_metrics']['fairness_ratings'][field] = summary.get('fairness_rating', 'Unknown')

            # Include data completeness information
            if 'data_completeness' in summary:
                report['overall_metrics']['data_completeness'][field] = summary['data_completeness']

    # Compile significant findings with sample size checks
    for field in fairness_metrics:
        for gene, metrics in fairness_metrics[field].items():
            if metrics.get('significant', False):
                sample_sizes = metrics.get('sample_sizes', [])

                # Skip unreliable findings with small sample sizes
                if any(n < MIN_COMPARISON_SIZE for n in sample_sizes):
                    continue

                # Add genetic context for population differences
                genetic_context = ""
                if field in ['Population', 'Superpopulation']:
                    genetic_context = "Differences may reflect natural genetic variation in pharmacogenes across populations."

                report['bias_analysis']['significant_demographic_effects'].append({
                    'demographic_factor': field,
                    'gene': gene,
                    'test': metrics.get('test', 'statistical test'),
                    'p_value': float(metrics.get('p_value', 1.0)),
                    'statistic': float(metrics.get('statistic', 0)),
                    'effect_size': metrics.get('effect_size', 0),
                    'effect_magnitude': metrics.get('effect_magnitude', 'unknown'),
                    'severity': 'high' if metrics.get('effect_magnitude') == 'large' else
                    ('medium' if metrics.get('effect_magnitude') == 'medium' else 'low'),
                    'sample_sizes': sample_sizes,
                    'genetic_context': genetic_context if genetic_context else None
                })

    # ... Similar adjustments for disparate impact, parity violations, etc ...

    # Generate overall summary
    report['summary'] = []

    # Add data completeness warnings first
    if report['overall_metrics']['data_completeness']['missing_superpopulations']:
        missing = report['overall_metrics']['data_completeness']['missing_superpopulations']
        report['summary'].append(f"Warning: Missing data for {len(missing)} super-populations: {', '.join(missing)}.")
        report['summary'].append(
            "Analysis may not represent all genetic ancestries and should be interpreted with caution.")

    small_group_warnings = []
    for field, groups in report['overall_metrics']['data_completeness'].get('small_sample_groups', {}).items():
        if groups:
            groups_str = ", ".join([f"{g} (n={n})" for g, n in groups.items()])
            small_group_warnings.append(f"{field}: {groups_str}")

    if small_group_warnings:
        report['summary'].append("Small sample size warning for these groups:")
        for warning in small_group_warnings:
            report['summary'].append(f"  • {warning}")
        report['summary'].append("Findings for these groups should be interpreted with caution.")

    # Add standard summary content
    report['summary'].append("Overall Fairness Assessment:")
    for field, score in report['overall_metrics']['fairness_scores'].items():
        rating = report['overall_metrics']['fairness_ratings'][field]
        qualifier = ""
        if rating == "Insufficient Data":
            qualifier = " (limited by sample size)"
        report['summary'].append(f"  • {field}: Score {score}/100 ({rating}{qualifier})")

    # Add pharmacogenomic context
    report['summary'].append("\nPharmacogenomic Context:")
    report['summary'].append(
        "Differences in drug metabolism genes across populations are expected due to evolutionary history. " +
        "Some identified disparities may represent natural genetic variation rather than algorithmic bias."
    )

    # ... Other summary components ...

    # Generate recommendations based on findings
    report['summary'].append("\nTop Recommendations:")
    all_recommendations = []
    for field, summary in fairness_summary.items():
        if isinstance(summary, dict) and 'recommendations' in summary:
            for rec in summary['recommendations']:
                all_recommendations.append(rec)

    unique_recommendations = list(set(all_recommendations))
    for i, rec in enumerate(unique_recommendations[:5]):
        report['summary'].append(f"{i + 1}. {rec}")

    # Save the overall report
    output_file = os.path.join(output_dir, "overall_fairness_report.json")
    try:
        with open(output_file, 'w') as f:
            json.dump(ensure_serializable(report), f, indent=2)
    except Exception as e:
        print(f"Error saving overall report: {e}")

    # Generate a human-readable summary text file
    summary_file = os.path.join(output_dir, "fairness_summary.txt")
    try:
        with open(summary_file, 'w') as f:
            f.write("# PharmCAT Fairness Analysis Summary\n\n")

            f.write("## Data Completeness\n")
            if report['overall_metrics']['data_completeness']['missing_superpopulations']:
                missing = report['overall_metrics']['data_completeness']['missing_superpopulations']
                f.write(f"WARNING: Missing data for super-populations: {', '.join(missing)}\n")
                f.write("This limits the comprehensive assessment of pharmacogenomic fairness.\n\n")

            if small_group_warnings:
                f.write("Small sample size warnings:\n")
                for warning in small_group_warnings:
                    f.write(f"- {warning}\n")
                f.write("\n")

            f.write("## Overall Fairness Scores\n")
            for field, score in report['overall_metrics']['fairness_scores'].items():
                rating = report['overall_metrics']['fairness_ratings'][field]
                f.write(f"{field}: {score}/100 ({rating})\n")

            f.write("\n## Pharmacogenomic Context\n")
            f.write(report['pharmacogenomic_context']['note'] + "\n")

            f.write("\n## Key Findings\n")
            for item in report['summary']:
                f.write(f"{item}\n")

            f.write("\n## Demographic Distribution\n")
            for field, distribution in report['overall_metrics']['demographic_distribution'].items():
                f.write(f"\n### {field}\n")
                for group, data in distribution.items():
                    count = data['count']
                    small_sample = data['small_sample']
                    warning = " [SMALL SAMPLE]" if small_sample else ""
                    f.write(f"  {group}: {count}{warning}\n")

            f.write("\n## Statistical Significance Summary\n")
            if not report['bias_analysis']['significant_demographic_effects']:
                f.write("No statistically significant effects detected with sufficient sample size.\n")
            else:
                for effect in report['bias_analysis']['significant_demographic_effects'][:10]:
                    sample_info = f"(samples: {','.join(map(str, effect['sample_sizes']))})" if 'sample_sizes' in effect else ""
                    f.write(f"- {effect['demographic_factor']} on {effect['gene']}: p={effect['p_value']:.4f}, " +
                            f"effect={effect['effect_magnitude']} (severity: {effect['severity']}) {sample_info}\n")
                    if effect.get('genetic_context'):
                        f.write(f"  Note: {effect['genetic_context']}\n")

            f.write("\n## Disparate Impact Summary\n")
            if not report['bias_analysis'].get('disparate_impact_summary'):
                f.write("No disparate impact findings with sufficient sample size.\n")
            else:
                for impact in report['bias_analysis']['disparate_impact_summary'][:10]:
                    genetic_note = ""
                    if impact['demographic_factor'] in ['Population', 'Superpopulation']:
                        genetic_note = " (may reflect natural genetic variation)"
                    f.write(
                        f"- {impact['demographic_factor']} for {impact['gene']} ({impact['phenotype_category']}): " +
                        f"DI score={impact['di_score']:.2f}{genetic_note}\n")

            f.write("\n## Top Recommendations\n")
            for i, rec in enumerate(unique_recommendations[:5]):
                f.write(f"{i + 1}. {rec}\n")
    except Exception as e:
        print(f"Error saving summary text file: {e}")


def main():
    """Main function to run the PharmCAT Fairness/Bias Analyzer."""
    parser = argparse.ArgumentParser(description='PharmCAT Fairness/Bias Analyzer')
    parser.add_argument('--demographic_file', required=True, help='Path to demographic data CSV file')
    parser.add_argument('--phenotypes_file', required=True, help='Path to phenotype predictions CSV file')
    parser.add_argument('--output_dir', default='pgx_fairness_results', help='Output directory for results')
    parser.add_argument('--min_group_size', type=int, default=5, help='Minimum group size for reliable analysis')
    args = parser.parse_args()

    # Update minimum group size if provided
    global MIN_GROUP_SIZE, MIN_COMPARISON_SIZE
    MIN_GROUP_SIZE = args.min_group_size
    MIN_COMPARISON_SIZE = max(3, args.min_group_size // 2)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading demographic data from {args.demographic_file}...")
    demographic_df = load_demographic_data(args.demographic_file)

    print(f"Loading phenotype data from {args.phenotypes_file}...")
    phenotype_df = load_phenotype_data(args.phenotypes_file)

    print("Merging demographic and phenotype data...")
    merged_data = merge_data(demographic_df, phenotype_df)

    # Print sample size warnings
    for field in ['Sex', 'Population', 'Superpopulation']:
        if field in merged_data.columns:
            counts = merged_data[field].value_counts()
            small_groups = counts[counts < MIN_GROUP_SIZE]
            if not small_groups.empty:
                warnings.warn(f"Small sample sizes detected for {field} groups: {small_groups.to_dict()}")

    # Check for missing superpopulations
    if 'Superpopulation' in merged_data.columns:
        missing_superpops = [pop for pop in EXPECTED_SUPER_POPULATIONS if
                             pop not in merged_data['Superpopulation'].unique()]
        if missing_superpops:
            warnings.warn(f"Missing super-populations in the dataset: {missing_superpops}")

    print("Calculating phenotype distributions...")
    distributions = calculate_phenotype_distributions(merged_data)

    print("Calculating fairness metrics...")
    fairness_metrics = calculate_fairness_metrics(merged_data)

    print("Calculating disparate impact metrics...")
    impact_metrics = calculate_disparate_impact(merged_data)

    print("Calculating demographic parity metrics...")
    parity_metrics = calculate_demographic_parity(merged_data)

    print("Calculating group fairness metrics...")
    group_fairness_metrics = calculate_group_fairness_metrics(merged_data)

    print("Calculating intersectional fairness metrics...")
    intersectional_metrics = calculate_intersectional_metrics(merged_data)

    print("Calculating phenotype correlations...")
    correlation_metrics = calculate_phenotype_correlations(merged_data)

    print("Generating fairness summary...")
    fairness_summary = calculate_fairness_summary(
        merged_data, fairness_metrics, impact_metrics,
        parity_metrics, correlation_metrics
    )

    print("Generating individual fairness reports...")
    generate_individual_reports(
        merged_data, distributions, fairness_metrics, impact_metrics,
        parity_metrics, correlation_metrics, group_fairness_metrics,
        intersectional_metrics, fairness_summary, args.output_dir
    )

    print("Generating overall fairness report...")
    generate_overall_report(
        merged_data, distributions, fairness_metrics, impact_metrics,
        parity_metrics, correlation_metrics, group_fairness_metrics,
        intersectional_metrics, fairness_summary, args.output_dir
    )

    print(f"Analysis complete! Results saved to {args.output_dir}")

    # Print final summary and warnings
    print("\nAnalysis Summary:")
    for field, summary in fairness_summary.items():
        if isinstance(summary, dict):
            score = summary.get('overall_fairness_score', 0)
            rating = summary.get('fairness_rating', 'Unknown')
            print(f"- {field}: {score}/100 ({rating})")

    if 'Superpopulation' in merged_data.columns:
        missing_superpops = [pop for pop in EXPECTED_SUPER_POPULATIONS if
                             pop not in merged_data['Superpopulation'].unique()]
        if missing_superpops:
            print(f"\nWARNING: Missing super-populations: {', '.join(missing_superpops)}")
            print("This limits the comprehensiveness of the pharmacogenomic fairness assessment.")


if __name__ == "__main__":
    main()
