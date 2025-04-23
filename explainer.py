import os
import pandas as pd
import numpy as np
import argparse
from scipy.stats import pointbiserialr, spearmanr
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from collections import defaultdict


def load_data(input_file, output_file):
    input_df = pd.read_csv(input_file)
    output_df = pd.read_csv(output_file)

    if len(input_df) != len(output_df):
        raise ValueError(f"Input and output files have different numbers of rows: {len(input_df)} vs {len(output_df)}")

    return input_df, output_df


def correlation_analysis(input_df, output_df):
    correlations = defaultdict(dict)

    if 'PATIENT_ID' in input_df.columns:
        input_numeric = input_df.drop(columns=['PATIENT_ID'])
    else:
        input_numeric = input_df

    input_numeric = input_numeric.select_dtypes(include=[np.number])

    if 'Sample ID' in output_df.columns:
        output_numeric = output_df.drop(columns=['Sample ID'])
    else:
        output_numeric = output_df

    for output_col in output_numeric.columns:
        output_values = output_numeric[output_col].values

        for input_col in input_numeric.columns:
            input_values = input_numeric[input_col].values

            if np.unique(input_values).size <= 2:
                try:
                    corr, p_value = pointbiserialr(input_values, output_values)
                    if not np.isnan(corr):
                        correlations[output_col][input_col] = (corr, p_value)
                except:
                    pass
            else:
                try:
                    corr, p_value = spearmanr(input_values, output_values)
                    if not np.isnan(corr):
                        correlations[output_col][input_col] = (corr, p_value)
                except:
                    pass

    return correlations


def mutual_information_analysis(input_df, output_df):
    mi_scores = defaultdict(dict)

    if 'PATIENT_ID' in input_df.columns:
        input_numeric = input_df.drop(columns=['PATIENT_ID'])
    else:
        input_numeric = input_df

    input_numeric = input_numeric.select_dtypes(include=[np.number])

    if 'Sample ID' in output_df.columns:
        output_numeric = output_df.drop(columns=['Sample ID'])
    else:
        output_numeric = output_df

    for output_col in output_numeric.columns:
        output_values = output_numeric[output_col].values

        if np.unique(output_values).size <= 5:
            mi_values = mutual_info_classif(input_numeric, output_values, discrete_features='auto')
        else:
            mi_values = mutual_info_regression(input_numeric, output_values, discrete_features='auto')

        for idx, input_col in enumerate(input_numeric.columns):
            mi_scores[output_col][input_col] = mi_values[idx]

    return mi_scores


def feature_permutation_analysis(input_df, output_df, n_permutations=20):
    if 'PATIENT_ID' in input_df.columns:
        input_numeric = input_df.drop(columns=['PATIENT_ID'])
    else:
        input_numeric = input_df

    input_numeric = input_numeric.select_dtypes(include=[np.number])

    if 'Sample ID' in output_df.columns:
        output_numeric = output_df.drop(columns=['Sample ID'])
    else:
        output_numeric = output_df

    baseline_concordance = calculate_prediction_concordance(input_numeric, output_numeric)
    importance_scores = defaultdict(dict)

    for input_col in input_numeric.columns:
        permutation_concordances = []

        for _ in range(n_permutations):
            permuted_input = input_numeric.copy()
            permuted_input[input_col] = np.random.permutation(permuted_input[input_col].values)

            perm_concordance = calculate_prediction_concordance(permuted_input, output_numeric)
            permutation_concordances.append(perm_concordance)

        mean_perm_concordance = np.mean(permutation_concordances)
        importance = baseline_concordance - mean_perm_concordance

        for output_col in output_numeric.columns:
            importance_scores[output_col][input_col] = importance

    return importance_scores


def calculate_prediction_concordance(input_df, output_df):
    return np.mean([output_df[col].nunique() == 1 for col in output_df.columns])


def run_feature_importance_analysis(input_file, output_file):
    print("Loading data...")
    input_df, output_df = load_data(input_file, output_file)
    print(f"Loaded data with {len(input_df)} rows")

    print("Running correlation analysis...")
    correlation_results = correlation_analysis(input_df, output_df)

    print("Running mutual information analysis...")
    mi_results = mutual_information_analysis(input_df, output_df)

    print("Running feature permutation analysis...")
    permutation_results = feature_permutation_analysis(input_df, output_df)

    # Combine all results for overall importance
    all_scores = defaultdict(lambda: defaultdict(dict))

    for output_col in correlation_results:
        for input_col in correlation_results[output_col]:
            corr, p_value = correlation_results[output_col][input_col]
            all_scores[output_col][input_col]["correlation"] = corr
            all_scores[output_col][input_col]["p_value"] = p_value
            all_scores[output_col][input_col]["abs_correlation"] = abs(corr)

    for output_col in mi_results:
        for input_col in mi_results[output_col]:
            all_scores[output_col][input_col]["mutual_info"] = mi_results[output_col][input_col]

    for output_col in permutation_results:
        for input_col in permutation_results[output_col]:
            all_scores[output_col][input_col]["permutation"] = permutation_results[output_col][input_col]

    return all_scores


def save_results(all_scores, output_path):
    os.makedirs(output_path, exist_ok=True)

    # Prepare detailed data for comprehensive analysis file
    detailed_data = []

    for output_col, features in all_scores.items():
        for input_col, metrics in features.items():
            # Calculate combined importance score (weighted average of normalized metrics)
            abs_corr = metrics.get("abs_correlation", 0)
            mi_score = metrics.get("mutual_info", 0)
            perm_score = metrics.get("permutation", 0)

            # Simple weighted average for combined importance
            combined_score = (0.4 * abs_corr +
                              0.4 * mi_score +
                              0.2 * perm_score)

            detailed_data.append({
                "Gene": output_col,
                "Feature": input_col,
                "Combined_Importance": combined_score,
                "Correlation": metrics.get("correlation", 0),
                "P_Value": metrics.get("p_value", 1.0),
                "Mutual_Information": metrics.get("mutual_info", 0),
                "Permutation_Importance": metrics.get("permutation", 0)
            })

    # Create comprehensive feature importance file
    comprehensive_df = pd.DataFrame(detailed_data)
    comprehensive_df.to_csv(os.path.join(output_path, 'comprehensive_feature_importance.csv'), index=False)

    # Create a summary file with top features per gene
    summary_data = []
    for gene, gene_data in comprehensive_df.groupby("Gene"):
        # Get top 10 features per gene
        top_features = gene_data.sort_values("Combined_Importance", ascending=False).head(10)

        # Add rank information
        top_features = top_features.copy()
        top_features["Rank"] = range(1, len(top_features) + 1)

        summary_data.append(top_features[["Gene", "Rank", "Feature", "Combined_Importance", "Correlation", "P_Value"]])

    summary_df = pd.concat(summary_data)
    summary_df.to_csv(os.path.join(output_path, 'top_features_summary.csv'), index=False)

    return comprehensive_df, summary_df


def main():
    parser = argparse.ArgumentParser(description='Analyze and explain feature importance')
    parser.add_argument('--input_file', required=True, help='Input file with encoded genetic data')
    parser.add_argument('--output_file', required=True, help='Output file with encoded phenotypes')
    parser.add_argument('--results_dir', default='explainer_results', help='Directory to save results')
    args = parser.parse_args()

    # Ensure results directory exists
    os.makedirs(args.results_dir, exist_ok=True)

    all_scores = run_feature_importance_analysis(args.input_file, args.output_file)
    comprehensive_df, summary_df = save_results(all_scores, args.results_dir)

    print("\nAnalysis complete! Results saved to", args.results_dir)
    print("\nTop features summary saved as 'top_features_summary.csv'")
    print("Complete feature importance metrics saved as 'comprehensive_feature_importance.csv'")

    print("\nTop 5 features by importance:")
    top_overall = comprehensive_df.sort_values("Combined_Importance", ascending=False).head(5)
    print(top_overall[["Gene", "Feature", "Combined_Importance"]].to_string(index=False))


if __name__ == "__main__":
    main()
