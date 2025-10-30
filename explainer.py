import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import chi2_contingency


def load_data(input_file, output_file):
    input_df = pd.read_csv(input_file)
    output_df = pd.read_csv(output_file)

    if len(input_df) != len(output_df):
        raise ValueError(f"Input and output files have different numbers of rows: {len(input_df)} vs {len(output_df)}")

    return input_df, output_df


def cramers_v(x, y):
    """
    Calculate Cramer's V statistic for categorical association between two variables.
    """
    contingency = pd.crosstab(x, y)

    chi2, p, dof, expected = chi2_contingency(contingency)

    n = contingency.sum().sum()
    phi2 = chi2 / n

    r, k = contingency.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)

    v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1))) if min((kcorr - 1), (rcorr - 1)) > 0 else 0
    return v, p


def categorical_association_analysis(input_df, output_df):
    associations = defaultdict(dict)

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

            mask = ~(np.isnan(input_values) | np.isnan(output_values))
            if np.sum(mask) < 5:
                continue

            try:
                v, p_value = cramers_v(input_values[mask], output_values[mask])

                if not np.isnan(v) and v > 0:
                    associations[output_col][input_col] = (v, p_value)
            except Exception as e:  # Skip cases that can't be processed
                pass

    return associations


def mutual_information_analysis(input_df, output_df, random_state=42):
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

        mi_values = mutual_info_classif(input_numeric, output_values, discrete_features=True, random_state=random_state)

        for idx, input_col in enumerate(input_numeric.columns):
            mi_scores[output_col][input_col] = mi_values[idx]

    return mi_scores


def run_analysis(input_file, output_file, sensitivity):
    print("Loading data...")
    input_df, output_df = load_data(input_file, output_file)
    print(f"Loaded data with {len(input_df)} rows")

    method = "categorical_association" if sensitivity >= 0.5 else "mutual_information"

    print(f"Using method: {method} based on sensitivity: {sensitivity}")

    if method == "categorical_association":
        return categorical_association_analysis(input_df, output_df), method
    else:
        return mutual_information_analysis(input_df, output_df), method


def save_results(results, output_path, method, sensitivity):
    os.makedirs(output_path, exist_ok=True)

    if method == "categorical_association":
        result_df = pd.DataFrame(columns=["Gene", "Feature", "Association", "P_Value"])

        for gene, features in results.items():
            for feature, (association, p_value) in features.items():
                result_df = pd.concat([result_df, pd.DataFrame({
                    "Gene": [gene],
                    "Feature": [feature],
                    "Association": [association],
                    "P_Value": [p_value if p_value is not None else float('nan')]
                })], ignore_index=True)

        result_df = result_df.sort_values(by=["Gene", "Association"], ascending=[True, False])
        output_file = os.path.join(output_path, "categorical_association_analysis.json")

        json_data = {
            "metadata": {
                "sensitivity": sensitivity,
                "method": method
            },
            "results": result_df.to_dict(orient='records')
        }
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)

    elif method == "mutual_information":
        result_df = pd.DataFrame(columns=["Gene", "Feature", "Importance"])

        for gene, features in results.items():
            for feature, importance in features.items():
                result_df = pd.concat([result_df, pd.DataFrame({
                    "Gene": [gene],
                    "Feature": [feature],
                    "Importance": [importance]
                })], ignore_index=True)

        result_df = result_df.sort_values(by=["Gene", "Importance"], ascending=[True, False])
        output_file = os.path.join(output_path, "mutual_information_analysis.json")

        json_data = {
            "metadata": {
                "sensitivity": sensitivity,
                "method": method
            },
            "results": result_df.to_dict(orient='records')
        }
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)

    print(f"Results saved as JSON to {output_file}")

    return result_df


def main():
    parser = argparse.ArgumentParser(description='Analyze and explain feature importance')
    parser.add_argument('--input_file', required=True, help='Input file with encoded genetic data')
    parser.add_argument('--output_file', required=True, help='Output file with encoded phenotypes')
    parser.add_argument('--results_dir', default='explainer_results', help='Directory to save results')
    parser.add_argument('--sensitivity', type=float, default=0.7,
                        help='Sensitivity value (0-1): <0.5 uses mutual_information, >=0.5 uses categorical_association')
    args = parser.parse_args()

    if not 0 <= args.sensitivity <= 1:
        raise ValueError("Sensitivity must be between 0 and 1")

    results, method = run_analysis(args.input_file, args.output_file, args.sensitivity)
    result_df = save_results(results, args.results_dir, method, args.sensitivity)

    print(f"Analysis complete using {method} method. Results saved to {args.results_dir} as JSON")

    if method == "categorical_association":
        top_features = result_df.sort_values("Association", ascending=False).head(10)
        print("\nTop 10 features by association strength:")
        print(top_features[["Gene", "Feature", "Association", "P_Value"]].to_string(index=False))
    else:
        top_features = result_df.sort_values("Importance", ascending=False).head(10)
        print("\nTop 10 features by mutual information:")
        print(top_features[["Gene", "Feature", "Importance"]].to_string(index=False))


if __name__ == "__main__":
    main()
