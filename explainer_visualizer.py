import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize feature importance analysis results')
    parser.add_argument('--input_file', required=True, help='Input file with analysis results (CSV or JSON)')
    parser.add_argument('--output_dir', required=True, help='Directory to save visualization results')
    return parser.parse_args()


def determine_analysis_type(df):
    if 'P_Value' in df.columns and 'Association' in df.columns:
        return 'categorical_association'
    elif 'Importance' in df.columns:
        return 'mutual_information'
    else:
        raise ValueError("Unable to determine analysis type from input file")


def load_data(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path), None, None
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)

        metadata = data.get('metadata', {})
        sensitivity = metadata.get('sensitivity')
        method = metadata.get('method')

        if isinstance(data, dict) and 'results' in data:
            return pd.DataFrame(data['results']), sensitivity, method
        elif isinstance(data, list):
            return pd.DataFrame(data), None, None
        else:
            flattened_data = []
            for key, value in data.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, dict):
                            row = {'Gene': key, 'Feature': subkey}
                            if 'Association' in subvalue:
                                row['Association'] = subvalue['Association']
                                row['P_Value'] = subvalue.get('P_Value', 0)
                            elif 'Importance' in subvalue:
                                row['Importance'] = subvalue['Importance']
                            flattened_data.append(row)
            return pd.DataFrame(flattened_data), None, None
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def plot_top_features_per_gene(df, gene, output_dir, analysis_type, sensitivity, method):
    gene_df = df[df['Gene'] == gene].copy()

    if len(gene_df) == 0:
        print(f"No data found for gene {gene}, skipping")
        return

    if analysis_type == 'categorical_association':
        gene_df = gene_df.loc[gene_df['Association'].abs().nlargest(10).index]
        importance_col = 'Association'
        color_map = gene_df['Association'].apply(lambda x: 'red' if x < 0 else 'blue')
    else:
        gene_df = gene_df.sort_values('Importance', ascending=False).head(10)
        importance_col = 'Importance'
        color_map = 'blue'

    if len(gene_df) == 0:
        print(f"No features found for gene {gene} after filtering")
        return

    plt.figure(figsize=(10, 6))
    plt.barh(gene_df['Feature'], gene_df[importance_col], color=color_map)

    if sensitivity is not None and method is not None:
        title = f'Sensitivity [0,1]: {sensitivity}, Methodology: {method}'
    else:
        title = f'Top Features for {gene} ({analysis_type})'

    plt.title(title)
    plt.xlabel('Importance' if analysis_type == 'mutual_information' else 'Correlation')
    plt.ylabel('Feature')
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f"{gene}_{analysis_type}.png"))
    plt.close()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        print(f"Loading data from {args.input_file}")
        df, sensitivity, method = load_data(args.input_file)

        if df.empty:
            print("Loaded dataframe is empty. Check the input file format.")
            return

        print(f"Loaded dataframe with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        if sensitivity is not None:
            print(f"Sensitivity: {sensitivity}")
        if method is not None:
            print(f"Method: {method}")

        required_columns = ['Gene', 'Feature']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            print(f"Missing required columns: {missing}")
            if 'gene' in df.columns and 'Feature' not in df.columns:
                df = df.rename(columns={'gene': 'Gene'})
            if 'feature' in df.columns and 'Feature' not in df.columns:
                df = df.rename(columns={'feature': 'Feature'})
            print(f"Adapted columns: {df.columns.tolist()}")

        try:
            analysis_type = determine_analysis_type(df)
            print(f"Detected analysis type: {analysis_type}")
        except ValueError as e:
            print(f"Error determining analysis type: {e}")
            if any(col for col in df.columns if 'importance' in col.lower()):
                analysis_type = 'mutual_information'
                if 'importance' in df.columns and 'Importance' not in df.columns:
                    df = df.rename(columns={'importance': 'Importance'})
            elif any(col for col in df.columns if 'corr' in col.lower()):
                analysis_type = 'correlation'
                if 'association' in df.columns and 'Association' not in df.columns:
                    df = df.rename(columns={'association': 'Association', 'p_value': 'P_Value'})
            else:
                print("Could not determine analysis type, assuming mutual_information")
                analysis_type = 'mutual_information'
                if 'value' in df.columns and 'Importance' not in df.columns:
                    df = df.rename(columns={'value': 'Importance'})
                elif 'importance' not in df.columns and 'Importance' not in df.columns:
                    df['Importance'] = 1.0
            print(f"Inferred analysis type: {analysis_type}")

        target_genes = ["CYP2B6", "CYP2C9", "CYP2C19", "CYP3A5", "SLCO1B1", "TPMT", "DPYD"]
        available_genes = df['Gene'].unique()
        genes_to_visualize = [gene for gene in target_genes if gene in available_genes]

        if not genes_to_visualize:
            print(f"Warning: None of the target genes found in data. Available genes: {available_genes}")
            genes_to_visualize = available_genes

        print(f"Generating visualizations for genes: {', '.join(genes_to_visualize)}")

        for gene in genes_to_visualize:
            try:
                plot_top_features_per_gene(df, gene, args.output_dir, analysis_type, sensitivity, method)
                print(f"Created visualization for {gene}")
            except Exception as e:
                print(f"Error creating visualization for {gene}: {e}")

        print(f"Visualizations saved to {args.output_dir}")
    except Exception as e:
        print(f"An error occurred: {e}")
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Error processing data: {e}",
                 horizontalalignment='center', fontsize=12)
        plt.axis('off')
        plt.savefig(os.path.join(args.output_dir, "error.png"))
        plt.close()
        print(f"Error image saved to {args.output_dir}/error.png")


if __name__ == "__main__":
    main()
