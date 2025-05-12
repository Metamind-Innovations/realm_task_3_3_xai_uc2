import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import json


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
    """Load data from either CSV or JSON format"""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
        # Convert JSON to DataFrame
        if isinstance(data, list):
            return pd.DataFrame(data)
        else:
            # Handle nested JSON structure
            # This assumes a specific structure - modify as needed for your actual JSON structure
            if 'results' in data:
                return pd.DataFrame(data['results'])
            else:
                # Try to flatten the JSON structure
                flattened_data = []
                for key, value in data.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, dict):
                                row = {'Gene': key, 'Feature': subkey}
                                if 'Association' in subvalue:
                                    row['Association'] = subvalue['Association']
                                    row['P_Value'] = subvalue.get('P_Value', 0)
                                    row['Abs_Association'] = abs(subvalue['Association'])
                                elif 'Importance' in subvalue:
                                    row['Importance'] = subvalue['Importance']
                                flattened_data.append(row)
                return pd.DataFrame(flattened_data)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def plot_top_features_per_gene(df, gene, output_dir, analysis_type):
    gene_df = df[df['Gene'] == gene].copy()

    if len(gene_df) == 0:
        print(f"No data found for gene {gene}, skipping")
        return

    if analysis_type == 'categorical_association':
        gene_df = gene_df.sort_values('Abs_Association', ascending=False).head(10)
        importance_col = 'Association'
        title = f'Top Features for {gene} (Association)'
        color_map = gene_df['Association'].apply(lambda x: 'red' if x < 0 else 'blue')
    else:
        gene_df = gene_df.sort_values('Importance', ascending=False).head(10)
        importance_col = 'Importance'
        title = f'Top Features for {gene} (Mutual Information)'
        color_map = 'blue'

    if len(gene_df) == 0:
        print(f"No features found for gene {gene} after filtering")
        return

    plt.figure(figsize=(10, 6))
    plt.barh(gene_df['Feature'], gene_df[importance_col], color=color_map)
    plt.title(title)
    plt.xlabel('Importance' if analysis_type == 'mutual_information' else 'Correlation')
    plt.ylabel('Feature')
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f"{gene}_{analysis_type}.png"))
    plt.close()


def create_heatmap(df, output_dir, analysis_type):
    target_genes = ["CYP2B6", "CYP2C9", "CYP2C19", "CYP3A5", "SLCO1B1", "TPMT", "DPYD"]
    available_genes = df['Gene'].unique()
    genes_to_use = [gene for gene in target_genes if gene in available_genes]

    if not genes_to_use:
        print("No target genes found in the data")
        return

    if analysis_type == 'categorical_association':
        top_features_per_gene = {}
        for gene in genes_to_use:
            gene_df = df[df['Gene'] == gene]
            if len(gene_df) == 0:
                continue
            top_features_per_gene[gene] = gene_df.nlargest(5, 'Abs_Association')['Feature'].tolist()

        value_col = 'Association'
        title = 'Feature Association Heatmap'
        cmap = 'coolwarm'
        center = 0
    else:
        top_features_per_gene = {}
        for gene in genes_to_use:
            gene_df = df[df['Gene'] == gene]
            if len(gene_df) == 0:
                continue
            top_features_per_gene[gene] = gene_df.nlargest(5, 'Importance')['Feature'].tolist()

        value_col = 'Importance'
        title = 'Feature Importance Heatmap'
        cmap = 'viridis'
        center = None

    top_features = []
    for features in top_features_per_gene.values():
        top_features.extend(features)
    top_features = list(set(top_features))

    if not top_features:
        print("No top features found to create heatmap")
        return

    # Filter dataframe for top features
    filtered_df = df[(df['Gene'].isin(genes_to_use)) & (df['Feature'].isin(top_features))]

    if len(filtered_df) == 0:
        print("No data available for heatmap after filtering")
        return

    # Create pivot table for heatmap
    try:
        pivot_df = filtered_df.pivot_table(index='Gene', columns='Feature', values=value_col)

        # Handle the case where the pivot table is empty
        if pivot_df.empty:
            print("Pivot table is empty, cannot create heatmap")
            return

        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, cmap=cmap, center=center, annot=True, fmt=".2f", linewidths=.5)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{analysis_type}_heatmap.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating heatmap: {e}")
        # Create a simple alternative visualization
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, f"Could not create heatmap: {e}",
                 horizontalalignment='center', fontsize=12)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"{analysis_type}_error.png"))
        plt.close()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        print(f"Loading data from {args.input_file}")
        df = load_data(args.input_file)

        if df.empty:
            print("Loaded dataframe is empty. Check the input file format.")
            return

        print(f"Loaded dataframe with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Check if required columns exist
        required_columns = ['Gene', 'Feature']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            print(f"Missing required columns: {missing}")
            # Try to adapt to the actual data format
            if 'gene' in df.columns and 'Feature' not in df.columns:
                df = df.rename(columns={'gene': 'Gene'})
            if 'feature' in df.columns and 'Feature' not in df.columns:
                df = df.rename(columns={'feature': 'Feature'})
            print(f"Adapted columns: {df.columns.tolist()}")

        # Determine analysis type
        try:
            analysis_type = determine_analysis_type(df)
            print(f"Detected analysis type: {analysis_type}")
        except ValueError as e:
            print(f"Error determining analysis type: {e}")
            # Try to infer the type from columns
            if any(col for col in df.columns if 'importance' in col.lower()):
                analysis_type = 'mutual_information'
                if 'importance' in df.columns and 'Importance' not in df.columns:
                    df = df.rename(columns={'importance': 'Importance'})
            elif any(col for col in df.columns if 'corr' in col.lower()):
                analysis_type = 'correlation'
                if 'association' in df.columns and 'Association' not in df.columns:
                    df = df.rename(columns={'association': 'Association', 'p_value': 'P_Value'})
                if 'Association' in df.columns and 'Abs_Association' not in df.columns:
                    df['Abs_Association'] = df['Association'].abs()
            else:
                print("Could not determine analysis type, assuming mutual_information")
                analysis_type = 'mutual_information'
                if 'value' in df.columns and 'Importance' not in df.columns:
                    df = df.rename(columns={'value': 'Importance'})
                elif 'importance' not in df.columns and 'Importance' not in df.columns:
                    # Create a placeholder importance column
                    df['Importance'] = 1.0
            print(f"Inferred analysis type: {analysis_type}")

        target_genes = ["CYP2B6", "CYP2C9", "CYP2C19", "CYP3A5", "SLCO1B1", "TPMT", "DPYD"]
        available_genes = df['Gene'].unique()
        genes_to_visualize = [gene for gene in target_genes if gene in available_genes]

        if not genes_to_visualize:
            print(f"Warning: None of the target genes found in data. Available genes: {available_genes}")
            # If no target genes found, use whatever genes are available
            genes_to_visualize = available_genes

        print(f"Generating visualizations for genes: {', '.join(genes_to_visualize)}")

        for gene in genes_to_visualize:
            try:
                plot_top_features_per_gene(df, gene, args.output_dir, analysis_type)
                print(f"Created visualization for {gene}")
            except Exception as e:
                print(f"Error creating visualization for {gene}: {e}")

        try:
            create_heatmap(df, args.output_dir, analysis_type)
            print("Created heatmap visualization")
        except Exception as e:
            print(f"Error creating heatmap: {e}")

        print(f"Visualizations saved to {args.output_dir}")
    except Exception as e:
        print(f"An error occurred: {e}")
        # Create an error image to ensure the component doesn't fail completely
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Error processing data: {e}",
                 horizontalalignment='center', fontsize=12)
        plt.axis('off')
        plt.savefig(os.path.join(args.output_dir, "error.png"))
        plt.close()
        print(f"Error image saved to {args.output_dir}/error.png")


if __name__ == "__main__":
    main()
