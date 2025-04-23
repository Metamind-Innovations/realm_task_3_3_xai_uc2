import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize feature importance analysis results')
    parser.add_argument('--input_file', required=True, help='Input CSV file with analysis results')
    parser.add_argument('--output_dir', required=True, help='Directory to save visualization results')
    return parser.parse_args()


def determine_analysis_type(df):
    if 'P_Value' in df.columns and 'Correlation' in df.columns:
        return 'correlation'
    elif 'Importance' in df.columns:
        return 'mutual_information'
    else:
        raise ValueError("Unable to determine analysis type from input file")


def plot_top_features_per_gene(df, gene, output_dir, analysis_type):
    gene_df = df[df['Gene'] == gene].copy()

    if analysis_type == 'correlation':
        gene_df = gene_df.sort_values('Abs_Correlation', ascending=False).head(10)
        importance_col = 'Correlation'
        title = f'Top Features for {gene} (Correlation)'
        color_map = gene_df['Correlation'].apply(lambda x: 'red' if x < 0 else 'blue')
    else:
        gene_df = gene_df.sort_values('Importance', ascending=False).head(10)
        importance_col = 'Importance'
        title = f'Top Features for {gene} (Mutual Information)'
        color_map = 'blue'

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

    if analysis_type == 'correlation':
        top_features_per_gene = {}
        for gene in genes_to_use:
            top_features_per_gene[gene] = df[(df['Gene'] == gene)].nlargest(5, 'Abs_Correlation')['Feature'].tolist()

        value_col = 'Correlation'
        title = 'Correlation Heatmap'
        cmap = 'coolwarm'
        center = 0
    else:
        top_features_per_gene = {}
        for gene in genes_to_use:
            top_features_per_gene[gene] = df[(df['Gene'] == gene)].nlargest(5, 'Importance')['Feature'].tolist()

        value_col = 'Importance'
        title = 'Feature Importance Heatmap'
        cmap = 'viridis'
        center = None

    top_features = []
    for features in top_features_per_gene.values():
        top_features.extend(features)
    top_features = list(set(top_features))

    filtered_df = df[(df['Gene'].isin(genes_to_use)) & (df['Feature'].isin(top_features))]

    pivot_df = filtered_df.pivot(index='Gene', columns='Feature', values=value_col)

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, cmap=cmap, center=center, annot=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{analysis_type}_heatmap.png"))
    plt.close()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_file)
    analysis_type = determine_analysis_type(df)
    print(f"Detected analysis type: {analysis_type}")

    target_genes = ["CYP2B6", "CYP2C9", "CYP2C19", "CYP3A5", "SLCO1B1", "TPMT", "DPYD"]
    available_genes = df['Gene'].unique()
    genes_to_visualize = [gene for gene in target_genes if gene in available_genes]

    print(f"Generating visualizations for genes: {', '.join(genes_to_visualize)}")

    for gene in genes_to_visualize:
        plot_top_features_per_gene(df, gene, args.output_dir, analysis_type)

    create_heatmap(df, args.output_dir, analysis_type)

    print(f"Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
