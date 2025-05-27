import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Color mappings for consistent visualization
PHENOTYPE_COLORS = {
    'PM': '#FF5555',  # Poor - Red
    'IM': '#FFA500',  # Intermediate - Orange
    'NM': '#55AA55',  # Normal - Green
    'RM': '#5555FF',  # Rapid - Blue
    'UM': '#AA55AA',  # Ultra Rapid - Purple
    'PF': '#FF5555',  # Poor - Red
    'DF': '#FFA500',  # Decreased - Orange
    'NF': '#55AA55',  # Normal - Green
    'IF': '#5555FF',  # Increased - Blue
    'INDETERMINATE': '#AAAAAA',  # Gray
    ' ': '#CCCCCC'  # Light Gray for empty phenotype
}

GENE_COLORS = {
    'CYP2B6': '#E41A1C',
    'CYP2C9': '#377EB8',
    'CYP2C19': '#4DAF4A',
    'CYP3A5': '#984EA3',
    'SLCO1B1': '#FF7F00',
    'TPMT': '#FFFF33',
    'DPYD': '#A65628'
}

SUPERPOPULATION_COLORS = {
    'AFR': '#66C2A5',  # African
    'AMR': '#FC8D62',  # Ad Mixed American
    'EAS': '#8DA0CB',  # East Asian
    'EUR': '#E78AC3',  # European
    'SAS': '#A6D854'  # South Asian
}


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize pharmacogenomics fairness analysis results')
    parser.add_argument('--input_file', required=True, help='Path to fairness analysis JSON file')
    parser.add_argument('--output_dir', required=True, help='Directory to save visualization outputs')
    return parser.parse_args()


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def ensure_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def visualize_equalized_odds(data, output_dir):
    """Visualize the equalized odds metrics (false positive rates by demographic groups)"""
    if 'equalized_odds_metrics' not in data:
        return

    odds_metrics = data['equalized_odds_metrics']

    for gene, gene_data in odds_metrics.items():
        for demographic, demo_data in gene_data.items():
            # Only visualize superpopulation for clarity
            if demographic != 'Superpopulation':
                continue

            for phenotype, pheno_data in demo_data.items():
                if 'error_rates_by_group' not in pheno_data or len(pheno_data['error_rates_by_group']) < 2:
                    continue

                # Extract false positive rates
                groups = []
                fpr_values = []

                for group, rates in pheno_data['error_rates_by_group'].items():
                    if 'false_positive_rate' in rates and rates['false_positive_rate'] is not None:
                        groups.append(group)
                        fpr_values.append(rates['false_positive_rate'])

                if not groups:
                    continue

                plt.figure(figsize=(10, 6))

                # Use superpopulation colors if available
                colors = [SUPERPOPULATION_COLORS.get(group, '#AAAAAA') for group in groups]

                # Create bar chart for false positive rates
                bars = plt.bar(groups, fpr_values, color=colors)

                # Annotate bars
                for bar, rate in zip(bars, fpr_values):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                             f"{rate:.2f}", ha='center', va='bottom')

                plt.title(f'{gene} {phenotype} False Positive Rate by {demographic}')
                plt.xlabel(demographic)
                plt.ylabel('False Positive Rate')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

                plt.savefig(os.path.join(output_dir, f'fpr_{gene}_{phenotype}_by_{demographic}.png'))
                plt.close()


def visualize_demographic_parity(data, output_dir):
    """Visualize the demographic parity metrics (prediction rates by demographic groups)"""
    if 'demographic_parity_metrics' not in data:
        return

    parity_metrics = data['demographic_parity_metrics']

    for gene, gene_data in parity_metrics.items():
        for demographic, demo_data in gene_data.items():
            # Only visualize superpopulation for clarity
            if demographic != 'Superpopulation':
                continue

            for phenotype, pheno_data in demo_data.items():
                if 'prediction_rates_by_group' not in pheno_data or len(pheno_data['prediction_rates_by_group']) < 2:
                    continue

                groups = list(pheno_data['prediction_rates_by_group'].keys())
                rates = list(pheno_data['prediction_rates_by_group'].values())

                plt.figure(figsize=(10, 6))

                # Use superpopulation colors if available
                colors = [SUPERPOPULATION_COLORS.get(group, '#AAAAAA') for group in groups]

                bars = plt.bar(groups, rates, color=colors)

                # Annotate bars
                for bar, rate in zip(bars, rates):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                             f"{rate:.2f}", ha='center', va='bottom')

                plt.title(f'{gene} {phenotype} Prediction Rate by {demographic}')
                plt.xlabel(demographic)
                plt.ylabel('Prediction Rate')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

                plt.savefig(os.path.join(output_dir, f'prediction_rate_{gene}_{phenotype}_by_{demographic}.png'))
                plt.close()


def create_simple_dashboard(data, output_dir):
    """Create a simple dashboard showing key insights from the data"""
    plt.figure(figsize=(12, 6))
    # Create a heatmap of equalized odds (FPR) for Superpopulation
    if 'equalized_odds_metrics' in data:
        plt.subplot(1, 2, 1)
        # Collect data for heatmap - keep track of gene-phenotype combinations
        heatmap_data = {}
        for gene in data['equalized_odds_metrics']:
            if 'Superpopulation' in data['equalized_odds_metrics'][gene]:
                demo_data = data['equalized_odds_metrics'][gene]['Superpopulation']
                if gene not in heatmap_data:
                    heatmap_data[gene] = {}
                for phenotype in demo_data:
                    if 'error_rates_by_group' in demo_data[phenotype]:
                        for group, rates in demo_data[phenotype]['error_rates_by_group'].items():
                            if 'false_positive_rate' in rates and rates['false_positive_rate'] is not None:
                                # Use maximum FPR across phenotypes for each gene-group combination
                                if group not in heatmap_data[gene]:
                                    heatmap_data[gene][group] = rates['false_positive_rate']
                                else:
                                    heatmap_data[gene][group] = max(heatmap_data[gene][group],
                                                                    rates['false_positive_rate'])
        if heatmap_data:
            # Convert to DataFrame for heatmap
            genes = sorted(heatmap_data.keys())
            groups = sorted(set(group for gene_data in heatmap_data.values() for group in gene_data.keys()))
            matrix_data = []
            for gene in genes:
                row = []
                for group in groups:
                    row.append(heatmap_data.get(gene, {}).get(group, 0))
                matrix_data.append(row)
            pivot_df = pd.DataFrame(matrix_data, index=genes, columns=groups)
            if not pivot_df.empty:
                sns.heatmap(pivot_df, cmap="YlOrRd", annot=True, fmt=".2f", linewidths=.5)
                plt.title('False Positive Rate by Gene & Superpopulation')
            else:
                plt.text(0.5, 0.5, 'Insufficient data for FPR heatmap',
                         ha='center', va='center')
                plt.axis('off')
        else:
            plt.text(0.5, 0.5, 'No FPR data available',
                     ha='center', va='center')
            plt.axis('off')
    # Create a heatmap of demographic parity for Superpopulation
    if 'demographic_parity_metrics' in data:
        plt.subplot(1, 2, 2)
        # Collect data for heatmap - for each gene, show the most common phenotype
        heatmap_data = {}
        for gene in data['demographic_parity_metrics']:
            if 'Superpopulation' in data['demographic_parity_metrics'][gene]:
                demo_data = data['demographic_parity_metrics'][gene]['Superpopulation']
                if gene not in heatmap_data:
                    heatmap_data[gene] = {}
                # Find the phenotype with the most complete data across groups
                best_phenotype = None
                best_coverage = 0
                for phenotype in demo_data:
                    if 'prediction_rates_by_group' in demo_data[phenotype]:
                        coverage = len(demo_data[phenotype]['prediction_rates_by_group'])
                        if coverage > best_coverage:
                            best_coverage = coverage
                            best_phenotype = phenotype
                # Use the best phenotype's data
                if best_phenotype:
                    for group, rate in demo_data[best_phenotype]['prediction_rates_by_group'].items():
                        heatmap_data[gene][group] = rate
        if heatmap_data:
            # Convert to DataFrame for heatmap
            genes = sorted(heatmap_data.keys())
            groups = sorted(set(group for gene_data in heatmap_data.values() for group in gene_data.keys()))
            matrix_data = []
            for gene in genes:
                row = []
                for group in groups:
                    row.append(heatmap_data.get(gene, {}).get(group, 0))
                matrix_data.append(row)
            pivot_df = pd.DataFrame(matrix_data, index=genes, columns=groups)
            if not pivot_df.empty:
                sns.heatmap(pivot_df, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=.5)
                plt.title('Prediction Rate by Gene & Superpopulation')
            else:
                plt.text(0.5, 0.5, 'Insufficient data for prediction rate heatmap',
                         ha='center', va='center')
                plt.axis('off')
        else:
            plt.text(0.5, 0.5, 'No prediction rate data available',
                     ha='center', va='center')
            plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fairness_summary_dashboard.png'))
    plt.close()


def create_metabolizer_dashboards(data, output_dir):
    """Create separate dashboards for each metabolizer type"""

    # Collect all metabolizers from the data
    metabolizers = set()

    # Extract metabolizers from equalized odds metrics
    if 'equalized_odds_metrics' in data:
        for gene in data['equalized_odds_metrics']:
            if 'Superpopulation' in data['equalized_odds_metrics'][gene]:
                demo_data = data['equalized_odds_metrics'][gene]['Superpopulation']
                metabolizers.update(demo_data.keys())

    # Extract metabolizers from demographic parity metrics
    if 'demographic_parity_metrics' in data:
        for gene in data['demographic_parity_metrics']:
            if 'Superpopulation' in data['demographic_parity_metrics'][gene]:
                demo_data = data['demographic_parity_metrics'][gene]['Superpopulation']
                metabolizers.update(demo_data.keys())

    # Create a dashboard for each metabolizer
    for metabolizer in metabolizers:
        create_metabolizer_dashboard(data, metabolizer, output_dir)


def create_metabolizer_dashboard(data, metabolizer, output_dir):
    """Create a dashboard for a specific metabolizer type"""
    plt.figure(figsize=(12, 6))

    # Create a heatmap of equalized odds (FPR) for Superpopulation
    if 'equalized_odds_metrics' in data:
        plt.subplot(1, 2, 1)

        # Collect data for heatmap
        heatmap_data = []

        for gene in data['equalized_odds_metrics']:
            if 'Superpopulation' in data['equalized_odds_metrics'][gene]:
                demo_data = data['equalized_odds_metrics'][gene]['Superpopulation']

                if metabolizer in demo_data and 'error_rates_by_group' in demo_data[metabolizer]:
                    for group, rates in demo_data[metabolizer]['error_rates_by_group'].items():
                        if 'false_positive_rate' in rates and rates['false_positive_rate'] is not None:
                            heatmap_data.append({
                                'Gene': gene,
                                'Group': group,
                                'FPR': rates['false_positive_rate']
                            })

        if heatmap_data:
            df = pd.DataFrame(heatmap_data)

            # Create pivot table for heatmap
            pivot_df = df.pivot_table(
                index='Gene',
                columns='Group',
                values='FPR',
                aggfunc='mean'
            )

            if not pivot_df.empty:
                sns.heatmap(pivot_df, cmap="YlOrRd", annot=True, fmt=".2f", linewidths=.5)
                plt.title(f'False Positive Rate by Gene & Superpopulation ({metabolizer})')
            else:
                plt.text(0.5, 0.5, f'Insufficient data for FPR heatmap for {metabolizer}',
                         ha='center', va='center')
                plt.axis('off')
        else:
            plt.text(0.5, 0.5, f'No FPR data available for {metabolizer}',
                     ha='center', va='center')
            plt.axis('off')

    # Create a heatmap of demographic parity for Superpopulation
    if 'demographic_parity_metrics' in data:
        plt.subplot(1, 2, 2)

        # Collect data for heatmap
        heatmap_data = []

        for gene in data['demographic_parity_metrics']:
            if 'Superpopulation' in data['demographic_parity_metrics'][gene]:
                demo_data = data['demographic_parity_metrics'][gene]['Superpopulation']

                if metabolizer in demo_data and 'prediction_rates_by_group' in demo_data[metabolizer]:
                    for group, rate in demo_data[metabolizer]['prediction_rates_by_group'].items():
                        heatmap_data.append({
                            'Gene': gene,
                            'Group': group,
                            'Rate': rate
                        })

        if heatmap_data:
            df = pd.DataFrame(heatmap_data)

            # Create pivot table for heatmap
            pivot_df = df.pivot_table(
                index='Gene',
                columns='Group',
                values='Rate',
                aggfunc='mean'
            )

            if not pivot_df.empty:
                sns.heatmap(pivot_df, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=.5)
                plt.title(f'Prediction Rate by Gene & Superpopulation ({metabolizer})')
            else:
                plt.text(0.5, 0.5, f'Insufficient data for prediction rate heatmap for {metabolizer}',
                         ha='center', va='center')
                plt.axis('off')
        else:
            plt.text(0.5, 0.5, f'No prediction rate data available for {metabolizer}',
                     ha='center', va='center')
            plt.axis('off')

    plt.tight_layout()
    safe_metabolizer = str(metabolizer).replace('/', '_').replace('\\', '_').replace(' ', '_')
    plt.savefig(os.path.join(output_dir, f'metabolizer_{safe_metabolizer}_dashboard.png'))
    plt.close()


def main():
    """Main function to run the visualizations"""
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    try:
        data = load_json(args.input_file)
        print(f"Loaded data from {args.input_file}")

        # Generate visualizations
        visualize_equalized_odds(data, output_dir)
        visualize_demographic_parity(data, output_dir)
        create_simple_dashboard(data, output_dir)
        create_metabolizer_dashboards(data, output_dir)

        print(f"All visualizations saved to {output_dir}")
    except Exception as e:
        print(f"Error processing the analysis file: {e}")
        # Create an error image
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"Error: {str(e)}",
                 ha='center', va='center', fontsize=12, color='red')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'error.png'))
        plt.close()


if __name__ == "__main__":
    main()
