import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
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


def visualize_demographic_distribution(data, output_dir):
    """Visualize the demographic distribution from the analysis"""
    if 'metadata' not in data or 'demographics' not in data['metadata']:
        return

    demographics = data['metadata']['demographics']

    for demo_key, distribution in demographics.items():
        plt.figure(figsize=(10, 6))

        # Sort by count for better visualization
        sorted_items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        groups = [item[0] for item in sorted_items]
        counts = [item[1] for item in sorted_items]

        # Determine colors based on the demographic type
        if 'superpopulation' in demo_key.lower():
            colors = [SUPERPOPULATION_COLORS.get(group, '#AAAAAA') for group in groups]
        else:
            colors = sns.color_palette('pastel', len(groups))

        # Create bars
        bars = plt.bar(groups, counts, color=colors)

        # Annotate bars with counts
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f"{count}", ha='center', va='bottom')

        # Add small sample warning for groups with less than 5 samples
        small_samples = [group for group, count in zip(groups, counts) if count < 5]
        if small_samples:
            plt.figtext(0.01, 0.01, f"* Small sample size (n < 5): {', '.join(small_samples)}", color='red', fontsize=8)

        plt.title(f'Sample Distribution by {demo_key.replace("_", " ").title()}')
        plt.xlabel(demo_key.replace("_", " ").title())
        plt.ylabel('Number of samples')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, f'demographic_distribution_{demo_key}.png'))
        plt.close()


def visualize_phenotype_distribution(data, output_dir):
    """Visualize the phenotype distribution for each gene"""
    if 'phenotype_distributions' not in data:
        return

    phenotype_dist = data['phenotype_distributions']

    for gene, gene_data in phenotype_dist.items():
        if 'overall' not in gene_data:
            continue

        plt.figure(figsize=(10, 6))

        # Get overall phenotype distribution
        distribution = gene_data['overall']

        # Sort phenotypes for better visualization (maintaining typical metabolizer order)
        metabolizer_order = ['PM', 'IM', 'NM', 'RM', 'UM', 'PF', 'DF', 'NF', 'IF', 'INDETERMINATE', ' ']
        phenotypes = sorted(distribution.keys(),
                            key=lambda x: metabolizer_order.index(x) if x in metabolizer_order else 999)
        frequencies = [distribution[p] for p in phenotypes]

        # Colors for phenotypes
        colors = [PHENOTYPE_COLORS.get(p, '#AAAAAA') for p in phenotypes]

        # Create bars
        bars = plt.bar(phenotypes, frequencies, color=colors)

        # Annotate bars with frequencies
        for bar, freq in zip(bars, frequencies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f"{freq:.2f}", ha='center', va='bottom')

        plt.title(f'Phenotype Distribution for {gene}')
        plt.xlabel('Phenotype')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.1)  # Set y-axis limit to accommodate annotations
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, f'phenotype_distribution_{gene}.png'))
        plt.close()

        # Create a demographic breakdown of phenotypes
        for demographic, demo_data in gene_data.items():
            if demographic == 'overall':
                continue

            # Only visualize superpopulation for clarity
            if demographic != 'Superpopulation':
                continue

            plt.figure(figsize=(12, 8))

            # Create a DataFrame for easier plotting
            demo_df = []
            for group, group_data in demo_data.items():
                for pheno, freq in group_data['distribution'].items():
                    demo_df.append({
                        'Group': group,
                        'Phenotype': pheno,
                        'Frequency': freq,
                        'Count': int(freq * group_data['sample_size']),
                        'Sample_Size': group_data['sample_size']
                    })

            demo_df = pd.DataFrame(demo_df)

            # Create pivot table
            pivot = demo_df.pivot(index='Group', columns='Phenotype', values='Frequency')
            pivot = pivot.fillna(0)

            # Sort columns by metabolizer order
            ordered_cols = [col for col in metabolizer_order if col in pivot.columns]
            other_cols = [col for col in pivot.columns if col not in metabolizer_order]
            pivot = pivot[ordered_cols + other_cols]

            # Create heatmap
            sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=.5, cbar_kws={'label': 'Frequency'})

            plt.title(f'{gene} Phenotype Distribution by {demographic}')
            plt.tight_layout()

            plt.savefig(os.path.join(output_dir, f'phenotype_distribution_{gene}_by_{demographic}.png'))
            plt.close()


def visualize_equalized_odds(data, output_dir):
    """Visualize the equalized odds metrics (error rates by demographic groups)"""
    if 'equalized_odds_metrics' not in data:
        return

    odds_metrics = data['equalized_odds_metrics']

    for gene, gene_data in odds_metrics.items():
        for demographic, demo_data in gene_data.items():
            # Only visualize superpopulation for clarity
            if demographic != 'Superpopulation':
                continue

            # Create a summary of disparities
            plt.figure(figsize=(12, 8))

            # Collect disparity data
            phenotypes = []
            tpr_disparities = []
            fpr_disparities = []

            for phenotype, pheno_data in demo_data.items():
                if 'disparity' not in pheno_data:
                    continue

                disparity = pheno_data['disparity']

                # Check if TPR disparity exists and is not null
                if 'true_positive_rate' in disparity and disparity['true_positive_rate'] is not None:
                    phenotypes.append(phenotype)
                    tpr_disparities.append(disparity['true_positive_rate'])

                    # Check if FPR disparity exists
                    if 'false_positive_rate' in disparity and disparity['false_positive_rate'] is not None:
                        fpr_disparities.append(disparity['false_positive_rate'])
                    else:
                        fpr_disparities.append(0)

            if not phenotypes:
                continue

            # Set up the bar positions
            x = np.arange(len(phenotypes))
            width = 0.35

            fig, ax = plt.subplots(figsize=(12, 8))

            # Create grouped bars
            bar1 = ax.bar(x - width / 2, tpr_disparities, width, label='True Positive Rate Disparity')
            bar2 = ax.bar(x + width / 2, fpr_disparities, width, label='False Positive Rate Disparity')

            # Add labels and title
            ax.set_xlabel('Phenotype')
            ax.set_ylabel('Disparity (0-1 scale)')
            ax.set_title(f'Equalized Odds Disparities for {gene} by {demographic}')
            ax.set_xticks(x)
            ax.set_xticklabels(phenotypes, rotation=45, ha='right')
            ax.legend()

            # Add reference line for typical acceptable disparity threshold
            ax.axhline(y=0.2, color='r', linestyle='--', alpha=0.5)
            ax.text(x[-1], 0.21, 'Typical threshold (0.2)', color='r', alpha=0.7)

            plt.tight_layout()

            plt.savefig(os.path.join(output_dir, f'equalized_odds_{gene}_by_{demographic}.png'))
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

            # Collect data for disparities
            phenotypes = []
            max_differences = []
            min_max_ratios = []

            for phenotype, pheno_data in demo_data.items():
                if 'disparity' not in pheno_data:
                    continue

                phenotypes.append(phenotype)
                max_differences.append(pheno_data['disparity']['maximum_difference'])
                min_max_ratios.append(pheno_data['disparity']['min_to_max_ratio'])

            if not phenotypes:
                continue

            # Create two subplots: one for max difference, one for min/max ratio
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Maximum difference subplot (lower is better)
            bars1 = ax1.bar(phenotypes, max_differences, color=[
                '#55AA55' if x < 0.2 else '#FFA500' if x < 0.4 else '#FF5555' for x in max_differences
            ])

            ax1.set_xlabel('Phenotype')
            ax1.set_ylabel('Maximum Difference')
            ax1.set_title('Maximum Prediction Rate Difference Between Groups')
            ax1.set_xticklabels(phenotypes, rotation=45, ha='right')

            # Add reference line
            ax1.axhline(y=0.2, color='r', linestyle='--', alpha=0.5)
            ax1.text(0, 0.21, 'Typical threshold (0.2)', color='r', alpha=0.7)

            # Min/max ratio subplot (higher is better)
            bars2 = ax2.bar(phenotypes, min_max_ratios, color=[
                '#55AA55' if x > 0.8 else '#FFA500' if x > 0.6 else '#FF5555' for x in min_max_ratios
            ])

            ax2.set_xlabel('Phenotype')
            ax2.set_ylabel('Min/Max Ratio')
            ax2.set_title('Min to Max Prediction Rate Ratio (Higher is Better)')
            ax2.set_xticklabels(phenotypes, rotation=45, ha='right')

            # Add reference line
            ax2.axhline(y=0.8, color='g', linestyle='--', alpha=0.5)
            ax2.text(0, 0.79, 'Typical threshold (0.8)', color='g', alpha=0.7)

            plt.tight_layout()

            plt.savefig(os.path.join(output_dir, f'demographic_parity_{gene}_by_{demographic}.png'))
            plt.close()

            # Create a visualization of prediction rates by group
            for phenotype, pheno_data in demo_data.items():
                if 'prediction_rates_by_group' not in pheno_data or len(pheno_data['prediction_rates_by_group']) < 2:
                    continue

                groups = list(pheno_data['prediction_rates_by_group'].keys())
                rates = list(pheno_data['prediction_rates_by_group'].values())

                plt.figure(figsize=(10, 6))

                # Use superpopulation colors if available
                colors = [SUPERPOPULATION_COLORS.get(group, '#AAAAAA') for group in groups]

                bars = plt.bar(groups, rates, color=colors)

                # Add a line for the overall rate
                plt.axhline(y=pheno_data['overall_rate'], color='r', linestyle='--', label='Overall Rate')

                # Annotate bars
                for bar, rate in zip(bars, rates):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                             f"{rate:.2f}", ha='center', va='bottom')

                plt.title(f'{gene} {phenotype} Prediction Rate by {demographic}')
                plt.xlabel(demographic)
                plt.ylabel('Prediction Rate')
                plt.xticks(rotation=45, ha='right')
                plt.legend()
                plt.tight_layout()

                plt.savefig(os.path.join(output_dir, f'prediction_rate_{gene}_{phenotype}_by_{demographic}.png'))
                plt.close()


def create_summary_dashboard(data, output_dir):
    """Create a summary dashboard of key findings"""
    plt.figure(figsize=(15, 10))

    # 2x2 grid for the dashboard
    plt.subplot(2, 2, 1)

    # 1. Demographic distribution (superpopulation)
    if 'metadata' in data and 'demographics' in data['metadata'] and 'superpopulation_distribution' in data['metadata'][
        'demographics']:
        superpop_dist = data['metadata']['demographics']['superpopulation_distribution']
        groups = list(superpop_dist.keys())
        counts = list(superpop_dist.values())

        colors = [SUPERPOPULATION_COLORS.get(group, '#AAAAAA') for group in groups]
        plt.bar(groups, counts, color=colors)

        plt.title('Sample Distribution by Superpopulation')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45, ha='right')

    # 2. Overall phenotype distributions
    plt.subplot(2, 2, 2)

    if 'phenotype_distributions' in data:
        # Count phenotypes across all genes
        all_phenotypes = {}

        for gene, gene_data in data['phenotype_distributions'].items():
            if 'overall' in gene_data:
                for phenotype, freq in gene_data['overall'].items():
                    if phenotype not in all_phenotypes:
                        all_phenotypes[phenotype] = 0

                    # Add weighted by sample count to get overall count
                    all_phenotypes[phenotype] += freq * data['metadata']['sample_count'] / len(
                        data['phenotype_distributions'])

        # Sort phenotypes
        metabolizer_order = ['PM', 'IM', 'NM', 'RM', 'UM', 'PF', 'DF', 'NF', 'IF', 'INDETERMINATE', ' ']
        ordered_phenotypes = sorted(all_phenotypes.keys(),
                                    key=lambda x: metabolizer_order.index(x) if x in metabolizer_order else 999)

        values = [all_phenotypes[p] for p in ordered_phenotypes]
        colors = [PHENOTYPE_COLORS.get(p, '#AAAAAA') for p in ordered_phenotypes]

        plt.bar(ordered_phenotypes, values, color=colors)
        plt.title('Overall Phenotype Distribution')
        plt.ylabel('Average Count')
        plt.xticks(rotation=45, ha='right')

    # 3. Equalized Odds Summary
    plt.subplot(2, 2, 3)

    if 'equalized_odds_metrics' in data:
        gene_disparities = {}

        for gene, gene_data in data['equalized_odds_metrics'].items():
            if 'Superpopulation' in gene_data:
                max_disparity = 0

                for phenotype, pheno_data in gene_data['Superpopulation'].items():
                    if 'disparity' in pheno_data:
                        disparity = pheno_data['disparity']
                        tpr_disp = disparity.get('true_positive_rate')
                        fpr_disp = disparity.get('false_positive_rate')

                        if tpr_disp is not None and tpr_disp > max_disparity:
                            max_disparity = tpr_disp

                        if fpr_disp is not None and fpr_disp > max_disparity:
                            max_disparity = fpr_disp

                gene_disparities[gene] = max_disparity

        if gene_disparities:
            genes = list(gene_disparities.keys())
            disparities = list(gene_disparities.values())

            colors = [GENE_COLORS.get(gene, '#AAAAAA') for gene in genes]
            plt.bar(genes, disparities, color=colors)

            plt.title('Maximum Equalized Odds Disparity by Gene')
            plt.ylabel('Maximum Disparity')
            plt.axhline(y=0.2, color='r', linestyle='--', alpha=0.5)
            plt.xticks(rotation=45, ha='right')

    # 4. Demographic Parity Summary
    plt.subplot(2, 2, 4)

    if 'demographic_parity_metrics' in data:
        gene_disparities = {}

        for gene, gene_data in data['demographic_parity_metrics'].items():
            if 'Superpopulation' in gene_data:
                max_diff = 0

                for phenotype, pheno_data in gene_data['Superpopulation'].items():
                    if 'disparity' in pheno_data:
                        diff = pheno_data['disparity'].get('maximum_difference', 0)

                        if diff > max_diff:
                            max_diff = diff

                gene_disparities[gene] = max_diff

        if gene_disparities:
            genes = list(gene_disparities.keys())
            disparities = list(gene_disparities.values())

            colors = [GENE_COLORS.get(gene, '#AAAAAA') for gene in genes]
            plt.bar(genes, disparities, color=colors)

            plt.title('Maximum Demographic Parity Disparity by Gene')
            plt.ylabel('Maximum Difference')
            plt.axhline(y=0.2, color='r', linestyle='--', alpha=0.5)
            plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fairness_summary_dashboard.png'))
    plt.close()


def main():
    """Main function to run the visualizations"""
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    try:
        data = load_json(args.input_file)
        print(f"Loaded data from {args.input_file}")

        # Generate visualizations
        visualize_demographic_distribution(data, output_dir)
        visualize_phenotype_distribution(data, output_dir)
        visualize_equalized_odds(data, output_dir)
        visualize_demographic_parity(data, output_dir)
        create_summary_dashboard(data, output_dir)

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
