import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SUPERPOPULATION_COLORS = {
    'AFR': '#66C2A5',
    'AMR': '#FC8D62',
    'EAS': '#8DA0CB',
    'EUR': '#E78AC3',
    'SAS': '#A6D854'
}

SEX_COLORS = {
    'F': '#FF69B4',
    'M': '#4169E1'
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


def extract_average_metrics(data, demographic_key):
    fpr_data = {}
    prediction_data = {}

    if 'equalized_odds_metrics' in data:
        for gene, gene_data in data['equalized_odds_metrics'].items():
            if demographic_key in gene_data:
                demo_data = gene_data[demographic_key]
                fpr_by_group = {}

                for phenotype, pheno_data in demo_data.items():
                    if 'error_rates_by_group' in pheno_data:
                        for group, rates in pheno_data['error_rates_by_group'].items():
                            if 'false_positive_rate' in rates and rates['false_positive_rate'] is not None:
                                if group not in fpr_by_group:
                                    fpr_by_group[group] = []
                                fpr_by_group[group].append(rates['false_positive_rate'])

                for group, values in fpr_by_group.items():
                    if gene not in fpr_data:
                        fpr_data[gene] = {}
                    fpr_data[gene][group] = np.mean(values)

    if 'demographic_parity_metrics' in data:
        for gene, gene_data in data['demographic_parity_metrics'].items():
            if demographic_key in gene_data:
                demo_data = gene_data[demographic_key]
                pred_by_group = {}

                for phenotype, pheno_data in demo_data.items():
                    if 'prediction_rates_by_group' in pheno_data:
                        for group, rate in pheno_data['prediction_rates_by_group'].items():
                            if group not in pred_by_group:
                                pred_by_group[group] = []
                            pred_by_group[group].append(rate)

                for group, values in pred_by_group.items():
                    if gene not in prediction_data:
                        prediction_data[gene] = {}
                    prediction_data[gene][group] = np.mean(values)

    return fpr_data, prediction_data


def calculate_disparity(metrics_dict):
    disparity = {}
    for gene, groups in metrics_dict.items():
        if len(groups) >= 2:
            values = list(groups.values())
            disparity[gene] = max(values) - min(values)
    return disparity


def create_consolidated_visualization(data, demographic_key, output_dir, ethnicity_mapping=None):
    fairness_method = data.get('metadata', {}).get('fairness_method', 'calculate_equalized_odds')
    bias_method = data.get('metadata', {}).get('bias_method', 'calculate_demographic_parity')

    fpr_data, prediction_data = extract_average_metrics(data, demographic_key)

    if not fpr_data and not prediction_data:
        print(f"No data available for demographic: {demographic_key}")
        return

    display_key = "Ethnicity" if demographic_key == "Superpopulation" else demographic_key

    fig, axes = plt.subplots(2, 2, figsize=(16, 14.5))
    fig.suptitle(f'Fairness and Bias Analysis by {display_key}', fontsize=16, fontweight='bold')

    target_genes = ["CYP2B6", "CYP2C9", "CYP2C19", "CYP3A5", "SLCO1B1", "TPMT", "DPYD"]

    if fpr_data:
        genes = sorted([g for g in target_genes if g in fpr_data])
        if demographic_key == 'Superpopulation':
            groups = sorted(set(group for gene_data in fpr_data.values() for group in gene_data.keys()))
            if ethnicity_mapping:
                group_labels = [ethnicity_mapping.get(g, g) for g in groups]
            else:
                group_labels = groups
        else:
            groups = sorted(set(group for gene_data in fpr_data.values() for group in gene_data.keys()))
            group_labels = groups

        matrix_data = []
        for gene in genes:
            row = []
            for group in groups:
                row.append(fpr_data.get(gene, {}).get(group, 0))
            matrix_data.append(row)

        if matrix_data:
            pivot_df = pd.DataFrame(matrix_data, index=genes, columns=group_labels)
            sns.heatmap(pivot_df, cmap="YlOrRd", annot=True, fmt=".3f", linewidths=.5, ax=axes[0, 0],
                        cbar_kws={'label': 'FPR'})
            axes[0, 0].set_title(f'Fairness calculation using {fairness_method}\nby Gene & {display_key}',
                                 fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel(display_key)
            axes[0, 0].set_ylabel('Gene')

            fig.text(0.25, 0.51, 'Shows average false positive rates across phenotypes; lower is better',
                     ha='center', fontsize=10, style='italic', color='#555555')
        else:
            axes[0, 0].text(0.5, 0.5, 'No FPR data available', ha='center', va='center')
            axes[0, 0].axis('off')
    else:
        axes[0, 0].text(0.5, 0.5, 'No FPR data available', ha='center', va='center')
        axes[0, 0].axis('off')

    if prediction_data:
        genes = sorted([g for g in target_genes if g in prediction_data])
        if demographic_key == 'Superpopulation':
            groups = sorted(set(group for gene_data in prediction_data.values() for group in gene_data.keys()))
            if ethnicity_mapping:
                group_labels = [ethnicity_mapping.get(g, g) for g in groups]
            else:
                group_labels = groups
        else:
            groups = sorted(set(group for gene_data in prediction_data.values() for group in gene_data.keys()))
            group_labels = groups

        matrix_data = []
        for gene in genes:
            row = []
            for group in groups:
                row.append(prediction_data.get(gene, {}).get(group, 0))
            matrix_data.append(row)

        if matrix_data:
            pivot_df = pd.DataFrame(matrix_data, index=genes, columns=group_labels)
            sns.heatmap(pivot_df, cmap="YlGnBu", annot=True, fmt=".3f", linewidths=.5, ax=axes[0, 1],
                        cbar_kws={'label': 'Prediction Rate'})
            axes[0, 1].set_title(f'Prediction Rate by Gene & {display_key}', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel(display_key)
            axes[0, 1].set_ylabel('Gene')

            fig.text(0.75, 0.51, 'Shows average prediction rates across phenotypes; closer to 0.5 is better',
                     ha='center', fontsize=10, style='italic', color='#555555')
        else:
            axes[0, 1].text(0.5, 0.5, 'No prediction data available', ha='center', va='center')
            axes[0, 1].axis('off')
    else:
        axes[0, 1].text(0.5, 0.5, 'No prediction data available', ha='center', va='center')
        axes[0, 1].axis('off')

    if prediction_data:
        pred_disparity = calculate_disparity(prediction_data)
        if pred_disparity:
            genes = sorted([g for g in target_genes if g in pred_disparity])
            disparities = [pred_disparity[g] for g in genes]

            colors = ['#FF6B6B' if d > 0.1 else '#4ECDC4' for d in disparities]
            bars = axes[1, 0].bar(genes, disparities, color=colors)
            axes[1, 0].set_title(f'Bias calculation using {bias_method}', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Gene')
            axes[1, 0].set_ylabel('Max Disparity in Prediction Rate')
            axes[1, 0].tick_params(axis='x', rotation=45)

            for bar, disp in zip(bars, disparities):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width() / 2., height,
                                f'{disp:.3f}', ha='center', va='bottom', fontsize=9)

            fig.text(0.25, 0.05, 'Shows maximum difference in prediction rates between groups; lower is better',
                     ha='center', fontsize=10, style='italic', color='#555555')
        else:
            axes[1, 0].text(0.5, 0.5, 'No disparity data available', ha='center', va='center')
            axes[1, 0].axis('off')
    else:
        axes[1, 0].text(0.5, 0.5, 'No disparity data available', ha='center', va='center')
        axes[1, 0].axis('off')

    if fpr_data:
        fpr_disparity = calculate_disparity(fpr_data)
        if fpr_disparity:
            genes = sorted([g for g in target_genes if g in fpr_disparity])
            disparities = [fpr_disparity[g] for g in genes]

            colors = ['#FF6B6B' if d > 0.1 else '#95E1D3' for d in disparities]
            bars = axes[1, 1].bar(genes, disparities, color=colors)
            axes[1, 1].set_title(f'Fairness Disparity (FPR) by Gene', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Gene')
            axes[1, 1].set_ylabel('Max Disparity in FPR')
            axes[1, 1].tick_params(axis='x', rotation=45)

            for bar, disp in zip(bars, disparities):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height,
                                f'{disp:.3f}', ha='center', va='bottom', fontsize=9)

            fig.text(0.75, 0.05, 'Shows maximum difference in false positive rates between groups; lower is better',
                     ha='center', fontsize=10, style='italic', color='#555555')
        else:
            axes[1, 1].text(0.5, 0.5, 'No FPR disparity data available', ha='center', va='center')
            axes[1, 1].axis('off')
    else:
        axes[1, 1].text(0.5, 0.5, 'No FPR disparity data available', ha='center', va='center')
        axes[1, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    if demographic_key == "Superpopulation":
        filename = f"fairness_bias_ethnicity.png"
    else:
        filename = f"fairness_bias_{demographic_key.lower()}.png"

    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def main():
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    try:
        data = load_json(args.input_file)
        print(f"Loaded data from {args.input_file}")

        ethnicity_mapping = data.get('metadata', {}).get('ethnicity_mapping')

        create_consolidated_visualization(data, 'Superpopulation', output_dir, ethnicity_mapping)

        create_consolidated_visualization(data, 'Sex', output_dir)

        print(f"All visualizations saved to {output_dir}")
    except Exception as e:
        print(f"Error processing the analysis file: {e}")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"Error: {str(e)}",
                 ha='center', va='center', fontsize=12, color='red')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'error.png'))
        plt.close()


if __name__ == "__main__":
    main()
