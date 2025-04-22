import argparse
import json
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

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
    'INDETERMINATE': '#AAAAAA'  # Gray
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


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize PharmCAT analysis results')
    parser.add_argument('--counterfactual', type=str, help='Path to counterfactual analysis JSON file')
    parser.add_argument('--decision_trees', type=str, help='Path to decision trees JSON file')
    parser.add_argument('--rules', type=str, help='Path to rule extraction JSON file')
    parser.add_argument('--pgx_results', type=str, help='Path to PGx results JSON file')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Directory to save visualization outputs')
    return parser.parse_args()


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def ensure_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def create_phenotype_distribution_panel(ax, gene, samples, gene_data=None):
    if gene_data and 'prediction_distribution' in gene_data:
        phenotype_counts = gene_data['prediction_distribution']
    else:
        phenotype_counts = {}
        for sample in samples:
            phenotype = sample['phenotype']
            if phenotype not in phenotype_counts:
                phenotype_counts[phenotype] = 0
            phenotype_counts[phenotype] += 1

    phenotypes = list(phenotype_counts.keys())
    counts = list(phenotype_counts.values())
    colors = [PHENOTYPE_COLORS.get(p, '#AAAAAA') for p in phenotypes]

    bars = ax.bar(phenotypes, counts, color=colors)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)

    ax.set_title(f'Phenotype Distribution', fontsize=10)
    ax.set_xlabel('Phenotype', fontsize=8)
    ax.set_ylabel('Sample Count', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=8)


def summarize_decision_tree(ax, gene, tree_data):
    if not tree_data:
        ax.text(0.5, 0.5, 'No decision tree data available',
                ha='center', va='center', fontsize=8)
        ax.set_axis_off()
        return

    ax.set_axis_off()
    tree = tree_data['tree']

    def add_node(node, x, y, width, depth=0, max_depth=2):
        if node is None or depth > max_depth:
            return

        class_name = node.get('class', 'Unknown')
        node_color = PHENOTYPE_COLORS.get(class_name, '#AAAAAA')

        circle = plt.Circle((x, y), 0.1, color=node_color, alpha=0.7)
        ax.add_patch(circle)

        if node['type'] == 'decision':
            feature = node['feature']
            if '_rs' in feature:
                feature = 'rs' + feature.split('_rs')[1].split('_')[0]
            ax.text(x, y, feature, ha='center', va='center', fontsize=7)
        else:
            ax.text(x, y, class_name, ha='center', va='center', fontsize=7)

        if 'left' in node and depth < max_depth:
            child_x_left = x - width / (2 ** (depth + 1))
            child_y = y - 0.3
            ax.plot([x, child_x_left], [y - 0.1, child_y + 0.1], 'k-', alpha=0.5)
            add_node(node['left'], child_x_left, child_y, width, depth + 1, max_depth)

        if 'right' in node and depth < max_depth:
            child_x_right = x + width / (2 ** (depth + 1))
            child_y = y - 0.3
            ax.plot([x, child_x_right], [y - 0.1, child_y + 0.1], 'k-', alpha=0.5)
            add_node(node['right'], child_x_right, child_y, width, depth + 1, max_depth)

    add_node(tree, 0.5, 0.9, 0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Decision Tree Preview', fontsize=10)


def summarize_rules(ax, gene, rules_data):
    if not rules_data:
        ax.text(0.5, 0.5, 'No rules data available',
                ha='center', va='center', fontsize=8)
        ax.set_axis_off()
        return

    rules = rules_data["rules"]
    ax.set_axis_off()

    max_rules = min(3, len(rules))
    shown_rules = rules[:max_rules]

    for i, rule in enumerate(shown_rules):
        phenotype = rule.split("phenotype is ")[1]
        color = PHENOTYPE_COLORS.get(phenotype, '#AAAAAA')

        rule_parts = rule.split(" THEN ")
        simplified_if = rule_parts[0].replace("IF ", "IF: ")
        simplified_then = rule_parts[1].replace("phenotype is ", "→ ")

        if len(simplified_if) > 40:
            parts = simplified_if.split(" AND ")
            simplified_if = parts[0] + "\n  AND " + " AND\n  ".join(parts[1:])

        full_text = f"{simplified_if}\n{simplified_then}"

        ax.text(0.05, 0.95 - i * 0.3, full_text, ha='left', va='top', fontsize=7,
                bbox=dict(facecolor=color, alpha=0.1, boxstyle='round'))

    if len(rules) > max_rules:
        ax.text(0.05, 0.05, f"+ {len(rules) - max_rules} more rules...",
                ha='left', va='bottom', fontsize=7, style='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Key Decision Rules', fontsize=10)


def top_variant_impacts(ax, gene, samples, all_genes_data=None):
    if not samples:
        ax.text(0.5, 0.5, 'No samples data available',
                ha='center', va='center', fontsize=8)
        ax.set_axis_off()
        return

    variant_counts = {}
    total_samples = len(samples)

    for sample in samples:
        for impact in sample.get('variant_impacts', []):
            feature = impact['feature']
            if feature not in variant_counts:
                variant_counts[feature] = 0
            variant_counts[feature] += 1

    sorted_variants = sorted(variant_counts.items(), key=lambda x: x[1], reverse=True)
    top_variants = sorted_variants[:10]

    variant_names = [v[0] if len(v[0]) < 25 else v[0][:22] + '...' for v in top_variants]
    counts = [v[1] for v in top_variants]

    percentages = [count / total_samples * 100 for count in counts]

    colors = []
    for variant in [v[0] for v in top_variants]:
        gene_part = variant.split('_')[0]
        if gene_part == gene:
            colors.append(GENE_COLORS.get(gene, '#FF4500'))
        else:
            colors.append('#1E90FF')

    y_pos = np.arange(len(variant_names))
    bars = ax.barh(y_pos, percentages, align='center', color=colors, alpha=0.7)

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height() / 2,
                f'{percentages[i]:.1f}% ({counts[i]})',
                va='center', fontsize=7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(variant_names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Percentage of Samples', fontsize=8)
    ax.set_title('Top Genetic Variants', fontsize=10)

    ax.legend([
        plt.Rectangle((0, 0), 1, 1, color=GENE_COLORS.get(gene, '#FF4500'), alpha=0.7),
        plt.Rectangle((0, 0), 1, 1, color='#1E90FF', alpha=0.7)
    ], [f'In {gene}', 'In other genes'], loc='lower right', fontsize=7)


def feature_importance_panel(ax, gene, feature_importance):
    if not feature_importance:
        ax.text(0.5, 0.5, 'No feature importance data available',
                ha='center', va='center', fontsize=8)
        ax.set_axis_off()
        return

    gene_data = feature_importance.get(gene, {})
    if not gene_data:
        ax.text(0.5, 0.5, 'No feature importance data available',
                ha='center', va='center', fontsize=8)
        ax.set_axis_off()
        return

    sorted_features = sorted(gene_data.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:10]

    feature_names = [f[0] for f in top_features]
    importance_values = [f[1] for f in top_features]

    colors = []
    for feature in feature_names:
        feature_gene = feature.split('_')[0]
        if feature_gene == gene:
            colors.append(GENE_COLORS.get(gene, '#FF4500'))
        else:
            colors.append('#1E90FF')

    y_pos = np.arange(len(feature_names))
    bars = ax.barh(y_pos, importance_values, align='center', color=colors, alpha=0.7)

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{width:.3f}', va='center', fontsize=7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, fontsize=7)
    ax.set_xlabel('Feature Importance', fontsize=8)
    ax.set_title('Top Feature Importance', fontsize=10)
    ax.invert_yaxis()

    ax.legend([
        plt.Rectangle((0, 0), 1, 1, color=GENE_COLORS.get(gene, '#FF4500'), alpha=0.7),
        plt.Rectangle((0, 0), 1, 1, color='#1E90FF', alpha=0.7)
    ], [f'In {gene}', 'In other genes'], loc='lower right', fontsize=7)


def create_gene_dashboard(gene, data, output_dir):
    plt.figure(figsize=(12, 10))

    gs = GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1.2])

    counterfactual_samples = data.get('counterfactual_analysis', {}).get(gene, [])
    decision_tree_data = data.get('decision_trees', {}).get(gene, None)
    rules_data = data.get('rule_extraction', {}).get(gene, None)
    gene_explanation = data.get('gene_explanations', {}).get(gene, None)
    feature_importance = data.get('feature_importance', {})

    ax1 = plt.subplot(gs[0, 0])
    create_phenotype_distribution_panel(ax1, gene, counterfactual_samples, gene_explanation)

    ax2 = plt.subplot(gs[0, 1])
    feature_importance_panel(ax2, gene, feature_importance)

    ax3 = plt.subplot(gs[1, 0])
    summarize_rules(ax3, gene, rules_data)

    ax4 = plt.subplot(gs[1, 1])
    summarize_decision_tree(ax4, gene, decision_tree_data)

    ax5 = plt.subplot(gs[2, :])
    top_variant_impacts(ax5, gene, counterfactual_samples)

    plt.suptitle(f'Pharmacogenomic Dashboard for {gene}', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_file = os.path.join(output_dir, f'dashboard_{gene}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    return output_file


def create_cross_gene_heatmaps(data, output_dir):
    if not data.get('feature_importance'):
        return None

    feature_importance = data['feature_importance']

    plt.figure(figsize=(14, 10))

    genes = list(feature_importance.keys())

    gene_features = {}
    for g, feats in feature_importance.items():
        gene_features[g] = set(feats.keys())

    common_features = set()
    for feature in set().union(*gene_features.values()):
        count = sum(1 for gene_set in gene_features.values() if feature in gene_set)
        if count >= 3:
            common_features.add(feature)

    features = sorted(list(common_features))

    matrix = np.zeros((len(genes), len(features)))

    for i, g in enumerate(genes):
        for j, f in enumerate(features):
            matrix[i, j] = feature_importance[g].get(f, 0)

    plt.figure(figsize=(max(12, len(features) * 0.3), max(8, len(genes) * 0.5)))

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(matrix, annot=False, fmt=".2f",
                xticklabels=features, yticklabels=genes,
                cmap=cmap)

    plt.title('Cross-Gene Feature Importance', fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    output_file = os.path.join(output_dir, 'cross_gene_feature_importance.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def create_gene_phenotype_summary(data, output_dir):
    if not data.get('gene_explanations'):
        return None

    gene_explanations = data['gene_explanations']

    phenotype_counts = {}
    genes = list(gene_explanations.keys())
    all_phenotypes = set()

    for gene, gene_data in gene_explanations.items():
        if 'prediction_distribution' in gene_data:
            phenotype_counts[gene] = gene_data['prediction_distribution']
            all_phenotypes.update(gene_data['prediction_distribution'].keys())

    if not phenotype_counts:
        return None

    all_phenotypes = sorted(list(all_phenotypes))

    matrix = np.zeros((len(genes), len(all_phenotypes)))

    for i, gene in enumerate(genes):
        for j, phenotype in enumerate(all_phenotypes):
            matrix[i, j] = phenotype_counts[gene].get(phenotype, 0)

    plt.figure(figsize=(max(8, len(all_phenotypes) * 1.2), max(6, len(genes) * 0.6)))

    matrix = matrix.astype(int)
    sns.heatmap(matrix, annot=True, fmt="d",
                xticklabels=all_phenotypes, yticklabels=genes,
                cmap='viridis')

    plt.title('Phenotype Distribution Across Genes', fontsize=14)
    plt.tight_layout()

    output_file = os.path.join(output_dir, 'gene_phenotype_summary.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def create_gene_interactions_network(data, output_dir):
    if not data.get('feature_importance'):
        return None

    feature_importance = data['feature_importance']

    G = nx.DiGraph()

    threshold = 0.1

    genes = list(feature_importance.keys())

    for source_gene in genes:
        G.add_node(source_gene, type='gene')

        for feature, importance in feature_importance[source_gene].items():
            if importance < threshold:
                continue

            target_gene = feature.split('_')[0]
            if target_gene in genes and target_gene != source_gene:
                if not G.has_edge(source_gene, target_gene):
                    G.add_edge(source_gene, target_gene, weight=0)

                G[source_gene][target_gene]['weight'] += importance

    plt.figure(figsize=(10, 8))

    pos = nx.spring_layout(G, k=0.5, seed=42)

    edge_weights = [G[u][v]['weight'] * 10 for u, v in G.edges()]

    node_colors = [GENE_COLORS.get(node, '#AAAAAA') for node in G.nodes()]

    nx.draw_networkx(
        G, pos,
        node_color=node_colors,
        node_size=800,
        font_size=10,
        width=edge_weights,
        edge_color='gray',
        arrows=True,
        arrowsize=15,
        with_labels=True
    )

    plt.title('Gene Interaction Network', fontsize=14)
    plt.axis('off')

    output_file = os.path.join(output_dir, 'gene_interaction_network.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def create_summary_dashboard(data, output_files, output_dir):
    network_file = create_gene_interactions_network(data, output_dir)
    heatmap_file = create_cross_gene_heatmaps(data, output_dir)
    phenotype_file = create_gene_phenotype_summary(data, output_dir)

    plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2)

    for i, ax_pos in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        ax = plt.subplot(gs[ax_pos[0], ax_pos[1]])
        ax.set_axis_off()

        if i == 0 and network_file:
            img = plt.imread(network_file)
            ax.imshow(img)
            ax.set_title('Gene Interaction Network', fontsize=12)
        elif i == 1 and heatmap_file:
            img = plt.imread(heatmap_file)
            ax.imshow(img)
            ax.set_title('Cross-Gene Feature Importance', fontsize=12)
        elif i == 2 and phenotype_file:
            img = plt.imread(phenotype_file)
            ax.imshow(img)
            ax.set_title('Phenotype Distribution', fontsize=12)
        elif i == 3:
            ax.text(0.5, 0.9, 'Gene Dashboards:', ha='center', va='top', fontsize=12, fontweight='bold')
            for j, (gene, file) in enumerate(output_files.items()):
                color = GENE_COLORS.get(gene, '#333333')
                ax.text(0.1, 0.8 - j * 0.1, f"• {gene}", ha='left', va='top', fontsize=11,
                        color=color, fontweight='bold')

    plt.suptitle('Pharmacogenomic Analysis Summary', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_file = os.path.join(output_dir, 'pgx_summary_dashboard.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    data = {}

    if args.counterfactual:
        data['counterfactual_analysis'] = load_json(args.counterfactual)

    if args.decision_trees:
        data['decision_trees'] = load_json(args.decision_trees)

    if args.rules:
        data['rule_extraction'] = load_json(args.rules)

    if args.pgx_results:
        pgx_results = load_json(args.pgx_results)
        data['gene_explanations'] = pgx_results.get('gene_explanations', {})
        data['feature_importance'] = pgx_results.get('feature_importance', {})
        data['sample_explanations'] = pgx_results.get('sample_explanations', [])

    genes = set()

    if 'counterfactual_analysis' in data:
        genes.update(data['counterfactual_analysis'].keys())

    if 'decision_trees' in data:
        genes.update(data['decision_trees'].keys())

    if 'rule_extraction' in data:
        genes.update(data['rule_extraction'].keys())

    if 'gene_explanations' in data:
        genes.update(data['gene_explanations'].keys())

    if 'feature_importance' in data:
        genes.update(data['feature_importance'].keys())

    genes = sorted(list(genes))

    output_files = {}

    for gene in genes:
        output_file = create_gene_dashboard(gene, data, output_dir)
        output_files[gene] = output_file

    create_summary_dashboard(data, output_files, output_dir)

    print(f"Generated {len(output_files) + 4} visualizations in {output_dir}")


if __name__ == "__main__":
    main()
