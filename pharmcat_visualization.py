import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_results(results_file='pharmcat_explanation.json'):
    """Load the analysis results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def visualize_feature_importance(results, output_dir='visualizations'):
    """Create feature importance visualizations for each gene."""
    Path(output_dir).mkdir(exist_ok=True)

    for gene, gene_data in results.items():
        # Skip genes with no feature importance data
        if not gene_data.get('feature_importance'):
            continue

        # Get feature importance values
        features = list(gene_data['feature_importance'].keys())
        importance = list(gene_data['feature_importance'].values())

        # If no features, skip
        if not features:
            continue

        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        })

        # Sort by importance
        df = df.sort_values('Importance', ascending=True)

        # Plot top 10 features (or fewer if not enough)
        n_features = min(10, len(df))
        plot_df = df.tail(n_features)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=plot_df)
        plt.title(f'Top {n_features} Features for {gene}')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{gene}_feature_importance.png')
        plt.close()


def create_comprehensive_report(results, output_file='pharmcat_report.html'):
    """Create a comprehensive HTML report from the analysis results."""
    html = ['<!DOCTYPE html>',
            '<html>',
            '<head>',
            '<title>PharmCAT XAI Analysis Report</title>',
            '<style>',
            'body { font-family: Arial, sans-serif; margin: 20px; }',
            'h1 { color: #2c3e50; }',
            'h2 { color: #3498db; }',
            'h3 { color: #2980b9; }',
            '.gene-section { border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 5px; }',
            '.accurate { background-color: #d5f5e3; }',  # Light green
            '.inaccurate { background-color: #fadbd8; }',  # Light red
            '.unknown { background-color: #f5f5f5; }',  # Light gray
            '.feature-table { width: 100%; border-collapse: collapse; }',
            '.feature-table th, .feature-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }',
            '.feature-table th { background-color: #f2f2f2; }',
            '.explanation { white-space: pre-wrap; font-family: monospace; background-color: #f8f9fa; padding: 10px; border-radius: 5px; }',
            '</style>',
            '</head>',
            '<body>',
            '<h1>PharmCAT XAI Analysis Report</h1>',
            '<p>This report provides an explanation of PharmCAT\'s decision-making process for determining pharmacogenomic phenotypes.</p>']

    # Summary section
    html.append('<h2>Summary</h2>')
    html.append('<table class="feature-table">')
    html.append(
        '<tr><th>Gene</th><th>PharmCAT Phenotype</th><th>Ground Truth</th><th>Match</th><th>Top Features</th></tr>')

    for gene, gene_data in results.items():
        match = gene_data.get('match_accuracy', 0) == 1
        match_str = "✓ Match" if match else "✗ Mismatch"

        # Get top 3 features
        top_features = []
        for feature, importance in list(gene_data.get('feature_importance', {}).items())[:3]:
            top_features.append(f"{feature} ({importance:.4f})")
        top_features_str = "<br>".join(top_features)

        # Add row to table
        html.append(f'<tr>')
        html.append(f'<td>{gene}</td>')
        html.append(f'<td>{gene_data.get("pharmcat_phenotype", "Unknown")}</td>')
        html.append(f'<td>{gene_data.get("ground_truth_phenotype", "Unknown")}</td>')
        html.append(f'<td>{match_str}</td>')
        html.append(f'<td>{top_features_str}</td>')
        html.append(f'</tr>')

    html.append('</table>')

    # Detailed sections for each gene
    html.append('<h2>Gene Analysis Details</h2>')

    for gene, gene_data in results.items():
        # Determine section class based on match accuracy
        if gene_data.get('match_accuracy', 0) == 1:
            section_class = "accurate"
        elif gene_data.get('match_accuracy', 0) == 0 and gene_data.get('ground_truth_phenotype') != "Unknown":
            section_class = "inaccurate"
        else:
            section_class = "unknown"

        html.append(f'<div class="gene-section {section_class}">')
        html.append(f'<h3>{gene}</h3>')

        # Basic information
        html.append('<h4>Basic Information</h4>')
        html.append('<table class="feature-table">')
        html.append(f'<tr><th>PharmCAT Phenotype</th><td>{gene_data.get("pharmcat_phenotype", "Unknown")}</td></tr>')
        html.append(
            f'<tr><th>Ground Truth Phenotype</th><td>{gene_data.get("ground_truth_phenotype", "Unknown")}</td></tr>')
        html.append(f'<tr><th>Diplotype</th><td>{gene_data.get("diplotype", "Unknown")}</td></tr>')
        html.append(f'<tr><th>Allele 1</th><td>{gene_data.get("allele1", "Unknown")}</td></tr>')
        html.append(f'<tr><th>Allele 2</th><td>{gene_data.get("allele2", "Unknown")}</td></tr>')
        html.append('</table>')

        # Feature importance
        html.append('<h4>Feature Importance</h4>')
        html.append('<table class="feature-table">')
        html.append(
            '<tr><th>Rank</th><th>Variant</th><th>Importance</th><th>Genotype</th><th>Position</th><th>Ref/Alt</th></tr>')

        for i, (feature, importance) in enumerate(list(gene_data.get('feature_importance', {}).items())[:10], 1):
            variant_details = gene_data.get('variant_details', {}).get(feature, {})
            genotype = variant_details.get('genotype', 'Unknown')
            position = variant_details.get('position', 'Unknown')
            ref_alt = f"{variant_details.get('ref', '?')}/{variant_details.get('alt', '?')}"

            html.append(f'<tr>')
            html.append(f'<td>{i}</td>')
            html.append(f'<td>{feature}</td>')
            html.append(f'<td>{importance:.4f}</td>')
            html.append(f'<td>{genotype}</td>')
            html.append(f'<td>{position}</td>')
            html.append(f'<td>{ref_alt}</td>')
            html.append(f'</tr>')

        html.append('</table>')

        # Decision Pathway
        pathway = gene_data.get('decision_pathway', {})
        if pathway:
            html.append('<h4>Decision Pathway</h4>')
            html.append('<p>How variants lead to phenotype prediction:</p>')
            html.append('<ol>')

            for variant in pathway.get('top_variants', []):
                alleles = pathway.get('variant_to_allele_map', {}).get(variant, [])
                alleles_str = ", ".join(alleles) if alleles else "No specific alleles"
                html.append(f'<li>Variant <strong>{variant}</strong> → Contributes to alleles: {alleles_str}</li>')

            html.append(f'<li>Alleles → Diplotype: <strong>{gene_data.get("diplotype", "Unknown")}</strong></li>')
            html.append(
                f'<li>Diplotype → Phenotype: <strong>{gene_data.get("pharmcat_phenotype", "Unknown")}</strong></li>')
            html.append('</ol>')

        # Explanation
        if 'explanation' in gene_data:
            html.append('<h4>Explanation</h4>')
            html.append(f'<div class="explanation">{gene_data["explanation"]}</div>')

        html.append('</div>')  # Close gene section

    # Close HTML
    html.append('</body>')
    html.append('</html>')

    # Write HTML to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(html))

    return output_file


def generate_decision_tree_visualization(results, output_dir='visualizations'):
    """Generate decision tree visualizations for the surrogate models."""
    # This would require additional code to recreate the models and visualize them
    # Left as a placeholder for future implementation
    pass


def main():
    # Load analysis results
    print("Loading analysis results...")
    results = load_results()

    # Create visualizations
    print("Creating feature importance visualizations...")
    visualize_feature_importance(results)

    # Create comprehensive report
    print("Generating comprehensive HTML report...")
    report_file = create_comprehensive_report(results)

    print(f"Visualization complete! Report saved to {report_file}")


if __name__ == "__main__":
    main()
