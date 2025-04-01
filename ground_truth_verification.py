import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pharmcat_xai import PharmcatExplainer


def load_ground_truth(excel_file):
    """Load ground truth data from Excel file."""
    try:
        # Check if the file exists
        if not os.path.exists(excel_file):
            print(f"Error: Ground truth file not found: {excel_file}")

            # Check if there are any Excel files in the Groundtruth directory
            groundtruth_dir = os.path.dirname(excel_file)
            if os.path.exists(groundtruth_dir):
                excel_files = [f for f in os.listdir(groundtruth_dir) if f.endswith('.xlsx') or f.endswith('.xls')]
                if excel_files:
                    print(f"Found these Excel files in {groundtruth_dir}:")
                    for file in excel_files:
                        print(f"  - {file}")

                    # Try using the first available Excel file
                    alt_file = os.path.join(groundtruth_dir, excel_files[0])
                    print(f"Attempting to use alternative file: {alt_file}")
                    return pd.read_excel(alt_file)

            return None

        # Try to read the Excel file
        data = pd.read_excel(excel_file)
        print(f"Successfully loaded ground truth data with {len(data)} rows and {len(data.columns)} columns")
        print(f"Column names: {', '.join(str(col) for col in data.columns)}")
        return data
    except Exception as e:
        print(f"Error loading ground truth data: {str(e)}")
        return None


def extract_gene_based_validation(gt_data, importance_df, focus_genes=None):
    """Extract gene-based validation data by comparing known important genes."""
    if gt_data is None or importance_df is None or importance_df.empty:
        return None

    try:
        # If focus_genes is provided, use it; otherwise use default key genes
        if focus_genes:
            key_genes = focus_genes
        else:
            # Key genes in pharmacogenomics
            key_genes = ["CYP2D6", "CYP2C19", "UGT1A1", "SLCO1B1", "CYP3A5", "VKORC1", "CYP4F2"]

        # Extract gene information from ground truth
        gt_genes = []

        # Print a sample of the ground truth data to help with debugging
        print("\nChecking ground truth data for gene mentions in COLUMN HEADERS:")

        # Convert column headers to strings for searching
        column_headers = [str(col) for col in gt_data.columns]

        # Initialize all genes as not found
        for gene in key_genes:
            found = False
            mentions = 0

            # First check if gene is a column header or part of a column header
            for header in column_headers:
                if gene.lower() in header.lower():
                    found = True
                    # Count non-empty cells in this column as mentions
                    gene_column = gt_data[header]
                    mentions = gene_column.dropna().shape[0]
                    print(f"  {gene}: Found in column header '{header}' with {mentions} non-empty cells")
                    break

            # If not found in headers, also check the data (as a backup)
            if not found:
                for col in gt_data.columns:
                    col_data = gt_data[col].astype(str).str.lower()
                    if col_data.str.contains(gene.lower()).any():
                        found = True
                        new_mentions = col_data.str.contains(gene.lower()).sum()
                        mentions += new_mentions
                        print(f"  {gene}: Found in data of column '{col}' with {new_mentions} mentions")

            if not found:
                print(f"  {gene}: Not found in column headers or data")

            gt_genes.append({
                'gene': gene,
                'in_ground_truth': found,
                'mentions': mentions,
                'ground_truth_data': None  # We don't need to store the data here
            })

        # Get gene information from our importance scores
        our_genes = []
        for gene in key_genes:
            gene_variants = importance_df[importance_df['gene'] == gene]
            if not gene_variants.empty:
                our_genes.append({
                    'gene': gene,
                    'in_our_results': True,
                    'variant_count': len(gene_variants),
                    'avg_importance': gene_variants['importance'].mean(),
                    'max_importance': gene_variants['importance'].max(),
                    'variants': gene_variants
                })
            else:
                our_genes.append({
                    'gene': gene,
                    'in_our_results': False,
                    'variant_count': 0,
                    'avg_importance': 0,
                    'max_importance': 0,
                    'variants': None
                })

        # Combine the information
        results = []
        for gene in key_genes:
            gt_entry = next((g for g in gt_genes if g['gene'] == gene), None)
            our_entry = next((g for g in our_genes if g['gene'] == gene), None)

            if gt_entry and our_entry:
                results.append({
                    'gene': gene,
                    'in_ground_truth': gt_entry['in_ground_truth'],
                    'gt_mentions': gt_entry['mentions'],
                    'in_our_results': our_entry['in_our_results'],
                    'our_variant_count': our_entry['variant_count'],
                    'our_avg_importance': our_entry['avg_importance'],
                    'our_max_importance': our_entry['max_importance']
                })

        return pd.DataFrame(results)
    except Exception as e:
        print(f"Error extracting gene-based validation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def visualize_gene_validation(gene_validation_df, output_dir):
    """Create visualizations for gene-based validation."""
    if gene_validation_df is None or gene_validation_df.empty:
        print("No gene validation data to visualize.")
        return

    try:
        os.makedirs(output_dir, exist_ok=True)

        # 1. Bar chart of variant counts by gene
        plt.figure(figsize=(12, 6))
        sns.barplot(x='gene', y='our_variant_count', data=gene_validation_df)

        # Add ground truth mention counts on top as text
        for i, row in enumerate(gene_validation_df.itertuples()):
            plt.text(i, row.our_variant_count + 0.1,
                     f"GT: {row.gt_mentions}",
                     ha='center', va='bottom',
                     fontweight='bold' if row.gt_mentions > 0 else 'normal')

        plt.title('Number of Variants by Gene')
        plt.xlabel('Gene')
        plt.ylabel('Variant Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gene_variant_counts.png'), dpi=150)
        plt.close()

        # 2. Bar chart of importance scores by gene
        plt.figure(figsize=(12, 6))

        # Create a combined data frame for maximum and average importance
        importance_data = []
        for _, row in gene_validation_df.iterrows():
            importance_data.append({
                'gene': row['gene'],
                'importance': row['our_max_importance'],
                'type': 'Maximum'
            })
            importance_data.append({
                'gene': row['gene'],
                'importance': row['our_avg_importance'],
                'type': 'Average'
            })

        importance_df = pd.DataFrame(importance_data)

        # Create the grouped bar chart
        sns.barplot(x='gene', y='importance', hue='type', data=importance_df)

        plt.title('Importance Scores by Gene')
        plt.xlabel('Gene')
        plt.ylabel('Importance Score')
        plt.xticks(rotation=45)
        plt.legend(title='Importance Type')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gene_importance_scores.png'), dpi=150)
        plt.close()

        # 3. Create a gene-based summary table as HTML
        with open(os.path.join(output_dir, 'gene_validation.html'), 'w', encoding='utf-8') as f:
            f.write("""
            <html>
            <head>
                <title>Gene-Based Validation</title>
                <meta charset="UTF-8">
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px 40px; line-height: 1.6; }
                    h1, h2, h3 { color: #2c3e50; }
                    table { border-collapse: collapse; width: 100%; margin: 15px 0; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    tr:nth-child(even) { background-color: #f9f9f9; }
                    .highlighted { background-color: #d4edda; }
                </style>
            </head>
            <body>
                <h1>Gene-Based Validation Report</h1>
                <p>This report compares the genes found in our XAI results with the ground truth data.</p>

                <h2>Gene Summary</h2>
                <table>
                    <tr>
                        <th>Gene</th>
                        <th>Found in Ground Truth</th>
                        <th>Ground Truth Mentions</th>
                        <th>Found in Our Results</th>
                        <th>Our Variant Count</th>
                        <th>Average Importance</th>
                        <th>Maximum Importance</th>
                    </tr>
            """)

            for _, row in gene_validation_df.iterrows():
                # Highlight rows where the gene is found in both datasets
                highlighted = "highlighted" if row['in_ground_truth'] and row['in_our_results'] else ""

                f.write(f"""
                    <tr class="{highlighted}">
                        <td>{row['gene']}</td>
                        <td>{"Yes" if row['in_ground_truth'] else "No"}</td>
                        <td>{row['gt_mentions']}</td>
                        <td>{"Yes" if row['in_our_results'] else "No"}</td>
                        <td>{row['our_variant_count']}</td>
                        <td>{row['our_avg_importance']:.2f}</td>
                        <td>{row['our_max_importance']:.2f}</td>
                    </tr>
                """)

            f.write("""
                </table>

                <h2>Visualization</h2>
                <div>
                    <h3>Variant Counts by Gene</h3>
                    <img src="gene_variant_counts.png" alt="Gene variant counts" style="max-width:100%;">
                </div>
                <div>
                    <h3>Importance Scores by Gene</h3>
                    <img src="gene_importance_scores.png" alt="Gene importance scores" style="max-width:100%;">
                </div>
            </body>
            </html>
            """)

    except Exception as e:
        print(f"Error visualizing gene validation: {str(e)}")


def run_validation():
    """Main function to run the validation process."""
    # Paths
    vcf_file = "pharmcat_processed/HG00276/HG00276_freebayes.preprocessed.vcf"
    match_json_file = "pharmcat_processed/HG00276/HG00276_freebayes.preprocessed.match.json"
    phenotype_json_file = "pharmcat_processed/HG00276/HG00276_freebayes.preprocessed.phenotype.json"
    ground_truth_file = "Groundtruth/getrm_updated_calls.xlsx"
    output_dir = "validation_results"

    # Focus genes - must match those in pharmcat_xai.py
    focus_genes = ["CYP2B6", "CYP2C19", "CYP2C9", "CYP3A4", "CYP3A5", "CYP4F2", "DPYD", "SLCO1B1", "TPMT", "UGT1A1"]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Run PharmCAT XAI analysis
    print("Running PharmCAT XAI analysis...")
    explainer = PharmcatExplainer(vcf_file, match_json_file, phenotype_json_file, output_dir='xai_results')
    explainer.focus_genes = focus_genes
    results = explainer.run()

    # Check if the XAI analysis generated the expected files
    required_files = [
        'xai_results/overall_variant_importance.png',
        'xai_results/pharmcat_xai_report.html',
        'xai_results/match_only_report.html',
        'xai_results/phenotype_only_report.html'
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("Warning: Some required output files are missing:")
        for file in missing_files:
            print(f"  - {file}")
    else:
        print("All required output files were generated successfully.")

    # Load ground truth data
    print("\nLoading ground truth data...")
    gt_data = load_ground_truth(ground_truth_file)

    if gt_data is None:
        print("Error: Could not load ground truth data. Validation aborted.")
        return

    # Perform gene-based validation with focus genes
    print("\nPerforming gene-based validation...")
    gene_validation = extract_gene_based_validation(gt_data, results['variant_importance'], focus_genes)

    if gene_validation is not None:
        print("\nVisualizing gene-based validation...")
        visualize_gene_validation(gene_validation, output_dir)

    # Create validation report
    print("\nCreating validation report...")
    with open(os.path.join(output_dir, 'validation_report.html'), 'w', encoding='utf-8') as f:
        f.write("""
        <html>
        <head>
            <title>PharmCAT XAI Validation Report</title>
            <meta charset="UTF-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px 40px; line-height: 1.6; }
                h1, h2, h3 { color: #2c3e50; }
                .section { margin-bottom: 30px; }
                iframe { border: none; width: 100%; }
            </style>
        </head>
        <body>
            <h1>PharmCAT XAI Validation Report</h1>
            <p>This report compares the XAI results with the ground truth data in getrm_updated_calls.xlsx.</p>

            <div class="section">
                <h2>Generated Reports</h2>
                <ul>
                    <li><a href="../xai_results/pharmcat_xai_report.html" target="_blank">Main XAI Report</a></li>
                    <li><a href="../xai_results/match_only_report.html" target="_blank">Match Data Report</a></li>
                    <li><a href="../xai_results/phenotype_only_report.html" target="_blank">Phenotype Report</a></li>
                </ul>
            </div>

            <div class="section">
                <h2>Gene-Based Validation</h2>
                <iframe src="gene_validation.html" height="600"></iframe>
            </div>

            <div class="section">
                <h2>Validation Summary</h2>
                <p>This validation compares our XAI analysis against the ground truth data for the following genes:</p>
                <ul>
        """)

        for gene in focus_genes:
            f.write(f"<li>{gene}</li>")

        f.write("""
                </ul>
                <p>The visualizations above show how our variant importance scoring aligns with known variants in the ground truth dataset.</p>
            </div>
        </body>
        </html>
        """)

    print(f"\nValidation complete! Reports saved to {output_dir}")
    print(f"Open {output_dir}/validation_report.html to view the complete validation results")


if __name__ == "__main__":
    run_validation()
