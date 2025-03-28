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
        return pd.read_excel(excel_file)
    except Exception as e:
        print(f"Error loading ground truth data: {str(e)}")
        return None


def extract_gene_based_validation(gt_data, importance_df):
    """Extract gene-based validation data by comparing known important genes."""
    if gt_data is None or importance_df is None or importance_df.empty:
        return None

    try:
        # Key genes in pharmacogenomics
        key_genes = ["CYP2D6", "CYP2C19", "UGT1A1", "SLCO1B1", "CYP3A5", "VKORC1", "CYP4F2"]

        # Extract gene information from ground truth
        gt_genes = []

        # Look for columns that contain gene names
        gene_cols = []
        for col in gt_data.columns:
            col_str = str(col).lower()
            if any(term in col_str for term in ['gene', 'symbol', 'name']):
                gene_cols.append(col)

        if gene_cols:
            for col in gene_cols:
                for gene in key_genes:
                    if gt_data[col].astype(str).str.contains(gene).any():
                        matches = gt_data[gt_data[col].astype(str).str.contains(gene)]
                        gt_genes.append({
                            'gene': gene,
                            'in_ground_truth': True,
                            'mentions': len(matches),
                            'ground_truth_data': matches
                        })

        # If no specific gene columns, search all columns
        if not gt_genes:
            for gene in key_genes:
                found = False
                for col in gt_data.columns:
                    if gt_data[col].astype(str).str.contains(gene).any():
                        found = True
                        matches = gt_data[gt_data[col].astype(str).str.contains(gene)]
                        gt_genes.append({
                            'gene': gene,
                            'in_ground_truth': True,
                            'mentions': len(matches),
                            'ground_truth_data': matches
                        })

                if not found:
                    gt_genes.append({
                        'gene': gene,
                        'in_ground_truth': False,
                        'mentions': 0,
                        'ground_truth_data': None
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
        return None


def extract_variant_based_validation(gt_data, importance_df):
    """Extract variant-based validation by identifying important variants in both datasets."""
    if gt_data is None or importance_df is None or importance_df.empty:
        return None

    try:
        # Find RS IDs in ground truth data
        rs_ids = []

        # Look for columns that might contain RS IDs
        for col in gt_data.columns:
            col_str = str(col).lower()
            if any(term in col_str for term in ['rs', 'id', 'variant', 'snp']):
                # Extract RS IDs using regex
                for val in gt_data[col].dropna():
                    if isinstance(val, str) and 'rs' in val.lower():
                        # Extract RS IDs using string methods
                        if 'rs' in val.lower():
                            start_idx = val.lower().find('rs')
                            # Extract the ID, assuming it's followed by digits
                            rs_id = ''
                            for char in val[start_idx:]:
                                if char.isdigit() or char.lower() == 'r' or char.lower() == 's':
                                    rs_id += char
                                else:
                                    break
                            if rs_id and rs_id.lower().startswith('rs') and len(rs_id) > 2:
                                rs_ids.append({
                                    'rsid': rs_id,
                                    'source_column': col,
                                    'source_value': val
                                })

        if not rs_ids:
            # If no RS IDs found in column names, search all text values
            for col in gt_data.columns:
                for val in gt_data[col].astype(str).dropna():
                    if 'rs' in val.lower():
                        # Extract the RS ID
                        start_idx = val.lower().find('rs')
                        rs_id = ''
                        for char in val[start_idx:]:
                            if char.isdigit() or char.lower() == 'r' or char.lower() == 's':
                                rs_id += char
                            else:
                                break
                        if rs_id and rs_id.lower().startswith('rs') and len(rs_id) > 2:
                            rs_ids.append({
                                'rsid': rs_id,
                                'source_column': col,
                                'source_value': val
                            })

        # Extract unique RS IDs
        unique_rs_ids = list(set([r['rsid'] for r in rs_ids]))
        print(f"Found {len(unique_rs_ids)} unique RS IDs in ground truth data")

        # Compare with our importance scores
        comparison = []
        for rs_id in unique_rs_ids:
            # Find matching variant in our results
            matching_variant = importance_df[importance_df['rsid'].str.lower() == rs_id.lower()]

            if not matching_variant.empty:
                # Extract information
                for _, row in matching_variant.iterrows():
                    comparison.append({
                        'rsid': rs_id,
                        'gene': row['gene'],
                        'in_ground_truth': True,
                        'in_our_results': True,
                        'our_importance': row['importance'],
                        'our_genotype': row['genotype']
                    })
            else:
                comparison.append({
                    'rsid': rs_id,
                    'gene': 'Unknown',
                    'in_ground_truth': True,
                    'in_our_results': False,
                    'our_importance': 0,
                    'our_genotype': 'Not found'
                })

        # Add variants from our results that are not in ground truth
        for _, row in importance_df.iterrows():
            if row['rsid'].lower() not in [r.lower() for r in unique_rs_ids]:
                comparison.append({
                    'rsid': row['rsid'],
                    'gene': row['gene'],
                    'in_ground_truth': False,
                    'in_our_results': True,
                    'our_importance': row['importance'],
                    'our_genotype': row['genotype']
                })

        return pd.DataFrame(comparison)
    except Exception as e:
        print(f"Error extracting variant-based validation: {str(e)}")
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


def visualize_variant_validation(variant_validation_df, output_dir):
    """Create visualizations for variant-based validation."""
    if variant_validation_df is None or variant_validation_df.empty:
        print("No variant validation data to visualize.")
        return

    try:
        os.makedirs(output_dir, exist_ok=True)

        # Filter to keep only variants found in both datasets
        common_variants = variant_validation_df[
            variant_validation_df['in_ground_truth'] & variant_validation_df['in_our_results']]

        if not common_variants.empty:
            # 1. Bar chart of importance scores for common variants
            plt.figure(figsize=(14, 8))

            # Sort by importance score
            sorted_variants = common_variants.sort_values('our_importance', ascending=False)

            # Create the bar chart
            bars = sns.barplot(x='rsid', y='our_importance', data=sorted_variants)

            plt.title('Importance Scores for Variants Found in Ground Truth')
            plt.xlabel('Variant (rsID)')
            plt.ylabel('Importance Score')
            plt.xticks(rotation=45, ha='right')

            # Add gene labels on top of bars
            for i, row in enumerate(sorted_variants.itertuples()):
                plt.text(i, row.our_importance + 0.1,
                         row.gene,
                         ha='center', va='bottom',
                         fontsize=8, rotation=45)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'common_variant_importance.png'), dpi=150)
            plt.close()

        # 2. Grouped bar chart comparing all variants
        plt.figure(figsize=(10, 6))

        # Create a summary of variant counts by gene and presence in datasets
        summary_data = []

        # Variants in both datasets
        both_count = len(
            variant_validation_df[variant_validation_df['in_ground_truth'] & variant_validation_df['in_our_results']])
        summary_data.append({
            'category': 'In Both',
            'count': both_count
        })

        # Variants only in ground truth
        gt_only_count = len(
            variant_validation_df[variant_validation_df['in_ground_truth'] & ~variant_validation_df['in_our_results']])
        summary_data.append({
            'category': 'Ground Truth Only',
            'count': gt_only_count
        })

        # Variants only in our results
        our_only_count = len(
            variant_validation_df[~variant_validation_df['in_ground_truth'] & variant_validation_df['in_our_results']])
        summary_data.append({
            'category': 'Our Results Only',
            'count': our_only_count
        })

        # Create the summary DataFrame
        summary_df = pd.DataFrame(summary_data)

        # Create the bar chart
        sns.barplot(x='category', y='count', data=summary_df)

        plt.title('Variant Distribution Between Datasets')
        plt.xlabel('Category')
        plt.ylabel('Variant Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'variant_distribution.png'), dpi=150)
        plt.close()

        # 3. Create a variant validation HTML report
        with open(os.path.join(output_dir, 'variant_validation.html'), 'w', encoding='utf-8') as f:
            # Calculate percentages first
            total_variants = len(variant_validation_df)
            both_percent = both_count / total_variants * 100 if total_variants > 0 else 0
            gt_only_percent = gt_only_count / total_variants * 100 if total_variants > 0 else 0
            our_only_percent = our_only_count / total_variants * 100 if total_variants > 0 else 0

            # Create HTML content with properly substituted values
            html_content = f"""
            <html>
            <head>
                <title>Variant-Based Validation</title>
                <meta charset="UTF-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px 40px; line-height: 1.6; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .highlighted {{ background-color: #d4edda; }}
                </style>
            </head>
            <body>
                <h1>Variant-Based Validation Report</h1>
                <p>This report compares the variants found in our XAI results with the ground truth data.</p>

                <h2>Variant Summary</h2>
                <table>
                    <tr>
                        <th>Category</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                    <tr>
                        <td>Variants in both datasets</td>
                        <td>{both_count}</td>
                        <td>{both_percent:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Variants only in ground truth</td>
                        <td>{gt_only_count}</td>
                        <td>{gt_only_percent:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Variants only in our results</td>
                        <td>{our_only_count}</td>
                        <td>{our_only_percent:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Total variants</td>
                        <td>{total_variants}</td>
                        <td>100%</td>
                    </tr>
                </table>

                <h2>Visualization</h2>
                <div>
                    <h3>Variant Distribution</h3>
                    <img src="variant_distribution.png" alt="Variant distribution" style="max-width:100%;">
                </div>
            """

            if not common_variants.empty:
                html_content += """
                <div>
                    <h3>Importance Scores for Common Variants</h3>
                    <img src="common_variant_importance.png" alt="Common variant importance" style="max-width:100%;">
                </div>
                """

            # Add a table of variants with high importance scores
            high_importance = variant_validation_df[variant_validation_df['our_importance'] >= 5].sort_values(
                'our_importance', ascending=False)
            if not high_importance.empty:
                html_content += """
                <h2>High Importance Variants</h2>
                <p>These variants received the highest importance scores in our analysis:</p>
                <table>
                    <tr>
                        <th>Variant</th>
                        <th>Gene</th>
                        <th>Importance Score</th>
                        <th>In Ground Truth</th>
                        <th>Genotype</th>
                    </tr>
                """

                for _, row in high_importance.iterrows():
                    highlighted = "highlighted" if row['in_ground_truth'] else ""
                    html_content += f"""
                    <tr class="{highlighted}">
                        <td>{row['rsid']}</td>
                        <td>{row['gene']}</td>
                        <td>{row['our_importance']:.2f}</td>
                        <td>{"Yes" if row['in_ground_truth'] else "No"}</td>
                        <td>{row['our_genotype']}</td>
                    </tr>
                    """

                html_content += "</table>"

            # Add a table of common variants
            if not common_variants.empty:
                html_content += """
                <h2>Variants Found in Both Datasets</h2>
                <table>
                    <tr>
                        <th>Variant</th>
                        <th>Gene</th>
                        <th>Importance Score</th>
                        <th>Genotype</th>
                    </tr>
                """

                for _, row in common_variants.sort_values(['gene', 'our_importance'],
                                                          ascending=[True, False]).iterrows():
                    html_content += f"""
                    <tr>
                        <td>{row['rsid']}</td>
                        <td>{row['gene']}</td>
                        <td>{row['our_importance']:.2f}</td>
                        <td>{row['our_genotype']}</td>
                    </tr>
                    """

                html_content += "</table>"

            html_content += """
            </body>
            </html>
            """

            # Write the fully rendered HTML content to the file
            f.write(html_content)

    except Exception as e:
        print(f"Error visualizing variant validation: {str(e)}")
        import traceback
        traceback.print_exc()


def run_validation():
    """Main function to run the validation process."""
    # Paths
    vcf_file = "Preprocessed/HG00276_freebayes.preprocessed.vcf"
    match_json_file = "Preprocessed/HG00276_freebayes.preprocessed.match.json"
    phenotype_json_file = "Preprocessed/HG00276_freebayes.preprocessed.phenotype.json"
    ground_truth_file = "Groundtruth/getrm_updated_calls.xlsx"
    output_dir = "validation_results"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Run PharmCAT XAI if needed or load existing results
    xai_results_file = os.path.join(output_dir, "xai_results.json")

    if os.path.exists(xai_results_file):
        print("Loading existing XAI results...")
        try:
            results = {
                'variant_importance': pd.read_json(xai_results_file)
            }
        except:
            print("Error loading existing results. Running PharmCAT XAI analysis...")
            explainer = PharmcatExplainer(vcf_file, match_json_file, phenotype_json_file)
            results = explainer.run()
            # Save results for future use
            if 'variant_importance' in results and not results['variant_importance'].empty:
                results['variant_importance'].to_json(xai_results_file)
    else:
        print("Running PharmCAT XAI analysis...")
        explainer = PharmcatExplainer(vcf_file, match_json_file, phenotype_json_file)
        results = explainer.run()
        # Save results for future use
        if 'variant_importance' in results and not results['variant_importance'].empty:
            results['variant_importance'].to_json(xai_results_file)

    # Load ground truth data
    print("\nLoading ground truth data...")
    gt_data = load_ground_truth(ground_truth_file)

    if gt_data is None:
        print("Error: Could not load ground truth data. Validation aborted.")
        return

    # Perform gene-based validation
    print("\nPerforming gene-based validation...")
    gene_validation = extract_gene_based_validation(gt_data, results['variant_importance'])

    if gene_validation is not None:
        print("\nVisualizing gene-based validation...")
        visualize_gene_validation(gene_validation, output_dir)

    # Perform variant-based validation
    print("\nPerforming variant-based validation...")
    variant_validation = extract_variant_based_validation(gt_data, results['variant_importance'])

    if variant_validation is not None:
        print("\nVisualizing variant-based validation...")
        visualize_variant_validation(variant_validation, output_dir)

    # Create a combined validation report
    print("\nCreating combined validation report...")
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
            <p>This report compares the XAI results with the ground truth data.</p>

            <div class="section">
                <h2>Gene-Based Validation</h2>
                <iframe src="gene_validation.html" height="600"></iframe>
            </div>

            <div class="section">
                <h2>Variant-Based Validation</h2>
                <iframe src="variant_validation.html" height="600"></iframe>
            </div>
        </body>
        </html>
        """)

    print(f"\nValidation complete! Reports saved to {output_dir}")


if __name__ == "__main__":
    run_validation()
