import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class PharmcatExplainer:
    def __init__(self, vcf_file, match_json_file, phenotype_json_file, output_dir='xai_results'):
        self.vcf_file = vcf_file
        self.match_json_file = match_json_file
        self.phenotype_json_file = phenotype_json_file
        self.output_dir = output_dir

        # Ensure output directory exists
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Output directory ensured: {output_dir}")
        except Exception as e:
            print(f"Warning: Could not create output directory {output_dir}: {str(e)}")
            # Try to use a default directory if specified directory fails
            self.output_dir = 'pharmcat_xai_output'
            try:
                os.makedirs(self.output_dir, exist_ok=True)
                print(f"Using alternative output directory: {self.output_dir}")
            except Exception as e2:
                print(f"Error: Could not create alternative output directory: {str(e2)}")

        # Data containers
        self.vcf_data = None
        self.match_data = None
        self.phenotype_data = None
        self.variant_importance = None
        self.gene_summaries = None

    def parse_vcf(self):
        variants = []
        try:
            with open(self.vcf_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue

                    fields = line.strip().split('\t')
                    if len(fields) < 10:  # Need at least 10 fields for FORMAT and sample data
                        continue

                    try:
                        chrom = fields[0]
                        pos = int(fields[1])
                        rsid = fields[2]
                        ref = fields[3]
                        alt = fields[4]
                        info = fields[7]

                        # Extract gene information
                        gene = None
                        if 'PX=' in info:
                            for item in info.split(';'):
                                if item.startswith('PX='):
                                    gene = item.split('=')[1]
                                    break

                        # Extract genotype safely
                        genotype = "0/0"  # Default
                        if ":" in fields[9]:
                            genotype = fields[9].split(':')[0]

                        variants.append({
                            'chrom': chrom,
                            'pos': pos,
                            'rsid': rsid if rsid != '.' else f"pos_{pos}",
                            'ref': ref,
                            'alt': alt.split(',')[0],  # Take first alt allele for simplicity
                            'gene': gene,
                            'genotype': genotype
                        })
                    except (IndexError, ValueError) as e:
                        print(f"Error processing VCF line: {line.strip()}")
                        print(f"Error details: {str(e)}")
                        continue
        except Exception as e:
            print(f"Error reading VCF file: {str(e)}")

        self.vcf_data = pd.DataFrame(variants)
        return self.vcf_data

    def load_match_data(self):
        try:
            with open(self.match_json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Match JSON file not found: {self.match_json_file}")
            print("Please make sure the file exists and the path is correct.")
            self.match_data = []
            return []
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in match file: {self.match_json_file}")
            print("Please make sure the file contains valid JSON data.")
            self.match_data = []
            return []
        except Exception as e:
            print(f"Error loading match data: {str(e)}")
            self.match_data = []
            return []

        # Extract gene-level results
        results = []
        for result in data.get('results', []):
            gene = result.get('gene')
            if not gene:
                continue

            # Extract variant information
            variants = []
            for variant in result.get('variants', []):
                if not isinstance(variant, dict):
                    continue
                variants.append({
                    'position': variant.get('position'),
                    'rsid': variant.get('dbSnpId'),
                    'call': variant.get('call'),
                    'alleles': variant.get('alleles', [])
                })

            # Extract missing positions
            missing_positions = []
            match_data = result.get('matchData', {})
            if match_data and 'missingPositions' in match_data:
                for pos in match_data['missingPositions']:
                    if not isinstance(pos, dict):
                        continue
                    missing_positions.append({
                        'position': pos.get('position'),
                        'rsid': pos.get('rsid'),
                        'ref': pos.get('ref'),
                        'alts': pos.get('alts', [])
                    })

            # Extract uncallable haplotypes
            uncallable_haplotypes = result.get('uncallableHaplotypes', [])

            results.append({
                'gene': gene,
                'variants': variants,
                'missing_positions': missing_positions,
                'uncallable_haplotypes': uncallable_haplotypes,
                'phased': result.get('phased', False),
                'match_data_phased': result.get('matchData', {}).get('phased',
                                                                     False) if 'matchData' in result else False,
                'match_data_homozygous': result.get('matchData', {}).get('homozygous',
                                                                         False) if 'matchData' in result else False
            })

        self.match_data = results
        return results

    def load_phenotype_data(self):
        try:
            with open(self.phenotype_json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Phenotype JSON file not found: {self.phenotype_json_file}")
            print("Please make sure the file exists and the path is correct.")
            self.phenotype_data = []
            return []
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in phenotype file: {self.phenotype_json_file}")
            print("Please make sure the file contains valid JSON data.")
            self.phenotype_data = []
            return []
        except Exception as e:
            print(f"Error loading phenotype data: {str(e)}")
            self.phenotype_data = []
            return []

        # Extract gene-level information
        gene_reports = []
        for source, genes in data.get('geneReports', {}).items():
            for gene_symbol, gene_info in genes.items():
                # Extract diplotype information
                diplotypes = []

                # Try getting diplotypes from recommendationDiplotypes first
                diplotype_sources = []
                if 'recommendationDiplotypes' in gene_info and gene_info['recommendationDiplotypes']:
                    diplotype_sources.extend(gene_info['recommendationDiplotypes'])

                # Then try sourceDiplotypes as a fallback
                if not diplotype_sources and 'sourceDiplotypes' in gene_info and gene_info['sourceDiplotypes']:
                    diplotype_sources.extend(gene_info['sourceDiplotypes'])

                for diplotype in diplotype_sources:
                    if not isinstance(diplotype, dict):
                        continue

                    # Handle potential None values with safer nested gets
                    allele1_obj = diplotype.get('allele1') or {}
                    allele2_obj = diplotype.get('allele2') or {}

                    allele1 = allele1_obj.get('name', 'Unknown') if isinstance(allele1_obj, dict) else 'Unknown'
                    allele2 = allele2_obj.get('name', 'Unknown') if isinstance(allele2_obj, dict) else 'Unknown'
                    phenotypes = diplotype.get('phenotypes', ['Unknown'])

                    if isinstance(phenotypes, str):
                        phenotypes = [phenotypes]

                    diplotypes.append({
                        'allele1': allele1,
                        'allele2': allele2,
                        'phenotypes': phenotypes,
                        'label': diplotype.get('label', f"{allele1}/{allele2}")
                    })

                # Extract variant information
                variants = []
                for variant in gene_info.get('variants', []):
                    variants.append({
                        'position': variant.get('position'),
                        'rsid': variant.get('dbSnpId'),
                        'call': variant.get('call'),
                        'alleles': variant.get('alleles', []),
                        'phased': variant.get('phased', False)
                    })

                gene_reports.append({
                    'source': source,
                    'gene_symbol': gene_symbol,
                    'diplotypes': diplotypes,
                    'variants': variants,
                    'uncalled_haplotypes': gene_info.get('uncalledHaplotypes', []),
                    'phased': gene_info.get('phased', False)
                })

        self.phenotype_data = gene_reports
        return gene_reports

    def calculate_variant_importance(self):
        """Calculate variant importance using a data-driven approach based on the system's behavior."""
        if self.vcf_data is None:
            self.parse_vcf()

        if self.match_data is None:
            self.load_match_data()

        if self.phenotype_data is None:
            self.load_phenotype_data()

        importance_scores = []

        for _, variant in self.vcf_data.iterrows():
            gene = variant['gene']
            if not gene:
                continue

            # Find corresponding match result
            match_result = next((r for r in self.match_data if r['gene'] == gene), None)
            if not match_result:
                continue

            # Find corresponding gene report
            gene_report = next((r for r in self.phenotype_data if r['gene_symbol'] == gene), None)
            if not gene_report:
                continue

            # Initialize importance and explanation
            importance = 0
            explanation = []

            # 1. Base evidence: Variant appears in the VCF
            importance += 1
            explanation.append("Present in VCF file")

            # 2. Evidence from PharmCAT's matcher component
            # Directly used in haplotype determination
            in_match_variants = False
            for match_variant in match_result.get('variants', []):
                if match_variant.get('rsid') == variant['rsid']:
                    importance += 2
                    explanation.append("Used in haplotype matching")
                    in_match_variants = True
                    break

            # 3. Evidence from being mentioned in uncallable haplotypes
            # These variants directly prevent certain haplotype calls
            in_uncallable = False
            for haplotype in match_result.get('uncallable_haplotypes', []):
                if variant['rsid'] in haplotype:
                    importance += 2
                    explanation.append(f"Prevents calling {haplotype}")
                    in_uncallable = True
                    break

            # 4. Evidence from missing data
            # Missing positions impact available haplotypes
            in_missing_positions = False
            for missing in match_result.get('missing_positions', []):
                if missing.get('rsid') == variant['rsid']:
                    importance += 1
                    explanation.append("Missing data affects phenotype predictions")
                    in_missing_positions = True
                    break

            # 5. Evidence from gene report
            # Variants explicitly mentioned in final phenotype report
            in_gene_report = False
            for report_variant in gene_report.get('variants', []):
                if report_variant.get('rsid') == variant['rsid']:
                    importance += 2
                    explanation.append("Referenced in final phenotype report")
                    in_gene_report = True
                    break

            # 6. Evidence from genetic effect
            # Heterozygous or homozygous status affects phenotype expression
            if variant['genotype'] == '0/1':
                importance += 1
                explanation.append("Heterozygous variant (one altered copy)")
            elif variant['genotype'] == '1/1':
                importance += 2
                explanation.append("Homozygous alternate (two altered copies)")

            # 7. Evidence from phenotype impact
            # If explicit phenotypes are assigned despite missing data
            phenotypes = []
            for diplotype in gene_report.get('diplotypes', []):
                phenotypes.extend(diplotype.get('phenotypes', []))

            has_distinct_phenotype = False
            if phenotypes and "Unknown" not in phenotypes and "No Result" not in phenotypes:
                if in_match_variants or in_gene_report:
                    importance += 1
                    explanation.append("Contributes to a definitive phenotype prediction")
                    has_distinct_phenotype = True

            # 8. Differentiate by position to avoid identical scores
            # Adds minimal noise (0.01-0.99) to create unique scores while preserving ranking
            pos_factor = 0.01 * (hash(str(variant['pos'])) % 100) / 100
            importance += pos_factor

            # Only include variants with some importance
            if importance > 0:
                importance_scores.append({
                    'gene': gene,
                    'rsid': variant['rsid'],
                    'pos': variant['pos'],
                    'ref': variant['ref'],
                    'alt': variant['alt'],
                    'genotype': variant['genotype'],
                    'importance': round(importance, 2),  # Round to 2 decimal places
                    'explanation': "; ".join(explanation),
                    'phenotypes': list(set(phenotypes)) if phenotypes else ["Unknown"],
                    'in_match': in_match_variants,
                    'in_report': in_gene_report,
                    'uncallable': in_uncallable,
                    'missing': in_missing_positions,
                    'has_phenotype': has_distinct_phenotype
                })

        self.variant_importance = pd.DataFrame(importance_scores)
        return self.variant_importance

    def generate_summaries(self):
        if self.variant_importance is None:
            self.calculate_variant_importance()

        summaries = []

        # Process all genes in match data even if they have no important variants
        processed_genes = set()

        # Add gene function map for explanations
        gene_function_map = {
            "CYP2D6": "metabolizes approximately 25% of clinically used drugs including antidepressants, antipsychotics, opioids, and beta blockers",
            "CYP2C19": "involved in the metabolism of several important drug classes including proton pump inhibitors, antiplatelet drugs, and antidepressants",
            "CYP3A5": "metabolizes approximately 50% of clinically used drugs, particularly important for immunosuppressants like tacrolimus",
            "SLCO1B1": "mediates the uptake of various drugs into hepatocytes, especially statins",
            "VKORC1": "encodes the target of warfarin and other vitamin K antagonists used as anticoagulants",
            "UGT1A1": "responsible for bilirubin conjugation and metabolism of certain drugs including irinotecan",
            "CYP4F2": "involved in vitamin K metabolism, affects warfarin dose requirements",
            "DPYD": "metabolizes fluoropyrimidine drugs like 5-fluorouracil (5-FU)",
            "TPMT": "metabolizes thiopurine drugs like azathioprine and mercaptopurine",
            "NUDT15": "involved in metabolism of thiopurine drugs",
            "CYP2C9": "metabolizes many drugs including warfarin, NSAIDs, and some antidiabetics",
            "G6PD": "involved in the hexose monophosphate shunt, deficiency can lead to hemolytic anemia with certain drugs",
            "CFTR": "encodes a chloride channel protein, mutations cause cystic fibrosis",
            "IFNL3": "involved in immune response, affects hepatitis C treatment response"
        }

        # First process genes with important variants
        for gene, group in self.variant_importance.groupby('gene'):
            processed_genes.add(gene)

            # Find gene report
            gene_report = next((r for r in self.phenotype_data if r['gene_symbol'] == gene), None)

            # Find match result
            match_result = next((r for r in self.match_data if r['gene'] == gene), None)

            # Get phenotypes
            phenotypes = []
            if gene_report:
                for diplotype in gene_report.get('diplotypes', []):
                    phenotypes.extend(diplotype.get('phenotypes', []))

            phenotype_str = ", ".join(set(phenotypes)) if phenotypes else "Unknown"

            # Get gene function
            gene_function = gene_function_map.get(gene, "affects drug metabolism or response")

            # Create gene summary
            gene_summary = {
                'gene': gene,
                'phenotypes': list(set(phenotypes)),
                'summary': f"Gene: {gene}\n"
                           f"Function: {gene_function}\n"
                           f"Phenotype(s): {phenotype_str}\n"
                           f"Number of important variants: {len(group)}\n"
            }

            # Add clinical significance if known
            clinical_significance = self.get_clinical_significance(gene, phenotypes)
            if clinical_significance:
                gene_summary['summary'] += f"Clinical Significance: {clinical_significance}\n"

            # Add variant details
            if not group.empty:
                variant_details = []
                for _, variant in group.sort_values('importance', ascending=False).iterrows():
                    genotype_desc = "homozygous reference (0/0)" if variant['genotype'] == '0/0' else \
                        "heterozygous (0/1)" if variant['genotype'] == '0/1' else \
                            "homozygous alternate (1/1)" if variant['genotype'] == '1/1' else \
                                variant['genotype']

                    # Add clinical impact of variant if known
                    variant_impact = self.get_variant_impact(gene, variant['rsid'], variant['genotype'])
                    if variant_impact:
                        impact_text = f"\n  Clinical Impact: {variant_impact}"
                    else:
                        impact_text = ""

                    variant_detail = f"- {variant['rsid']} ({variant['ref']} to {variant['alt']}): {genotype_desc}\n" \
                                     f"  Importance Score: {variant['importance']}\n" \
                                     f"  Reason: {variant['explanation']}{impact_text}"
                    variant_details.append(variant_detail)

                gene_summary['summary'] += "\nImportant variants:\n" + "\n\n".join(variant_details)

            # Add uncallable haplotypes if any
            if match_result and match_result.get('uncallable_haplotypes'):
                gene_summary['summary'] += "\n\nUncallable haplotypes:\n- " + "\n- ".join(
                    match_result['uncallable_haplotypes'])

            # Add missing positions if any
            if match_result and match_result.get('missing_positions'):
                missing_positions = []
                for pos in match_result['missing_positions']:
                    rsid = pos.get('rsid', 'unknown')
                    position = pos.get('position', 'unknown')
                    missing_positions.append(f"{rsid} at position {position}")

                if missing_positions:
                    gene_summary['summary'] += "\n\nMissing positions:\n- " + "\n- ".join(missing_positions)

            # Add explanation
            gene_summary['summary'] += "\n\nExplanation:\n"
            if group.empty and match_result and not match_result.get('uncallable_haplotypes') and not match_result.get(
                    'missing_positions'):
                gene_summary[
                    'summary'] += f"No significant variants or uncallable haplotypes found for this gene. {gene} {gene_function}, but no variants affecting function were detected in this sample."
            elif group.empty and match_result and (
                    match_result.get('uncallable_haplotypes') or match_result.get('missing_positions')):
                gene_summary[
                    'summary'] += f"The phenotype determination for {gene} is affected by missing data (uncallable haplotypes or missing positions). {gene} {gene_function}, but complete phenotype prediction is not possible due to missing genetic information."
            else:
                gene_summary[
                    'summary'] += f"The {phenotype_str} phenotype for {gene} is determined by the presence of the variants listed above. {gene} {gene_function}."
                if clinical_significance:
                    gene_summary['summary'] += f" {clinical_significance}"
                if match_result and (
                        match_result.get('uncallable_haplotypes') or match_result.get('missing_positions')):
                    gene_summary[
                        'summary'] += f" Additionally, there are uncallable haplotypes or missing positions which affect the confidence of this determination."

            summaries.append(gene_summary)

        # Then process remaining genes in match data
        for match_result in self.match_data:
            gene = match_result['gene']
            if gene in processed_genes:
                continue

            # Find gene report
            gene_report = next((r for r in self.phenotype_data if r['gene_symbol'] == gene), None)

            # Get phenotypes
            phenotypes = []
            if gene_report:
                for diplotype in gene_report.get('diplotypes', []):
                    phenotypes.extend(diplotype.get('phenotypes', []))

            phenotype_str = ", ".join(set(phenotypes)) if phenotypes else "Unknown"

            # Get gene function
            gene_function = gene_function_map.get(gene, "affects drug metabolism or response")

            # Create gene summary
            gene_summary = {
                'gene': gene,
                'phenotypes': list(set(phenotypes)),
                'summary': f"Gene: {gene}\n"
                           f"Function: {gene_function}\n"
                           f"Phenotype(s): {phenotype_str}\n"
                           f"Number of important variants: 0\n"
            }

            # Add clinical significance if known
            clinical_significance = self.get_clinical_significance(gene, phenotypes)
            if clinical_significance:
                gene_summary['summary'] += f"Clinical Significance: {clinical_significance}\n"

            # Add uncallable haplotypes if any
            if match_result.get('uncallable_haplotypes'):
                gene_summary['summary'] += "\n\nUncallable haplotypes:\n- " + "\n- ".join(
                    match_result['uncallable_haplotypes'])

            # Add missing positions if any
            if match_result.get('missing_positions'):
                missing_positions = []
                for pos in match_result['missing_positions']:
                    rsid = pos.get('rsid', 'unknown')
                    position = pos.get('position', 'unknown')
                    missing_positions.append(f"{rsid} at position {position}")

                if missing_positions:
                    gene_summary['summary'] += "\n\nMissing positions:\n- " + "\n- ".join(missing_positions)

            # Add explanation
            gene_summary['summary'] += "\n\nExplanation:\n"
            if not match_result.get('uncallable_haplotypes') and not match_result.get('missing_positions'):
                gene_summary[
                    'summary'] += f"No significant variants found for this gene in the input data. {gene} {gene_function}, but no variants affecting function were detected in this sample."
            else:
                gene_summary[
                    'summary'] += f"The phenotype determination for {gene} is affected by missing data (uncallable haplotypes or missing positions). {gene} {gene_function}, but complete phenotype prediction is not possible due to missing genetic information."

            summaries.append(gene_summary)

        self.gene_summaries = summaries
        return summaries

    def get_clinical_significance(self, gene, phenotypes):
        """Return clinical significance explanation based on gene and phenotypes."""
        if not phenotypes or phenotypes == ["Unknown"] or phenotypes == ["No Result"]:
            return ""

        # Common clinical significance patterns
        if gene == "CYP2D6":
            if "Poor Metabolizer" in phenotypes:
                return "Poor metabolizers may require dose reductions for affected drugs or alternative medications."
            elif "Intermediate Metabolizer" in phenotypes:
                return "Intermediate metabolizers may have reduced efficacy with prodrugs like codeine or tamoxifen."
            elif "Ultrarapid Metabolizer" in phenotypes:
                return "Ultrarapid metabolizers may experience treatment failure at standard doses of affected drugs."

        elif gene == "CYP2C19":
            if "Poor Metabolizer" in phenotypes:
                return "Poor metabolizers may have reduced efficacy with clopidogrel and increased exposure to PPIs."
            elif "Rapid Metabolizer" in phenotypes or "Ultrarapid Metabolizer" in phenotypes:
                return "Rapid/ultrarapid metabolizers may have increased efficacy with clopidogrel and reduced exposure to PPIs."

        elif gene == "SLCO1B1":
            if "Decreased Function" in phenotypes or "Poor Function" in phenotypes:
                return "Decreased function may lead to higher statin exposure and increased risk of myopathy."

        elif gene == "VKORC1":
            if "Low Warfarin Sensitivity" in phenotypes:
                return "Lower sensitivity may require higher warfarin doses to achieve therapeutic anticoagulation."
            elif "High Warfarin Sensitivity" in phenotypes:
                return "Higher sensitivity may require lower warfarin doses to avoid over-anticoagulation."

        elif gene == "CYP3A5":
            if "Poor Metabolizer" in phenotypes:
                return "Poor metabolizers generally require lower doses of tacrolimus, cyclosporine, and other affected drugs."
            elif "Intermediate Metabolizer" in phenotypes or "Normal Metabolizer" in phenotypes:
                return "Normal/intermediate metabolizers may require higher doses of tacrolimus and similar drugs."

        elif gene == "UGT1A1":
            if "*28/*28" in str(phenotypes) or "Poor Metabolizer" in phenotypes:
                return "Reduced function may increase risk of irinotecan toxicity and unconjugated hyperbilirubinemia."

        # Generic significance by phenotype pattern
        if any(p for p in phenotypes if "Poor" in p and "Metabolizer" in p):
            return "Poor metabolizer status may require dose adjustments for affected medications."
        elif any(p for p in phenotypes if "Intermediate" in p and "Metabolizer" in p):
            return "Intermediate metabolizer status may require monitoring for efficacy and side effects."
        elif any(p for p in phenotypes if "Ultrarapid" in p and "Metabolizer" in p):
            return "Ultrarapid metabolizer status may require dose increases or alternative medications."

        return ""

    def get_variant_impact(self, gene, rsid, genotype):
        """Return clinical impact of specific variant."""
        # Create a dictionary mapping gene + rsid + genotype to clinical impact
        impact_map = {
            # CYP2D6 variants
            ("CYP2D6", "rs3745274", "0/1"): "May reduce metabolism of CYP2D6 substrates",
            ("CYP2D6", "rs3745274", "1/1"): "Significantly reduces metabolism of CYP2D6 substrates",
            ("CYP2D6", "rs2279343", "0/1"): "May alter CYP2D6 activity",
            ("CYP2D6", "rs2279343", "1/1"): "Associated with altered metabolism of CYP2D6 substrates",

            # CYP2C19 variants
            ("CYP2C19", "rs4244285", "0/1"): "Reduced function, may impair clopidogrel activation",
            ("CYP2C19", "rs4244285", "1/1"): "Loss of function, significantly impairs clopidogrel activation",
            ("CYP2C19", "rs12248560", "0/1"): "Enhanced function, may increase clopidogrel response",
            ("CYP2C19", "rs12248560", "1/1"): "Significantly enhanced function, increases clopidogrel response",

            # SLCO1B1 variants
            ("SLCO1B1", "rs4149056", "0/1"): "Reduced transport, moderate increase in statin exposure",
            ("SLCO1B1", "rs4149056", "1/1"): "Significantly reduced transport, higher risk of statin myopathy",
            ("SLCO1B1", "rs2306283", "0/1"): "Possible increased transporter activity",

            # VKORC1 variants
            ("VKORC1", "rs9923231", "0/1"): "Intermediate warfarin sensitivity",
            ("VKORC1", "rs9923231", "1/1"): "High warfarin sensitivity, lower dose requirements",

            # CYP3A5 variants
            ("CYP3A5", "rs776746", "0/1"): "Intermediate metabolizer, affects tacrolimus exposure",
            ("CYP3A5", "rs776746", "1/1"): "Non-expresser (*3/*3), higher tacrolimus exposure",

            # UGT1A1 variants
            ("UGT1A1", "rs887829", "0/1"): "Reduced enzyme activity, moderate risk of toxicity",
            ("UGT1A1", "rs887829", "1/1"): "Significantly reduced activity, higher risk of toxicity",
            ("UGT1A1", "rs3064744", "0/1"): "Reduced enzyme activity, may affect irinotecan metabolism",
            ("UGT1A1", "rs3064744", "1/1"): "Significant reduction in enzyme activity, higher toxicity risk",

            # CYP4F2 variants
            ("CYP4F2", "rs2108622", "0/1"): "Moderately decreased vitamin K metabolism, affects warfarin dosing",
            ("CYP4F2", "rs2108622", "1/1"): "Significantly decreased vitamin K metabolism, higher warfarin dose needs"
        }

        return impact_map.get((gene, rsid, genotype), "")

    def visualize_importance(self):
        """Create visualizations of variant importance based on data-driven metrics."""
        if self.variant_importance is None or self.variant_importance.empty:
            print("No variant importance data to visualize.")
            return

        try:
            # Use a consistent color palette
            colors = sns.color_palette("husl", len(self.variant_importance['gene'].unique()))
            gene_color_map = dict(zip(self.variant_importance['gene'].unique(), colors))

            # Create overall importance plot with sorted values
            plt.figure(figsize=(14, 8))

            # Sort by importance score
            sorted_data = self.variant_importance.sort_values('importance', ascending=False)

            # Create bars with custom colors
            bars = sns.barplot(data=sorted_data, x='rsid', y='importance', hue='gene')

            # Customize the plot
            plt.title("Data-Driven Variant Importance for Phenotype Predictions", fontsize=16)
            plt.xlabel("Variant (rsID)", fontsize=12)
            plt.ylabel("Importance Score", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title="Gene", fontsize=10)

            # Add value labels on top of bars
            for i, bar in enumerate(bars.patches):
                bars.text(
                    bar.get_x() + bar.get_width() / 2.,
                    bar.get_height() + 0.1,
                    f"{bar.get_height():.2f}",
                    ha='center', fontsize=9
                )

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "overall_variant_importance.png"), dpi=150)
            plt.close()

            # Create a stacked bar chart showing the composition of importance scores
            if len(sorted_data) > 0:
                plt.figure(figsize=(14, 8))

                # Create a new DataFrame for the components
                component_data = []
                for _, row in sorted_data.iterrows():
                    # Base score
                    component_data.append({
                        'rsid': row['rsid'],
                        'gene': row['gene'],
                        'component': 'Base',
                        'value': 1.0
                    })

                    # Matching
                    if row.get('in_match'):
                        component_data.append({
                            'rsid': row['rsid'],
                            'gene': row['gene'],
                            'component': 'Matching',
                            'value': 2.0
                        })

                    # Report
                    if row.get('in_report'):
                        component_data.append({
                            'rsid': row['rsid'],
                            'gene': row['gene'],
                            'component': 'Report',
                            'value': 2.0
                        })

                    # Uncallable
                    if row.get('uncallable'):
                        component_data.append({
                            'rsid': row['rsid'],
                            'gene': row['gene'],
                            'component': 'Uncallable',
                            'value': 2.0
                        })

                    # Missing
                    if row.get('missing'):
                        component_data.append({
                            'rsid': row['rsid'],
                            'gene': row['gene'],
                            'component': 'Missing',
                            'value': 1.0
                        })

                    # Genotype
                    if row['genotype'] == '0/1':
                        component_data.append({
                            'rsid': row['rsid'],
                            'gene': row['gene'],
                            'component': 'Heterozygous',
                            'value': 1.0
                        })
                    elif row['genotype'] == '1/1':
                        component_data.append({
                            'rsid': row['rsid'],
                            'gene': row['gene'],
                            'component': 'Homozygous',
                            'value': 2.0
                        })

                    # Phenotype
                    if row.get('has_phenotype'):
                        component_data.append({
                            'rsid': row['rsid'],
                            'gene': row['gene'],
                            'component': 'Phenotype',
                            'value': 1.0
                        })

                component_df = pd.DataFrame(component_data)

                # Create a pivot table for the stacked bar chart
                pivot_df = component_df.pivot_table(
                    index='rsid',
                    columns='component',
                    values='value',
                    aggfunc='sum',
                    fill_value=0
                )

                # Sort by total importance
                total_importance = pivot_df.sum(axis=1)
                pivot_df = pivot_df.loc[total_importance.sort_values(ascending=False).index]

                # Create stacked bar plot
                ax = pivot_df.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis')

                plt.title("Composition of Variant Importance Scores", fontsize=16)
                plt.xlabel("Variant (rsID)", fontsize=12)
                plt.ylabel("Importance Score Components", fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.legend(title="Evidence Component", bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, "importance_composition.png"), dpi=150)
                plt.close()

            # Create gene-specific plots
            for gene, group in self.variant_importance.groupby('gene'):
                if len(group) > 1:  # Only create plot if there's more than one variant
                    plt.figure(figsize=(10, 6))

                    # Sort by importance
                    group_sorted = group.sort_values('importance', ascending=False)

                    # Use consistent color for this gene
                    gene_color = gene_color_map[gene]

                    # Create barplot with custom color
                    bars = sns.barplot(data=group_sorted, x='rsid', y='importance', color=gene_color)

                    # Customize the plot
                    plt.title(f"Variant Importance for {gene}", fontsize=16)
                    plt.xlabel("Variant (rsID)", fontsize=12)
                    plt.ylabel("Importance Score", fontsize=12)
                    plt.xticks(rotation=45, ha='right')

                    # Add annotations with importance values
                    for i, bar in enumerate(bars.patches):
                        bars.text(
                            bar.get_x() + bar.get_width() / 2.,
                            bar.get_height() + 0.05,
                            f"{bar.get_height():.2f}",
                            ha='center', fontsize=10
                        )

                    # Add the genotype information below each bar
                    for i, (_, row) in enumerate(group_sorted.iterrows()):
                        plt.text(
                            i, -0.2,
                            f"{row['genotype']}",
                            ha='center', fontsize=9
                        )

                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, f"{gene}_importance.png"), dpi=150)
                    plt.close()

                    # Create a stacked bar chart for this gene
                    component_data = []
                    for _, row in group_sorted.iterrows():
                        # Base score
                        component_data.append({
                            'rsid': row['rsid'],
                            'component': 'Base',
                            'value': 1.0
                        })

                        # Matching
                        if row.get('in_match'):
                            component_data.append({
                                'rsid': row['rsid'],
                                'component': 'Matching',
                                'value': 2.0
                            })

                        # Report
                        if row.get('in_report'):
                            component_data.append({
                                'rsid': row['rsid'],
                                'component': 'Report',
                                'value': 2.0
                            })

                        # Uncallable
                        if row.get('uncallable'):
                            component_data.append({
                                'rsid': row['rsid'],
                                'component': 'Uncallable',
                                'value': 2.0
                            })

                        # Missing
                        if row.get('missing'):
                            component_data.append({
                                'rsid': row['rsid'],
                                'component': 'Missing',
                                'value': 1.0
                            })

                        # Genotype
                        if row['genotype'] == '0/1':
                            component_data.append({
                                'rsid': row['rsid'],
                                'component': 'Heterozygous',
                                'value': 1.0
                            })
                        elif row['genotype'] == '1/1':
                            component_data.append({
                                'rsid': row['rsid'],
                                'component': 'Homozygous',
                                'value': 2.0
                            })

                        # Phenotype
                        if row.get('has_phenotype'):
                            component_data.append({
                                'rsid': row['rsid'],
                                'component': 'Phenotype',
                                'value': 1.0
                            })

                    if component_data:
                        gene_component_df = pd.DataFrame(component_data)

                        # Create a pivot table for the stacked bar chart
                        gene_pivot_df = gene_component_df.pivot_table(
                            index='rsid',
                            columns='component',
                            values='value',
                            aggfunc='sum',
                            fill_value=0
                        )

                        # Sort by total importance
                        gene_total_importance = gene_pivot_df.sum(axis=1)
                        gene_pivot_df = gene_pivot_df.loc[gene_total_importance.sort_values(ascending=False).index]

                        # Create stacked bar plot
                        plt.figure(figsize=(10, 6))
                        ax = gene_pivot_df.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')

                        plt.title(f"Composition of {gene} Variant Importance Scores", fontsize=16)
                        plt.xlabel("Variant (rsID)", fontsize=12)
                        plt.ylabel("Importance Score Components", fontsize=12)
                        plt.xticks(rotation=45, ha='right')
                        plt.legend(title="Evidence Component", bbox_to_anchor=(1.05, 1), loc='upper left')
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.output_dir, f"{gene}_importance_composition.png"), dpi=150)
                        plt.close()

            # Create a heatmap of variant importance by gene
            if len(self.variant_importance) > 3:  # Only create if we have enough data
                plt.figure(figsize=(12, 8))

                # Pivot data for heatmap
                pivot_data = self.variant_importance.pivot_table(
                    index='gene', columns='rsid', values='importance', fill_value=0
                )

                # Create heatmap
                sns.heatmap(pivot_data, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5)

                plt.title("Variant Importance Heatmap by Gene", fontsize=16)
                plt.ylabel("Gene", fontsize=12)
                plt.xlabel("Variant (rsID)", fontsize=12)
                plt.xticks(rotation=45, ha='right')

                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, "importance_heatmap.png"), dpi=150)
                plt.close()

        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
            import traceback
            traceback.print_exc()

    def create_counterfactual_analysis(self):
        if self.variant_importance is None:
            self.calculate_variant_importance()

        counterfactuals = []

        for _, variant in self.variant_importance.iterrows():
            gene = variant['gene']
            genotype = variant['genotype']

            # Define alternative genotypes
            alt_genotypes = []
            if genotype == '0/0':
                alt_genotypes = ['0/1', '1/1']
            elif genotype == '0/1':
                alt_genotypes = ['0/0', '1/1']
            elif genotype == '1/1':
                alt_genotypes = ['0/0', '0/1']

            for alt_genotype in alt_genotypes:
                impact = "unknown"
                confidence = "low"
                detail = ""

                # Assess impact by gene and variant
                if gene == "CYP2D6":
                    if genotype == '0/0' and alt_genotype == '0/1':
                        impact = "moderate decrease in metabolism"
                        confidence = "medium"
                        detail = "May reduce metabolism of certain drugs (codeine, tamoxifen)"
                    elif genotype == '0/0' and alt_genotype == '1/1':
                        impact = "significant decrease in metabolism"
                        confidence = "high"
                        detail = "Could significantly impair drug metabolism, increasing risk of adverse effects"
                    elif genotype == '0/1' and alt_genotype == '0/0':
                        impact = "moderate increase in metabolism"
                        confidence = "medium"
                        detail = "May enhance drug clearance, potentially reducing efficacy"
                    elif genotype == '0/1' and alt_genotype == '1/1':
                        impact = "moderate decrease in metabolism"
                        confidence = "medium"
                        detail = "Further reduction in enzyme activity"
                    elif genotype == '1/1' and alt_genotype == '0/0':
                        impact = "significant increase in metabolism"
                        confidence = "high"
                        detail = "Normal enzyme function restored"
                    elif genotype == '1/1' and alt_genotype == '0/1':
                        impact = "moderate increase in metabolism"
                        confidence = "medium"
                        detail = "Partial restoration of enzyme function"

                elif gene == "CYP2C19":
                    if genotype == '0/0' and alt_genotype in ['0/1', '1/1']:
                        impact = "decreased metabolism"
                        confidence = "high"
                        detail = "May affect clopidogrel activation, PPI metabolism"
                    elif genotype in ['0/1', '1/1'] and alt_genotype == '0/0':
                        impact = "increased metabolism"
                        confidence = "high"
                        detail = "Could enhance drug clearance"

                elif gene == "CYP3A5":
                    if genotype == '0/0' and alt_genotype in ['0/1', '1/1']:
                        impact = "increased metabolism"
                        confidence = "high"
                        detail = "May affect tacrolimus, cyclosporine dosing"
                    elif genotype in ['0/1', '1/1'] and alt_genotype == '0/0':
                        impact = "decreased metabolism"
                        confidence = "high"
                        detail = "Could lead to higher drug exposure"

                elif gene == "SLCO1B1":
                    if genotype == '0/0' and alt_genotype in ['0/1', '1/1']:
                        impact = "decreased transport"
                        confidence = "high"
                        detail = "May increase statin-related muscle toxicity risk"
                    elif genotype in ['0/1', '1/1'] and alt_genotype == '0/0':
                        impact = "increased transport"
                        confidence = "high"
                        detail = "Could reduce adverse effects risk"

                elif gene == "VKORC1":
                    if genotype == '0/0' and alt_genotype in ['0/1', '1/1']:
                        impact = "increased warfarin sensitivity"
                        confidence = "high"
                        detail = "May require lower warfarin doses"
                    elif genotype in ['0/1', '1/1'] and alt_genotype == '0/0':
                        impact = "decreased warfarin sensitivity"
                        confidence = "high"
                        detail = "Could require higher warfarin doses"

                elif gene == "UGT1A1":
                    if variant['rsid'] == 'rs887829' or variant['rsid'] == 'rs3064744':
                        if genotype == '0/0' and alt_genotype in ['0/1', '1/1']:
                            impact = "decreased enzyme activity"
                            confidence = "high"
                            detail = "May increase irinotecan toxicity risk"
                        elif genotype in ['0/1', '1/1'] and alt_genotype == '0/0':
                            impact = "increased enzyme activity"
                            confidence = "high"
                            detail = "Could reduce hyperbilirubinemia risk"

                # Generic assessment if no specific rule matched
                if impact == "unknown":
                    if genotype == '0/0' and alt_genotype == '0/1':
                        impact = "moderate effect change"
                        confidence = "low"
                    elif genotype == '0/0' and alt_genotype == '1/1':
                        impact = "significant effect change"
                        confidence = "medium"
                    elif genotype == '0/1' and alt_genotype == '0/0':
                        impact = "moderate reversal effect"
                        confidence = "low"
                    elif genotype == '0/1' and alt_genotype == '1/1':
                        impact = "moderate enhancement effect"
                        confidence = "low"
                    elif genotype == '1/1' and alt_genotype == '0/0':
                        impact = "significant reversal effect"
                        confidence = "medium"
                    elif genotype == '1/1' and alt_genotype == '0/1':
                        impact = "moderate reduction effect"
                        confidence = "low"

                explanation = f"If {variant['rsid']} had genotype {alt_genotype} instead of {genotype}, there might be a {impact} on phenotype"
                if detail:
                    explanation += f". {detail}"

                counterfactuals.append({
                    'gene': gene,
                    'rsid': variant['rsid'],
                    'current_genotype': genotype,
                    'alternative_genotype': alt_genotype,
                    'potential_impact': impact,
                    'confidence': confidence,
                    'detail': detail,
                    'explanation': explanation
                })

        return pd.DataFrame(counterfactuals)

    def generate_html_report(self):
        if self.gene_summaries is None:
            self.generate_summaries()

        counterfactuals = self.create_counterfactual_analysis()

        # Create HTML report with explicit UTF-8 encoding
        with open(os.path.join(self.output_dir, "pharmcat_xai_report.html"), 'w', encoding='utf-8') as f:
            f.write("""
            <html>
            <head>
                <title>PharmCAT XAI Report</title>
                <meta charset="UTF-8">
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px 40px; line-height: 1.6; }
                    h1, h2, h3 { color: #2c3e50; }
                    .gene-section { border: 1px solid #ddd; padding: 20px; margin-bottom: 30px; border-radius: 5px; }
                    .variant-info { margin-left: 20px; margin-bottom: 15px; position: relative; }
                    .importance-bar { 
                        height: 8px; 
                        background-color: #3498db; 
                        margin-top: 5px;
                        border-radius: 4px;
                    }
                    .explanation { background-color: #f8f9fa; padding: 15px; border-left: 4px solid #007bff; }
                    .counterfactual { background-color: #fff3cd; padding: 10px; margin-top: 20px; border-radius: 5px; }
                    .missing { color: #dc3545; }
                    .present { color: #28a745; }
                    table { border-collapse: collapse; width: 100%; margin: 15px 0; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    tr:nth-child(even) { background-color: #f9f9f9; }
                    .tooltip { position: relative; display: inline-block; cursor: pointer; border-bottom: 1px dotted #666; }
                    .tooltip .tooltiptext { 
                        visibility: hidden; 
                        width: 300px; 
                        background-color: #555; 
                        color: #fff; 
                        text-align: left; 
                        border-radius: 6px; 
                        padding: 10px; 
                        position: absolute; 
                        z-index: 1; 
                        bottom: 125%; 
                        left: 50%; 
                        margin-left: -150px; 
                        opacity: 0; 
                        transition: opacity 0.3s; 
                    }
                    .tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
                    .badge {
                        display: inline-block;
                        padding: 3px 7px;
                        font-size: 12px;
                        font-weight: bold;
                        line-height: 1;
                        color: #fff;
                        text-align: center;
                        white-space: nowrap;
                        vertical-align: baseline;
                        border-radius: 10px;
                        margin-left: 5px;
                    }
                    .badge-high { background-color: #dc3545; }
                    .badge-medium { background-color: #fd7e14; }
                    .badge-low { background-color: #6c757d; }
                </style>
            </head>
            <body>
                <h1>PharmCAT XAI Report</h1>
                <p>This report explains how genetic variants in the input VCF file influence pharmacogenomic phenotype predictions made by PharmCAT.</p>
            """)

            # Add importance score explanation
            f.write("""
                <div class="explanation">
                    <h3>How to Read Importance Scores</h3>
                    <p>Importance scores indicate how much each variant contributes to the phenotype prediction based on <strong>data-driven analysis</strong> of PharmCAT's behavior:</p>
                    <ul>
                        <li><strong>Base evidence (1.0)</strong>: Every variant found in the VCF gets this score</li>
                        <li><strong>Matching evidence (2.0)</strong>: Variant is directly used in PharmCAT's haplotype matching</li>
                        <li><strong>Uncallable haplotype evidence (2.0)</strong>: Variant prevents calling specific haplotypes</li>
                        <li><strong>Missing position evidence (1.0)</strong>: Missing data affects available haplotype calls</li>
                        <li><strong>Report evidence (2.0)</strong>: Variant is referenced in the final phenotype report</li>
                        <li><strong>Genotype evidence</strong>: Heterozygous (1.0) or Homozygous alternate (2.0)</li>
                        <li><strong>Phenotype impact (1.0)</strong>: Contributes to a definitive phenotype prediction</li>
                    </ul>
                    <p>Higher scores (5.0+) indicate variants that significantly influence PharmCAT's phenotype determination.</p>
                </div>
            """)

            # Summary table
            f.write("""
                <h2>Gene Summary</h2>
                <table>
                    <tr>
                        <th>Gene</th>
                        <th>Phenotype(s)</th>
                        <th>Important Variants</th>
                        <th>Top Variant</th>
                        <th>Missing Data</th>
                    </tr>
            """)

            # Get all variants for importance info
            all_variants = self.variant_importance if self.variant_importance is not None else pd.DataFrame()

            for summary in self.gene_summaries:
                gene = summary['gene']
                phenotypes = ", ".join(summary['phenotypes']) if summary['phenotypes'] else "Unknown"

                # Find match result for this gene
                match_result = next((r for r in self.match_data if r['gene'] == gene), None)
                has_missing_data = False
                if match_result and (
                        match_result.get('uncallable_haplotypes') or match_result.get('missing_positions')):
                    has_missing_data = True

                # Count variants for this gene
                gene_variants = all_variants[all_variants['gene'] == gene] if not all_variants.empty else pd.DataFrame()
                variants_count = len(gene_variants)

                # Get top variant
                if not gene_variants.empty:
                    top_variant = gene_variants.sort_values('importance', ascending=False).iloc[0]
                    top_variant_str = f"{top_variant['rsid']} ({top_variant['importance']:.2f})"
                else:
                    top_variant_str = "None"

                f.write(f"""
                    <tr>
                        <td>{gene}</td>
                        <td>{phenotypes}</td>
                        <td class="{'present' if variants_count > 0 else ''}">{variants_count}</td>
                        <td>{top_variant_str}</td>
                        <td class="{'missing' if has_missing_data else ''}">{('Yes' if has_missing_data else 'No')}</td>
                    </tr>
                """)

            f.write("</table>")

            # Gene details
            for summary in self.gene_summaries:
                gene = summary['gene']
                f.write(f"""
                    <div class="gene-section">
                        <h2>Gene: {gene}</h2>
                        <p><strong>Phenotype(s):</strong> {', '.join(summary['phenotypes']) if summary['phenotypes'] else 'Unknown'}</p>
                """)

                # Get variants for this gene
                gene_variants = all_variants[all_variants['gene'] == gene] if not all_variants.empty else pd.DataFrame()

                if not gene_variants.empty:
                    f.write("<h3>Important Variants:</h3>")

                    # Find maximum importance for scaling bars
                    max_importance = gene_variants['importance'].max() if len(gene_variants) > 0 else 0

                    for _, variant in gene_variants.sort_values('importance', ascending=False).iterrows():
                        genotype_desc = "homozygous reference (0/0)" if variant['genotype'] == '0/0' else \
                            "heterozygous (0/1)" if variant['genotype'] == '0/1' else \
                                "homozygous alternate (1/1)" if variant['genotype'] == '1/1' else \
                                    variant['genotype']

                        # Get variant impact if available
                        variant_impact = self.get_variant_impact(gene, variant['rsid'], variant['genotype'])
                        impact_html = f"<br><strong>Clinical Impact:</strong> {variant_impact}" if variant_impact else ""

                        # Determine importance level for badge based on data-driven scoring
                        if variant['importance'] >= 7:
                            badge_class = "badge-high"
                            importance_level = "High"
                        elif variant['importance'] >= 5:
                            badge_class = "badge-medium"
                            importance_level = "Medium"
                        else:
                            badge_class = "badge-low"
                            importance_level = "Low"

                        # Calculate importance bar width as percentage of maximum
                        bar_width = (variant['importance'] / max(max_importance, 8)) * 100

                        # Create color coding based on evidence types
                        evidence_badges = ""
                        if variant.get('in_match'):
                            evidence_badges += '<span class="badge" style="background-color:#2ecc71">Match</span> '
                        if variant.get('in_report'):
                            evidence_badges += '<span class="badge" style="background-color:#3498db">Report</span> '
                        if variant.get('uncallable'):
                            evidence_badges += '<span class="badge" style="background-color:#e74c3c">Uncallable</span> '
                        if variant.get('missing'):
                            evidence_badges += '<span class="badge" style="background-color:#f39c12">Missing</span> '
                        if variant.get('has_phenotype'):
                            evidence_badges += '<span class="badge" style="background-color:#9b59b6">Phenotype</span> '

                        # Use HTML entity for arrow to avoid encoding issues
                        f.write(f"""
                            <div class="variant-info">
                                <p>
                                    <strong>{variant['rsid']}</strong> 
                                    <span class="badge {badge_class}">{importance_level}</span>
                                    <span class="tooltip">[?]
                                        <span class="tooltiptext">
                                            Importance score components:<br>
                                            {variant['explanation'].replace('; ', '<br>')}
                                        </span>
                                    </span>
                                    <br>
                                    <strong>Importance Score:</strong> {variant['importance']:.2f}
                                    <div class="importance-bar" style="width: {bar_width}%;"></div>
                                    <br>
                                    <strong>Genotype:</strong> {variant['ref']} to {variant['alt']} - {genotype_desc}
                                    {impact_html}
                                    <br><br>
                                    <strong>Evidence Types:</strong> {evidence_badges}
                                </p>
                            </div>
                        """)
                else:
                    f.write("<p>No significant variants found for this gene.</p>")

                # Find match result for this gene
                match_result = next((r for r in self.match_data if r['gene'] == gene), None)

                # Add uncallable haplotypes
                if match_result and match_result.get('uncallable_haplotypes'):
                    f.write("<h3>Uncallable Haplotypes:</h3><ul>")
                    for haplotype in match_result['uncallable_haplotypes']:
                        f.write(f"<li>{haplotype}</li>")
                    f.write("</ul>")

                # Add missing positions
                if match_result and match_result.get('missing_positions'):
                    f.write("<h3>Missing Positions:</h3><ul>")
                    for pos in match_result['missing_positions']:
                        rsid = pos.get('rsid', 'unknown')
                        position = pos.get('position', 'unknown')
                        f.write(f"<li>{rsid} at position {position}</li>")
                    f.write("</ul>")

                # Add explanation section
                f.write("<h3>Explanation:</h3>")
                explanation_text = summary['summary'].split("Explanation:\n")[1] if "Explanation:\n" in summary[
                    'summary'] else ""
                f.write(f"""<div class="explanation"><p>{explanation_text}</p></div>""")

                # Add counterfactual analysis
                gene_counterfactuals = counterfactuals[counterfactuals['gene'] == gene]
                if not gene_counterfactuals.empty:
                    f.write("<h3>What-If Analysis:</h3>")
                    f.write("<p>This section explores how different genotypes might affect the phenotype:</p>")

                    # Get only the top 3 counterfactuals by confidence
                    top_counterfactuals = gene_counterfactuals.sort_values(['confidence', 'alternative_genotype'],
                                                                           key=lambda x: x.map(
                                                                               {'high': 2, 'medium': 1, 'low': 0})
                                                                           if x.name == 'confidence' else x)

                    for _, cf in top_counterfactuals.head(3).iterrows():
                        conf_color = "#dc3545" if cf['confidence'] == "high" else "#fd7e14" if cf[
                                                                                                   'confidence'] == "medium" else "#6c757d"
                        f.write(f"""
                            <div class="counterfactual">
                                <p><strong style="color:{conf_color};">{cf['confidence'].title()} confidence:</strong> {cf['explanation']}</p>
                            </div>
                        """)

                # Add visualization if it exists
                if len(gene_variants) > 1:
                    if os.path.exists(os.path.join(self.output_dir, f"{gene}_importance.png")):
                        f.write(f"""
                            <h3>Variant Importance Visualization:</h3>
                            <img src="{gene}_importance.png" alt="Variant importance for {gene}" style="max-width:100%;">
                        """)

                    if os.path.exists(os.path.join(self.output_dir, f"{gene}_importance_composition.png")):
                        f.write(f"""
                            <h3>Importance Score Composition:</h3>
                            <p>This chart shows how different evidence types contribute to each variant's importance score:</p>
                            <img src="{gene}_importance_composition.png" alt="Importance composition for {gene}" style="max-width:100%;">
                        """)

                f.write("</div>")  # Close gene-section

            # Add overall importance plot if it exists
            if os.path.exists(os.path.join(self.output_dir, "overall_variant_importance.png")):
                f.write(f"""
                    <h2>Overall Variant Importance</h2>
                    <img src="overall_variant_importance.png" alt="Overall variant importance" style="max-width:100%;">
                """)

            # Add importance composition plot if it exists
            if os.path.exists(os.path.join(self.output_dir, "importance_composition.png")):
                f.write(f"""
                    <h2>Importance Score Composition</h2>
                    <p>This chart shows how different types of evidence from PharmCAT's behavior contribute to each variant's importance score:</p>
                    <img src="importance_composition.png" alt="Importance score composition" style="max-width:100%;">
                """)

            # Add importance heatmap if it exists
            if os.path.exists(os.path.join(self.output_dir, "importance_heatmap.png")):
                f.write(f"""
                    <h2>Variant Importance Heatmap</h2>
                    <p>This heatmap shows the importance of each variant across different genes:</p>
                    <img src="importance_heatmap.png" alt="Importance heatmap" style="max-width:100%;">
                """)

            f.write("""
                <hr>
                <footer>
                    <p><em>This report was generated using PharmCAT XAI Explainer to analyze pharmacogenomic predictions.</em></p>
                </footer>
            </body>
            </html>
            """)

    def run(self):
        try:
            print("Starting XAI analysis...")
            print(f"Parsing VCF file: {self.vcf_file}")
            self.parse_vcf()

            print(f"Loading match data from: {self.match_json_file}")
            self.load_match_data()

            print(f"Loading phenotype data from: {self.phenotype_json_file}")
            self.load_phenotype_data()

            print("Calculating variant importance...")
            self.calculate_variant_importance()

            print("Generating gene summaries...")
            self.generate_summaries()

            print("Creating visualizations...")
            self.visualize_importance()

            print("Generating HTML report...")
            self.generate_html_report()

            print(f"XAI analysis complete. Results saved to {self.output_dir}/")
            return {
                'vcf_data': self.vcf_data,
                'match_data': self.match_data,
                'phenotype_data': self.phenotype_data,
                'variant_importance': self.variant_importance,
                'gene_summaries': self.gene_summaries
            }
        except Exception as e:
            print(f"Error during XAI analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'vcf_data': self.vcf_data,
                'match_data': self.match_data,
                'phenotype_data': self.phenotype_data,
                'variant_importance': self.variant_importance,
                'gene_summaries': self.gene_summaries
            }


if __name__ == "__main__":
    # Paths to input files
    vcf_file = "Preprocessed/HG00436_freebayes.preprocessed.vcf"
    match_json_file = "Preprocessed/HG00436_freebayes.preprocessed.match.json"
    phenotype_json_file = "Preprocessed/HG00436_freebayes.preprocessed.phenotype.json"

    # Create and run the explainer
    explainer = PharmcatExplainer(vcf_file, match_json_file, phenotype_json_file)
    results = explainer.run()
