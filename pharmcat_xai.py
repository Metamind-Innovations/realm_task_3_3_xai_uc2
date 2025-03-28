import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class PharmcatExplainer:
    def __init__(self, vcf_file=None, match_json_file=None, phenotype_json_file=None, output_dir='xai_results'):
        self.vcf_file = vcf_file
        self.match_json_file = match_json_file
        self.phenotype_json_file = phenotype_json_file
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        self.vcf_data = None
        self.match_data = None
        self.phenotype_data = None
        self.variant_importance = None
        self.gene_summaries = None

    def parse_vcf(self):
        if not self.vcf_file:
            print("No VCF file specified")
            return pd.DataFrame()

        variants = []
        with open(self.vcf_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue

                fields = line.strip().split('\t')
                if len(fields) < 10:
                    continue

                try:
                    chrom = fields[0]
                    pos = int(fields[1])
                    rsid = fields[2]
                    ref = fields[3]
                    alt = fields[4]
                    info = fields[7]

                    gene = None
                    if 'PX=' in info:
                        for item in info.split(';'):
                            if item.startswith('PX='):
                                gene = item.split('=')[1]
                                break

                    genotype = "0/0"
                    if ":" in fields[9]:
                        genotype = fields[9].split(':')[0]

                    variants.append({
                        'chrom': chrom,
                        'pos': pos,
                        'rsid': rsid if rsid != '.' else f"pos_{pos}",
                        'ref': ref,
                        'alt': alt.split(',')[0],
                        'gene': gene,
                        'genotype': genotype
                    })
                except Exception as e:
                    print(f"Error processing VCF line: {str(e)}")
                    continue

        self.vcf_data = pd.DataFrame(variants)
        return self.vcf_data

    def load_match_data(self):
        if not self.match_json_file:
            print("No match JSON file specified")
            return []

        try:
            with open(self.match_json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading match data: {str(e)}")
            self.match_data = []
            return []

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
                    'rsid': variant.get('rsid'),
                    'call': variant.get('vcfCall'),
                    'phased': variant.get('phased', False)
                })

            # Extract diplotype information
            diplotypes = []
            for diplotype in result.get('diplotypes', []):
                if not isinstance(diplotype, dict):
                    continue

                diplotype_info = {
                    'name': diplotype.get('name'),
                    'score': diplotype.get('score'),
                    'allele1': None,
                    'allele2': None
                }

                haplotype1 = diplotype.get('haplotype1', {})
                haplotype2 = diplotype.get('haplotype2', {})

                if haplotype1:
                    diplotype_info['allele1'] = {
                        'name': haplotype1.get('name'),
                        'sequences': haplotype1.get('sequences', [])
                    }

                if haplotype2:
                    diplotype_info['allele2'] = {
                        'name': haplotype2.get('name'),
                        'sequences': haplotype2.get('sequences', [])
                    }

                diplotypes.append(diplotype_info)

            # Extract haplotype information
            haplotypes = []
            for haplotype in result.get('haplotypes', []):
                if not isinstance(haplotype, dict):
                    continue

                haplotype_info = {
                    'name': haplotype.get('name'),
                    'sequences': haplotype.get('sequences', []),
                    'reference': haplotype.get('haplotype', {}).get('reference', False)
                }

                haplotypes.append(haplotype_info)

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

            uncallable_haplotypes = result.get('uncallableHaplotypes', [])

            results.append({
                'gene': gene,
                'variants': variants,
                'diplotypes': diplotypes,
                'haplotypes': haplotypes,
                'missing_positions': missing_positions,
                'uncallable_haplotypes': uncallable_haplotypes,
                'phased': result.get('phased', False),
                'match_data_phased': match_data.get('phased', False),
                'match_data_homozygous': match_data.get('homozygous', False),
                'match_data_effectively_phased': match_data.get('effectivelyPhased', False)
            })

        self.match_data = results
        return results

    def load_match_data_from_string(self, json_string):
        try:
            data = json.loads(json_string)
        except Exception as e:
            print(f"Error loading match data from string: {str(e)}")
            self.match_data = []
            return []

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
                    'rsid': variant.get('rsid'),
                    'call': variant.get('vcfCall'),
                    'phased': variant.get('phased', False)
                })

            # Extract diplotype information
            diplotypes = []
            for diplotype in result.get('diplotypes', []):
                if not isinstance(diplotype, dict):
                    continue

                diplotype_info = {
                    'name': diplotype.get('name'),
                    'score': diplotype.get('score'),
                    'allele1': None,
                    'allele2': None
                }

                haplotype1 = diplotype.get('haplotype1', {})
                haplotype2 = diplotype.get('haplotype2', {})

                if haplotype1:
                    diplotype_info['allele1'] = {
                        'name': haplotype1.get('name'),
                        'sequences': haplotype1.get('sequences', [])
                    }

                if haplotype2:
                    diplotype_info['allele2'] = {
                        'name': haplotype2.get('name'),
                        'sequences': haplotype2.get('sequences', [])
                    }

                diplotypes.append(diplotype_info)

            # Extract haplotype information
            haplotypes = []
            for haplotype in result.get('haplotypes', []):
                if not isinstance(haplotype, dict):
                    continue

                haplotype_info = {
                    'name': haplotype.get('name'),
                    'sequences': haplotype.get('sequences', []),
                    'reference': haplotype.get('haplotype', {}).get('reference', False)
                }

                haplotypes.append(haplotype_info)

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

            uncallable_haplotypes = result.get('uncallableHaplotypes', [])

            results.append({
                'gene': gene,
                'variants': variants,
                'diplotypes': diplotypes,
                'haplotypes': haplotypes,
                'missing_positions': missing_positions,
                'uncallable_haplotypes': uncallable_haplotypes,
                'phased': result.get('phased', False),
                'match_data_phased': match_data.get('phased', False),
                'match_data_homozygous': match_data.get('homozygous', False),
                'match_data_effectively_phased': match_data.get('effectivelyPhased', False)
            })

        self.match_data = results
        return results

    def load_phenotype_data(self):
        if not self.phenotype_json_file:
            print("No phenotype JSON file specified")
            return []

        try:
            with open(self.phenotype_json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading phenotype data: {str(e)}")
            self.phenotype_data = []
            return []

        gene_reports = []
        for source, genes in data.get('geneReports', {}).items():
            for gene_symbol, gene_info in genes.items():
                diplotypes = []

                diplotype_sources = []
                if 'recommendationDiplotypes' in gene_info and gene_info['recommendationDiplotypes']:
                    diplotype_sources.extend(gene_info['recommendationDiplotypes'])
                elif 'sourceDiplotypes' in gene_info and gene_info['sourceDiplotypes']:
                    diplotype_sources.extend(gene_info['sourceDiplotypes'])

                for diplotype in diplotype_sources:
                    if not isinstance(diplotype, dict):
                        continue

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

            match_result = next((r for r in self.match_data if r['gene'] == gene), None)
            if not match_result:
                continue

            gene_report = next((r for r in self.phenotype_data if r['gene_symbol'] == gene), None)

            importance = 0
            explanation = []

            # Base evidence
            importance += 1
            explanation.append("Present in VCF file")

            # Match evidence
            in_match_variants = False
            for match_variant in match_result.get('variants', []):
                if match_variant.get('rsid') == variant['rsid']:
                    importance += 2
                    explanation.append("Used in haplotype matching")
                    in_match_variants = True
                    break

            # Diplotype evidence
            in_diplotype = False
            for diplotype in match_result.get('diplotypes', []):
                allele1 = diplotype.get('allele1', {})
                allele2 = diplotype.get('allele2', {})

                for sequence in allele1.get('sequences', []) + allele2.get('sequences', []):
                    if str(variant['pos']) in sequence:
                        importance += 1.5
                        explanation.append(f"Part of diplotype {diplotype.get('name')}")
                        in_diplotype = True
                        break

                if in_diplotype:
                    break

            # Uncallable haplotype evidence
            in_uncallable = False
            for haplotype in match_result.get('uncallable_haplotypes', []):
                if variant['rsid'] in haplotype:
                    importance += 2
                    explanation.append(f"Prevents calling {haplotype}")
                    in_uncallable = True
                    break

            # Missing position evidence
            in_missing_positions = False
            for missing in match_result.get('missing_positions', []):
                if missing.get('rsid') == variant['rsid']:
                    importance += 1
                    explanation.append("Missing data affects phenotype predictions")
                    in_missing_positions = True
                    break

            # Report evidence
            in_gene_report = False
            if gene_report:
                for report_variant in gene_report.get('variants', []):
                    if report_variant.get('rsid') == variant['rsid']:
                        importance += 2
                        explanation.append("Referenced in final phenotype report")
                        in_gene_report = True
                        break

            # Genotype evidence
            if variant['genotype'] == '0/1':
                importance += 1
                explanation.append("Heterozygous variant (one altered copy)")
            elif variant['genotype'] == '1/1':
                importance += 2
                explanation.append("Homozygous alternate (two altered copies)")

            # Phasing evidence
            is_phased = False
            for match_variant in match_result.get('variants', []):
                if match_variant.get('rsid') == variant['rsid'] and match_variant.get('phased', False):
                    importance += 1
                    explanation.append("Phased variant (precise haplotype determination)")
                    is_phased = True
                    break

            # Phenotype impact evidence
            phenotypes = []
            if gene_report:
                for diplotype in gene_report.get('diplotypes', []):
                    phenotypes.extend(diplotype.get('phenotypes', []))

            has_distinct_phenotype = False
            if phenotypes and "Unknown" not in phenotypes and "No Result" not in phenotypes:
                if in_match_variants or in_gene_report:
                    importance += 1
                    explanation.append("Contributes to a definitive phenotype prediction")
                    has_distinct_phenotype = True

            # Position factor for uniqueness
            pos_factor = 0.01 * (hash(str(variant['pos'])) % 100) / 100
            importance += pos_factor

            if importance > 0:
                importance_scores.append({
                    'gene': gene,
                    'rsid': variant['rsid'],
                    'pos': variant['pos'],
                    'ref': variant['ref'],
                    'alt': variant['alt'],
                    'genotype': variant['genotype'],
                    'importance': round(importance, 2),
                    'explanation': "; ".join(explanation),
                    'phenotypes': list(set(phenotypes)) if phenotypes else ["Unknown"],
                    'in_match': in_match_variants,
                    'in_diplotype': in_diplotype,
                    'in_report': in_gene_report,
                    'uncallable': in_uncallable,
                    'missing': in_missing_positions,
                    'phased': is_phased,
                    'has_phenotype': has_distinct_phenotype
                })

        self.variant_importance = pd.DataFrame(importance_scores)
        return self.variant_importance

    def generate_summaries(self):
        if self.match_data is None:
            print("No match data available")
            return []

        if self.variant_importance is None and self.vcf_file:
            self.calculate_variant_importance()

        summaries = []
        processed_genes = set()

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
            "IFNL3": "involved in immune response, affects hepatitis C treatment response",
            "ABCG2": "transporter protein involved in drug disposition and multidrug resistance",
            "CACNA1S": "calcium channel subunit, mutations associated with malignant hyperthermia susceptibility"
        }

        # Process genes with variants if we have variant importance data
        if self.variant_importance is not None and not self.variant_importance.empty:
            for gene, group in self.variant_importance.groupby('gene'):
                processed_genes.add(gene)

                gene_report = next((r for r in self.phenotype_data if r['gene_symbol'] == gene), None)
                match_result = next((r for r in self.match_data if r['gene'] == gene), None)

                phenotypes = []
                if gene_report:
                    for diplotype in gene_report.get('diplotypes', []):
                        phenotypes.extend(diplotype.get('phenotypes', []))

                diplotypes = []
                if match_result:
                    for diplotype in match_result.get('diplotypes', []):
                        diplotypes.append(diplotype.get('name', 'Unknown'))

                phenotype_str = ", ".join(set(phenotypes)) if phenotypes else "Unknown"
                diplotype_str = ", ".join(diplotypes) if diplotypes else "Unknown"

                gene_function = gene_function_map.get(gene, "affects drug metabolism or response")

                gene_summary = {
                    'gene': gene,
                    'phenotypes': list(set(phenotypes)),
                    'diplotypes': diplotypes,
                    'summary': f"Gene: {gene}\n"
                               f"Function: {gene_function}\n"
                               f"Diplotype(s): {diplotype_str}\n"
                               f"Phenotype(s): {phenotype_str}\n"
                               f"Number of important variants: {len(group)}\n"
                }

                if match_result:
                    phasing_status = "Phased" if match_result.get('phased', False) else "Unphased"
                    homozygous = "Homozygous" if match_result.get('match_data_homozygous', False) else "Heterozygous"
                    gene_summary['summary'] += f"Phasing status: {phasing_status}\n"
                    gene_summary['summary'] += f"Zygosity: {homozygous}\n"

                clinical_significance = self.get_clinical_significance(gene, phenotypes)
                if clinical_significance:
                    gene_summary['summary'] += f"Clinical Significance: {clinical_significance}\n"

                if not group.empty:
                    variant_details = []
                    for _, variant in group.sort_values('importance', ascending=False).iterrows():
                        genotype_desc = "homozygous reference (0/0)" if variant['genotype'] == '0/0' else \
                            "heterozygous (0/1)" if variant['genotype'] == '0/1' else \
                                "homozygous alternate (1/1)" if variant['genotype'] == '1/1' else \
                                    variant['genotype']

                        variant_impact = self.get_variant_impact(gene, variant['rsid'], variant['genotype'])
                        impact_text = f"\n  Clinical Impact: {variant_impact}" if variant_impact else ""

                        phased_text = " (phased)" if variant.get('phased', False) else ""

                        variant_detail = f"- {variant['rsid']} ({variant['ref']} to {variant['alt']}): {genotype_desc}{phased_text}\n" \
                                         f"  Importance Score: {variant['importance']}\n" \
                                         f"  Reason: {variant['explanation']}{impact_text}"
                        variant_details.append(variant_detail)

                    gene_summary['summary'] += "\nImportant variants:\n" + "\n\n".join(variant_details)

                if match_result and match_result.get('haplotypes'):
                    haplotype_info = []
                    for haplotype in match_result['haplotypes']:
                        reference_status = " (reference)" if haplotype.get('reference', False) else ""
                        haplotype_info.append(f"{haplotype.get('name', 'Unknown')}{reference_status}")

                    if haplotype_info:
                        gene_summary['summary'] += "\n\nIdentified haplotypes:\n- " + "\n- ".join(haplotype_info)

                if match_result and match_result.get('uncallable_haplotypes'):
                    gene_summary['summary'] += "\n\nUncallable haplotypes:\n- " + "\n- ".join(
                        match_result['uncallable_haplotypes'])

                if match_result and match_result.get('missing_positions'):
                    missing_positions = []
                    for pos in match_result['missing_positions']:
                        rsid = pos.get('rsid', 'unknown')
                        position = pos.get('position', 'unknown')
                        missing_positions.append(f"{rsid} at position {position}")

                    if missing_positions:
                        gene_summary['summary'] += "\n\nMissing positions:\n- " + "\n- ".join(missing_positions)

                gene_summary['summary'] += "\n\nExplanation:\n"
                if group.empty and match_result and not match_result.get(
                        'uncallable_haplotypes') and not match_result.get(
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

        # Process remaining genes from match data
        for match_result in self.match_data:
            gene = match_result['gene']
            if gene in processed_genes:
                continue

            gene_report = next((r for r in self.phenotype_data if r['gene_symbol'] == gene), None)

            phenotypes = []
            if gene_report:
                for diplotype in gene_report.get('diplotypes', []):
                    phenotypes.extend(diplotype.get('phenotypes', []))

            diplotypes = []
            for diplotype in match_result.get('diplotypes', []):
                diplotypes.append(diplotype.get('name', 'Unknown'))

            phenotype_str = ", ".join(set(phenotypes)) if phenotypes else "Unknown"
            diplotype_str = ", ".join(diplotypes) if diplotypes else "Unknown"

            gene_function = gene_function_map.get(gene, "affects drug metabolism or response")

            gene_summary = {
                'gene': gene,
                'phenotypes': list(set(phenotypes)),
                'diplotypes': diplotypes,
                'summary': f"Gene: {gene}\n"
                           f"Function: {gene_function}\n"
                           f"Diplotype(s): {diplotype_str}\n"
                           f"Phenotype(s): {phenotype_str}\n"
                           f"Number of important variants: 0\n"
            }

            phasing_status = "Phased" if match_result.get('phased', False) else "Unphased"
            homozygous = "Homozygous" if match_result.get('match_data_homozygous', False) else "Heterozygous"
            gene_summary['summary'] += f"Phasing status: {phasing_status}\n"
            gene_summary['summary'] += f"Zygosity: {homozygous}\n"

            clinical_significance = self.get_clinical_significance(gene, phenotypes)
            if clinical_significance:
                gene_summary['summary'] += f"Clinical Significance: {clinical_significance}\n"

            if match_result.get('haplotypes'):
                haplotype_info = []
                for haplotype in match_result['haplotypes']:
                    reference_status = " (reference)" if haplotype.get('reference', False) else ""
                    haplotype_info.append(f"{haplotype.get('name', 'Unknown')}{reference_status}")

                if haplotype_info:
                    gene_summary['summary'] += "\n\nIdentified haplotypes:\n- " + "\n- ".join(haplotype_info)

            if match_result.get('uncallable_haplotypes'):
                gene_summary['summary'] += "\n\nUncallable haplotypes:\n- " + "\n- ".join(
                    match_result['uncallable_haplotypes'])

            if match_result.get('missing_positions'):
                missing_positions = []
                for pos in match_result['missing_positions']:
                    rsid = pos.get('rsid', 'unknown')
                    position = pos.get('position', 'unknown')
                    missing_positions.append(f"{rsid} at position {position}")

                if missing_positions:
                    gene_summary['summary'] += "\n\nMissing positions:\n- " + "\n- ".join(missing_positions)

            if match_result.get('variants'):
                gene_summary['summary'] += "\n\nVariants considered:\n"
                for variant in match_result['variants']:
                    rsid = variant.get('rsid', 'unknown')
                    call = variant.get('call', 'unknown')
                    phased_status = " (phased)" if variant.get('phased', False) else ""
                    gene_summary['summary'] += f"- {rsid}: {call}{phased_status}\n"

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
        if not phenotypes or phenotypes == ["Unknown"] or phenotypes == ["No Result"]:
            return ""

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

        elif gene == "ABCG2":
            if "rs2231142 reference (G)/rs2231142 reference (G)" in str(phenotypes):
                return "Normal ABCG2 function, standard dosing for affected drugs like rosuvastatin."

        elif gene == "CFTR":
            if "ivacaftor non-responsive CFTR sequence" in str(phenotypes):
                return "No response expected to CFTR modulator therapy with ivacaftor."

        if any(p for p in phenotypes if "Poor" in p and "Metabolizer" in p):
            return "Poor metabolizer status may require dose adjustments for affected medications."
        elif any(p for p in phenotypes if "Intermediate" in p and "Metabolizer" in p):
            return "Intermediate metabolizer status may require monitoring for efficacy and side effects."
        elif any(p for p in phenotypes if "Ultrarapid" in p and "Metabolizer" in p):
            return "Ultrarapid metabolizer status may require dose increases or alternative medications."

        return ""

    def get_variant_impact(self, gene, rsid, genotype):
        impact_map = {
            ("CYP2D6", "rs3745274", "0/1"): "May reduce metabolism of CYP2D6 substrates",
            ("CYP2D6", "rs3745274", "1/1"): "Significantly reduces metabolism of CYP2D6 substrates",
            ("CYP2D6", "rs2279343", "0/1"): "May alter CYP2D6 activity",
            ("CYP2D6", "rs2279343", "1/1"): "Associated with altered metabolism of CYP2D6 substrates",
            ("CYP2C19", "rs4244285", "0/1"): "Reduced function, may impair clopidogrel activation",
            ("CYP2C19", "rs4244285", "1/1"): "Loss of function, significantly impairs clopidogrel activation",
            ("CYP2C19", "rs12248560", "0/1"): "Enhanced function, may increase clopidogrel response",
            ("CYP2C19", "rs12248560", "1/1"): "Significantly enhanced function, increases clopidogrel response",
            ("SLCO1B1", "rs4149056", "0/1"): "Reduced transport, moderate increase in statin exposure",
            ("SLCO1B1", "rs4149056", "1/1"): "Significantly reduced transport, higher risk of statin myopathy",
            ("SLCO1B1", "rs2306283", "0/1"): "Possible increased transporter activity",
            ("VKORC1", "rs9923231", "0/1"): "Intermediate warfarin sensitivity",
            ("VKORC1", "rs9923231", "1/1"): "High warfarin sensitivity, lower dose requirements",
            ("CYP3A5", "rs776746", "0/1"): "Intermediate metabolizer, affects tacrolimus exposure",
            ("CYP3A5", "rs776746", "1/1"): "Non-expresser (*3/*3), higher tacrolimus exposure",
            ("UGT1A1", "rs887829", "0/1"): "Reduced enzyme activity, moderate risk of toxicity",
            ("UGT1A1", "rs887829", "1/1"): "Significantly reduced activity, higher risk of toxicity",
            ("UGT1A1", "rs3064744", "0/1"): "Reduced enzyme activity, may affect irinotecan metabolism",
            ("UGT1A1", "rs3064744", "1/1"): "Significant reduction in enzyme activity, higher toxicity risk",
            ("CYP4F2", "rs2108622", "0/1"): "Moderately decreased vitamin K metabolism, affects warfarin dosing",
            ("CYP4F2", "rs2108622", "1/1"): "Significantly decreased vitamin K metabolism, higher warfarin dose needs"
        }

        return impact_map.get((gene, rsid, genotype), "")

    def visualize_importance(self):
        if self.variant_importance is None or self.variant_importance.empty:
            print("No variant importance data to visualize.")
            return

        colors = sns.color_palette("husl", len(self.variant_importance['gene'].unique()))
        gene_color_map = dict(zip(self.variant_importance['gene'].unique(), colors))

        plt.figure(figsize=(14, 8))
        sorted_data = self.variant_importance.sort_values('importance', ascending=False)
        bars = sns.barplot(data=sorted_data, x='rsid', y='importance', hue='gene')
        plt.title("Data-Driven Variant Importance for Phenotype Predictions", fontsize=16)
        plt.xlabel("Variant (rsID)", fontsize=12)
        plt.ylabel("Importance Score", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="Gene", fontsize=10)

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

        # Create gene-specific plots
        for gene, group in self.variant_importance.groupby('gene'):
            if len(group) > 1:
                plt.figure(figsize=(10, 6))
                group_sorted = group.sort_values('importance', ascending=False)
                gene_color = gene_color_map[gene]
                bars = sns.barplot(data=group_sorted, x='rsid', y='importance', color=gene_color)
                plt.title(f"Variant Importance for {gene}", fontsize=16)
                plt.xlabel("Variant (rsID)", fontsize=12)
                plt.ylabel("Importance Score", fontsize=12)
                plt.xticks(rotation=45, ha='right')

                for i, bar in enumerate(bars.patches):
                    bars.text(
                        bar.get_x() + bar.get_width() / 2.,
                        bar.get_height() + 0.05,
                        f"{bar.get_height():.2f}",
                        ha='center', fontsize=10
                    )

                for i, (_, row) in enumerate(group_sorted.iterrows()):
                    plt.text(
                        i, -0.2,
                        f"{row['genotype']}",
                        ha='center', fontsize=9
                    )

                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f"{gene}_importance.png"), dpi=150)
                plt.close()

    def generate_match_only_report(self):
        if self.match_data is None:
            print("No match data available. Load match data first.")
            return None

        output_path = os.path.join(self.output_dir, "match_only_report.html")

        if self.gene_summaries is None:
            self.generate_summaries()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("""
            <html>
            <head>
                <title>PharmCAT Match Data Report</title>
                <meta charset="UTF-8">
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px 40px; line-height: 1.6; }
                    h1, h2, h3 { color: #2c3e50; }
                    .gene-section { border: 1px solid #ddd; padding: 20px; margin-bottom: 30px; border-radius: 5px; }
                    .explanation { background-color: #f8f9fa; padding: 15px; border-left: 4px solid #007bff; }
                    .missing { color: #dc3545; }
                    .present { color: #28a745; }
                    table { border-collapse: collapse; width: 100%; margin: 15px 0; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    tr:nth-child(even) { background-color: #f9f9f9; }
                </style>
            </head>
            <body>
                <h1>PharmCAT Match Data Report</h1>
                <p>This report explains pharmacogenomic diplotypes determined by PharmCAT based on the match data.</p>
            """)

            # Summary table
            f.write("""
                <h2>Gene Summary</h2>
                <table><tr>
                        <th>Gene</th>
                        <th>Diplotype</th>
                        <th>Phasing Status</th>
                        <th>Zygosity</th>
                        <th>Variants Count</th>
                        <th>Missing Data</th>
                    </tr>
            """)

            for summary in self.gene_summaries:
                gene = summary['gene']
                match_result = next((r for r in self.match_data if r['gene'] == gene), None)

                if not match_result:
                    continue

                diplotype = summary['diplotypes'][0] if summary['diplotypes'] else "Unknown"
                phased = "Phased" if match_result.get('phased', False) else "Unphased"
                homozygous = "Homozygous" if match_result.get('match_data_homozygous', False) else "Heterozygous"
                variants_count = len(match_result.get('variants', []))
                has_missing_data = bool(
                    match_result.get('uncallable_haplotypes') or match_result.get('missing_positions'))

                f.write(f"""
                    <tr>
                        <td>{gene}</td>
                        <td>{diplotype}</td>
                        <td>{phased}</td>
                        <td>{homozygous}</td>
                        <td>{variants_count}</td>
                        <td class="{'missing' if has_missing_data else ''}">{('Yes' if has_missing_data else 'No')}</td>
                    </tr>
                """)

            f.write("</table>")

            # Gene details
            for summary in self.gene_summaries:
                gene = summary['gene']
                match_result = next((r for r in self.match_data if r['gene'] == gene), None)

                if not match_result:
                    continue

                diplotypes = summary['diplotypes']
                diplotype_str = ", ".join(diplotypes) if diplotypes else "Unknown"
                phenotype_str = ", ".join(summary['phenotypes']) if summary['phenotypes'] else "Unknown"

                f.write(f"""
                    <div class="gene-section">
                        <h2>Gene: {gene}</h2>
                        <p><strong>Diplotype(s):</strong> {diplotype_str}</p>
                        <p><strong>Phenotype(s):</strong> {phenotype_str}</p>
                        <p><strong>Phasing Status:</strong> {"Phased" if match_result.get('phased', False) else "Unphased"}</p>
                        <p><strong>Zygosity:</strong> {"Homozygous" if match_result.get('match_data_homozygous', False) else "Heterozygous"}</p>
                """)

                # Haplotypes
                if match_result.get('haplotypes'):
                    f.write("<h3>Identified Haplotypes:</h3><ul>")
                    for haplotype in match_result['haplotypes']:
                        reference_status = " (reference)" if haplotype.get('reference', False) else ""
                        f.write(f"<li>{haplotype.get('name', 'Unknown')}{reference_status}</li>")
                    f.write("</ul>")

                # Variants
                if match_result.get('variants'):
                    f.write("<h3>Variants:</h3><table>")
                    f.write("<tr><th>RSID</th><th>Position</th><th>Call</th><th>Phased</th></tr>")

                    for variant in match_result['variants']:
                        rsid = variant.get('rsid', 'Unknown')
                        position = variant.get('position', 'Unknown')
                        call = variant.get('call', 'Unknown')
                        phased = "Yes" if variant.get('phased', False) else "No"

                        f.write(f"<tr><td>{rsid}</td><td>{position}</td><td>{call}</td><td>{phased}</td></tr>")

                    f.write("</table>")

                # Diplotypes in detail
                if match_result.get('diplotypes'):
                    f.write("<h3>Diplotype Details:</h3>")

                    for idx, diplotype in enumerate(match_result['diplotypes']):
                        name = diplotype.get('name', 'Unknown')
                        score = diplotype.get('score', 'Unknown')

                        f.write(f"<div class='diplotype-detail'>")
                        f.write(f"<p><strong>Name:</strong> {name}</p>")
                        f.write(f"<p><strong>Score:</strong> {score}</p>")

                        if diplotype.get('allele1') or diplotype.get('allele2'):
                            f.write("<p><strong>Alleles:</strong></p><ul>")

                            if diplotype.get('allele1'):
                                f.write(f"<li>Allele 1: {diplotype['allele1'].get('name', 'Unknown')}</li>")

                            if diplotype.get('allele2'):
                                f.write(f"<li>Allele 2: {diplotype['allele2'].get('name', 'Unknown')}</li>")

                            f.write("</ul>")

                        f.write("</div>")

                # Uncallable haplotypes
                if match_result.get('uncallable_haplotypes'):
                    f.write("<h3>Uncallable Haplotypes:</h3><ul>")
                    for haplotype in match_result['uncallable_haplotypes']:
                        f.write(f"<li>{haplotype}</li>")
                    f.write("</ul>")

                # Missing positions
                if match_result.get('missing_positions'):
                    f.write("<h3>Missing Positions:</h3><ul>")
                    for pos in match_result['missing_positions']:
                        rsid = pos.get('rsid', 'Unknown')
                        position = pos.get('position', 'Unknown')
                        f.write(f"<li>{rsid} at position {position}</li>")
                    f.write("</ul>")

                # Clinical significance
                clinical_significance = self.get_clinical_significance(gene, summary['phenotypes'])
                if clinical_significance:
                    f.write(f"""
                        <div class="explanation">
                            <h3>Clinical Significance:</h3>
                            <p>{clinical_significance}</p>
                        </div>
                    """)

                f.write("</div>")  # Close gene-section

            f.write("""
                <hr>
                <footer>
                    <p><em>This report was generated using PharmCAT XAI Explainer to analyze pharmacogenomic predictions.</em></p>
                </footer>
            </body>
            </html>
            """)

        return output_path

    def create_counterfactual_analysis(self):
        if self.variant_importance is None and self.vcf_file:
            self.calculate_variant_importance()
        elif self.variant_importance is None:
            print("No variant importance data available. Cannot create counterfactual analysis.")
            return pd.DataFrame()

        counterfactuals = []

        for _, variant in self.variant_importance.iterrows():
            gene = variant['gene']
            genotype = variant['genotype']

            alt_genotypes = []
            if genotype == '0/0':
                alt_genotypes = ['0/1', '1/1']
            elif genotype == '0/1':
                alt_genotypes = ['0/0', '1/1']
            elif genotype == '1/1':
                alt_genotypes = ['0/0', '0/1']
            elif '|' in genotype:  # For phased data
                ref_genotype = genotype.replace('|', '/')
                if ref_genotype == '0/0':
                    alt_genotypes = ['0/1', '1/1']
                elif ref_genotype == '0/1':
                    alt_genotypes = ['0/0', '1/1']
                elif ref_genotype == '1/1':
                    alt_genotypes = ['0/0', '0/1']

            for alt_genotype in alt_genotypes:
                impact = "unknown"
                confidence = "low"
                detail = ""

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
                elif gene == "CFTR":
                    if genotype in ['0/0', '0|0'] and alt_genotype in ['0/1', '1/1']:
                        impact = "increased risk of CFTR-related disorder"
                        confidence = "high"
                        detail = "May affect response to CFTR modulator therapy"
                elif gene == "ABCG2":
                    if genotype in ['0/0', '0|0'] and alt_genotype in ['0/1', '1/1']:
                        impact = "decreased drug transport"
                        confidence = "medium"
                        detail = "Could affect drug disposition, especially statins and uric acid transport"

                # Generic assessment if no specific rule
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
                <p>This report explains how genetic variants influence pharmacogenomic phenotype predictions made by PharmCAT.</p>
            """)

            f.write("""
                <div class="explanation">
                    <h3>How to Read Importance Scores</h3>
                    <p>Importance scores indicate how much each variant contributes to the phenotype prediction based on <strong>data-driven analysis</strong> of PharmCAT's behavior:</p>
                    <ul>
                        <li><strong>Base evidence (1.0)</strong>: Every variant found in the VCF gets this score</li>
                        <li><strong>Matching evidence (2.0)</strong>: Variant is directly used in PharmCAT's haplotype matching</li>
                        <li><strong>Diplotype evidence (1.5)</strong>: Variant contributes to specific diplotype call</li>
                        <li><strong>Uncallable haplotype evidence (2.0)</strong>: Variant prevents calling specific haplotypes</li>
                        <li><strong>Missing position evidence (1.0)</strong>: Missing data affects available haplotype calls</li>
                        <li><strong>Report evidence (2.0)</strong>: Variant is referenced in the final phenotype report</li>
                        <li><strong>Genotype evidence</strong>: Heterozygous (1.0) or Homozygous alternate (2.0)</li>
                        <li><strong>Phasing evidence (1.0)</strong>: Phased variants provide precise haplotype information</li>
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
                        <th>Diplotype(s)</th>
                        <th>Phenotype(s)</th>
                        <th>Important Variants</th>
                        <th>Phasing Status</th>
                        <th>Missing Data</th>
                    </tr>
            """)

            all_variants = self.variant_importance if self.variant_importance is not None else pd.DataFrame()

            for summary in self.gene_summaries:
                gene = summary['gene']
                phenotypes = ", ".join(summary['phenotypes']) if summary['phenotypes'] else "Unknown"
                diplotypes = ", ".join(summary['diplotypes']) if summary['diplotypes'] else "Unknown"

                match_result = next((r for r in self.match_data if r['gene'] == gene), None)
                has_missing_data = False
                if match_result and (
                        match_result.get('uncallable_haplotypes') or match_result.get('missing_positions')):
                    has_missing_data = True

                gene_variants = all_variants[all_variants['gene'] == gene] if not all_variants.empty else pd.DataFrame()
                variants_count = len(gene_variants)

                phasing_status = "Phased" if match_result and match_result.get('phased') else "Unphased"

                f.write(f"""
                    <tr>
                        <td>{gene}</td>
                        <td>{diplotypes}</td>
                        <td>{phenotypes}</td>
                        <td class="{'present' if variants_count > 0 else ''}">{variants_count}</td>
                        <td>{phasing_status}</td>
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
                        <p><strong>Diplotype(s):</strong> {', '.join(summary['diplotypes']) if summary['diplotypes'] else 'Unknown'}</p>
                        <p><strong>Phenotype(s):</strong> {', '.join(summary['phenotypes']) if summary['phenotypes'] else 'Unknown'}</p>
                """)

                match_result = next((r for r in self.match_data if r['gene'] == gene), None)
                if match_result:
                    phasing_status = "Phased" if match_result.get('phased', False) else "Unphased"
                    homozygous = "Homozygous" if match_result.get('match_data_homozygous', False) else "Heterozygous"
                    f.write(f"<p><strong>Phasing status:</strong> {phasing_status}</p>")
                    f.write(f"<p><strong>Zygosity:</strong> {homozygous}</p>")

                # Get variants for this gene
                gene_variants = all_variants[all_variants['gene'] == gene] if not all_variants.empty else pd.DataFrame()

                if not gene_variants.empty:
                    f.write("<h3>Important Variants:</h3>")

                    max_importance = gene_variants['importance'].max() if len(gene_variants) > 0 else 0

                    for _, variant in gene_variants.sort_values('importance', ascending=False).iterrows():
                        genotype_desc = "homozygous reference (0/0)" if variant['genotype'] == '0/0' else \
                            "heterozygous (0/1)" if variant['genotype'] == '0/1' else \
                                "homozygous alternate (1/1)" if variant['genotype'] == '1/1' else \
                                    variant['genotype']

                        variant_impact = self.get_variant_impact(gene, variant['rsid'], variant['genotype'])
                        impact_html = f"<br><strong>Clinical Impact:</strong> {variant_impact}" if variant_impact else ""

                        if variant['importance'] >= 7:
                            badge_class = "badge-high"
                            importance_level = "High"
                        elif variant['importance'] >= 5:
                            badge_class = "badge-medium"
                            importance_level = "Medium"
                        else:
                            badge_class = "badge-low"
                            importance_level = "Low"

                        bar_width = (variant['importance'] / max(max_importance, 8)) * 100

                        evidence_badges = ""
                        if variant.get('in_match'):
                            evidence_badges += '<span class="badge" style="background-color:#2ecc71">Match</span> '
                        if variant.get('in_diplotype'):
                            evidence_badges += '<span class="badge" style="background-color:#9b59b6">Diplotype</span> '
                        if variant.get('in_report'):
                            evidence_badges += '<span class="badge" style="background-color:#3498db">Report</span> '
                        if variant.get('uncallable'):
                            evidence_badges += '<span class="badge" style="background-color:#e74c3c">Uncallable</span> '
                        if variant.get('missing'):
                            evidence_badges += '<span class="badge" style="background-color:#f39c12">Missing</span> '
                        if variant.get('phased'):
                            evidence_badges += '<span class="badge" style="background-color:#1abc9c">Phased</span> '
                        if variant.get('has_phenotype'):
                            evidence_badges += '<span class="badge" style="background-color:#9b59b6">Phenotype</span> '

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

                # Add haplotype information
                if match_result and match_result.get('haplotypes'):
                    f.write("<h3>Identified Haplotypes:</h3><ul>")
                    for haplotype in match_result['haplotypes']:
                        reference_status = " (reference)" if haplotype.get('reference', False) else ""
                        f.write(f"<li>{haplotype.get('name', 'Unknown')}{reference_status}</li>")
                    f.write("</ul>")

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
                gene_counterfactuals = counterfactuals[
                    counterfactuals['gene'] == gene] if not counterfactuals.empty else pd.DataFrame()
                if not gene_counterfactuals.empty:
                    f.write("<h3>What-If Analysis:</h3>")
                    f.write("<p>This section explores how different genotypes might affect the phenotype:</p>")

                    conf_map = {'high': 2, 'medium': 1, 'low': 0}
                    top_counterfactuals = gene_counterfactuals.sort_values('confidence', key=lambda x: x.map(conf_map),
                                                                           ascending=False)

                    for _, cf in top_counterfactuals.head(3).iterrows():
                        conf_color = "#dc3545" if cf['confidence'] == "high" else "#fd7e14" if cf[
                                                                                                   'confidence'] == "medium" else "#6c757d"
                        f.write(f"""
                            <div class="counterfactual">
                                <p><strong style="color:{conf_color};">{cf['confidence'].title()} confidence:</strong> {cf['explanation']}</p>
                            </div>
                        """)

                # Add visualization if it exists
                if len(gene_variants) > 1 and os.path.exists(os.path.join(self.output_dir, f"{gene}_importance.png")):
                    f.write(f"""
                        <h3>Variant Importance Visualization:</h3>
                        <img src="{gene}_importance.png" alt="Variant importance for {gene}" style="max-width:100%;">
                    """)

                f.write("</div>")  # Close gene-section

            # Add overall importance plot if it exists
            if os.path.exists(os.path.join(self.output_dir, "overall_variant_importance.png")):
                f.write(f"""
                    <h2>Overall Variant Importance</h2>
                    <img src="overall_variant_importance.png" alt="Overall variant importance" style="max-width:100%;">
                """)

            f.write("""
                <hr>
                <footer>
                    <p><em>This report was generated using PharmCAT XAI Explainer to analyze pharmacogenomic predictions.</em></p>
                </footer>
            </body>
            </html>
            """)

    def run(self, json_string=None):
        try:
            print("Starting XAI analysis...")

            if json_string:
                print("Loading match data from provided JSON string")
                self.load_match_data_from_string(json_string)
            else:
                if self.vcf_file:
                    print(f"Parsing VCF file: {self.vcf_file}")
                    self.parse_vcf()

                if self.match_json_file:
                    print(f"Loading match data from: {self.match_json_file}")
                    self.load_match_data()

                if self.phenotype_json_file:
                    print(f"Loading phenotype data from: {self.phenotype_json_file}")
                    self.load_phenotype_data()

            print("Generating gene summaries...")
            self.generate_summaries()

            if self.vcf_file or json_string:
                print("Calculating variant importance...")
                self.calculate_variant_importance()

                print("Creating visualizations...")
                self.visualize_importance()

            print("Generating HTML report...")
            self.generate_html_report()

            if self.match_data:
                print("Generating match-only report...")
                self.generate_match_only_report()

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
    # Check if we have paste.txt to use directly
    if os.path.exists("paste.txt"):
        with open("paste.txt", "r") as f:
            json_string = f.read()

        # Create and run the explainer with the JSON string
        explainer = PharmcatExplainer()
        results = explainer.run(json_string=json_string)
    else:
        # Paths to input files
        vcf_file = "Preprocessed/HG00276_freebayes.preprocessed.vcf"
        match_json_file = "Preprocessed/HG00276_freebayes.preprocessed.match.json"
        phenotype_json_file = "Preprocessed/HG00276_freebayes.preprocessed.phenotype.json"

        # Create and run the explainer
        explainer = PharmcatExplainer(vcf_file, match_json_file, phenotype_json_file)
        results = explainer.run()
