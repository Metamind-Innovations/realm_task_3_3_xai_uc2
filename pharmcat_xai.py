import argparse
import glob
import json
import os
import re

import pandas as pd


class PharmcatXAI:
    def __init__(self, vcf_dir, phenotypes_file, output_file='pharmcat_xai_results.json'):
        self.vcf_dir = vcf_dir
        self.phenotypes_file = phenotypes_file
        self.output_file = output_file
        self.focus_genes = ["CYP2B6", "CYP2C9", "CYP2C19", "CYP3A5", "SLCO1B1", "TPMT", "DPYD"]

        self.phenotypes = pd.read_csv(phenotypes_file)
        self.vcf_data = {}
        self.sample_to_vcf = self._map_samples_to_vcf()
        self.variant_phenotype_mapping = self._load_variant_phenotype_knowledge()

        self.phenotype_categories = {
            'decreased': ['PM', 'LPM', 'IM', 'LIM', 'DF', 'PF', 'PDF'],
            'normal': ['NM', 'LNM', 'NF'],
            'increased': ['UM', 'LUM', 'RM', 'LRM', 'IF'],
            'indeterminate': ['INDETERMINATE']
        }

    def _map_samples_to_vcf(self):
        sample_to_vcf = {}
        vcf_files = glob.glob(os.path.join(self.vcf_dir, "*.vcf"))
        csv_files = glob.glob(os.path.join(self.vcf_dir, "*.csv"))

        for file in vcf_files + csv_files:
            filename = os.path.basename(file)
            if "_freebayes" in filename:
                sample_id = filename.split('_')[0]
                sample_to_vcf[sample_id] = file

        return sample_to_vcf

    def _load_variant_phenotype_knowledge(self):
        return {
            # CYP2B6 variants
            'rs3745274': {'gene': 'CYP2B6', 'effect': {'0/1': 'IM', '1/1': 'PM'}, 'importance': 0.9},
            'rs2279343': {'gene': 'CYP2B6', 'effect': {'0/1': 'IM', '1/1': 'RM'}, 'importance': 0.8},
            'rs8192709': {'gene': 'CYP2B6', 'effect': {'0/1': 'NM', '1/1': 'IM'}, 'importance': 0.7},

            # CYP2C19 variants
            'rs12248560': {'gene': 'CYP2C19', 'effect': {'0/1': 'RM', '1/1': 'UM'}, 'importance': 0.85},
            'rs4244285': {'gene': 'CYP2C19', 'effect': {'0/1': 'IM', '1/1': 'PM'}, 'importance': 0.9},
            'rs3758581': {'gene': 'CYP2C19', 'effect': {'0/1': 'NM', '1/1': 'NM'}, 'importance': 0.5},

            # CYP2C9 variants
            'rs1799853': {'gene': 'CYP2C9', 'effect': {'0/1': 'IM', '1/1': 'PM'}, 'importance': 0.85},
            'rs1057910': {'gene': 'CYP2C9', 'effect': {'0/1': 'IM', '1/1': 'PM'}, 'importance': 0.85},

            # CYP3A5 variants
            'rs776746': {'gene': 'CYP3A5', 'effect': {'0/1': 'IM', '1/1': 'PM'}, 'importance': 0.95},

            # SLCO1B1 variants
            'rs4149056': {'gene': 'SLCO1B1', 'effect': {'0/1': 'DF', '1/1': 'PF'}, 'importance': 0.9},
            'rs2306283': {'gene': 'SLCO1B1', 'effect': {'0/1': 'NF', '1/1': 'IF'}, 'importance': 0.8},

            # TPMT variants
            'rs144041067': {'gene': 'TPMT', 'effect': {'0/1': 'IM', '1/1': 'PM', '0/2': 'IM', '2/2': 'PM'},
                            'importance': 0.9},

            # DPYD variants
            'rs3918290': {'gene': 'DPYD', 'effect': {'0/1': 'IM', '1/1': 'PM'}, 'importance': 0.95},
            'rs55886062': {'gene': 'DPYD', 'effect': {'0/1': 'IM', '1/1': 'PM'}, 'importance': 0.9}
        }

    def load_data(self):
        for sample_id, file_path in self.sample_to_vcf.items():
            if file_path.endswith('.csv'):
                self.vcf_data[sample_id] = pd.read_csv(file_path)
            else:
                self.vcf_data[sample_id] = self._parse_vcf(file_path)
            print(f"Loaded data for sample {sample_id}")

    def _parse_vcf(self, vcf_file):
        rows = []

        with open(vcf_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue

                parts = line.strip().split('\t')
                if len(parts) < 8:
                    continue

                chrom, pos, id, ref, alt, qual, filter_val, info = parts[:8]

                gene_match = re.search(r'PX=([^;]+)', info)
                gene = gene_match.group(1) if gene_match else ""

                # Get genotype
                genotype = "0/0"
                if len(parts) > 9:
                    format_parts = parts[8].split(':')
                    sample_parts = parts[9].split(':')

                    if 'GT' in format_parts:
                        gt_index = format_parts.index('GT')
                        if gt_index < len(sample_parts):
                            genotype = sample_parts[gt_index]

                rows.append({
                    'CHROM': chrom,
                    'POS': pos,
                    'ID': id,
                    'REF': ref,
                    'ALT': alt,
                    'QUAL': qual,
                    'FILTER': filter_val,
                    'Gene': gene,
                    'Genotype': genotype
                })

        return pd.DataFrame(rows)

    def calculate_variant_importance(self):
        results = {}

        for sample_id, sample_data in self.vcf_data.items():
            results[sample_id] = {}

            for gene in self.focus_genes:
                if gene not in results[sample_id]:
                    results[sample_id][gene] = {'variants': [], 'phenotype': 'Unknown'}

                # Get phenotype for this gene
                phenotype = "Unknown"
                sample_row = self.phenotypes[self.phenotypes['Sample ID'] == sample_id]
                if not sample_row.empty and gene in sample_row.columns:
                    phenotype = sample_row[gene].iloc[0]

                results[sample_id][gene]['phenotype'] = phenotype

                # Extract variants for this gene
                gene_variants = sample_data[sample_data['Gene'] == gene]

                for _, variant in gene_variants.iterrows():
                    if pd.isna(variant['ID']) or variant['ID'] == '.':
                        continue

                    rsid = variant['ID']

                    # Get genotype from appropriate column
                    genotype = None
                    if 'Genotype' in variant:
                        genotype = variant['Genotype']
                    else:
                        gt_cols = [col for col in variant.index if '_GT' in col]
                        if gt_cols:
                            genotype = variant[gt_cols[0]]

                    if genotype is None:
                        continue

                    # Calculate importance
                    importance = self._calculate_importance(rsid, gene, genotype, phenotype)
                    contribution = "supporting" if importance > 0 else "opposing" if importance < 0 else "neutral"

                    # Get additional variant info
                    known_effect = "Unknown"
                    if rsid in self.variant_phenotype_mapping and genotype in self.variant_phenotype_mapping[rsid].get(
                            'effect', {}):
                        known_effect = self.variant_phenotype_mapping[rsid]['effect'][genotype]

                    results[sample_id][gene]['variants'].append({
                        'rsid': rsid,
                        'genotype': genotype,
                        'importance': importance,
                        'contribution': contribution,
                        'known_effect': known_effect,
                        'position': str(variant['POS']),
                        'reference': variant['REF'],
                        'alternate': variant['ALT']
                    })

                # Sort variants by absolute importance
                results[sample_id][gene]['variants'].sort(key=lambda x: abs(x['importance']), reverse=True)

                # Add summary explanation
                results[sample_id][gene]['explanation'] = self._generate_gene_explanation(
                    gene,
                    phenotype,
                    results[sample_id][gene]['variants']
                )

        return results

    def _calculate_importance(self, rsid, gene, genotype, phenotype):
        # Base importance
        importance = 0.0

        # Check if we have knowledge about this variant
        if rsid in self.variant_phenotype_mapping and self.variant_phenotype_mapping[rsid]['gene'] == gene:
            variant_info = self.variant_phenotype_mapping[rsid]
            base_importance = variant_info.get('importance', 0.5)

            # Check if genotype is relevant
            if genotype in variant_info.get('effect', {}):
                expected_phenotype = variant_info['effect'][genotype]

                # If phenotype matches expectation, positive importance
                if expected_phenotype == phenotype:
                    importance = base_importance
                # If in same category (e.g., both decreased), partial importance
                elif self._in_same_category(expected_phenotype, phenotype):
                    importance = base_importance * 0.5
                # If in opposite categories, negative importance
                elif self._in_opposite_categories(expected_phenotype, phenotype):
                    importance = -base_importance * 0.5

        return importance

    def _in_same_category(self, phenotype1, phenotype2):
        for category, phenotypes in self.phenotype_categories.items():
            if phenotype1 in phenotypes and phenotype2 in phenotypes:
                return True
        return False

    def _in_opposite_categories(self, phenotype1, phenotype2):
        if (phenotype1 in self.phenotype_categories['increased'] and
                phenotype2 in self.phenotype_categories['decreased']):
            return True
        if (phenotype1 in self.phenotype_categories['decreased'] and
                phenotype2 in self.phenotype_categories['increased']):
            return True
        return False

    def _generate_gene_explanation(self, gene, phenotype, variants):
        if not variants:
            return f"No significant variants found for {gene}. Phenotype is {phenotype}."

        # Get category description
        phenotype_category = "unknown"
        for category, phenotypes in self.phenotype_categories.items():
            if phenotype in phenotypes:
                phenotype_category = category
                break

        # Start with phenotype summary
        explanation = f"{gene} phenotype is {phenotype} ({phenotype_category} function). "

        # Focus on top 3 variants
        top_variants = variants[:3]
        if top_variants:
            explanation += "Key variants: "

            variant_descriptions = []
            for variant in top_variants:
                effect_direction = "supporting" if variant['importance'] > 0 else "opposing" if variant[
                                                                                                    'importance'] < 0 else "neutral"
                variant_descriptions.append(
                    f"{variant['rsid']} (genotype {variant['genotype']}, {effect_direction}, impact: {abs(variant['importance']):.2f})"
                )

            explanation += ", ".join(variant_descriptions)

        return explanation

    def run(self):
        print("Loading data...")
        self.load_data()

        print("Analyzing variant importance...")
        results = self.calculate_variant_importance()

        # Save results to JSON
        print(f"Saving results to {self.output_file}")
        with open(self.output_file, 'w') as f:
            json.dump(results, f, indent=4)

        return results


def main():
    parser = argparse.ArgumentParser(description='PharmCAT XAI - Explainable AI for PGx predictions')
    parser.add_argument('--vcf_dir', required=True, help='Directory containing VCF and CSV files')
    parser.add_argument('--phenotypes', required=True, help='CSV file with phenotype predictions')
    parser.add_argument('--output', default='pharmcat_xai_results.json', help='Output JSON file')

    args = parser.parse_args()

    xai = PharmcatXAI(args.vcf_dir, args.phenotypes, args.output)
    xai.run()


if __name__ == "__main__":
    main()
