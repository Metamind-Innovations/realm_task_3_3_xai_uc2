import argparse
import glob
import json
import os
import re
from collections import defaultdict

import pandas as pd


class VCFParser:
    def __init__(self):
        self.meta_info = {}
        self.header = None

    def parse_vcf(self, vcf_file):
        try:
            with open(vcf_file, 'r') as f:
                lines = f.readlines()

            data_lines = []
            for line in lines:
                line = line.strip()
                if line.startswith('##'):
                    self._parse_meta_line(line)
                elif line.startswith('#CHROM'):
                    self.header = line[1:].split('\t')
                else:
                    data_lines.append(line.split('\t'))

            if not self.header or not data_lines:
                return None

            df = pd.DataFrame(data_lines, columns=self.header)
            df['Gene'] = df['INFO'].apply(self._extract_gene)

            sample_columns = self.header[9:]
            if sample_columns:
                for sample in sample_columns:
                    df[f'{sample}_GT'] = df.apply(
                        lambda row: self._extract_genotype(row['FORMAT'], row[sample]),
                        axis=1
                    )

            return df

        except Exception as e:
            print(f"Error parsing VCF file: {e}")
            return None

    def _parse_meta_line(self, line):
        if line.startswith('##'):
            line = line[2:]
            if '=' in line:
                key, value = line.split('=', 1)
                self.meta_info[key] = value

    def _extract_gene(self, info_field):
        gene_match = re.search(r'PX=([^;]+)', info_field)
        return gene_match.group(1) if gene_match else ''

    def _extract_genotype(self, format_field, sample_field):
        format_parts = format_field.split(':')
        sample_parts = sample_field.split(':')

        if 'GT' in format_parts:
            gt_index = format_parts.index('GT')
            if gt_index < len(sample_parts):
                return sample_parts[gt_index]

        return ''

    def vcf_to_csv(self, vcf_file, output_csv=None):
        if output_csv is None:
            output_csv = os.path.splitext(vcf_file)[0] + '.csv'

        df = self.parse_vcf(vcf_file)
        if df is not None:
            df.to_csv(output_csv, index=False)
            return output_csv
        return None


class PharmcatExplainer:
    def __init__(self, input_path, output_file, output_dir='explanations'):
        self.input_path = input_path
        self.output_file = output_file
        self.output_dir = output_dir
        self.focus_genes = ["CYP2B6", "CYP2C9", "CYP2C19", "CYP3A5", "SLCO1B1", "TPMT", "DPYD"]

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Map of phenotype codes to descriptions
        self.phenotype_descriptions = {
            'NM': 'Normal Metabolizer',
            'LNM': 'Likely Normal Metabolizer',
            'IM': 'Intermediate Metabolizer',
            'LIM': 'Likely Intermediate Metabolizer',
            'PM': 'Poor Metabolizer',
            'LPM': 'Likely Poor Metabolizer',
            'UM': 'Ultra Rapid Metabolizer',
            'LUM': 'Likely Ultra Rapid Metabolizer',
            'RM': 'Rapid Metabolizer',
            'LRM': 'Likely Rapid Metabolizer',
            'NF': 'Normal Function',
            'DF': 'Decreased Function',
            'IF': 'Increased Function',
            'PF': 'Poor Function',
            'PDF': 'Possible Decrease Function',
            'INDETERMINATE': 'Indeterminate'
        }

        # Define phenotype categories
        self.phenotype_categories = {
            'decreased': ['PM', 'LPM', 'IM', 'LIM', 'DF', 'PF', 'PDF'],
            'normal': ['NM', 'LNM', 'NF'],
            'increased': ['UM', 'LUM', 'RM', 'LRM', 'IF'],
            'indeterminate': ['INDETERMINATE']
        }

        # Variant-phenotype association data
        self.variant_phenotype_effects = self._load_variant_phenotype_effects()

        # Load phenotypes data
        try:
            self.phenotypes = pd.read_csv(output_file)
        except Exception as e:
            print(f"Error loading phenotypes file: {e}")
            self.phenotypes = None

        # Cache for storing variant distribution across phenotypes
        self.variant_phenotype_distribution = {}

        # Sample ID to file mapping
        self.sample_to_file = {}

        # All sample data for variant-phenotype analysis
        self.all_sample_data = {}

        # Process input path to find VCF/CSV files
        self._process_input_path()

    def _process_input_path(self):
        if os.path.isdir(self.input_path):
            # Find all VCF files in the directory
            vcf_files = glob.glob(os.path.join(self.input_path, "*.vcf"))
            csv_files = glob.glob(os.path.join(self.input_path, "*.csv"))

            # Process VCF files
            for vcf_file in vcf_files:
                filename = os.path.basename(vcf_file)
                sample_id = filename.split('_')[0]
                self.sample_to_file[sample_id] = vcf_file

            # Process existing CSV files
            for csv_file in csv_files:
                filename = os.path.basename(csv_file)
                if not (filename.endswith('.match.csv') or filename.endswith('.phenotype.csv')):
                    sample_id = filename.split('_')[0]
                    self.sample_to_file[sample_id] = csv_file

        elif os.path.isfile(self.input_path):
            # Single file processing
            filename = os.path.basename(self.input_path)
            sample_id = filename.split('_')[0]
            self.sample_to_file[sample_id] = self.input_path

        # Preload all sample data for analysis
        if self.phenotypes is not None:
            self._preload_all_samples()

    def _load_variant_phenotype_effects(self):
        return {
            # CYP2B6
            'rs3745274': {
                'gene': 'CYP2B6',
                'allele': 'CYP2B6*6',
                'effect': 'Decreased enzyme activity',
                'phenotype_association': {
                    'homozygous': 'PM',
                    'heterozygous': 'IM'
                },
                'clinical_significance': 'Affects metabolism of efavirenz, nevirapine, and other drugs'
            },
            'rs2279343': {
                'gene': 'CYP2B6',
                'allele': 'CYP2B6*4',
                'effect': 'Increased enzyme activity',
                'phenotype_association': {
                    'homozygous': 'RM',
                    'heterozygous': 'IM'
                }
            },
            'rs8192709': {
                'gene': 'CYP2B6',
                'allele': 'CYP2B6*2',
                'effect': 'Altered enzyme activity',
                'phenotype_association': {
                    'homozygous': 'IM',
                    'heterozygous': 'NM'
                }
            },

            # CYP2C9
            'rs1799853': {
                'gene': 'CYP2C9',
                'allele': 'CYP2C9*2',
                'effect': 'Decreased enzyme activity',
                'phenotype_association': {
                    'homozygous': 'PM',
                    'heterozygous': 'IM'
                },
                'clinical_significance': 'Affects metabolism of warfarin, phenytoin, NSAIDs'
            },
            'rs1057910': {
                'gene': 'CYP2C9',
                'allele': 'CYP2C9*3',
                'effect': 'Decreased enzyme activity',
                'phenotype_association': {
                    'homozygous': 'PM',
                    'heterozygous': 'IM'
                },
                'clinical_significance': 'Significant impact on warfarin dosing'
            },

            # CYP2C19
            'rs4244285': {
                'gene': 'CYP2C19',
                'allele': 'CYP2C19*2',
                'effect': 'Loss of function',
                'phenotype_association': {
                    'homozygous': 'PM',
                    'heterozygous': 'IM'
                },
                'clinical_significance': 'Affects metabolism of clopidogrel, PPIs, antidepressants'
            },
            'rs4986893': {
                'gene': 'CYP2C19',
                'allele': 'CYP2C19*3',
                'effect': 'Loss of function',
                'phenotype_association': {
                    'homozygous': 'PM',
                    'heterozygous': 'IM'
                }
            },
            'rs12248560': {
                'gene': 'CYP2C19',
                'allele': 'CYP2C19*17',
                'effect': 'Increased expression',
                'phenotype_association': {
                    'homozygous': 'UM',
                    'heterozygous': 'RM'
                },
                'clinical_significance': 'Enhanced conversion of clopidogrel to active metabolite'
            },
            'rs3758581': {
                'gene': 'CYP2C19',
                'allele': 'CYP2C19*1',
                'effect': 'Normal function',
                'phenotype_association': {
                    'homozygous': 'NM',
                    'heterozygous': 'NM'
                }
            },

            # CYP3A5
            'rs776746': {
                'gene': 'CYP3A5',
                'allele': 'CYP3A5*3',
                'effect': 'Non-functional enzyme',
                'phenotype_association': {
                    'homozygous': 'PM',
                    'heterozygous': 'IM'
                },
                'clinical_significance': 'Affects metabolism of tacrolimus and other immunosuppressants'
            },

            # SLCO1B1
            'rs4149056': {
                'gene': 'SLCO1B1',
                'allele': 'SLCO1B1*5',
                'effect': 'Decreased transporter function',
                'phenotype_association': {
                    'homozygous': 'PF',
                    'heterozygous': 'DF'
                },
                'clinical_significance': 'Increased risk of statin-induced myopathy'
            },
            'rs2306283': {
                'gene': 'SLCO1B1',
                'allele': 'SLCO1B1*1B',
                'effect': 'Increased transporter function',
                'phenotype_association': {
                    'homozygous': 'IF',
                    'heterozygous': 'NF'
                }
            },

            # TPMT
            'rs1800462': {
                'gene': 'TPMT',
                'allele': 'TPMT*2',
                'effect': 'Decreased enzyme activity',
                'phenotype_association': {
                    'homozygous': 'PM',
                    'heterozygous': 'IM'
                },
                'clinical_significance': 'Increased risk of thiopurine toxicity'
            },
            'rs1800460': {
                'gene': 'TPMT',
                'allele': 'TPMT*3B',
                'effect': 'Decreased enzyme activity',
                'phenotype_association': {
                    'homozygous': 'PM',
                    'heterozygous': 'IM'
                }
            },
            'rs1142345': {
                'gene': 'TPMT',
                'allele': 'TPMT*3C',
                'effect': 'Decreased enzyme activity',
                'phenotype_association': {
                    'homozygous': 'PM',
                    'heterozygous': 'IM'
                },
                'clinical_significance': 'Most common decreased function allele'
            },

            # DPYD
            'rs3918290': {
                'gene': 'DPYD',
                'allele': 'DPYD*2A',
                'effect': 'Complete loss of function, splice site mutation',
                'phenotype_association': {
                    'homozygous': 'PM',
                    'heterozygous': 'IM'
                },
                'clinical_significance': 'High risk of severe fluoropyrimidine toxicity'
            },
            'rs55886062': {
                'gene': 'DPYD',
                'allele': 'DPYD*13',
                'effect': 'Decreased enzyme activity',
                'phenotype_association': {
                    'homozygous': 'PM',
                    'heterozygous': 'IM'
                },
                'clinical_significance': 'Increased risk of fluoropyrimidine toxicity'
            },
            'rs67376798': {
                'gene': 'DPYD',
                'allele': 'DPYD c.2846A>T',
                'effect': 'Decreased enzyme activity',
                'phenotype_association': {
                    'homozygous': 'PM',
                    'heterozygous': 'IM'
                },
                'clinical_significance': 'Moderate risk of fluoropyrimidine toxicity'
            }
        }

    def _preload_all_samples(self):
        for _, row in self.phenotypes.iterrows():
            sample_id = row['Sample ID']
            if sample_id not in self.sample_to_file:
                continue

            try:
                # Load the variant data
                vcf_data = self._load_sample_data(sample_id)
                if vcf_data is None:
                    continue

                # Store phenotypes and variant data
                phenotypes = {}
                for gene in self.focus_genes:
                    if gene in row:
                        phenotypes[gene] = row[gene]

                variants = {}
                for gene in self.focus_genes:
                    gene_variants = vcf_data[vcf_data['Gene'] == gene] if 'Gene' in vcf_data.columns else pd.DataFrame()

                    if not gene_variants.empty:
                        variants[gene] = []
                        for _, variant_row in gene_variants.iterrows():
                            if 'ID' in variant_row and not pd.isna(variant_row['ID']):
                                # Get genotype
                                genotype = "Unknown"
                                for col in variant_row.index:
                                    if '_GT' in col:
                                        genotype = variant_row[col]
                                        break

                                variants[gene].append({
                                    'rsid': variant_row['ID'],
                                    'genotype': genotype
                                })

                self.all_sample_data[sample_id] = {
                    'phenotypes': phenotypes,
                    'variants': variants
                }

            except Exception as e:
                print(f"Error preloading data for sample {sample_id}: {e}")

    def _load_sample_data(self, sample_id):
        if sample_id not in self.sample_to_file:
            return None

        file_path = self.sample_to_file[sample_id]

        try:
            if file_path.endswith('.vcf'):
                # Parse VCF file
                parser = VCFParser()
                return parser.parse_vcf(file_path)
            elif file_path.endswith('.csv'):
                # Load CSV file
                return pd.read_csv(file_path)
            else:
                print(f"Unsupported file format for {file_path}")
                return None
        except Exception as e:
            print(f"Error loading data for sample {sample_id}: {e}")
            return None

    def _analyze_variant_phenotype_correlations(self):
        variant_correlations = {}

        for gene in self.focus_genes:
            variant_correlations[gene] = defaultdict(lambda: defaultdict(int))

            # Count occurrences of each variant-phenotype combination
            for sample_id, data in self.all_sample_data.items():
                if gene in data['phenotypes'] and gene in data['variants']:
                    phenotype = data['phenotypes'][gene]

                    # Skip indeterminate phenotypes for correlation analysis
                    if phenotype == 'INDETERMINATE':
                        continue

                    for variant in data['variants'][gene]:
                        rsid = variant['rsid']
                        genotype = variant['genotype']

                        # Categorize genotype as homozygous/heterozygous
                        if genotype in ['1/1', '2/2']:
                            gtype = 'homozygous'
                        elif genotype in ['0/1', '1/0', '0/2', '2/0']:
                            gtype = 'heterozygous'
                        else:
                            gtype = 'other'

                        # Increment counter for this variant-genotype-phenotype combination
                        variant_correlations[gene][rsid][f"{gtype}_{phenotype}"] += 1

        return variant_correlations

    def get_phenotype(self, sample_id, gene):
        if self.phenotypes is None:
            return "Unknown"

        row = self.phenotypes[self.phenotypes['Sample ID'] == sample_id]
        if not row.empty and gene in row.columns:
            return row[gene].iloc[0]
        return "Unknown"

    def _calculate_variant_importance(self, sample_id, rsid, genotype, gene, phenotype):
        importance = 0.5
        confidence = 0.5
        reasons = []

        # Check variant-phenotype association knowledge base
        if rsid in self.variant_phenotype_effects and self.variant_phenotype_effects[rsid]['gene'] == gene:
            association = self.variant_phenotype_effects[rsid]

            # Determine genotype category
            if genotype in ['1/1', '2/2']:
                gtype = 'homozygous'
            elif genotype in ['0/1', '1/0', '0/2', '2/0']:
                gtype = 'heterozygous'
            else:
                gtype = 'unknown'

            # If we have phenotype association data for this genotype
            if 'phenotype_association' in association and gtype in association['phenotype_association']:
                expected_phenotype = association['phenotype_association'][gtype]

                # Phenotype match increases importance significantly
                if expected_phenotype == phenotype:
                    importance += 0.4
                    confidence += 0.3
                    reasons.append(f"Direct match with known {gtype} variant effect")
                # Phenotype category match (e.g., both indicate decreased function)
                elif self._same_phenotype_category(expected_phenotype, phenotype):
                    importance += 0.2
                    confidence += 0.1
                    reasons.append(f"Partial match with known {gtype} variant effect (same category)")
                else:
                    importance -= 0.1
                    reasons.append(f"Known variant but phenotype doesn't match expected {expected_phenotype}")
            else:
                # Known variant but no specific genotype-phenotype data
                importance += 0.1
                reasons.append("Known functional variant but specific phenotype impact unknown")

        # Check for data-driven insights
        variant_phenotype_counts = self._get_variant_phenotype_distribution(gene, rsid)
        total_same_phenotype = sum(1 for s_id, data in self.all_sample_data.items()
                                   if gene in data['phenotypes'] and data['phenotypes'][gene] == phenotype)

        # Get count of this variant in samples with the same phenotype
        same_phenotype_count = variant_phenotype_counts.get(phenotype, 0)

        if total_same_phenotype > 0:
            # If this variant appears in >50% of samples with this phenotype, it's likely important
            if same_phenotype_count / total_same_phenotype > 0.5:
                importance += 0.2
                confidence += 0.2
                reasons.append(f"Present in {same_phenotype_count}/{total_same_phenotype} samples with {phenotype}")
            # If this variant appears in >80% of samples with this phenotype, it's very important
            if same_phenotype_count / total_same_phenotype > 0.8:
                importance += 0.2
                confidence += 0.2
                reasons.append("Strong statistical association with this phenotype")

        # Adjust for homozygous vs heterozygous
        if genotype in ['1/1', '2/2']:  # Homozygous alternate
            importance += 0.1
            reasons.append("Homozygous variant typically has stronger effect")

        # Adjust importance based on existing clinical knowledge for specific variants
        if gene == 'CYP3A5' and rsid == 'rs776746' and phenotype in ['PM', 'IM']:
            importance += 0.1
            confidence += 0.1
            reasons.append("Key determinant of CYP3A5 metabolizer status")
        elif gene == 'CYP2C19' and rsid in ['rs4244285', 'rs4986893'] and phenotype in ['PM', 'IM']:
            importance += 0.1
            confidence += 0.1
            reasons.append("Major loss-of-function variant for CYP2C19")
        elif gene == 'DPYD' and rsid == 'rs3918290' and phenotype in ['PM', 'IM']:
            importance += 0.2
            confidence += 0.2
            reasons.append("Critical DPYD variant with strong clinical impact")

        # Normalize values to 0-1 range
        importance = min(max(importance, 0), 1)
        confidence = min(max(confidence, 0), 1)

        # Combine reasons into a single string
        reason = "; ".join(reasons) if reasons else "Based on general variant assessment"

        return importance, confidence, reason

    def _same_phenotype_category(self, phenotype1, phenotype2):
        category1 = None
        category2 = None

        for category, phenotypes in self.phenotype_categories.items():
            if phenotype1 in phenotypes:
                category1 = category
            if phenotype2 in phenotypes:
                category2 = category

        return category1 is not None and category1 == category2

    def _get_variant_phenotype_distribution(self, gene, rsid):
        # Create key for caching
        cache_key = f"{gene}_{rsid}"

        # Check if we've already calculated this
        if cache_key in self.variant_phenotype_distribution:
            return self.variant_phenotype_distribution[cache_key]

        # Count phenotype occurrences for this variant
        phenotype_counts = defaultdict(int)

        for sample_id, data in self.all_sample_data.items():
            if gene in data['phenotypes'] and gene in data['variants']:
                phenotype = data['phenotypes'][gene]

                # Check if this sample has the variant
                has_variant = any(v['rsid'] == rsid for v in data['variants'][gene])

                if has_variant:
                    phenotype_counts[phenotype] += 1

        # Cache result
        self.variant_phenotype_distribution[cache_key] = phenotype_counts

        return phenotype_counts

    def _get_variant_effect(self, rsid, gene, genotype, phenotype):
        if rsid in self.variant_phenotype_effects and self.variant_phenotype_effects[rsid]['gene'] == gene:
            variant_info = self.variant_phenotype_effects[rsid]

            effect = variant_info['effect']
            allele = variant_info.get('allele', '')
            clinical_significance = variant_info.get('clinical_significance', '')

            # Format genotype-specific effect
            if genotype in ['1/1', '2/2']:  # Homozygous alternate
                genotype_effect = "homozygous variant"
            elif genotype in ['0/1', '1/0', '0/2', '2/0']:  # Heterozygous
                genotype_effect = "heterozygous variant"
            else:
                genotype_effect = genotype

            result = f"{effect} ({allele}, {genotype_effect})"

            if clinical_significance:
                result += f"\nClinical impact: {clinical_significance}"

            return result

        return "Unknown effect"

    def analyze_variants(self, sample_id, vcf_data, gene, phenotype):
        if vcf_data is None or 'Gene' not in vcf_data.columns:
            return {
                'phenotype': phenotype,
                'phenotype_description': self.phenotype_descriptions.get(phenotype, phenotype),
                'variants': [],
                'explanation': f"No data found for gene {gene}. Phenotype determined to be {phenotype}."
            }

        gene_variants = vcf_data[vcf_data['Gene'] == gene]
        if gene_variants.empty:
            return {
                'phenotype': phenotype,
                'phenotype_description': self.phenotype_descriptions.get(phenotype, phenotype),
                'variants': [],
                'explanation': f"No variants found for gene {gene}. Phenotype determined to be {phenotype}."
            }

        # Extract relevant variant information
        variants = []
        for _, row in gene_variants.iterrows():
            if 'ID' not in row or pd.isna(row['ID']):
                continue

            # Get genotype
            genotype = "Unknown"
            for col in row.index:
                if '_GT' in col:
                    genotype = row[col]
                    break

            # Calculate importance based on genotype and variant known effects
            importance, confidence, reason = self._calculate_variant_importance(
                sample_id, row['ID'], genotype, gene, phenotype
            )

            variant = {
                'rsid': row['ID'],
                'position': str(row['POS']) if 'POS' in row else "Unknown",
                'reference': str(row['REF']) if 'REF' in row else "Unknown",
                'alternate': str(row['ALT']) if 'ALT' in row else "Unknown",
                'genotype': genotype,
                'importance': importance,
                'confidence': confidence,
                'importance_reason': reason,
                'effect': self._get_variant_effect(row['ID'], gene, genotype, phenotype)
            }
            variants.append(variant)

        # Sort variants by importance
        variants.sort(key=lambda x: x['importance'], reverse=True)

        # Generate explanation
        explanation = self._generate_explanation(gene, phenotype, variants)

        return {
            'phenotype': phenotype,
            'phenotype_description': self.phenotype_descriptions.get(phenotype, phenotype),
            'variants': variants,
            'explanation': explanation
        }

    def _generate_explanation(self, gene, phenotype, variants):
        phenotype_desc = self.phenotype_descriptions.get(phenotype, phenotype)

        if not variants:
            return f"The {gene} phenotype was determined to be {phenotype} ({phenotype_desc}), but no specific variants were identified that contribute to this phenotype."

        # Start with a general explanation
        explanation = [
            f"The {gene} phenotype was determined to be {phenotype} ({phenotype_desc}). This prediction is based on the following variants:"]

        # Add details for top variants (up to 3 or fewer if not available)
        num_variants = min(3, len(variants))
        for i, variant in enumerate(variants[:num_variants], 1):
            explanation.append(f"{i}. {variant['rsid']} - Genotype: {variant['genotype']}")
            explanation.append(f"   Effect: {variant['effect']}")
            explanation.append(f"   Importance: {variant['importance']:.2f} (Confidence: {variant['confidence']:.2f})")
            explanation.append(f"   Reasoning: {variant['importance_reason']}")

        # Add interpretation based on phenotype category
        if phenotype in self.phenotype_categories['decreased']:
            explanation.append(
                f"\nThe combination of these variants suggests reduced {gene} function, leading to a {phenotype_desc} phenotype.")

            # Add medication implications for decreased function
            drug_implications = self._get_drug_implications(gene, 'decreased')
            if drug_implications:
                explanation.append(f"Clinical implications: {drug_implications}")

        elif phenotype in self.phenotype_categories['increased']:
            explanation.append(
                f"\nThe combination of these variants suggests increased {gene} function, leading to a {phenotype_desc} phenotype.")

            # Add medication implications for increased function
            drug_implications = self._get_drug_implications(gene, 'increased')
            if drug_implications:
                explanation.append(f"Clinical implications: {drug_implications}")

        elif phenotype in self.phenotype_categories['normal']:
            explanation.append(
                f"\nThe variants detected do not significantly impact {gene} function, resulting in a {phenotype_desc} phenotype.")

        elif phenotype in self.phenotype_categories['indeterminate']:
            explanation.append(
                f"\nThere is insufficient or conflicting variant information to determine a clear {gene} phenotype.")

        # Add diplotype information if available
        diplotype = self._infer_diplotype(gene, variants)
        if diplotype:
            explanation.append(f"\nInferred diplotype: {diplotype}")

        return "\n".join(explanation)

    def _get_drug_implications(self, gene, function_category):
        implications = {
            'CYP2B6': {
                'decreased': "May require reduced doses of drugs metabolized by CYP2B6 (e.g., efavirenz, bupropion).",
                'increased': "May require increased doses of drugs metabolized by CYP2B6 due to faster clearance."
            },
            'CYP2C9': {
                'decreased': "May require reduced doses of warfarin, phenytoin, and NSAIDs. Higher risk of adverse effects.",
                'increased': "May have reduced efficacy with standard doses of CYP2C9 substrates."
            },
            'CYP2C19': {
                'decreased': "May have reduced efficacy of clopidogrel. May require lower doses of PPIs and certain antidepressants.",
                'increased': "May have increased risk of bleeding with clopidogrel. May require higher doses of PPIs."
            },
            'CYP3A5': {
                'decreased': "May require lower doses of tacrolimus, cyclosporine and other CYP3A5 substrates.",
                'increased': "May require higher doses of tacrolimus and other CYP3A5 substrates."
            },
            'SLCO1B1': {
                'decreased': "Increased risk of statin-induced myopathy. Consider lower statin doses or alternative statins.",
                'increased': "May have lower plasma concentrations of statins and other SLCO1B1 substrates."
            },
            'TPMT': {
                'decreased': "Increased risk of thiopurine toxicity. Requires significant dose reduction of azathioprine, 6-MP, and 6-TG.",
                'increased': "May require higher doses of thiopurines to achieve therapeutic effect."
            },
            'DPYD': {
                'decreased': "Increased risk of severe toxicity with fluoropyrimidines. Requires dose reduction or alternative therapy.",
                'increased': "May have reduced efficacy with standard doses of fluoropyrimidines."
            }
        }

        if gene in implications and function_category in implications[gene]:
            return implications[gene][function_category]
        return ""

    def _infer_diplotype(self, gene, variants):
        if not variants:
            return ""

        # Get known allele-defining variants
        allele_variants = {}
        for variant in variants:
            rsid = variant['rsid']
            genotype = variant['genotype']

            if rsid in self.variant_phenotype_effects and self.variant_phenotype_effects[rsid]['gene'] == gene:
                allele = self.variant_phenotype_effects[rsid].get('allele', '')
                if allele:
                    if genotype in ['1/1', '2/2']:  # Homozygous
                        allele_variants[allele] = 'homozygous'
                    elif genotype in ['0/1', '1/0', '0/2', '2/0']:  # Heterozygous
                        allele_variants[allele] = 'heterozygous'

        # Construct diplotype string
        if not allele_variants:
            return f"{gene}*1/*1 (presumed wild type)"

        if len(allele_variants) == 1:
            allele, zygosity = list(allele_variants.items())[0]

            if zygosity == 'homozygous':
                # Extract allele number from string (e.g., CYP2C9*2 -> *2)
                allele_num = allele.split('*')[1] if '*' in allele else allele
                return f"{gene}*{allele_num}/*{allele_num}"
            else:
                # Extract allele number from string
                allele_num = allele.split('*')[1] if '*' in allele else allele
                return f"{gene}*1/*{allele_num}"

        elif len(allele_variants) > 1:
            # For multiple allele variants, construct a compound diplotype
            allele_strings = []
            for allele, zygosity in allele_variants.items():
                # Extract allele number
                allele_num = allele.split('*')[1] if '*' in allele else allele
                allele_strings.append(f"*{allele_num}")

            return f"{gene}{'/'.join(allele_strings)}"

        return ""

    def run_analysis(self):
        if self.phenotypes is None:
            return "Failed to load phenotypes data. Please check output file."

        all_results = {}

        # Process each sample
        for _, row in self.phenotypes.iterrows():
            sample_id = row['Sample ID']

            if sample_id not in self.sample_to_file:
                print(f"No input file found for sample {sample_id}, skipping")
                continue

            print(f"Analyzing sample: {sample_id}")

            # Load sample data
            vcf_data = self._load_sample_data(sample_id)
            if vcf_data is None:
                print(f"Could not load data for sample {sample_id}, skipping")
                continue

            sample_results = {}

            # Analyze each gene
            for gene in self.focus_genes:
                if gene in row:
                    phenotype = row[gene]
                    gene_results = self.analyze_variants(sample_id, vcf_data, gene, phenotype)
                    sample_results[gene] = gene_results

            all_results[sample_id] = sample_results

        # Save results to a JSON file
        json_output_file = os.path.join(self.output_dir, "pharmcat_explanations.json")
        with open(json_output_file, 'w') as f:
            json.dump(all_results, f, indent=4)

        print(f"Detailed results saved to {json_output_file}")

        # Generate unified text report
        self._generate_unified_report(all_results)

        return all_results

    def _generate_unified_report(self, results):
        # Create the unified report content
        unified_report = ["# PHARMCAT EXPLAINER UNIFIED REPORT", "=" * 80, ""]
        unified_report.append(f"Number of samples analyzed: {len(results)}")
        unified_report.append("")

        # Add a summary table
        unified_report.append("## Summary Table")
        unified_report.append("-" * 80)
        unified_report.append(f"{'Sample ID':<20} | {'File':<50}")
        unified_report.append("-" * 80)

        for sample_id in results:
            file_path = os.path.basename(self.sample_to_file.get(sample_id, "Unknown"))
            unified_report.append(f"{sample_id:<20} | {file_path:<50}")

        unified_report.append("-" * 80)
        unified_report.append("")

        # Add a phenotype summary table
        unified_report.append("## Phenotype Summary")
        unified_report.append("-" * 80)
        gene_header = "Sample ID".ljust(20)
        for gene in self.focus_genes:
            gene_header += f" | {gene:<10}"
        unified_report.append(gene_header)
        unified_report.append("-" * 80)

        for sample_id, sample_data in results.items():
            row = sample_id.ljust(20)
            for gene in self.focus_genes:
                if gene in sample_data:
                    phenotype = sample_data[gene]['phenotype']
                    row += f" | {phenotype:<10}"
                else:
                    row += f" | {'N/A':<10}"
            unified_report.append(row)

        unified_report.append("-" * 80)
        unified_report.append("")

        # Add detailed analyses
        unified_report.append("## Detailed Analysis by Sample")

        for sample_id, sample_data in results.items():
            unified_report.append("")
            unified_report.append("=" * 80)
            unified_report.append(f"SAMPLE: {sample_id}")
            unified_report.append("=" * 80)

            for gene, gene_data in sample_data.items():
                unified_report.append(f"\n### {gene} - {gene_data['phenotype']} ({gene_data['phenotype_description']})")

                if 'explanation' in gene_data:
                    unified_report.append(gene_data['explanation'])

                if 'variants' in gene_data and gene_data['variants']:
                    unified_report.append("\nVariants Details:")
                    for variant in gene_data['variants']:
                        unified_report.append(f"- {variant['rsid']} (Importance: {variant['importance']:.2f})")

            unified_report.append("\n" + "-" * 80)

        # Save the unified report
        unified_file = os.path.join(self.output_dir, "pharmcat_unified_report.txt")
        with open(unified_file, 'w') as f:
            f.write("\n".join(unified_report))

        print(f"Unified report saved to {unified_file}")


def main():
    parser = argparse.ArgumentParser(description='PharmCAT Decision Explainer')
    parser.add_argument('--input', required=True, help='Input VCF file or directory')
    parser.add_argument('--output', required=True, help='Output phenotypes CSV file')
    parser.add_argument('--outdir', default='explanations', help='Output directory for the explanations')

    args = parser.parse_args()

    # Create explainer and run analysis
    explainer = PharmcatExplainer(args.input, args.output, args.outdir)
    explainer.run_analysis()

    print(f"Analysis complete! Results saved to {args.outdir}")


if __name__ == "__main__":
    main()
