import glob
import json
import os
from collections import defaultdict

import pandas as pd


class PharmcatExplainer:
    def __init__(self, vcf_csv_dir='Preprocessed', phenotypes_file='result/phenotypes.csv', output_dir='xai_results'):
        self.vcf_csv_dir = vcf_csv_dir
        self.phenotypes_file = phenotypes_file
        self.output_dir = output_dir
        self.focus_genes = ["CYP2B6", "CYP2C9", "CYP2C19", "CYP3A5", "SLCO1B1", "TPMT", "DPYD"]

        # Map of phenotype codes to descriptions
        self.phenotype_descriptions = {
            # Metabolizer phenotypes
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

            # Function phenotypes
            'NF': 'Normal Function',
            'DF': 'Decreased Function',
            'IF': 'Increased Function',
            'PF': 'Poor Function',
            'PDF': 'Possible Decrease Function',

            # Special cases
            'INDETERMINATE': 'Indeterminate'
        }

        # Define the phenotype categories (for importance analysis)
        self.phenotype_categories = {
            # Decreased metabolism/function
            'decreased': ['PM', 'LPM', 'IM', 'LIM', 'DF', 'PF', 'PDF'],
            # Normal metabolism/function
            'normal': ['NM', 'LNM', 'NF'],
            # Increased metabolism/function
            'increased': ['UM', 'LUM', 'RM', 'LRM', 'IF'],
            # Special case
            'indeterminate': ['INDETERMINATE']
        }

        # Variant-phenotype association data
        self.variant_phenotype_effects = self._load_variant_phenotype_effects()

        os.makedirs(output_dir, exist_ok=True)

        # Load phenotypes data
        self.phenotypes = pd.read_csv(phenotypes_file)

        # Cache for storing variant distribution across phenotypes
        self.variant_phenotype_distribution = {}

        # Load all sample data to build variant-phenotype associations
        self.all_sample_data = self._preload_all_samples()

        # Sample ID to file mapping
        self.sample_to_csv = self._map_samples_to_files()

    def _load_variant_phenotype_effects(self):
        """Load known associations between variants and phenotypes"""
        # This would ideally come from a database, but for now we'll use a static mapping
        return {
            # CYP2B6
            'rs3745274': {
                'gene': 'CYP2B6',
                'allele': 'CYP2B6*6',
                'effect': 'Decreased enzyme activity',
                'phenotype_association': {
                    'homozygous': 'PM',  # Poor metabolizer when homozygous
                    'heterozygous': 'IM'  # Intermediate metabolizer when heterozygous
                },
                'clinical_significance': 'Affects metabolism of efavirenz, nevirapine, and other drugs'
            },
            'rs2279343': {
                'gene': 'CYP2B6',
                'allele': 'CYP2B6*4',
                'effect': 'Increased enzyme activity',
                'phenotype_association': {
                    'homozygous': 'RM',  # Rapid metabolizer when homozygous
                    'heterozygous': 'IM'  # Intermediate effect when heterozygous
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

            # CYP3A5
            'rs776746': {
                'gene': 'CYP3A5',
                'allele': 'CYP3A5*3',
                'effect': 'Non-functional enzyme',
                'phenotype_association': {
                    'homozygous': 'PM',  # Non-expressor
                    'heterozygous': 'IM'  # Intermediate expressor
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
            },
            'rs75017182': {
                'gene': 'DPYD',
                'allele': 'DPYD c.1679T>G',
                'effect': 'Decreased enzyme function',
                'phenotype_association': {
                    'homozygous': 'PM',
                    'heterozygous': 'IM'
                }
            }
        }

    def _map_samples_to_files(self):
        mapping = {}
        vcf_csv_files = glob.glob(os.path.join(self.vcf_csv_dir, '*.csv'))

        for csv_file in vcf_csv_files:
            filename = os.path.basename(csv_file)
            sample_id = filename.split('_')[0]
            mapping[sample_id] = csv_file

        return mapping

    def _preload_all_samples(self):
        """Load all sample data to analyze variant-phenotype correlations"""
        sample_data = {}

        # For each sample in the phenotypes file
        for _, row in self.phenotypes.iterrows():
            sample_id = row['Sample ID']

            # Find the corresponding VCF file
            csv_path = None
            for file_path in glob.glob(os.path.join(self.vcf_csv_dir, '*.csv')):
                if sample_id in os.path.basename(file_path):
                    csv_path = file_path
                    break

            if not csv_path:
                continue

            try:
                # Load the VCF data
                vcf_df = pd.read_csv(csv_path)

                # Store phenotypes and variant data
                phenotypes = {}
                for gene in self.focus_genes:
                    if gene in row:
                        phenotypes[gene] = row[gene]

                variants = {}
                for gene in self.focus_genes:
                    gene_variants = vcf_df[vcf_df['Gene'] == gene] if 'Gene' in vcf_df.columns else pd.DataFrame()

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

                sample_data[sample_id] = {
                    'phenotypes': phenotypes,
                    'variants': variants
                }

            except Exception as e:
                print(f"Error loading data for sample {sample_id}: {e}")

        return sample_data

    def _analyze_variant_phenotype_correlations(self):
        """Analyze correlations between variants and phenotypes across all samples"""
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

    def load_vcf_csv(self, sample_id):
        if sample_id in self.sample_to_csv:
            try:
                return pd.read_csv(self.sample_to_csv[sample_id])
            except Exception as e:
                print(f"Error reading VCF CSV for {sample_id}: {e}")
        else:
            print(f"No CSV file found for sample {sample_id}")
        return None

    def get_phenotype(self, sample_id, gene):
        row = self.phenotypes[self.phenotypes['Sample ID'] == sample_id]
        if not row.empty and gene in row.columns:
            return row[gene].iloc[0]
        return "Unknown"

    def extract_gene_variants(self, vcf_df, gene):
        if vcf_df is None or 'Gene' not in vcf_df.columns:
            return pd.DataFrame()

        return vcf_df[vcf_df['Gene'] == gene]

    def analyze_variants(self, sample_id, vcf_df, gene, phenotype):
        gene_variants = self.extract_gene_variants(vcf_df, gene)
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

    def _calculate_variant_importance(self, sample_id, rsid, genotype, gene, phenotype):
        """Calculate variant importance based on multiple factors with explanation"""
        # Start with moderate importance and confidence
        importance = 0.5
        confidence = 0.5
        reasons = []

        # 1. Check variant-phenotype association knowledge base
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
                # Phenotype category match (e.g., both indicate decreased function but different degrees)
                elif self._same_phenotype_category(expected_phenotype, phenotype):
                    importance += 0.2
                    confidence += 0.1
                    reasons.append(f"Partial match with known {gtype} variant effect (same category)")
                else:
                    # If expected phenotype doesn't match, this variant might be less important
                    # for this specific phenotype assignment
                    importance -= 0.1
                    reasons.append(f"Known variant but phenotype doesn't match expected {expected_phenotype}")
            else:
                # Known variant but no specific genotype-phenotype data
                importance += 0.1
                reasons.append("Known functional variant but specific phenotype impact unknown")

        # 2. Check variant frequency across samples with same phenotype - data-driven approach
        # This helps identify variants that consistently appear in samples with the same phenotype
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
                reasons.append(
                    f"Present in {same_phenotype_count}/{total_same_phenotype} samples with {phenotype} phenotype")
            # If this variant appears in >80% of samples with this phenotype, it's very important
            if same_phenotype_count / total_same_phenotype > 0.8:
                importance += 0.2
                confidence += 0.2
                reasons.append("Strong statistical association with this phenotype")

        # 3. Adjust for homozygous vs heterozygous
        if genotype in ['1/1', '2/2']:  # Homozygous alternate
            importance += 0.1
            reasons.append("Homozygous variant typically has stronger effect")

        # 4. Adjust importance based on existing clinical knowledge for specific variants
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
        """Check if two phenotypes fall into the same general category"""
        # Find which category each phenotype belongs to
        category1 = None
        category2 = None

        for category, phenotypes in self.phenotype_categories.items():
            if phenotype1 in phenotypes:
                category1 = category
            if phenotype2 in phenotypes:
                category2 = category

        # Return True if they're in the same category
        return category1 is not None and category1 == category2

    def _get_variant_phenotype_distribution(self, gene, rsid):
        """Get distribution of phenotypes for a specific variant across all samples"""
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
        """Get detailed effect information for a variant"""
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

    def _generate_explanation(self, gene, phenotype, variants):
        """Generate a detailed human-readable explanation of the phenotype prediction"""
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
        """Return clinical implications based on gene and function category"""
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
        """Attempt to infer diplotype from variant information"""
        # This is a simplified approach - real diplotype calling is more complex

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
            # This is simplified and may not be accurate for all cases
            allele_strings = []
            for allele, zygosity in allele_variants.items():
                # Extract allele number
                allele_num = allele.split('*')[1] if '*' in allele else allele
                allele_strings.append(f"*{allele_num}")

            return f"{gene}{'/'.join(allele_strings)}"

        return ""

    def run(self):
        results = {}

        # Process each sample in the phenotypes file
        for _, row in self.phenotypes.iterrows():
            sample_id = row['Sample ID']
            print(f"Processing sample: {sample_id}")

            # Load VCF CSV data
            vcf_df = self.load_vcf_csv(sample_id)
            if vcf_df is None:
                print(f"Skipping sample {sample_id} due to missing data")
                continue

            sample_results = {}

            # Analyze each gene
            for gene in self.focus_genes:
                if gene in row:
                    phenotype = row[gene]
                    gene_results = self.analyze_variants(sample_id, vcf_df, gene, phenotype)
                    sample_results[gene] = gene_results

            results[sample_id] = sample_results

        # Save all results to a single JSON file
        output_file = os.path.join(self.output_dir, "pharmcat_explanations.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"Explanations saved to {output_file}")
        return results


def main():
    # Initialize with appropriate directories
    explainer = PharmcatExplainer()
    explainer.run()
    print("Analysis complete!")


if __name__ == "__main__":
    main()
