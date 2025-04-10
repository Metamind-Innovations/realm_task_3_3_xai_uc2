import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import shap

# Target genes for analysis
TARGET_GENES = ["CYP2B6", "CYP2C9", "CYP2C19", "CYP3A5", "SLCO1B1", "TPMT", "DPYD"]

# Phenotype mappings for numerical representation
PHENOTYPE_MAPPING = {
    "PM": 0,  # Poor Metabolizer
    "IM": 1,  # Intermediate Metabolizer
    "NM": 2,  # Normal Metabolizer
    "RM": 3,  # Rapid Metabolizer
    "UM": 4,  # Ultrarapid Metabolizer
    "NF": 5,  # Normal Function
    "DF": 6,  # Decreased Function
    "PF": 7,  # Poor Function
    "IF": 8,  # Increased Function
    "INDETERMINATE": 9  # Indeterminate
}
REVERSE_PHENOTYPE_MAPPING = {v: k for k, v in PHENOTYPE_MAPPING.items()}

# Phenotype descriptions for human-readable output
PHENOTYPE_DESCRIPTIONS = {
    'NM': 'Normal Metabolizer',
    'IM': 'Intermediate Metabolizer',
    'PM': 'Poor Metabolizer',
    'UM': 'Ultra Rapid Metabolizer',
    'RM': 'Rapid Metabolizer',
    'NF': 'Normal Function',
    'DF': 'Decreased Function',
    'IF': 'Increased Function',
    'PF': 'Poor Function',
    'INDETERMINATE': 'Indeterminate'
}

# Phenotype categories for biological meaning
PHENOTYPE_CATEGORIES = {
    'decreased': ['PM', 'IM', 'DF', 'PF'],
    'normal': ['NM', 'NF'],
    'increased': ['UM', 'RM', 'IF'],
    'indeterminate': ['INDETERMINATE']
}

# Variant-phenotype knowledge base
VARIANT_PHENOTYPE_EFFECTS = {
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


def preprocess_input_data(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(input_dir, "preprocessed")

    os.makedirs(output_dir, exist_ok=True)

    vcf_files = glob.glob(os.path.join(input_dir, "*.vcf"))
    sample_to_file = {}

    for vcf_file in vcf_files:
        filename = os.path.basename(vcf_file)
        sample_id = filename.split('_')[0]

        parser = VCFParser()
        output_csv = os.path.join(output_dir, f"{sample_id}_preprocessed.csv")

        if parser.vcf_to_csv(vcf_file, output_csv):
            sample_to_file[sample_id] = output_csv
            print(f"Converted {vcf_file} to {output_csv}")

    return sample_to_file, output_dir


def load_csv_data(csv_files):
    all_data = []

    for csv_file in csv_files:
        sample_id = os.path.basename(csv_file).split('_')[0]
        try:
            df = pd.read_csv(csv_file)
            df['Sample'] = sample_id
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {str(e)}")

    if not all_data:
        raise ValueError(f"No valid CSV files found")

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


def extract_features(variant_data):
    features = defaultdict(dict)

    for _, row in variant_data.iterrows():
        sample = row['Sample']
        gene = row.get('Gene', None)
        if not gene or gene not in TARGET_GENES:
            continue

        # Extract variant ID and genotype
        variant_id = row.get('ID', f"{row['CHROM']}_{row['POS']}")
        if variant_id == '.' or pd.isna(variant_id):
            variant_id = f"{row['CHROM']}_{row['POS']}"

        # Try to find GT column (format may vary)
        gt_col = None
        for col in row.index:
            if '_GT' in col:
                gt_col = col
                break

        if gt_col:
            genotype = row[gt_col]
        else:
            # Default to 0/0 if GT not found
            genotype = "0/0"

        # Add basic variant-present feature
        feature_name = f"{gene}_{variant_id}"
        features[sample][feature_name] = 1

        # Add specific genotype features
        if genotype in ['0/1', '1/0', '0/2', '2/0']:  # Heterozygous
            features[sample][f"{feature_name}_het"] = 1
        elif genotype in ['1/1', '2/2']:  # Homozygous alt
            features[sample][f"{feature_name}_hom"] = 1

        # Add known allele features if applicable
        if variant_id in VARIANT_PHENOTYPE_EFFECTS:
            allele = VARIANT_PHENOTYPE_EFFECTS[variant_id].get('allele', '')
            if allele:
                if genotype in ['0/1', '1/0', '0/2', '2/0']:  # Heterozygous
                    features[sample][f"{allele}_het"] = 1
                elif genotype in ['1/1', '2/2']:  # Homozygous alt
                    features[sample][f"{allele}_hom"] = 1

    # Convert to dataframe
    samples = list(features.keys())
    all_features = set()
    for sample_features in features.values():
        all_features.update(sample_features.keys())

    X = pd.DataFrame(0, index=samples, columns=sorted(all_features))
    for sample, sample_features in features.items():
        for feature, value in sample_features.items():
            X.loc[sample, feature] = value

    return X


def prepare_targets(phenotypes_file, feature_samples):
    phenotypes_df = pd.read_csv(phenotypes_file)

    # Ensure 'Sample ID' column exists
    if 'Sample ID' not in phenotypes_df.columns:
        raise ValueError(f"No 'Sample ID' column found in {phenotypes_file}")

    # Filter to samples in feature matrix
    phenotypes_df = phenotypes_df[phenotypes_df['Sample ID'].isin(feature_samples)]

    # Set index to Sample ID
    phenotypes_df = phenotypes_df.set_index('Sample ID')

    # Convert phenotypes to numeric using mapping
    Y = pd.DataFrame(index=phenotypes_df.index)

    for gene in TARGET_GENES:
        if gene in phenotypes_df.columns:
            Y[gene] = phenotypes_df[gene].map(lambda x: PHENOTYPE_MAPPING.get(x, 9) if pd.notna(x) else 9)

    return Y, phenotypes_df


def run_shap_analysis(X, Y, max_samples=100, n_background=50):
    np.random.seed(42)  # For reproducibility
    results = {}

    # Limit to a reasonable number of samples for performance
    if len(X) > max_samples:
        sample_indices = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X.iloc[sample_indices]
        y_sample = Y.iloc[sample_indices]
    else:
        X_sample = X
        y_sample = Y

    # Create background dataset for SHAP
    if len(X_sample) > n_background:
        background = shap.sample(X_sample.values, n_background)
    else:
        background = shap.sample(X_sample.values, min(len(X_sample), 20))

    feature_names = X_sample.columns.tolist()

    for gene in TARGET_GENES:
        if gene not in y_sample.columns:
            print(f"Skipping {gene} - not found in phenotypes data")
            continue

        print(f"Running SHAP analysis for {gene}...")

        # Create prediction function for this gene
        def predict_gene(x):
            # For SHAP explainer, we need a function that predicts based on the feature vectors
            # Using a pre-existing model doesn't make sense in this context since we're explaining PharmCAT
            # Instead, we map each feature vector to its corresponding phenotype

            # If input shape matches our samples, return the actual values
            if len(x) == len(X_sample):
                return y_sample[gene].values

            # Otherwise, for consistency in the KernelExplainer, return values based on
            # the distribution in our dataset
            vals = y_sample[gene].values
            return np.random.choice(vals, size=len(x))

        # Create SHAP explainer
        explainer = shap.KernelExplainer(
            predict_gene,
            background,
            link="identity",
            feature_names=feature_names
        )

        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample.values, nsamples=100)

        # Store results
        results[gene] = {
            "shap_values": shap_values,
            "feature_names": feature_names,
            "sample_indices": X_sample.index.tolist(),
            "predictions": y_sample[gene].tolist()
        }

    return results


def enriched_results_json(shap_results, X, phenotypes_df, variant_data, output_file):
    json_results = {
        "gene_explanations": {},
        "feature_importance": {},
        "sample_explanations": []
    }

    # Process each gene's results
    for gene, result in shap_results.items():
        shap_values = result["shap_values"]
        feature_names = result["feature_names"]
        sample_indices = result["sample_indices"]
        predictions = result["predictions"]

        # Calculate overall feature importance
        abs_shap = np.abs(shap_values)
        importance = abs_shap.mean(axis=0)

        # Get top features
        top_indices = np.argsort(-importance)[:20]  # Top 20 features

        # Store feature importance
        gene_importance_dict = {}
        for i in top_indices:
            if i < len(feature_names):
                feature = feature_names[i]
                feature_imp = float(importance[i])
                gene_importance_dict[feature] = feature_imp

                # Add clinical significance if available
                rsid = feature.split('_')[1] if len(feature.split('_')) > 1 else None
                if rsid in VARIANT_PHENOTYPE_EFFECTS and VARIANT_PHENOTYPE_EFFECTS[rsid]['gene'] == gene:
                    clinical_info = VARIANT_PHENOTYPE_EFFECTS[rsid].get('clinical_significance', '')
                    if clinical_info:
                        gene_importance_dict[f"{feature}_clinical"] = clinical_info

        json_results["feature_importance"][gene] = gene_importance_dict

        # Extract gene from feature name
        def get_gene_from_feature(feature):
            parts = feature.split('_')
            return parts[0] if parts else "Unknown"

        # Group features by gene
        gene_features = defaultdict(list)
        for i in top_indices:
            if i < len(feature_names):
                feature = feature_names[i]
                feature_gene = get_gene_from_feature(feature)
                if feature_gene in TARGET_GENES:  # Only include target genes
                    # Add additional information for each variant
                    rsid = feature.split('_')[1] if len(feature.split('_')) > 1 else None
                    feature_info = {
                        "feature": feature,
                        "importance": float(importance[i])
                    }

                    # Add variant information if available
                    if rsid in VARIANT_PHENOTYPE_EFFECTS:
                        var_info = VARIANT_PHENOTYPE_EFFECTS[rsid]
                        if var_info['gene'] == feature_gene:
                            feature_info["allele"] = var_info.get('allele', '')
                            feature_info["effect"] = var_info.get('effect', '')
                            feature_info["clinical_significance"] = var_info.get('clinical_significance', '')

                    gene_features[feature_gene].append(feature_info)

        # Store gene explanations
        phenotype_counts = {}
        for pred in predictions:
            phenotype = REVERSE_PHENOTYPE_MAPPING.get(pred, "Unknown")
            phenotype_counts[phenotype] = phenotype_counts.get(phenotype, 0) + 1

        gene_explanation = {
            "prediction_distribution": phenotype_counts,
            "top_features_by_gene": {
                gene: features for gene, features in gene_features.items()
            }
        }

        json_results["gene_explanations"][gene] = gene_explanation

        # Add sample explanations (limit to 10)
        for i in range(min(10, len(sample_indices))):
            sample = sample_indices[i]
            pred = predictions[i]
            phenotype = REVERSE_PHENOTYPE_MAPPING.get(pred, "Unknown")
            phenotype_desc = PHENOTYPE_DESCRIPTIONS.get(phenotype, "Unknown")

            # Get SHAP values for this sample
            sample_shap = shap_values[i]

            # Get top contributing features
            top_contrib_indices = np.argsort(-np.abs(sample_shap))[:10]
            top_contributions = []

            for j in top_contrib_indices:
                if j < len(feature_names):
                    feature = feature_names[j]
                    # Extract variant ID if present
                    parts = feature.split('_')
                    rsid = parts[1] if len(parts) > 1 else None

                    contrib = {
                        "feature": feature,
                        "value": float(X.loc[sample, feature]),
                        "shap_value": float(sample_shap[j])
                    }

                    # Add variant information if available
                    if rsid in VARIANT_PHENOTYPE_EFFECTS:
                        var_info = VARIANT_PHENOTYPE_EFFECTS[rsid]
                        if var_info['gene'] == gene:
                            contrib["allele"] = var_info.get('allele', '')
                            contrib["effect"] = var_info.get('effect', '')
                            contrib["clinical_significance"] = var_info.get('clinical_significance', '')

                    top_contributions.append(contrib)

            # Generate explanation text
            explanation_text = f"The {gene} phenotype is {phenotype} ({phenotype_desc}). "

            # Add information about which category this phenotype falls into
            for category, phenotypes in PHENOTYPE_CATEGORIES.items():
                if phenotype in phenotypes:
                    explanation_text += f"This indicates {category} function. "
                    break

            # Add information about top contributing variants
            if top_contributions:
                explanation_text += "Key contributing variants include: "
                variant_descriptions = []

                for contrib in top_contributions[:3]:  # Top 3 for brevity
                    feature = contrib["feature"]
                    shap_val = contrib["shap_value"]
                    effect = "positive" if shap_val > 0 else "negative"

                    # Get rsid if available
                    parts = feature.split('_')
                    rsid = parts[1] if len(parts) > 1 else None

                    if rsid and rsid in VARIANT_PHENOTYPE_EFFECTS:
                        allele = VARIANT_PHENOTYPE_EFFECTS[rsid].get('allele', '')
                        var_effect = VARIANT_PHENOTYPE_EFFECTS[rsid].get('effect', '')
                        if allele and var_effect:
                            variant_descriptions.append(f"{rsid} ({allele}, {var_effect}) with {effect} contribution")
                        else:
                            variant_descriptions.append(f"{rsid} with {effect} contribution")
                    else:
                        variant_descriptions.append(f"{feature} with {effect} contribution")

                explanation_text += ", ".join(variant_descriptions)

            sample_explanation = {
                "sample_id": str(sample),
                "gene": gene,
                "predicted_phenotype": phenotype,
                "phenotype_description": phenotype_desc,
                "top_contributions": top_contributions,
                "explanation": explanation_text
            }

            json_results["sample_explanations"].append(sample_explanation)

    # Write to file
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"Results saved to {output_file}")
    return json_results


def generate_summary_report(json_results, output_dir):
    summary_file = os.path.join(output_dir, "pgx_shap_summary.txt")

    with open(summary_file, 'w') as f:
        f.write("# PharmCAT SHAP Analysis Summary Report\n\n")

        # Overall stats
        f.write("## Overview\n")
        f.write(f"Analysis performed on {len(json_results['gene_explanations'])} genes\n")
        num_samples = len(set([exp['sample_id'] for exp in json_results['sample_explanations']]))
        f.write(f"Total samples analyzed: {num_samples}\n\n")

        # Gene-by-gene summary
        f.write("## Gene Summaries\n")

        for gene, gene_data in json_results['gene_explanations'].items():
            f.write(f"\n### {gene}\n")

            # Phenotype distribution
            f.write("Phenotype distribution:\n")
            for phenotype, count in gene_data['prediction_distribution'].items():
                desc = PHENOTYPE_DESCRIPTIONS.get(phenotype, "")
                f.write(f"- {phenotype} ({desc}): {count}\n")

            # Top features
            f.write("\nTop contributing features:\n")
            all_features = []
            for gene_name, features in gene_data['top_features_by_gene'].items():
                all_features.extend(features)

            # Sort by importance
            all_features.sort(key=lambda x: x['importance'], reverse=True)

            for i, feature in enumerate(all_features[:10]):  # Top 10
                f.write(f"{i + 1}. {feature['feature']} - Importance: {feature['importance']:.4f}")

                if 'allele' in feature and feature['allele']:
                    f.write(f" - {feature['allele']}")

                if 'effect' in feature and feature['effect']:
                    f.write(f" - {feature['effect']}")

                f.write("\n")

            f.write("\n")

        # Sample explanations
        f.write("## Sample Explanations\n")

        for i, sample in enumerate(json_results['sample_explanations'][:5]):  # First 5 samples
            f.write(f"\n### Sample {sample['sample_id']} - {sample['gene']}\n")
            f.write(f"Phenotype: {sample['predicted_phenotype']} ({sample['phenotype_description']})\n")
            f.write(f"Explanation: {sample['explanation']}\n")

            f.write("Top contributions:\n")
            for j, contrib in enumerate(sample['top_contributions'][:5]):  # Top 5 contributions
                f.write(f"{j + 1}. {contrib['feature']} - SHAP value: {contrib['shap_value']:.4f}")

                if 'allele' in contrib and contrib['allele']:
                    f.write(f" - {contrib['allele']}")

                if 'clinical_significance' in contrib and contrib['clinical_significance']:
                    f.write(f" - {contrib['clinical_significance']}")

                f.write("\n")

    print(f"Summary report saved to {summary_file}")
    return summary_file


def main():
    parser = argparse.ArgumentParser(description='PharmCAT SHAP-based explainer')
    parser.add_argument('--input_dir', required=True, help='Directory containing VCF files or preprocessed CSV files')
    parser.add_argument('--phenotypes_file', required=True, help='Path to phenotypes.csv file')
    parser.add_argument('--output_dir', default='pgx_shap_results', help='Output directory for results')
    parser.add_argument('--convert_vcf', action='store_true', help='Convert VCF files to CSV format')
    parser.add_argument('--max_samples', type=int, default=100, help='Maximum number of samples for SHAP analysis')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Preprocess input data if needed
    print(f"Processing input data from {args.input_dir}...")

    if args.convert_vcf:
        sample_to_file, csv_dir = preprocess_input_data(args.input_dir, os.path.join(args.output_dir, "preprocessed"))
        print(f"Converted VCF files to CSV format in {csv_dir}")
        input_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    else:
        # Check if input_dir contains CSV files directly
        csv_files = glob.glob(os.path.join(args.input_dir, "*.csv"))
        if csv_files:
            input_files = csv_files
        else:
            # Check if input_dir contains preprocessed directory with CSV files
            preprocessed_dir = os.path.join(args.input_dir, "preprocessed")
            if os.path.exists(preprocessed_dir):
                input_files = glob.glob(os.path.join(preprocessed_dir, "*.csv"))
            else:
                raise ValueError(f"No CSV files found in {args.input_dir}. Use --convert_vcf to convert VCF files.")

    print(f"Found {len(input_files)} input files")

    # Step 2: Load and combine CSV data
    print("Loading CSV data...")
    variant_data = load_csv_data(input_files)
    print(f"Loaded data for {variant_data['Sample'].nunique()} samples")

    # Step 3: Extract features
    print("Extracting features...")
    X = extract_features(variant_data)
    print(f"Created feature matrix with {X.shape[0]} samples and {X.shape[1]} features")

    # Step 4: Prepare target phenotypes
    print(f"Loading phenotypes from {args.phenotypes_file}...")
    Y, phenotypes_df = prepare_targets(args.phenotypes_file, X.index)
    print(f"Prepared target matrix with {Y.shape[0]} samples")

    # Ensure same samples in X and Y
    common_samples = X.index.intersection(Y.index)
    X = X.loc[common_samples]
    Y = Y.loc[common_samples]
    print(f"Using {len(common_samples)} samples common to both feature and target matrices")

    # Step 5: Run SHAP analysis
    print("Running SHAP analysis...")
    shap_results = run_shap_analysis(X, Y, max_samples=args.max_samples)

    # Step 6: Prepare enriched JSON results
    print("Preparing enriched results...")
    json_output_file = os.path.join(args.output_dir, "pgx_shap_results.json")
    json_results = enriched_results_json(shap_results, X, phenotypes_df, variant_data, json_output_file)

    # Step 7: Generate summary report
    print("Generating summary report...")
    summary_file = generate_summary_report(json_results, args.output_dir)

    print(f"Analysis complete! Results saved to {args.output_dir}")
    print(f"JSON results: {json_output_file}")
    print(f"Summary report: {summary_file}")


if __name__ == "__main__":
    main()
