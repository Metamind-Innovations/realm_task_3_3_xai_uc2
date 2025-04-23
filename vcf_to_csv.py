import pandas as pd
import re
import os
import argparse

important_genes = ["CYP2B6", "CYP2C9", "CYP2C19", "CYP3A5", "SLCO1B1", "TPMT", "DPYD"]


def vcf_to_csv(vcf_file, patient_id=None, output_csv=None):
    variants = []
    with open(vcf_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            chrom = fields[0]
            pos = fields[1]
            rsid = fields[2]
            ref = fields[3]
            alt = fields[4]
            qual = fields[5]
            filter_field = fields[6]
            info = fields[7]
            format_field = fields[8] if len(fields) > 8 else ''
            sample = fields[9] if len(fields) > 9 else ''
            gene = re.search(r'PX=([^;]+)', info)
            gene = gene.group(1) if gene else 'Unknown'
            genotype = 'Unknown'
            if format_field and 'GT' in format_field.split(':') and sample:
                genotype_idx = format_field.split(':').index('GT')
                sample_fields = sample.split(':')
                if genotype_idx < len(sample_fields):
                    genotype = sample_fields[genotype_idx]
            variant = {
                'PATIENT_ID': patient_id if patient_id else os.path.basename(vcf_file).split('_')[0],
                'CHROM': chrom,
                'POS': pos,
                'ID': rsid,
                'REF': ref,
                'ALT': alt,
                'QUAL': qual,
                'FILTER': filter_field,
                'GENE': gene,
                'GENOTYPE': genotype
            }
            variants.append(variant)
    df = pd.DataFrame(variants)
    if output_csv:
        df.to_csv(output_csv, index=False)
    return df


def preprocess_for_ai(df, output_encoded_csv=None):
    def encode_genotype(row):
        genotype = row['GENOTYPE']
        if genotype == 'Unknown' or genotype == './.':
            return -1
        alleles = genotype.split('/')
        if len(alleles) != 2:
            return -1
        left, right = alleles
        if left == '.' or right == '.':
            return -1
        left = int(left)
        right = int(right)
        if left == 0 and right == 0:
            return 0
        elif (left == 0 and right != 0) or (left != 0 and right == 0):
            return 1
        else:
            return 2

    df['GENOTYPE_ENCODED'] = df.apply(encode_genotype, axis=1)
    pivoted_df = df.pivot_table(
        index='PATIENT_ID',
        columns=['GENE', 'ID'],
        values='GENOTYPE_ENCODED',
        aggfunc='first',
        fill_value=-1
    )
    pivoted_df.columns = [f"{gene}_{rsid}" for gene, rsid in pivoted_df.columns]
    pivoted_df = pivoted_df.reset_index()
    for gene in important_genes:
        pivoted_df[f"HAS_GENE_{gene}"] = pivoted_df.apply(
            lambda row: 1 if any(col.startswith(f"{gene}_") and row[col] != -1 for col in pivoted_df.columns) else 0,
            axis=1
        )
    if output_encoded_csv:
        pivoted_df.to_csv(output_encoded_csv, index=False)
    return pivoted_df


def main():
    parser = argparse.ArgumentParser(description='Convert VCF files to CSV format and encode for AI processing')
    parser.add_argument('--input_dir', default='data/', help='Directory containing VCF files')
    parser.add_argument('--output_csv', default='encoded.csv', help='Output file for encoded data')
    args = parser.parse_args()

    all_dataframes = []
    vcf_files = [f for f in os.listdir(args.input_dir) if f.endswith('.vcf')]

    print(f"Found {len(vcf_files)} VCF files in {args.input_dir}")
    for vcf_file in vcf_files:
        file_path = os.path.join(args.input_dir, vcf_file)
        patient_id = os.path.basename(vcf_file).split('_')[0]
        df = vcf_to_csv(file_path, patient_id=patient_id)
        all_dataframes.append(df)

    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Directly process the combined DataFrame in memory
    def encode_genotype(row):
        genotype = row['GENOTYPE']
        if genotype == 'Unknown' or genotype == './.':
            return -1
        alleles = genotype.split('/')
        if len(alleles) != 2:
            return -1
        left, right = alleles
        if left == '.' or right == '.':
            return -1
        left = int(left)
        right = int(right)
        if left == 0 and right == 0:
            return 0
        elif (left == 0 and right != 0) or (left != 0 and right == 0):
            return 1
        else:
            return 2

    combined_df['GENOTYPE_ENCODED'] = combined_df.apply(encode_genotype, axis=1)

    pivoted_df = combined_df.pivot_table(
        index='PATIENT_ID',
        columns=['GENE', 'ID'],
        values='GENOTYPE_ENCODED',
        aggfunc='first',
        fill_value=-1
    )

    pivoted_df.columns = [f"{gene}_{rsid}" for gene, rsid in pivoted_df.columns]
    pivoted_df = pivoted_df.reset_index()

    for gene in important_genes:
        pivoted_df[f"HAS_GENE_{gene}"] = pivoted_df.apply(
            lambda row: 1 if any(col.startswith(f"{gene}_") and row[col] != -1 for col in pivoted_df.columns) else 0,
            axis=1
        )

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pivoted_df.to_csv(args.output_csv, index=False)
    print(f"Encoded data saved to {args.output_csv}")


if __name__ == "__main__":
    main()
