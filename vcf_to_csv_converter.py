import argparse
import glob
import os

import pandas as pd


def parse_vcf(vcf_file):
    header_lines = []
    data_lines = []

    with open(vcf_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                if line.startswith('#CHROM'):
                    header_lines.append(line.strip())
                continue
            else:
                data_lines.append(line.strip())

    if not header_lines:
        raise ValueError(f"No header line found in {vcf_file}")

    header = header_lines[-1].lstrip('#').split('\t')

    data = []
    for line in data_lines:
        row = line.split('\t')
        if len(row) != len(header):
            print(f"Warning: Line has {len(row)} columns but header has {len(header)} columns")
            continue
        data.append(row)

    df = pd.DataFrame(data, columns=header)

    # Extract the gene information from the INFO field
    if 'INFO' in df.columns:
        df['Gene'] = df['INFO'].apply(lambda x: extract_gene_from_info(x))

    # If sample column exists (usually the last column), parse it to get GT
    sample_col = header[-1]
    if sample_col not in ['INFO', 'FORMAT']:
        df[f'{sample_col}_GT'] = df[sample_col].apply(lambda x: extract_gt_from_sample(x))

    return df


def extract_gene_from_info(info_field):
    if 'PX=' in info_field:
        parts = info_field.split(';')
        for part in parts:
            if part.startswith('PX='):
                return part[3:]
    return None


def extract_gt_from_sample(sample_field):
    if ':' in sample_field:
        return sample_field.split(':')[0]
    return sample_field


def convert_vcf_to_csv(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    vcf_files = glob.glob(os.path.join(input_dir, "*.vcf"))

    for vcf_file in vcf_files:
        try:
            print(f"Processing {vcf_file}...")
            df = parse_vcf(vcf_file)

            # Generate output filename
            base_name = os.path.basename(vcf_file)
            output_file = os.path.join(output_dir, base_name.replace('.vcf', '.csv'))

            df.to_csv(output_file, index=False)
            print(f"Saved to {output_file}")
        except Exception as e:
            print(f"Error processing {vcf_file}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Convert VCF files to CSV format')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing VCF files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save CSV files')
    args = parser.parse_args()

    convert_vcf_to_csv(args.input_dir, args.output_dir)
    print("Conversion complete!")


if __name__ == "__main__":
    main()
