import csv
import os
import re
import sys


def parse_vcf_to_csv(vcf_file, output_csv=None):
    if output_csv is None:
        output_csv = os.path.splitext(vcf_file)[0] + '.csv'

    data = []
    header = None
    sample_names = []

    with open(vcf_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('##'):
                continue
            elif line.startswith('#CHROM'):
                header = line[1:].split('\t')
                if len(header) > 9:
                    sample_names = header[9:]
            else:
                data.append(line.split('\t'))

    csv_header = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'Gene']

    info_fields = ['DP', 'TYPE', 'AC', 'AF', 'AN', 'AB', 'ABP', 'AO', 'CIGAR', 'MQM', 'PAIRED']
    for field in info_fields:
        csv_header.append(f"{field}")

    format_fields = ['GT', 'DP', 'AD', 'AO', 'QA', 'RO', 'QR']

    for sample in sample_names:
        for field in format_fields:
            csv_header.append(f"{sample}_{field}")

    csv_data = []
    for row in data:
        if len(row) < 8:
            continue

        csv_row = row[:7]

        info_field = row[7]
        gene_match = re.search(r'PX=([^;]+)', info_field)
        gene = gene_match.group(1) if gene_match else ''
        csv_row.append(gene)

        info_dict = {}
        info_parts = info_field.split(';')
        for part in info_parts:
            if '=' in part:
                key, value = part.split('=', 1)
                info_dict[key] = value
            else:
                info_dict[part] = 'Yes'

        for field in info_fields:
            value = info_dict.get(field, '')
            if ',' in value:
                value = value.replace(',', '|')
            csv_row.append(value)

        if len(row) > 8:
            format_field = row[8]
            format_keys = format_field.split(':')

            for i, sample in enumerate(sample_names, 9):
                if i < len(row):
                    sample_data = row[i]
                    sample_values = sample_data.split(':')

                    sample_dict = {}
                    for j, key in enumerate(format_keys):
                        if j < len(sample_values):
                            sample_dict[key] = sample_values[j]

                    for field in format_fields:
                        value = sample_dict.get(field, '')
                        if ',' in value:
                            value = value.replace(',', '|')
                        csv_row.append(value)
                else:
                    for field in format_fields:
                        csv_row.append('')
        else:
            for sample in sample_names:
                for field in format_fields:
                    csv_row.append('')

        csv_data.append(csv_row)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_data)

    return output_csv


def process_pharmcat_folders(base_dir='pharmcat_processed'):
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist.")
        return []

    processed_files = []

    # Get all sample folders
    sample_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    for sample_folder in sample_folders:
        sample_path = os.path.join(base_dir, sample_folder)

        # Find preprocessed VCF files in the sample folder
        vcf_files = [f for f in os.listdir(sample_path) if f.endswith('preprocessed.vcf')]

        for vcf_file in vcf_files:
            vcf_path = os.path.join(sample_path, vcf_file)
            csv_path = os.path.splitext(vcf_path)[0] + '.csv'

            # Convert VCF to CSV
            output_csv = parse_vcf_to_csv(vcf_path, csv_path)
            processed_files.append(output_csv)
            print(f"Converted {vcf_path} to {output_csv}")

    return processed_files


def main():
    if len(sys.argv) > 1:
        input_path = sys.argv[1]

        if os.path.isdir(input_path):
            if input_path == 'pharmcat_processed' or os.path.basename(input_path) == 'pharmcat_processed':
                processed_files = process_pharmcat_folders(input_path)
                print(f"Processed {len(processed_files)} files under {input_path}")
            else:
                vcf_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.vcf')]
                for vcf_file in vcf_files:
                    output_csv = parse_vcf_to_csv(vcf_file)
                    print(f"Converted {vcf_file} to {output_csv}")
        else:
            for vcf_file in sys.argv[1:]:
                if os.path.isfile(vcf_file) and vcf_file.endswith('.vcf'):
                    output_csv = parse_vcf_to_csv(vcf_file)
                    print(f"Converted {vcf_file} to {output_csv}")
                else:
                    print(f"Skipping {vcf_file}: not a VCF file")
    else:
        # By default, process the pharmcat_processed directory
        processed_files = process_pharmcat_folders()
        if processed_files:
            print(f"Processed {len(processed_files)} files under pharmcat_processed")
        else:
            print("Please provide VCF files or a directory as arguments.")
            print("Usage: python vcf_to_csv.py file1.vcf file2.vcf ...")
            print("Or: python vcf_to_csv.py directory_with_vcfs")
            print("Or run without arguments to process all files under pharmcat_processed/")


if __name__ == "__main__":
    main()
