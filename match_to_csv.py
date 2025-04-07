import csv
import json
import os
import sys


def match_json_to_csv(input_file, output_file=None):
    if not output_file:
        output_file = os.path.splitext(input_file)[0] + '.csv'

    with open(input_file, 'r') as f:
        data = json.load(f)

    csv_rows = []
    headers = ['gene', 'diplotype_name', 'diplotype_score', 'phased', 'homozygous',
               'haplotype1_name', 'haplotype2_name', 'variant_positions', 'variant_rsids',
               'variant_calls', 'missing_positions', 'uncallable_haplotypes']

    for result in data.get('results', []):
        gene = result.get('gene', '')
        phased = result.get('phased', False)

        match_data = result.get('matchData', {})
        homozygous = match_data.get('homozygous', False)

        missing_positions = [f"{p.get('rsid', '')}:{p.get('position', '')}"
                             for p in match_data.get('missingPositions', [])]

        missing_pos_str = ';'.join(missing_positions) if missing_positions else ''

        uncallable_haplotypes = ';'.join(result.get('uncallableHaplotypes', []))

        variant_positions = []
        variant_rsids = []
        variant_calls = []

        for variant in result.get('variants', []):
            variant_positions.append(str(variant.get('position', '')))
            rsid = variant.get('rsid')
            variant_rsids.append(rsid if rsid is not None else '')
            variant_calls.append(variant.get('vcfCall', ''))

        variant_positions_str = ';'.join(variant_positions)
        variant_rsids_str = ';'.join(variant_rsids)
        variant_calls_str = ';'.join(variant_calls)

        for diplotype in result.get('diplotypes', []):
            row = {
                'gene': gene,
                'diplotype_name': diplotype.get('name', ''),
                'diplotype_score': diplotype.get('score', ''),
                'phased': str(phased),
                'homozygous': str(homozygous),
                'variant_positions': variant_positions_str,
                'variant_rsids': variant_rsids_str,
                'variant_calls': variant_calls_str,
                'missing_positions': missing_pos_str,
                'uncallable_haplotypes': uncallable_haplotypes
            }

            haplotype1 = diplotype.get('haplotype1', {})
            haplotype2 = diplotype.get('haplotype2', {})

            row['haplotype1_name'] = haplotype1.get('name', '')
            row['haplotype2_name'] = haplotype2.get('name', '')

            csv_rows.append(row)

        # If no diplotypes, still add a row for the gene
        if not result.get('diplotypes'):
            row = {
                'gene': gene,
                'diplotype_name': '',
                'diplotype_score': '',
                'phased': str(phased),
                'homozygous': str(homozygous),
                'haplotype1_name': '',
                'haplotype2_name': '',
                'variant_positions': variant_positions_str,
                'variant_rsids': variant_rsids_str,
                'variant_calls': variant_calls_str,
                'missing_positions': missing_pos_str,
                'uncallable_haplotypes': uncallable_haplotypes
            }
            csv_rows.append(row)

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(csv_rows)

    return output_file


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python match_json_to_csv.py input_file.json [output_file.csv]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    result_file = match_json_to_csv(input_file, output_file)
    print(f"Converted {input_file} to {result_file}")
