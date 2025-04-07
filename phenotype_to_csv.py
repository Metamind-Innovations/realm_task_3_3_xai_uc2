import csv
import json
import os
import sys


def safe_str(value):
    if value is None:
        return ""
    return str(value)


def phenotype_to_csv(input_file, output_file=None):
    if not output_file:
        output_file = os.path.splitext(input_file)[0] + '.csv'

    with open(input_file, 'r') as f:
        data = json.load(f)

    csv_rows = []
    headers = [
        'gene', 'geneSymbol', 'source', 'phenotype', 'diplotype_label',
        'allele1_name', 'allele2_name', 'allele1_function', 'allele2_function',
        'allele1_reference', 'allele2_reference', 'activity_score',
        'call_source', 'has_undocumented_variations',
        'treat_undocumented_variations_as_reference', 'num_variants',
        'variant_rsids', 'variant_calls', 'variant_positions',
        'variant_alleles', 'chr', 'phased', 'effectively_phased',
        'messages', 'uncalled_haplotypes'
    ]

    gene_reports = data.get('geneReports', {})

    for report_source, genes in gene_reports.items():
        for gene_symbol, gene_data in genes.items():
            geneSymbol = gene_data.get('geneSymbol', gene_symbol)

            variants = gene_data.get('variants', [])
            variant_rsids = []
            variant_calls = []
            variant_positions = []
            variant_alleles = []

            for variant in variants:
                rsid = variant.get('dbSnpId', '')
                variant_rsids.append(safe_str(rsid))
                variant_calls.append(safe_str(variant.get('call', '')))
                variant_positions.append(safe_str(variant.get('position', '')))
                alleles = variant.get('alleles', [])
                variant_alleles.append(
                    '|'.join(safe_str(a) for a in alleles) if alleles else ''
                )

            variant_rsids_str = ';'.join(variant_rsids)
            variant_calls_str = ';'.join(variant_calls)
            variant_positions_str = ';'.join(variant_positions)
            variant_alleles_str = ';'.join(variant_alleles)

            messages = [
                safe_str(msg.get('message', ''))
                for msg in gene_data.get('messages', [])
            ]
            messages_str = ';'.join(messages) if messages else ""

            uncalled_haplotypes = [
                safe_str(h) for h in gene_data.get('uncalledHaplotypes', [])
            ]
            uncalled_haplotypes_str = (
                ';'.join(uncalled_haplotypes) if uncalled_haplotypes else ""
            )

            diplotype_data = gene_data.get('recommendationDiplotypes', [])
            if not diplotype_data:
                diplotype_data = gene_data.get('sourceDiplotypes', [])

            if diplotype_data:
                for diplotype in diplotype_data:
                    phenotypes = [
                        safe_str(p) for p in diplotype.get('phenotypes', [])
                    ]
                    phenotype_str = ';'.join(phenotypes) if phenotypes else ""

                    allele1 = diplotype.get('allele1', {})
                    allele2 = diplotype.get('allele2', {})

                    row = {
                        'gene': safe_str(gene_symbol),
                        'geneSymbol': safe_str(geneSymbol),
                        'source': safe_str(report_source),
                        'phenotype': safe_str(phenotype_str),
                        'diplotype_label': safe_str(diplotype.get('label', "")),
                        'allele1_name': safe_str(allele1.get('name', "")),
                        'allele2_name': (
                            safe_str(allele2.get('name', "")) if allele2 else ""
                        ),
                        'allele1_function': safe_str(allele1.get('function', "")),
                        'allele2_function': (
                            safe_str(allele2.get('function', ""))
                            if allele2 else ""
                        ),
                        'allele1_reference': (
                            safe_str(allele1.get('reference', False))
                        ),
                        'allele2_reference': (
                            safe_str(allele2.get('reference', False))
                            if allele2 else "False"
                        ),
                        'activity_score': (
                            safe_str(diplotype.get('activityScore', ""))
                        ),
                        'call_source': safe_str(gene_data.get('callSource', "")),
                        'has_undocumented_variations': (
                            safe_str(gene_data.get('hasUndocumentedVariations', False))
                        ),
                        'treat_undocumented_variations_as_reference': (
                            safe_str(gene_data.get(
                                'treatUndocumentedVariationsAsReference', False))
                        ),
                        'num_variants': safe_str(len(variants)),
                        'variant_rsids': variant_rsids_str,
                        'variant_calls': variant_calls_str,
                        'variant_positions': variant_positions_str,
                        'variant_alleles': variant_alleles_str,
                        'chr': safe_str(gene_data.get('chr', "")),
                        'phased': safe_str(gene_data.get('phased', False)),
                        'effectively_phased': (
                            safe_str(gene_data.get('effectivelyPhased', False))
                        ),
                        'messages': messages_str,
                        'uncalled_haplotypes': uncalled_haplotypes_str
                    }

                    csv_rows.append(row)
            else:
                row = {
                    'gene': safe_str(gene_symbol),
                    'geneSymbol': safe_str(geneSymbol),
                    'source': safe_str(report_source),
                    'phenotype': "",
                    'diplotype_label': "",
                    'allele1_name': "",
                    'allele2_name': "",
                    'allele1_function': "",
                    'allele2_function': "",
                    'allele1_reference': "False",
                    'allele2_reference': "False",
                    'activity_score': "",
                    'call_source': safe_str(gene_data.get('callSource', "")),
                    'has_undocumented_variations': (
                        safe_str(gene_data.get('hasUndocumentedVariations', False))
                    ),
                    'treat_undocumented_variations_as_reference': (
                        safe_str(gene_data.get(
                            'treatUndocumentedVariationsAsReference', False))
                    ),
                    'num_variants': safe_str(len(variants)),
                    'variant_rsids': variant_rsids_str,
                    'variant_calls': variant_calls_str,
                    'variant_positions': variant_positions_str,
                    'variant_alleles': variant_alleles_str,
                    'chr': safe_str(gene_data.get('chr', "")),
                    'phased': safe_str(gene_data.get('phased', False)),
                    'effectively_phased': (
                        safe_str(gene_data.get('effectivelyPhased', False))
                    ),
                    'messages': messages_str,
                    'uncalled_haplotypes': uncalled_haplotypes_str
                }

                csv_rows.append(row)

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(csv_rows)

    return output_file


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python phenotype_to_csv.py input_file.json [output_file.csv]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    result_file = phenotype_to_csv(input_file, output_file)
    print(f"Converted {input_file} to {result_file}")
