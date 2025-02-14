import json
import pandas as pd


def df_from_json_phenotype(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    flat_records = []
    for source, genes in data.get('geneReports', {}).items():
        for gene_symbol, gene_info in genes.items():
            gene_info['source'] = source
            gene_info['geneSymbol'] = gene_symbol
            flat_records.append(gene_info)

    df = pd.json_normalize(flat_records, sep='_')
    print("Phenotype DataFrame Head:")
    print(df.head())
    return df


def df_from_json_match(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    results = data.get('results', [])

    # Extract and preserve all relevant phased values before normalization
    for record in results:
        # Top level phased
        record['top_level_phased'] = record.get('phased')

        # MatchData level phased and related fields
        if 'matchData' in record:
            record['matchData_phased'] = record['matchData'].get('phased')
            record['matchData_homozygous'] = record['matchData'].get('homozygous')
            record['matchData_effectivelyPhased'] = record['matchData'].get('effectivelyPhased')

    # Normalize with explicit field specification
    df = pd.json_normalize(
        results,
        sep='_',
        meta=[
            'top_level_phased',
            'gene',
            'source',
            'version',
            'matchData_phased',
            'matchData_homozygous',
            'matchData_effectivelyPhased'
        ]
    )

    print("Phased values verification:")
    print(df[['gene', 'top_level_phased', 'matchData_phased', 'matchData_homozygous',
              'matchData_effectivelyPhased']].head())
    return df


# Process the phenotype JSON
df_phenotype = df_from_json_phenotype('dataset_sample/HG00436_freebayes.preprocessed.phenotype.json')
# Process the match JSON
df_match = df_from_json_match('dataset_sample/HG00436_freebayes.preprocessed.match.json')

# Save to CSV files without the index
df_phenotype.to_csv('dataset_sample/HG00436_freebayes_preprocessed.phenotype.csv', index=False)
df_match.to_csv('dataset_sample/HG00436_freebayes_preprocessed.match.csv', index=False)

print("DataFrames saved as CSV files successfully.")

