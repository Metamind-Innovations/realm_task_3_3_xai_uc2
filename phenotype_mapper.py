import pandas as pd
import argparse
import os


def map_phenotypes(input_csv, output_csv=None):
    phenotype_map = {
        'NM': 0,
        'LNM': 1,
        'IM': 2,
        'LIM': 3,
        'PM': 4,
        'LPM': 5,
        'UM': 6,
        'LUM': 7,
        'RM': 8,
        'LRM': 9,
        'INTEDERMINATE': -1,
        'NF': 10,
        'DF': 11,
        'IF': 12,
        'PF': 13,
        'PDF': 14
    }

    df = pd.read_csv(input_csv)

    gene_columns = ['CYP2B6', 'CYP2C19', 'CYP2C9', 'CYP3A5', 'DPYD', 'SLCO1B1', 'TPMT']

    for column in gene_columns:
        if column in df.columns:
            df[column] = df[column].map(phenotype_map).fillna(-1).astype(int)

    if output_csv:
        df.to_csv(output_csv, index=False)

    return df


def main():
    parser = argparse.ArgumentParser(description='Map phenotypes to numeric values')
    parser.add_argument('--input_csv', default='phenotypes.csv', help='Input CSV file with phenotypes')
    parser.add_argument('--output_csv', default='phenotypes_encoded.csv',
                        help='Output CSV file with encoded phenotypes')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mapped_df = map_phenotypes(args.input_csv, args.output_csv)
    print(f"Phenotypes mapped and saved to {args.output_csv}")


if __name__ == "__main__":
    main()
