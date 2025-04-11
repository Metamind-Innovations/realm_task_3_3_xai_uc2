#!/usr/bin/env python3
import json
import argparse

# This currently doesn't have DPYD as the sourceDiplotype is not the same as the recommendationDiplotype 
APPLICABLE_GENES = ["CYP2B6", "CYP2C9", "CYP2C19", "CYP3A5", "DPYD", "SLCO1B1", "TPMT"] #, "UGT1A1"] # Leave this one out for now  

MAPPING = {
    'Normal Metabolizer': 'NM',
    'Likely Normal Metabolizer': 'LNM',
    'Intermediate Metabolizer': 'IM',
    'Likely Intermediate Metabolizer': 'LIM',
    'Poor Metabolizer': 'PM',
    'Likely Poor Metabolizer': 'LPM',
    'Ultra Rapid Metabolizer': 'UM',
    'Ultrarapid Metabolizer': 'UM', # Needed for Pharmcat
    'Likely Ultra Rapid Metabolizer': 'LUM',
    'Rapid Metabolizer': 'RM',
    'Likely Rapid Metabolizer': 'LRM',
    'Indeterminate': 'INDETERMINATE',
    'No Result': 'INDETERMINATE',
    'n/a': 'INDETERMINATE',
}

MAPPING_FUNCTION = {
    'Normal Function': 'NF',
    'Decreased Function': 'DF',
    'Increased Function': 'IF',
    'Poor Function': 'PF',
    'Indeterminate': 'INDETERMINATE',
    'Possible Decrease Function': 'PDF',
}

def process_phenotype_data(file_path, applicable_genes=APPLICABLE_GENES, mapping=MAPPING, mapping_function=MAPPING_FUNCTION):
    with open(file_path, "r") as file:
        data = json.load(file)
  
    phenotype_map = {}

    for source, genes in data.get("geneReports", {}).items():
        if source != "CPIC":
            continue
        for gene, details in genes.items():
            if gene not in applicable_genes:
                continue
            phenotype = details.get("recommendationDiplotypes", [])[0].get("phenotypes", [])[0]
            mapped_phenotype = mapping.get(phenotype, phenotype)
            mapped_phenotype = mapping_function.get(phenotype, mapped_phenotype)
            phenotype_map[gene] = mapped_phenotype

    return phenotype_map

def main():
    parser = argparse.ArgumentParser(description="Map pharmacogenomic phenotypes from a JSON file.")
    parser.add_argument("file", help="Path to PharmCAT JSON file")
    args = parser.parse_args()

    result = process_phenotype_data(args.file, APPLICABLE_GENES, MAPPING, MAPPING_FUNCTION)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
