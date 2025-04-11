import argparse
import subprocess
import os
from pathlib import Path
import pandas as pd
import shutil as sh
from helper_scripts.extract_phenotypes import process_phenotype_data

def main():
    parser = argparse.ArgumentParser(description="Process files through PharmCAT.")
    parser.add_argument("--input_folder", required=True, help="Folder with input files")
    parser.add_argument("--result_folder", required=True, help="Folder for result file")
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    working_folder = Path('/tmp/pharmcat')
    output_folder = Path(args.result_folder)
    
    print("Starting PharmCAT processing...")
    print(f"Input folder: {args.input_folder}")
    print(f"Result folder: {args.result_folder}")

    [f.unlink() for f in working_folder.glob("*") if f.is_file()]

    [sh.copy(f, working_folder / f.name) for f in input_folder.glob("*")]

    for filename in os.listdir(working_folder):
        if filename.endswith(".vcf"):
            input_path = working_folder / filename
            print(f"Processing {filename}...")
            subprocess.run(
                ['/pharmcat/pharmcat_pipeline', input_path, '-o', '/tmp/pharmcat', '-reporterJson', '--missing-to-ref', '-matcher', '-phenotyper'],
                capture_output=True, text=True, check=True
            )

    all_samples = {}
    for filename in os.listdir('/tmp/pharmcat'):
        if filename.endswith(".phenotype.json"):
            sh.copy(working_folder / filename, output_folder / filename)
            print(f"Post-processing {filename}...")
            ID = filename.split('_')[0]
            input_path = working_folder / filename
            gene_dict = process_phenotype_data(input_path)
            all_samples[ID] = gene_dict

    df = pd.DataFrame.from_dict(all_samples, orient="index")
    df.index.name = "Sample ID"
    df.reset_index(inplace=True)
    df.to_csv(output_folder / "phenotypes.csv", index=False)

if __name__ == "__main__":
    main()
