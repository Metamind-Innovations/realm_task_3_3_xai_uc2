import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd

from helper_scripts.extract_phenotypes import process_phenotype_data


def setup_pharmcat():
    """Setup PharmCAT environment"""
    os.makedirs('/tmp/pharmcat', exist_ok=True)
    os.makedirs('/pharmcat', exist_ok=True)

    # Set permissions
    for directory in ['/tmp/pharmcat', '/pharmcat']:
        os.chmod(directory, 0o777)

    # Call the PharmCAT setup from Dockerfile
    subprocess.run(['apt-get', 'update'], check=True)
    subprocess.run([
        'apt-get', 'install', '-y',
        'wget',
        'openjdk-17-jre-headless'
    ], check=True)

    # Download PharmCAT
    subprocess.run([
        'wget', '-O', '/pharmcat/pharmcat.jar',
        'https://github.com/PharmGKB/PharmCAT/releases/download/v2.15.1/pharmcat-2.15.1.jar'
    ], check=True)

    # Create PharmCAT wrapper script
    with open('/pharmcat/pharmcat_pipeline', 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('java -jar /pharmcat/pharmcat.jar "$@"\n')

    os.chmod('/pharmcat/pharmcat_pipeline', 0o755)


def process_vcf(file_path, output_dir):
    """Process a single VCF file"""
    try:
        result = subprocess.run(
            ['/pharmcat/pharmcat_pipeline', str(file_path),
             '-o', str(output_dir),
             '-reporterJson', '--missing-to-ref',
             '-matcher', '-phenotyper'],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"PharmCAT output for {file_path}:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing {file_path}:")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Process files through PharmCAT")
    parser.add_argument("--input_folder", required=True, help="Input folder path")
    parser.add_argument("--result_folder", required=True, help="Output folder path")
    args = parser.parse_args()

    print("Starting PharmCAT processing...")
    print(f"Input folder: {args.input_folder}")
    print(f"Result folder: {args.result_folder}")

    # Setup PharmCAT
    print("\nSetting up PharmCAT environment...")
    setup_pharmcat()

    # Create output directory
    os.makedirs(args.result_folder, exist_ok=True)

    # Process each VCF file
    results = {}
    vcf_files = [f for f in Path(args.input_folder).glob("*.vcf")]
    print(f"\nFound {len(vcf_files)} VCF files to process")

    for vcf_file in vcf_files:
        print(f"\nProcessing {vcf_file.name}...")
        if process_vcf(vcf_file, args.result_folder):
            sample_id = vcf_file.stem.split('_')[0]
            json_file = Path(args.result_folder) / f"{vcf_file.stem}.phenotype.json"

            if json_file.exists():
                results[sample_id] = process_phenotype_data(str(json_file))
            else:
                print(f"Warning: No phenotype JSON found for {sample_id}")

    # Generate final CSV
    if results:
        df = pd.DataFrame.from_dict(results, orient='index')
        df.index.name = "Sample ID"
        df.reset_index(inplace=True)

        csv_path = Path(args.result_folder) / "phenotypes.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSuccessfully generated {csv_path}")
    else:
        raise Exception("No results were generated")


if __name__ == "__main__":
    main()
