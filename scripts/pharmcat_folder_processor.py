import argparse
import os
import subprocess
import sys  # Import sys to potentially add helper_scripts to path
from pathlib import Path

import pandas as pd

# Attempt to import from helper_scripts, adjusting path if necessary
try:
    from helper_scripts.extract_phenotypes import process_phenotype_data
except ImportError:
    # If run directly, helper_scripts might not be in the Python path.
    # Assuming it's in the same directory level as this script.
    current_dir = Path(__file__).parent.resolve()
    helper_scripts_path = current_dir / "helper_scripts"
    if helper_scripts_path.exists():
        sys.path.append(str(current_dir))
        from helper_scripts.extract_phenotypes import process_phenotype_data
    else:
        # If run from the KFP component context, it might be in /scripts/helper_scripts
        helper_scripts_path = Path("/scripts/helper_scripts")
        if helper_scripts_path.exists():
            sys.path.append("/scripts")
            from helper_scripts.extract_phenotypes import process_phenotype_data
        else:
            print("Error: Could not find helper_scripts directory.")


            # Depending on requirements, you might raise an exception here
            # For now, define a placeholder if import fails and it's critical
            def process_phenotype_data(json_file_path):
                print(f"Warning: process_phenotype_data function not loaded. Cannot process {json_file_path}")
                return None  # Or return an empty dict/structure


def setup_pharmcat():
    """Setup PharmCAT environment. This function is now fully responsible for setup."""
    print("Creating PharmCAT directories...")
    os.makedirs('/tmp/pharmcat', exist_ok=True)
    os.makedirs('/pharmcat', exist_ok=True)

    # Set permissions (consider if 0o777 is necessary)
    try:
        os.chmod('/tmp/pharmcat', 0o777)
        os.chmod('/pharmcat', 0o777)
    except OSError as e:
        print(f"Warning: Could not set permissions on /tmp/pharmcat or /pharmcat: {e}")

    print("Updating package list and installing dependencies (wget, openjdk)...")
    try:
        # Update package list
        subprocess.run(['apt-get', 'update'], check=True, capture_output=True, text=True)
        # Install necessary packages
        subprocess.run([
            'apt-get', 'install', '-y', '--no-install-recommends',  # Added --no-install-recommends
            'wget',
            'openjdk-17-jre-headless'
        ], check=True, capture_output=True, text=True)
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise  # Re-raise the exception to stop execution if dependencies fail

    # Define PharmCAT JAR path and URL
    pharmcat_jar_path = '/pharmcat/pharmcat.jar'
    pharmcat_jar_url = 'https://github.com/PharmGKB/PharmCAT/releases/download/v2.15.1/pharmcat-2.15.1-all.jar'

    # Download PharmCAT only if it doesn't exist (Idempotency)
    if not os.path.exists(pharmcat_jar_path):
        print(f"Downloading PharmCAT JAR from {pharmcat_jar_url}...")
        try:
            subprocess.run([
                'wget', '-O', pharmcat_jar_path, pharmcat_jar_url
            ], check=True, capture_output=True, text=True)
            print("PharmCAT JAR downloaded successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading PharmCAT JAR: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            # Clean up potentially incomplete download
            if os.path.exists(pharmcat_jar_path):
                os.remove(pharmcat_jar_path)
            raise  # Re-raise the exception
    else:
        print(f"PharmCAT JAR already exists at {pharmcat_jar_path}. Skipping download.")

    # Create PharmCAT wrapper script
    wrapper_script_path = '/pharmcat/pharmcat_pipeline'
    print(f"Creating PharmCAT wrapper script at {wrapper_script_path}...")
    try:
        with open(wrapper_script_path, 'w') as f:
            f.write('#!/bin/bash\n')
            # Ensure the path to java is correct or rely on PATH
            # Using just 'java' assumes it's in the system's PATH
            f.write(f'java -jar {pharmcat_jar_path} "$@"\n')

        os.chmod(wrapper_script_path, 0o755)  # Make it executable
        print("PharmCAT wrapper script created successfully.")
    except IOError as e:
        print(f"Error creating wrapper script: {e}")
        raise  # Re-raise the exception

    print("PharmCAT setup complete.")


def process_vcf(file_path, output_dir):
    """Process a single VCF file using the PharmCAT wrapper script."""
    # Ensure output directory exists for this specific file's results if needed
    # PharmCAT might handle this, but being explicit can help debugging.
    # os.makedirs(output_dir, exist_ok=True) # Usually PharmCAT creates it

    # Use the wrapper script created in setup_pharmcat
    pharmcat_executable = '/pharmcat/pharmcat_pipeline'
    command = [pharmcat_executable,
               '-vcf', str(file_path),
               '-o', str(output_dir),
               '-reporterJson',  # Ensure these flags are correct for your version
               '-matcher',
               '-phenotyper']

    print(f"Executing PharmCAT command: {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            capture_output=True,  # Capture stdout/stderr
            text=True,  # Decode as text
            check=True  # Raise exception on non-zero exit code
        )
        print(f"PharmCAT ran successfully for {file_path.name}.")
        # Print stdout only if needed for debugging, can be verbose
        # print("PharmCAT stdout:")
        # print(result.stdout)
        if result.stderr:  # Print stderr if it contains anything (warnings etc.)
            print("PharmCAT stderr:")
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        # Log detailed error information
        print(f"Error processing {file_path.name} with PharmCAT.")
        print(f"Exit Code: {e.returncode}")
        print(f"stdout:\n{e.stdout}")
        print(f"stderr:\n{e.stderr}")
        return False
    except FileNotFoundError:
        print(f"Error: PharmCAT executable not found at {pharmcat_executable}. Was setup successful?")
        return False


def main():
    parser = argparse.ArgumentParser(description="Process VCF files in a folder using PharmCAT.")
    parser.add_argument("--input_folder", required=True, help="Path to the folder containing input VCF files.")
    parser.add_argument("--result_folder", required=True,
                        help="Path to the folder where PharmCAT results should be saved.")
    args = parser.parse_args()

    print("--- Starting PharmCAT Folder Processor ---")
    print(f"Input folder: {args.input_folder}")
    print(f"Result folder: {args.result_folder}")

    # --- Setup PharmCAT Environment ---
    # This function now handles installation, download, and wrapper script creation.
    print("\n--- Setting up PharmCAT Environment ---")
    try:
        setup_pharmcat()
    except Exception as e:
        print(f"Fatal error during PharmCAT setup: {e}")
        sys.exit(1)  # Exit if setup fails

    # --- Prepare Output Directory ---
    print(f"\n--- Preparing Output Directory ---")
    output_path = Path(args.result_folder)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Ensured output directory exists: {output_path}")
    except OSError as e:
        print(f"Error creating output directory {output_path}: {e}")
        sys.exit(1)

    # --- Process VCF Files ---
    print("\n--- Processing VCF Files ---")
    input_path = Path(args.input_folder)
    vcf_files = list(input_path.glob("*.vcf"))

    if not vcf_files:
        print(f"Warning: No VCF files found in {input_path}. Exiting.")
        # Decide if this is an error or just an empty run
        sys.exit(0)  # Exit normally if no files is acceptable

    print(f"Found {len(vcf_files)} VCF files to process.")

    phenotype_results = {}
    processed_count = 0
    error_count = 0

    for vcf_file in vcf_files:
        print(f"\nProcessing {vcf_file.name}...")
        # Pass the main result folder; PharmCAT creates subdirs/files within it
        if process_vcf(vcf_file, output_path):
            processed_count += 1
            # Construct the expected JSON output path based on PharmCAT naming conventions
            # Adjust filename pattern if PharmCAT output differs (e.g., includes '.report')
            sample_id = vcf_file.stem  # Use stem (filename without extension) as default ID
            # Example: If VCF is sample1.vcf, expect sample1.phenotyper.json
            # Check PharmCAT docs for exact output naming conventions for v2.15.1
            json_file_name = f"{sample_id}.report.json"  # Common pattern, adjust if needed
            json_file = output_path / json_file_name

            if json_file.exists():
                print(f"Found phenotype JSON: {json_file}")
                # Extract phenotype data using the helper script function
                pheno_data = process_phenotype_data(str(json_file))
                if pheno_data is not None:
                    phenotype_results[sample_id] = pheno_data
                else:
                    print(f"Warning: Phenotype processing failed for {json_file}")
            else:
                # Log if the expected JSON is missing after successful PharmCAT run
                print(
                    f"Warning: Expected phenotype JSON '{json_file}' not found for {vcf_file.name} despite successful run.")
        else:
            error_count += 1
            print(f"PharmCAT processing failed for {vcf_file.name}.")
            # Optionally, collect names of failed files

    print(f"\n--- VCF Processing Summary ---")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed to process: {error_count}")

    # --- Aggregate Results ---
    if phenotype_results:
        print("\n--- Aggregating Phenotype Results ---")
        try:
            # Create DataFrame from the collected phenotype results
            df = pd.DataFrame.from_dict(phenotype_results, orient='index')
            df.index.name = "Sample ID"  # Assign the index name
            df.reset_index(inplace=True)  # Convert index (sample_id) to a column

            # Define the output CSV file path
            csv_path = output_path / "phenotypes.csv"
            df.to_csv(csv_path, index=False)  # Save without pandas index column
            print(f"Successfully aggregated results to {csv_path}")
        except Exception as e:
            print(f"Error aggregating results or saving CSV: {e}")
            # Decide if this should be a fatal error
    elif processed_count > 0:
        print("\nWarning: PharmCAT processed files, but no valid phenotype data could be extracted.")
    else:
        print("\nNo VCF files were successfully processed or no phenotype data found.")
        # Consider exiting with an error if processing was expected
        if error_count > 0:
            sys.exit(1)  # Exit with error if files failed

    print("\n--- PharmCAT Folder Processor Finished ---")


if __name__ == "__main__":
    main()
