import argparse
import json  # Import json for loading report
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

# --- Helper Script Import Logic ---
try:
    # Assumes helper_scripts is in a directory structure accessible via sys.path
    # In KFP, this relies on the scripts being copied to /scripts
    from helper_scripts.extract_phenotypes import process_phenotype_data
except ImportError:
    # Fallback logic if the import fails initially
    current_dir = Path(__file__).parent.resolve()
    helper_scripts_path_local = current_dir / "helper_scripts"
    helper_scripts_path_kfp = Path("/scripts/helper_scripts")  # Standard path in KFP component

    if helper_scripts_path_local.is_dir():
        # If running locally and helper_scripts is relative
        sys.path.append(str(current_dir))
        print(f"Added {current_dir} to sys.path for helper_scripts")
        try:
            from helper_scripts.extract_phenotypes import process_phenotype_data

            print("Imported process_phenotype_data from local helper_scripts.")
        except ImportError as e:
            print(f"ImportError even after adding local path: {e}")
            process_phenotype_data = None  # Mark as unavailable
    elif helper_scripts_path_kfp.is_dir():
        # If running in KFP context where scripts are copied to /scripts
        sys.path.append("/scripts")
        print(f"Added /scripts to sys.path for helper_scripts")
        try:
            from helper_scripts.extract_phenotypes import process_phenotype_data

            print("Imported process_phenotype_data from KFP /scripts/helper_scripts.")
        except ImportError as e:
            print(f"ImportError even after adding /scripts path: {e}")
            process_phenotype_data = None  # Mark as unavailable
    else:
        print("Error: Could not find helper_scripts directory.")
        process_phenotype_data = None  # Mark as unavailable

    # Define a placeholder if import failed completely
    if process_phenotype_data is None:
        def process_phenotype_data(json_file_path):
            print(f"Warning: process_phenotype_data function not loaded. Cannot process {json_file_path}")
            # Basic fallback extraction attempt
            try:
                with open(json_file_path, 'r') as f:
                    data = json.load(f)
                phenotypes = data.get("geneResults", {})
                flat_phenos = {gene: result.get("phenotype") for gene, result in phenotypes.items() if
                               isinstance(result, dict)}
                if flat_phenos:
                    print("Performed basic fallback phenotype extraction.")
                    return flat_phenos
            except Exception as e:
                print(f"Fallback extraction failed: {e}")
            return None


# --- PharmCAT Setup Function ---
def setup_pharmcat():
    """Setup PharmCAT environment."""
    print("Creating PharmCAT directories...")
    os.makedirs('/tmp/pharmcat', exist_ok=True)
    os.makedirs('/pharmcat', exist_ok=True)

    try:
        # Set permissions cautiously. 0o777 might be too permissive.
        # Consider 0o755 if sufficient.
        os.chmod('/tmp/pharmcat', 0o777)
        os.chmod('/pharmcat', 0o777)
    except OSError as e:
        print(f"Warning: Could not set permissions on /tmp/pharmcat or /pharmcat: {e}")

    print("Updating package list and installing dependencies (wget, openjdk)...")
    try:
        # Run apt-get quietly, show output only on error
        update_result = subprocess.run(['apt-get', 'update', '-qq'], check=True, capture_output=True, text=True)
        install_result = subprocess.run([
            'apt-get', 'install', '-y', '-qq', '--no-install-recommends',
            'wget', 'openjdk-17-jre-headless'
        ], check=True, capture_output=True, text=True)
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        # Log output from the failed command
        print(f"apt-get stdout: {e.stdout}")
        print(f"apt-get stderr: {e.stderr}")
        raise  # Stop execution if dependencies fail

    pharmcat_jar_path = '/pharmcat/pharmcat.jar'
    pharmcat_jar_url = 'https://github.com/PharmGKB/PharmCAT/releases/download/v2.15.1/pharmcat-2.15.1-all.jar'

    # Download PharmCAT only if it doesn't exist
    if not os.path.exists(pharmcat_jar_path):
        print(f"Downloading PharmCAT JAR from {pharmcat_jar_url}...")
        try:
            # Use wget with progress bar, capture stderr for errors
            wget_result = subprocess.run(
                ['wget', '--progress=bar:force:noscroll', '-O', pharmcat_jar_path, pharmcat_jar_url],
                check=True, capture_output=True, text=True, stderr=subprocess.PIPE, encoding='utf-8'
            )
            print("PharmCAT JAR downloaded successfully.")
            # print(f"wget stderr:\n{wget_result.stderr}") # Optional: print stderr even on success
        except subprocess.CalledProcessError as e:
            print(f"Error downloading PharmCAT JAR: {e}")
            print(f"wget stdout:\n{e.stdout}")
            print(f"wget stderr:\n{e.stderr}")
            # Clean up potentially incomplete download
            if os.path.exists(pharmcat_jar_path): os.remove(pharmcat_jar_path)
            raise  # Stop execution if download fails
    else:
        print(f"PharmCAT JAR already exists at {pharmcat_jar_path}. Skipping download.")

    # Create PharmCAT wrapper script
    wrapper_script_path = '/pharmcat/pharmcat_pipeline'
    print(f"Creating PharmCAT wrapper script at {wrapper_script_path}...")
    try:
        with open(wrapper_script_path, 'w') as f:
            f.write('#!/bin/bash\n')
            # Use exec to replace the shell process with the java process
            f.write(f'exec java -jar {pharmcat_jar_path} "$@"\n')
        os.chmod(wrapper_script_path, 0o755)  # Make executable
        print("PharmCAT wrapper script created successfully.")
    except IOError as e:
        print(f"Error creating wrapper script: {e}")
        raise  # Stop execution if wrapper fails

    print("PharmCAT setup complete.")


# --- VCF Processing Function (Corrected expected JSON filename) ---
def process_vcf(file_path: Path, output_dir: Path) -> bool:
    """Process a single VCF file using the PharmCAT wrapper script."""
    pharmcat_executable = '/pharmcat/pharmcat_pipeline'
    # Define expected output JSON based on input filename stem
    # ***** CORRECTED AGAIN: Use .phenotype.json based on user logs *****
    expected_json_filename = f"{file_path.stem}.phenotype.json"
    expected_json_path = output_dir / expected_json_filename

    # Construct the command to run PharmCAT
    command = [pharmcat_executable,
               '-vcf', str(file_path),
               '-o', str(output_dir),
               # Flags used previously, assuming they generate .phenotype.json now
               '-reporterJson',
               '-matcher',
               '-phenotyper']

    print(f"Executing PharmCAT for {file_path.name}: {' '.join(command)}")
    try:
        # Run PharmCAT subprocess
        result = subprocess.run(
            command,
            capture_output=True,  # Capture stdout/stderr
            text=True,  # Decode output as text
            check=True,  # Raise error on non-zero exit code
            encoding='utf-8',  # Specify encoding
            timeout=600  # Set timeout (10 minutes)
        )
        # If check=True passes, PharmCAT exited with 0
        print(f"  PharmCAT ran successfully for {file_path.name} (exit code 0).")

        # Explicitly check if the expected JSON file exists *after* successful run
        if not expected_json_path.is_file():
            print(f"  Error: PharmCAT completed successfully, but expected output JSON not found: {expected_json_path}")
            # Log details to help diagnose why the file is missing
            print(f"  Listing files in output directory ({output_dir}):")
            try:
                files_in_output = list(output_dir.iterdir())
                if not files_in_output:
                    print("    Output directory is empty.")
                else:
                    for item in files_in_output: print(f"    - {item.name}")
            except Exception as list_err:
                print(f"    Could not list directory: {list_err}")
            # Log PharmCAT output which might contain clues
            print(f"  PharmCAT stdout:\n{result.stdout}")
            if result.stderr: print(f"  PharmCAT stderr:\n{result.stderr}")
            return False  # Indicate failure for this file

        # If the file exists, verification passed
        print(f"  Verified expected output JSON exists: {expected_json_path}")
        return True

    # Handle specific errors from subprocess.run
    except subprocess.TimeoutExpired as e:
        print(f"  Error: PharmCAT process timed out for {file_path.name} after {e.timeout} seconds.")
        print(f"  stdout (partial):\n{e.stdout}")
        print(f"  stderr (partial):\n{e.stderr}")
        return False
    except subprocess.CalledProcessError as e:
        # This catches non-zero exit codes if check=True is used
        print(f"  Error: PharmCAT process failed for {file_path.name} with exit code {e.returncode}.")
        print(f"  stdout:\n{e.stdout}")
        print(f"  stderr:\n{e.stderr}")
        return False
    except FileNotFoundError:
        # This means the pharmcat_executable itself wasn't found
        print(f"  Fatal Error: PharmCAT executable not found at {pharmcat_executable}. Was setup successful?")
        raise  # Re-raise to stop the entire script, as this is a setup issue
    except Exception as e:
        # Catch any other unexpected errors during subprocess execution
        print(f"  An unexpected error occurred during PharmCAT execution for {file_path.name}: {e}")
        return False


# --- Main Function (Corrected expected JSON filename) ---
def main():
    parser = argparse.ArgumentParser(description="Process VCF files in a folder using PharmCAT.")
    parser.add_argument("--input_folder", required=True, help="Path to the folder containing input VCF files.")
    parser.add_argument("--result_folder", required=True,
                        help="Path to the folder where PharmCAT results should be saved.")
    args = parser.parse_args()

    input_path = Path(args.input_folder)
    output_path = Path(args.result_folder)

    print("--- Starting PharmCAT Folder Processor ---")
    print(f"Input folder: {input_path}")
    print(f"Result folder: {output_path}")

    # --- Setup PharmCAT Environment ---
    print("\n--- Setting up PharmCAT Environment ---")
    try:
        setup_pharmcat()
    except Exception as e:
        print(f"Fatal error during PharmCAT setup: {e}")
        sys.exit(1)  # Exit if setup fails

    # --- Prepare Output Directory ---
    print(f"\n--- Preparing Output Directory ---")
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Ensured output directory exists: {output_path}")
    except OSError as e:
        print(f"Error creating output directory {output_path}: {e}")
        sys.exit(1)  # Exit if output dir can't be created

    # --- Process VCF Files ---
    print("\n--- Processing VCF Files ---")
    try:
        vcf_files = list(input_path.glob("*.vcf"))
    except Exception as e:
        print(f"Error listing VCF files in {input_path}: {e}")
        sys.exit(1)

    if not vcf_files:
        print(f"Warning: No VCF files found in {input_path}.")
        # Create empty phenotypes.csv and exit successfully if no input is okay
        phenotypes_csv_path = output_path / "phenotypes.csv"
        print(f"Creating empty {phenotypes_csv_path} as no VCFs were found.")
        pd.DataFrame(columns=['Sample ID']).to_csv(phenotypes_csv_path, index=False)
        print("--- PharmCAT Folder Processor Finished (No VCFs) ---")
        sys.exit(0)

    print(f"Found {len(vcf_files)} VCF files to process.")

    phenotype_results = {}  # Dictionary to store extracted phenotypes {sample_id: {gene: phenotype, ...}}
    processed_count = 0  # Count of VCFs where PharmCAT ran successfully
    error_count = 0  # Count of VCFs where PharmCAT failed or JSON was missing
    extraction_success_count = 0  # Count of successfully extracted phenotype dicts

    # Loop through each VCF file found
    for vcf_file in vcf_files:
        print(f"\nProcessing {vcf_file.name}...")
        # Call process_vcf to run PharmCAT and check for the output JSON
        if process_vcf(vcf_file, output_path):
            processed_count += 1
            # If PharmCAT ran and JSON exists, try to extract data
            sample_id = vcf_file.stem  # Use filename stem (e.g., "NA18952_freebayes") as ID
            # ***** CORRECTED AGAIN: Use .phenotype.json based on user logs *****
            json_file_name = f"{sample_id}.phenotype.json"
            json_file = output_path / json_file_name

            # Double-check file exists before processing (should be guaranteed by process_vcf)
            if json_file.is_file():
                print(f"  Attempting to extract phenotypes from {json_file.name}...")
                # Check if the helper function was imported successfully
                if process_phenotype_data is None:
                    print(f"  Error: Phenotype extraction function (process_phenotype_data) not available.")
                    # Decide how to handle this - maybe count as an error?
                    continue  # Skip to next VCF file

                try:
                    # Call the imported helper function
                    pheno_data = process_phenotype_data(str(json_file))
                    # Validate the returned data
                    if pheno_data is not None and isinstance(pheno_data, dict) and pheno_data:
                        phenotype_results[sample_id] = pheno_data
                        extraction_success_count += 1
                        print(f"  Successfully extracted phenotypes for {sample_id}.")
                    # Log cases where extraction didn't yield expected data
                    elif pheno_data is None:
                        print(f"  Warning: Phenotype extraction function returned None for {json_file.name}.")
                    else:
                        print(
                            f"  Warning: Phenotype extraction for {json_file.name} returned unexpected data type or empty dict: {type(pheno_data)}")
                except Exception as e:
                    # Catch errors specifically from the extraction function
                    print(f"  Error during phenotype extraction for {json_file.name}: {e}")
                    # Optionally, count this as an error depending on requirements
            else:
                # This block should ideally not be reached if process_vcf logic is correct
                print(
                    f"  Internal Error: Expected JSON file {json_file.name} not found after process_vcf reported success.")
                error_count += 1  # Count this logic error
        else:
            # process_vcf returned False (PharmCAT failed or JSON missing)
            error_count += 1
            print(f"  PharmCAT processing failed or output JSON was missing for {vcf_file.name}.")

    # --- Processing Summary ---
    print(f"\n--- VCF Processing Summary ---")
    print(f"Total VCFs found: {len(vcf_files)}")
    print(f"PharmCAT runs successful (exit 0 + JSON found): {processed_count}")
    print(f"PharmCAT runs failed (exit non-0 or JSON missing): {error_count}")
    print(f"Phenotype extractions successful: {extraction_success_count}")

    # --- Aggregate Results ---
    phenotypes_csv_path = output_path / "phenotypes.csv"
    # Only proceed if phenotype data was actually extracted
    if phenotype_results:
        print("\n--- Aggregating Phenotype Results ---")
        try:
            # Create DataFrame from the dictionary of dictionaries
            df = pd.DataFrame.from_dict(phenotype_results, orient='index')
            df.index.name = "Sample ID"  # Name the index column
            df.reset_index(inplace=True)  # Turn index into a regular column
            # Save to CSV
            df.to_csv(phenotypes_csv_path, index=False)
            print(f"Successfully aggregated {len(phenotype_results)} samples to {phenotypes_csv_path}")
        except Exception as e:
            print(f"Error aggregating results or saving CSV: {e}")
            # If aggregation fails, the CSV might be missing or incomplete. Exit with error.
            sys.exit(1)
    # Handle cases where processing occurred but no data was extracted
    elif processed_count > 0:
        print("\nWarning: PharmCAT processed files, but no phenotype data could be extracted.")
        print(f"Creating empty {phenotypes_csv_path} as placeholder.")
        pd.DataFrame(columns=['Sample ID']).to_csv(phenotypes_csv_path, index=False)
    # Handle cases where no processing happened or all failed
    else:
        print("\nNo VCF files were successfully processed or no phenotype data was extracted.")
        print(f"Creating empty {phenotypes_csv_path} as placeholder.")

    # --- Final Check and Exit ---
    print("\n--- Final Check ---")
    exit_code = 1  # Default to failure
    if phenotypes_csv_path.is_file():
        print(f"Verified final output file exists: {phenotypes_csv_path}")
        # Basic check: is the file non-empty if we expected results?
        file_size = os.path.getsize(phenotypes_csv_path)
        if file_size > 50 and phenotype_results:  # Approx header size + some data
            print("Output CSV appears valid and contains data.")
            exit_code = 0  # Success
        elif file_size <= 50 and not phenotype_results:
            print("Output CSV is empty or header-only, as expected (no data extracted).")
            # Consider this success if processing happened but extraction failed,
            # or if no VCFs were processed. If errors occurred, maybe still fail?
            if error_count == 0:  # Success if no PharmCAT errors occurred
                exit_code = 0
            else:
                print("Exiting with error code due to PharmCAT processing failures.")
                exit_code = 1
        elif file_size > 50 and not phenotype_results:
            print(
                "Warning: Output CSV is not empty, but no phenotypes were extracted (check helper script/aggregation).")
            # Treat as success for now, but indicates potential issue.
            exit_code = 0
        elif file_size <= 50 and phenotype_results:
            print(
                "Error: Output CSV is empty or header-only, but phenotypes were extracted (check aggregation/saving).")
            exit_code = 1  # Treat as failure
        else:  # Should not happen
            print("Could not determine validity of output CSV.")
            exit_code = 1

    else:
        print(f"Error: Final output file {phenotypes_csv_path} was NOT created.")
        exit_code = 1  # Definite failure

    if exit_code == 0:
        print("\n--- PharmCAT Folder Processor Finished Successfully ---")
    else:
        print("\n--- PharmCAT Folder Processor Finished with Errors ---")

    sys.exit(exit_code)  # Exit with 0 for success, non-zero for failure


if __name__ == "__main__":
    main()
