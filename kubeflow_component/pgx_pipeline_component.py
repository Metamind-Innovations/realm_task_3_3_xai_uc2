import kfp
from kfp.dsl import Dataset, Input, Output, Model


# Component 1: Download project files (Revised)
@kfp.dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["requests"]
)
def download_pharmcat_project(
        github_repo_url: str,
        project_files: Output[Model],  # For scripts, requirements.txt
        input_data: Output[Dataset],  # For 'data' folder contents (VCFs)
        demographic_data: Output[Dataset],  # For 'Demographics' folder contents
        branch: str = "main"
):
    import os
    import subprocess
    import shutil
    import re
    from pathlib import Path  # Use pathlib for easier path manipulation

    # --- Git Clone Logic (remains similar) ---
    # Extract branch from URL if present
    branch_match = re.search(r"/tree/([^/]+)", github_repo_url)
    if branch_match:
        url_branch = branch_match.group(1)
        print(f"Extracted branch '{url_branch}' from URL")
        branch = url_branch

    # Clean URL for git clone format
    repo_url = re.sub(r"/tree/[^/]+/?$", "", github_repo_url.strip())
    if not repo_url.endswith(".git"):
        repo_url = repo_url.rstrip("/") + ".git"

    print(f"Original URL: {github_repo_url}")
    print(f"Using branch: {branch}")
    print(f"Using repository URL: {repo_url}")

    # Install git
    print("Installing git...")
    subprocess.run(["apt-get", "update"], check=True, capture_output=True, text=True)
    subprocess.run(["apt-get", "install", "-y", "git"], check=True, capture_output=True, text=True)
    print("Git installed.")

    # Clone the repository
    temp_dir = Path("/tmp/pharmcat_repo")
    if temp_dir.exists():
        print(f"Removing existing temp directory: {temp_dir}")
        shutil.rmtree(temp_dir)

    clone_cmd = ["git", "clone", "--depth", "1", "-b", branch, repo_url, str(temp_dir)]
    print(f"Running: {' '.join(clone_cmd)}")
    try:
        subprocess.run(clone_cmd, check=True, capture_output=True, text=True)
        print(f"Successfully cloned branch '{branch}' to {temp_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Git clone failed with error: {e.stderr}")
        raise Exception(f"Failed to clone repository: {repo_url}, branch: {branch}")
    # --- End Git Clone Logic ---

    # Define KFP output paths using pathlib
    project_files_path = Path(project_files.path)
    input_data_path = Path(input_data.path)
    demographic_data_path = Path(demographic_data.path)

    # Create output directories
    project_files_path.mkdir(parents=True, exist_ok=True)
    input_data_path.mkdir(parents=True, exist_ok=True)
    demographic_data_path.mkdir(parents=True, exist_ok=True)

    print(f"Created KFP output directories:")
    print(f"  Project Files: {project_files_path}")
    print(f"  Input Data: {input_data_path}")
    print(f"  Demographic Data: {demographic_data_path}")

    # --- Find and Copy Required Files/Folders ---
    items_to_copy = {
        "project_files": [
            "pgx_fairness_analyzer.py",
            "pgx_shap_analyzer.py",
            "requirements.txt",
            "scripts"  # Copy the entire scripts folder here
        ],
        "input_data": ["data"],  # Copy the 'data' folder here (contains VCFs)
        "demographic_data": ["Demographics"]  # Copy the 'Demographics' folder here
    }
    found_items = {key: [] for key in items_to_copy}
    copied_items_count = 0

    print(f"\nSearching for required items in cloned repo: {temp_dir}")
    for root, dirs, files in os.walk(temp_dir):
        current_path = Path(root)
        # Check directories
        for dirname in list(dirs):  # Iterate over a copy to allow removal
            dirpath = current_path / dirname
            rel_path_str = str(dirpath.relative_to(temp_dir))  # Relative path for matching

            for target_list_key, target_items in items_to_copy.items():
                if dirname in target_items:
                    src = dirpath
                    # Determine destination based on key
                    if target_list_key == "project_files":
                        dst = project_files_path / dirname
                    elif target_list_key == "input_data":
                        dst = input_data_path  # Copy *contents* of 'data'
                    elif target_list_key == "demographic_data":
                        dst = demographic_data_path  # Copy *contents* of 'Demographics'
                    else:
                        continue  # Should not happen

                    try:
                        if dst.is_dir():  # If copying contents, ensure target exists
                            print(f"Copying contents of '{src.name}' from {src} to {dst}...")
                            shutil.copytree(src, dst, dirs_exist_ok=True)
                        else:  # If copying the directory itself (like 'scripts')
                            print(f"Copying directory '{src.name}' from {src} to {dst}...")
                            shutil.copytree(src, dst)

                        found_items[target_list_key].append(dirname)
                        copied_items_count += 1
                        # Prevent descending into already copied directories like 'scripts'
                        if dirname in dirs:
                            dirs.remove(dirname)
                        print(f"  Successfully copied '{dirname}' to {dst}")
                    except Exception as e:
                        print(f"  Error copying directory {src} to {dst}: {e}")
                    # Break inner loop once matched to avoid duplicate copies if name exists elsewhere
                    break

        # Check files
        for filename in files:
            filepath = current_path / filename
            rel_path_str = str(filepath.relative_to(temp_dir))

            for target_list_key, target_items in items_to_copy.items():
                if filename in target_items:
                    src = filepath
                    # Determine destination based on key
                    if target_list_key == "project_files":
                        dst = project_files_path / filename
                    # Add other keys if files need to go elsewhere
                    else:
                        continue

                    try:
                        print(f"Copying file '{filename}' from {src} to {dst}...")
                        shutil.copy2(src, dst)
                        found_items[target_list_key].append(filename)
                        copied_items_count += 1
                        print(f"  Successfully copied '{filename}' to {dst}")
                    except Exception as e:
                        print(f"  Error copying file {src} to {dst}: {e}")
                    # Break inner loop once matched
                    break

    print(f"\nFinished search. Copied {copied_items_count} items.")

    # --- Verification ---
    print("\nVerifying copied items:")
    missing_items = False
    for target_list_key, target_items in items_to_copy.items():
        print(f"  Checking {target_list_key}:")
        for item in target_items:
            # Adjust check based on destination structure
            if target_list_key == "project_files":
                expected_path = project_files_path / item
            elif target_list_key == "input_data":
                # Check if input_data_path has content (assuming 'data' was copied there)
                expected_path = input_data_path
                if not any(expected_path.iterdir()):  # Check if directory is not empty
                    print(f"    - {item}: Not found or empty in {expected_path}!")
                    missing_items = True
                    continue  # Skip exists check below if checking for content
            elif target_list_key == "demographic_data":
                # Check if demographic_data_path has content
                expected_path = demographic_data_path
                if not any(expected_path.iterdir()):
                    print(f"    - {item}: Not found or empty in {expected_path}!")
                    missing_items = True
                    continue
            else:
                expected_path = Path("invalid")  # Should not happen

            if expected_path.exists():
                print(f"    + {item}: Found at {expected_path}")
            else:
                print(f"    - {item}: Not found at {expected_path}!")
                missing_items = True

    if missing_items:
        print("\nWarning: Some required files/folders were not found in the repository or failed to copy.")
        # Decide if this should be a fatal error
        # raise Exception("Failed to find all required project files/folders.")
    else:
        print("\nAll required items successfully copied.")

    # Clean up cloned repo
    print(f"Removing temporary clone directory: {temp_dir}")
    shutil.rmtree(temp_dir)
    print("Download component finished.")


# Component 2: Run PharmCAT analysis (Updated Args/Paths)
@kfp.dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas"]
)
def run_pharmcat_analysis(
        project_files: Input[Model],  # Contains 'scripts' folder
        input_data: Input[Dataset],  # Contains VCFs (from 'data' folder)
        pharmcat_results: Output[Dataset]  # Output dir for JSONs and phenotypes.csv
):
    import subprocess
    import shutil
    from pathlib import Path

    project_files_path = Path(project_files.path)
    input_data_path = Path(input_data.path)
    pharmcat_results_path = Path(pharmcat_results.path)

    # Create results directory (pharmcat_folder_processor also does this, but safe to do here)
    pharmcat_results_path.mkdir(parents=True, exist_ok=True)
    print(f"Ensured PharmCAT results directory exists: {pharmcat_results_path}")

    # --- Copy necessary scripts ---
    scripts_src_dir = project_files_path / "scripts"
    scripts_dst_dir = Path("/scripts")  # Destination inside component container

    if not scripts_src_dir.is_dir():
        raise Exception(f"Scripts directory not found in project_files artifact: {scripts_src_dir}")

    # Copy the entire scripts directory
    if scripts_dst_dir.exists():
        shutil.rmtree(scripts_dst_dir)  # Remove existing if necessary
    shutil.copytree(scripts_src_dir, scripts_dst_dir)
    print(f"Copied scripts from {scripts_src_dir} to {scripts_dst_dir}")

    processor_script_dst = scripts_dst_dir / "pharmcat_folder_processor.py"
    if not processor_script_dst.is_file():
        raise Exception(f"PharmCAT processor script not found after copy: {processor_script_dst}")

    # --- Run the PharmCAT processor script ---
    # It handles its own setup (Java, wget, PharmCAT download)
    try:
        print("\nRunning PharmCAT processor script...")
        print(f"  Input VCFs folder: {input_data_path}")
        print(f"  Output results folder: {pharmcat_results_path}")

        # Execute the copied script using its path inside the container
        result = subprocess.run([
            "python", str(processor_script_dst),
            "--input_folder", str(input_data_path),
            "--result_folder", str(pharmcat_results_path)
        ], capture_output=True, text=True, check=True, encoding='utf-8')  # Added encoding

        print("\n--- PharmCAT Processor Output ---")
        print(result.stdout)
        if result.stderr:
            print("\n--- PharmCAT Processor Warnings/Errors ---")
            print(result.stderr)
        print("--- End PharmCAT Processor Output ---")

    except subprocess.CalledProcessError as e:
        print("\n--- Error running PharmCAT processor script ---")
        print(f"Exit Code: {e.returncode}")
        print("stdout:")
        print(e.stdout)
        print("stderr:")
        print(e.stderr)
        print("------------------------------------------------")
        # List files in output dir upon error for debugging
        print(f"Contents of results directory ({pharmcat_results_path}) upon error:")
        try:
            for item in pharmcat_results_path.iterdir():
                print(f"  - {item.name}")
        except Exception as list_err:
            print(f"  Could not list directory contents: {list_err}")
        raise Exception("PharmCAT processor script failed.")
    except Exception as e:
        print(f"\nAn unexpected error occurred while running the processor script: {e}")
        raise

    # --- Final Verification ---
    print("\nVerifying PharmCAT analysis output...")
    output_files = list(pharmcat_results_path.glob("*"))
    print("Files generated in results directory:")
    for f in output_files:
        print(f"  - {f.name}")

    phenotypes_csv = pharmcat_results_path / "phenotypes.csv"
    if not phenotypes_csv.is_file():
        print(f"Error: Expected phenotypes.csv not found in results directory: {phenotypes_csv}")
        print("PharmCAT processing likely failed to produce the final aggregated CSV.")
        raise Exception("Expected phenotypes.csv not found in results")
    else:
        print("Found phenotypes.csv.")

    print("\nPharmCAT analysis component completed successfully.")


# Component 3: Run SHAP analysis (Updated Args/Paths)
@kfp.dsl.component(
    base_image="python:3.10-slim",
    # Ensure all necessary packages for pgx_shap_analyzer.py are listed
    packages_to_install=["matplotlib", "numpy", "pandas", "seaborn", "shap"]
)
def run_shap_analysis(
        project_files: Input[Model],  # Contains pgx_shap_analyzer.py, requirements.txt
        input_data: Input[Dataset],  # Contains original VCFs (if needed by SHAP script)
        pharmcat_results: Input[Dataset],  # Contains phenotypes.csv
        shap_results: Output[Dataset]  # Output dir for SHAP results
):
    import subprocess
    from pathlib import Path

    project_files_path = Path(project_files.path)
    input_data_path = Path(input_data.path)  # Original VCFs location
    pharmcat_results_path = Path(pharmcat_results.path)  # PharmCAT output location
    shap_results_path = Path(shap_results.path)  # SHAP output location

    # Create SHAP output directory
    shap_results_path.mkdir(parents=True, exist_ok=True)
    print(f"Ensured SHAP results directory exists: {shap_results_path}")

    # --- Prepare environment ---
    # Check if analysis script exists
    shap_script = project_files_path / "pgx_shap_analyzer.py"
    if not shap_script.is_file():
        raise Exception(f"SHAP analyzer script not found at {shap_script}")
    print(f"Found SHAP script: {shap_script}")

    # Install requirements if requirements.txt exists
    req_file = project_files_path / "requirements.txt"
    if req_file.is_file():
        print(f"Installing requirements from {req_file}...")
        try:
            subprocess.run(["pip", "install", "-r", str(req_file)], check=True, capture_output=True, text=True)
            print("Requirements installed.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing requirements: {e.stderr}")
            raise Exception("Failed to install requirements for SHAP analysis.")
    else:
        print("No requirements.txt found in project_files, skipping pip install.")

    # Check if phenotypes file exists
    phenotypes_csv = pharmcat_results_path / "phenotypes.csv"
    if not phenotypes_csv.is_file():
        # This check might be redundant if the previous step already failed, but good practice
        raise Exception(f"Phenotypes CSV not found at {phenotypes_csv}. Cannot run SHAP analysis.")
    print(f"Found phenotypes file: {phenotypes_csv}")

    # --- Run SHAP analysis script ---
    # Adjust command based on the actual arguments needed by pgx_shap_analyzer.py
    command = [
        "python", str(shap_script),
        "--phenotypes_file", str(phenotypes_csv),  # Input phenotypes
        "--output_dir", str(shap_results_path),  # Output directory for SHAP results
        # Optional: Pass original data if needed by the script
        "--input_dir", str(input_data_path),
        # Optional: Add other flags like --convert_vcf if required
        # "--convert_vcf"
    ]
    print(f"\nRunning SHAP analysis command: {' '.join(command)}")

    try:
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print("SHAP analysis script executed successfully.")
    except subprocess.CalledProcessError as e:
        print("\n--- Error running SHAP analysis script ---")
        print(f"Exit Code: {e.returncode}")
        print("stdout:")
        print(e.stdout)
        print("stderr:")
        print(e.stderr)
        print("-----------------------------------------")
        raise Exception("SHAP analysis script failed.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during SHAP analysis: {e}")
        raise

    # --- Verification ---
    # Check for expected output files/folders as per user requirement
    print("\nVerifying SHAP analysis output...")
    expected_json = shap_results_path / "pgx_shap_results.json"
    expected_preprocessed_dir = shap_results_path / "preprocessed"

    if not expected_json.is_file():
        print(f"Error: Expected SHAP output file not found: {expected_json}")
        raise Exception("SHAP analysis failed: pgx_shap_results.json not found")
    else:
        print(f"Found expected output file: {expected_json}")

    if not expected_preprocessed_dir.is_dir():
        print(f"Warning: Expected 'preprocessed' directory not found in SHAP results: {expected_preprocessed_dir}")
        # Decide if this is critical - maybe the script doesn't always create it?
    else:
        print(f"Found expected output directory: {expected_preprocessed_dir}")
        # Optionally, check for CSV files inside 'preprocessed'
        preprocessed_files = list(expected_preprocessed_dir.glob("*.csv"))
        if not preprocessed_files:
            print(f"  Warning: 'preprocessed' directory exists but contains no CSV files.")
        else:
            print(f"  Found {len(preprocessed_files)} CSV files in 'preprocessed'.")

    print("\nSHAP analysis component completed successfully.")


# Component 4: Run fairness analysis (Updated Args/Paths)
@kfp.dsl.component(
    base_image="python:3.10-slim",
    # Ensure all necessary packages for pgx_fairness_analyzer.py are listed
    packages_to_install=["numpy", "pandas", "scipy"]
)
def run_fairness_analysis(
        project_files: Input[Model],  # Contains pgx_fairness_analyzer.py, requirements.txt
        demographic_data: Input[Dataset],  # Contains 'Demographics' folder contents
        pharmcat_results: Input[Dataset],  # Contains phenotypes.csv
        fairness_results: Output[Dataset]  # Output dir for fairness results
):
    import subprocess
    from pathlib import Path

    project_files_path = Path(project_files.path)
    demographic_data_path = Path(demographic_data.path)  # Root path for demographics artifact
    pharmcat_results_path = Path(pharmcat_results.path)  # PharmCAT output location
    fairness_results_path = Path(fairness_results.path)  # Fairness output location

    # Create Fairness output directory
    # Note: The user wants the output in "pgx_fairness_results", but the artifact name is "fairness_results"
    # The component will write to fairness_results.path. The internal script should handle naming if needed.
    fairness_results_path.mkdir(parents=True, exist_ok=True)
    print(f"Ensured Fairness results directory exists: {fairness_results_path}")

    # --- Prepare environment ---
    # Check if analysis script exists
    fairness_script = project_files_path / "pgx_fairness_analyzer.py"
    if not fairness_script.is_file():
        raise Exception(f"Fairness analyzer script not found at {fairness_script}")
    print(f"Found Fairness script: {fairness_script}")

    # Install requirements if requirements.txt exists
    req_file = project_files_path / "requirements.txt"
    if req_file.is_file():
        print(f"Installing requirements from {req_file}...")
        try:
            subprocess.run(["pip", "install", "-r", str(req_file)], check=True, capture_output=True, text=True)
            print("Requirements installed.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing requirements: {e.stderr}")
            raise Exception("Failed to install requirements for Fairness analysis.")
    else:
        print("No requirements.txt found in project_files, skipping pip install.")

    # Find demographic file (pgx_cohort.csv) within the demographic_data artifact path
    # The download step now copies the *contents* of the repo's Demographics folder here.
    demo_file_path = demographic_data_path / "pgx_cohort.csv"
    if not demo_file_path.is_file():
        # Fallback: search recursively within the artifact path (shouldn't be needed if download is correct)
        print(f"pgx_cohort.csv not found directly in {demographic_data_path}, searching recursively...")
        found_files = list(demographic_data_path.rglob("pgx_cohort.csv"))
        if not found_files:
            raise Exception(f"Demographic file pgx_cohort.csv not found within {demographic_data_path}")
        demo_file_path = found_files[0]  # Use the first one found

    print(f"Using demographic file: {demo_file_path}")

    # Check if phenotypes file exists
    phenotypes_csv = pharmcat_results_path / "phenotypes.csv"
    if not phenotypes_csv.is_file():
        raise Exception(f"Phenotypes CSV not found at {phenotypes_csv}. Cannot run Fairness analysis.")
    print(f"Found phenotypes file: {phenotypes_csv}")

    # --- Run fairness analysis script ---
    # Adjust command based on the actual arguments needed by pgx_fairness_analyzer.py
    command = [
        "python", str(fairness_script),
        "--demographic_file", str(demo_file_path),  # Input demographics
        "--phenotypes_file", str(phenotypes_csv),  # Input phenotypes
        "--output_dir", str(fairness_results_path)  # Output directory for fairness results
    ]
    print(f"\nRunning Fairness analysis command: {' '.join(command)}")

    try:
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print("Fairness analysis script executed successfully.")
    except subprocess.CalledProcessError as e:
        print("\n--- Error running Fairness analysis script ---")
        print(f"Exit Code: {e.returncode}")
        print("stdout:")
        print(e.stdout)
        print("stderr:")
        print(e.stderr)
        print("-------------------------------------------")
        raise Exception("Fairness analysis script failed.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during Fairness analysis: {e}")
        raise

    # --- Verification ---
    # Check for expected output files/folders as per user requirement
    # User wants output in "pgx_fairness_results" folder. The script should create this *inside* fairness_results.path
    # Or, the script simply writes files directly into fairness_results.path. Check for a key file.
    print("\nVerifying Fairness analysis output...")
    expected_json = fairness_results_path / "overall_fairness_report.json"  # Assuming this is the key output file

    if not expected_json.is_file():
        print(f"Error: Expected Fairness output file not found: {expected_json}")
        # Check if the script created a subfolder named "pgx_fairness_results"
        expected_subdir = fairness_results_path / "pgx_fairness_results"
        if expected_subdir.is_dir():
            print(f"Found directory {expected_subdir}, checking for report inside...")
            expected_json_in_subdir = expected_subdir / "overall_fairness_report.json"
            if not expected_json_in_subdir.is_file():
                raise Exception(
                    f"Fairness analysis failed: overall_fairness_report.json not found in {expected_subdir}")
            else:
                print(f"Found expected output file in subdirectory: {expected_json_in_subdir}")
        else:
            raise Exception("Fairness analysis failed: overall_fairness_report.json not found")
    else:
        print(f"Found expected output file: {expected_json}")

    print("\nFairness analysis component completed successfully.")


# Pipeline definition (Updated connections)
@kfp.dsl.pipeline(
    name="PharmCAT PGx Analysis Pipeline v3",
    description="Pipeline for pharmacogenomics analysis with PharmCAT, SHAP, and fairness evaluation (Updated paths)"
)
def pharmcat_pipeline(
        github_repo_url: str,
        branch: str = "main"
):
    # Component 1: Download (Outputs project_files, input_data, demographic_data)
    download_task = download_pharmcat_project(
        github_repo_url=github_repo_url,
        branch=branch
    )
    # Caching disabled for download ensures fresh clone if repo changes
    download_task.set_caching_options(False)

    # Component 2: Run PharmCAT
    # Takes project_files (for scripts) and input_data (for VCFs)
    # Outputs pharmcat_results (containing phenotypes.csv)
    pharmcat_task = run_pharmcat_analysis(
        project_files=download_task.outputs["project_files"],
        input_data=download_task.outputs["input_data"]
    )
    # Disable caching if PharmCAT run should always execute
    pharmcat_task.set_caching_options(False)
    # Resource requests
    pharmcat_task.set_cpu_request("2")
    pharmcat_task.set_cpu_limit("4")
    pharmcat_task.set_memory_request("4G")
    pharmcat_task.set_memory_limit("8G")

    # Component 3: Run SHAP
    # Takes project_files (for script), input_data (if needed), pharmcat_results (for phenotypes.csv)
    # Outputs shap_results
    shap_task = run_shap_analysis(
        project_files=download_task.outputs["project_files"],
        input_data=download_task.outputs["input_data"],  # Pass original VCFs if needed
        pharmcat_results=pharmcat_task.outputs["pharmcat_results"]
    )
    shap_task.set_caching_options(False)  # Disable caching if needed
    # Resource requests
    shap_task.set_cpu_request("2")
    shap_task.set_cpu_limit("4")
    shap_task.set_memory_request("4G")
    shap_task.set_memory_limit("8G")

    # Component 4: Run Fairness
    # Takes project_files (for script), demographic_data, pharmcat_results (for phenotypes.csv)
    # Outputs fairness_results
    fairness_task = run_fairness_analysis(
        project_files=download_task.outputs["project_files"],
        demographic_data=download_task.outputs["demographic_data"],
        pharmcat_results=pharmcat_task.outputs["pharmcat_results"]
    )
    fairness_task.set_caching_options(False)  # Disable caching if needed
    # Resource requests
    fairness_task.set_cpu_request("2")
    fairness_task.set_cpu_limit("4")
    fairness_task.set_memory_request("4G")
    fairness_task.set_memory_limit("8G")


# Compile the pipeline
if __name__ == "__main__":
    compiler = kfp.compiler.Compiler()
    # Consider using v2 compiler features if needed/available
    # from kfp import dsl
    # compiler.compile(pipeline_func=pharmcat_pipeline, package_path="pharmcat_pipeline_v3.yaml", type_check=True)
    compiler.compile(pipeline_func=pharmcat_pipeline, package_path="pharmcat_pipeline_v3.yaml")
    print("Pipeline compiled to pharmcat_pipeline_v3.yaml")
