import kfp
from kfp import dsl
from kfp.dsl import Dataset, Input, Output, Model


@kfp.dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["requests"]
)
def download_pharmcat_project(
        github_repo_url: str,
        project_files: Output[Model],
        input_data: Output[Dataset],
        demographic_data: Output[Dataset],
        branch: str = "main"
):
    import subprocess
    import shutil
    import re
    from pathlib import Path

    branch_match = re.search(r"/tree/([^/]+)", github_repo_url)
    if branch_match:
        url_branch = branch_match.group(1)
        print(f"Extracted branch '{url_branch}' from URL")
        branch = url_branch
    repo_url = re.sub(r"/tree/[^/]+/?$", "", github_repo_url.strip())
    if not repo_url.endswith(".git"):
        repo_url = repo_url.rstrip("/") + ".git"
    print(f"Original URL: {github_repo_url}")
    print(f"Using branch: {branch}")
    print(f"Using repository URL: {repo_url}")
    print("Installing git...")
    subprocess.run(["apt-get", "update"], check=True, capture_output=True, text=True)
    subprocess.run(["apt-get", "install", "-y", "git"], check=True, capture_output=True, text=True)
    print("Git installed.")
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

    project_files_path = Path(project_files.path)
    input_data_path = Path(input_data.path)
    demographic_data_path = Path(demographic_data.path)

    project_files_path.mkdir(parents=True, exist_ok=True)
    input_data_path.mkdir(parents=True, exist_ok=True)
    demographic_data_path.mkdir(parents=True, exist_ok=True)
    print(f"Created KFP output directories.")

    items_to_copy = {
        "project_files": [
            "pgx_fairness_analyzer.py",
            "pgx_shap_analyzer.py",
            "requirements.txt"
        ],
        "input_data": ["data"],
        "demographic_data": ["Demographics"]
    }
    found_items = {key: [] for key in items_to_copy}
    copied_items_count = 0

    print(f"\nSearching for required items in cloned repo: {temp_dir}")
    for target_list_key, target_items in items_to_copy.items():
        for item_name in target_items:
            src_path = temp_dir / item_name
            dst_path = None
            is_dir = src_path.is_dir()
            is_file = src_path.is_file()

            if not (is_dir or is_file):
                print(f"  Warning: Item '{item_name}' not found at {src_path}")
                continue

            try:
                if target_list_key == "project_files":
                    dst_path = project_files_path / item_name
                    if is_dir:
                        print(f"Copying directory '{item_name}' from {src_path} to {dst_path}...")
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    else:
                        print(f"Copying file '{item_name}' from {src_path} to {dst_path}...")
                        shutil.copy2(src_path, dst_path)

                elif target_list_key == "input_data":
                    dst_path = input_data_path
                    if is_dir:
                        print(f"Copying contents of '{item_name}' from {src_path} to {dst_path}...")
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    else:
                        print(f"  Warning: Expected '{item_name}' to be a directory, but it's a file.")
                        continue

                elif target_list_key == "demographic_data":
                    dst_path = demographic_data_path
                    if is_dir:
                        print(f"Copying contents of '{item_name}' from {src_path} to {dst_path}...")
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    else:
                        print(f"  Warning: Expected '{item_name}' to be a directory, but it's a file.")
                        continue

                if dst_path:
                    found_items[target_list_key].append(item_name)
                    copied_items_count += 1
                    print(f"  Successfully copied '{item_name}' to {dst_path}")

            except Exception as e:
                print(f"  Error copying {item_name} from {src_path} to {dst_path}: {e}")

    print(f"\nFinished search. Copied {copied_items_count} items.")

    print("\nVerifying copied items:")
    missing_items = False
    for target_list_key, target_items in items_to_copy.items():
        print(f"  Checking {target_list_key}:")
        for item in target_items:
            expected_path = None
            is_content_check = False
            if target_list_key == "project_files":
                expected_path = project_files_path / item
            elif target_list_key == "input_data":
                expected_path = input_data_path
                is_content_check = True
            elif target_list_key == "demographic_data":
                expected_path = demographic_data_path
                is_content_check = True

            if expected_path:
                if is_content_check:
                    if not any(expected_path.iterdir()):
                        print(f"    - {item}: Content not found or empty in {expected_path}!")
                        missing_items = True
                    else:
                        print(f"    + {item}: Content found in {expected_path}")
                elif expected_path.exists():
                    print(f"    + {item}: Found at {expected_path}")
                else:
                    print(f"    - {item}: Not found at {expected_path}!")
                    missing_items = True
            else:
                print(f"    ? {item}: Could not determine expected path.")
                missing_items = True

    if missing_items:
        print("\nWarning: Some required files/folders were not found in the repository or failed to copy.")
    else:
        print("\nAll required items successfully copied.")

    print(f"Removing temporary clone directory: {temp_dir}")
    shutil.rmtree(temp_dir)
    print("Download component finished.")


@dsl.container_component
def run_pharmcat_analysis_docker(
        input_folder: dsl.Input[dsl.Dataset],
        result_folder: dsl.Output[dsl.Dataset]
):
    command_str = f"mkdir -p '{result_folder.path}' && python3 -u /scripts/pharmcat_folder_processor.py --input_folder '{input_folder.path}' --result_folder '{result_folder.path}'"
    return dsl.ContainerSpec(
        image="<docker_image_link>",  # Insert your Docker image here (e.g. "docker.io/<username>/pharmcat-realm:latest")
        command=["sh", "-c"],
        args=[command_str]
    )


@kfp.dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["matplotlib", "numpy", "pandas", "scikit-learn", "seaborn", "shap"]
)
def run_shap_analysis(
        project_files: Input[Model],
        input_data: Input[Dataset],
        pharmcat_results: Input[Dataset],
        shap_results: Output[Dataset],
        sensitivity: float = 0.7
):
    import subprocess
    from pathlib import Path
    project_files_path = Path(project_files.path)
    input_data_path = Path(input_data.path)
    pharmcat_results_path = Path(pharmcat_results.path)
    shap_results_path = Path(shap_results.path)
    shap_results_path.mkdir(parents=True, exist_ok=True)
    print(f"Ensured SHAP results directory exists: {shap_results_path}")
    shap_script = project_files_path / "pgx_shap_analyzer.py"
    if not shap_script.is_file(): raise Exception(f"SHAP analyzer script not found at {shap_script}")
    print(f"Found SHAP script: {shap_script}")
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
    phenotypes_csv = pharmcat_results_path / "phenotypes.csv"
    if not phenotypes_csv.is_file(): raise Exception(
        f"Phenotypes CSV not found at {phenotypes_csv}. Cannot run SHAP analysis.")
    print(f"Found phenotypes file: {phenotypes_csv}")

    # Added sensitivity parameter to the command
    command = [
        "python", str(shap_script),
        "--phenotypes_file", str(phenotypes_csv),
        "--output_dir", str(shap_results_path),
        "--input_dir", str(input_data_path),
        "--convert_vcf",  # Convert VCF to CSV
        "--sensitivity", str(sensitivity)  # Add sensitivity parameter for fuzzy logic
    ]

    print(f"\nRunning explanability analysis command with sensitivity={sensitivity}: {' '.join(command)}")
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print("Explanability analysis script executed successfully.")
    except subprocess.CalledProcessError as e:
        print("\n--- Error running explanability analysis script ---")
        print(f"Exit Code: {e.returncode}")
        print(f"stdout:\n{e.stdout}")
        print(f"stderr:\n{e.stderr}")
        raise Exception("Explanability analysis script failed.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during explanability analysis: {e}")
        raise
    print("\nVerifying explanability analysis output...")
    expected_json = shap_results_path / "pgx_shap_results.json"
    expected_preprocessed_dir = shap_results_path / "preprocessed"
    if not expected_json.is_file():
        raise Exception("Explanability analysis failed: pgx_shap_results.json not found")
    else:
        print(f"Found expected output file: {expected_json}")
    if not expected_preprocessed_dir.is_dir():
        print(f"Warning: Expected 'preprocessed' directory not found in results: {expected_preprocessed_dir}")
    else:
        print(f"Found expected output directory: {expected_preprocessed_dir}")
        preprocessed_files = list(expected_preprocessed_dir.glob("*.csv"))
        if not preprocessed_files:
            print(f"  Warning: 'preprocessed' directory exists but contains no CSV files.")
        else:
            print(f"  Found {len(preprocessed_files)} CSV files in 'preprocessed'.")
    print("\nExplanability analysis component completed successfully.")


@kfp.dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["numpy", "pandas", "scipy"]
)
def run_fairness_analysis(
        project_files: Input[Model],
        demographic_data: Input[Dataset],
        pharmcat_results: Input[Dataset],
        fairness_results: Output[Dataset]
):
    import subprocess
    from pathlib import Path
    project_files_path = Path(project_files.path)
    demographic_data_path = Path(demographic_data.path)
    pharmcat_results_path = Path(pharmcat_results.path)
    fairness_results_path = Path(fairness_results.path)
    fairness_results_path.mkdir(parents=True, exist_ok=True)
    print(f"Ensured Fairness results directory exists: {fairness_results_path}")
    fairness_script = project_files_path / "pgx_fairness_analyzer.py"
    if not fairness_script.is_file(): raise Exception(f"Fairness analyzer script not found at {fairness_script}")
    print(f"Found Fairness script: {fairness_script}")
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
    demo_file_path = demographic_data_path / "pgx_cohort.csv"
    if not demo_file_path.is_file():
        print(f"pgx_cohort.csv not found directly in {demographic_data_path}, searching recursively...")
        found_files = list(demographic_data_path.rglob("pgx_cohort.csv"))
        if not found_files: raise Exception(f"Demographic file pgx_cohort.csv not found within {demographic_data_path}")
        demo_file_path = found_files[0]
    print(f"Using demographic file: {demo_file_path}")
    phenotypes_csv = pharmcat_results_path / "phenotypes.csv"
    if not phenotypes_csv.is_file(): raise Exception(
        f"Phenotypes CSV not found at {phenotypes_csv}. Cannot run Fairness analysis.")
    print(f"Found phenotypes file: {phenotypes_csv}")
    command = [
        "python", str(fairness_script),
        "--demographic_file", str(demo_file_path),
        "--phenotypes_file", str(phenotypes_csv),
        "--output_dir", str(fairness_results_path)
    ]
    print(f"\nRunning Fairness analysis command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print("Fairness analysis script executed successfully.")
    except subprocess.CalledProcessError as e:
        print("\n--- Error running Fairness analysis script ---")
        print(f"Exit Code: {e.returncode}")
        print(f"stdout:\n{e.stdout}")
        print(f"stderr:\n{e.stderr}")
        raise Exception("Fairness analysis script failed.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during Fairness analysis: {e}")
        raise
    print("\nVerifying Fairness analysis output...")
    expected_json = fairness_results_path / "overall_fairness_report.json"
    if not expected_json.is_file():
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


@kfp.dsl.pipeline(
    name="PharmCAT PGx Analysis Pipeline (Dockerized)",
    description="Pipeline using pre-built Docker image for PharmCAT analysis"
)
def pharmcat_pipeline(
        github_repo_url: str,
        branch: str = "main",
        sensitivity: float = 0.7  # Add sensitivity parameter with default value of 0.7
):
    download_task = download_pharmcat_project(
        github_repo_url=github_repo_url,
        branch=branch
    )
    download_task.set_caching_options(False)

    pharmcat_task = run_pharmcat_analysis_docker(
        input_folder=download_task.outputs["input_data"]
    )
    pharmcat_task.set_caching_options(False)
    pharmcat_task.set_cpu_request("2")
    pharmcat_task.set_cpu_limit("4")
    pharmcat_task.set_memory_request("4G")
    pharmcat_task.set_memory_limit("8G")

    shap_task = run_shap_analysis(
        project_files=download_task.outputs["project_files"],
        input_data=download_task.outputs["input_data"],
        pharmcat_results=pharmcat_task.outputs["result_folder"],
        sensitivity=sensitivity
    )
    shap_task.set_caching_options(False)
    shap_task.set_cpu_request("2")
    shap_task.set_cpu_limit("4")
    shap_task.set_memory_request("4G")
    shap_task.set_memory_limit("8G")

    fairness_task = run_fairness_analysis(
        project_files=download_task.outputs["project_files"],
        demographic_data=download_task.outputs["demographic_data"],
        pharmcat_results=pharmcat_task.outputs["result_folder"]
    )
    fairness_task.set_caching_options(False)
    fairness_task.set_cpu_request("2")
    fairness_task.set_cpu_limit("4")
    fairness_task.set_memory_request("4G")
    fairness_task.set_memory_limit("8G")


if __name__ == "__main__":
    compiler = kfp.compiler.Compiler()
    compiler.compile(pipeline_func=pharmcat_pipeline, package_path="pharmcat_pipeline_.yaml")
