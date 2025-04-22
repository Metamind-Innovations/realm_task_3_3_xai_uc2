import kfp
from kfp import dsl
from kfp.dsl import Dataset, Input, Output, Model


@kfp.dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["requests"]
)
def download_project(
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
            "pgx_analyzer.py",
            "requirements.txt",
            "pgx_visualizer.py"
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
def pharmcat_analysis_docker(
        input_folder: dsl.Input[dsl.Dataset],
        result_folder: dsl.Output[dsl.Dataset]
):
    command_str = f"mkdir -p '{result_folder.path}' && python3 -u /scripts/pharmcat_folder_processor.py --input_folder '{input_folder.path}' --result_folder '{result_folder.path}'"
    return dsl.ContainerSpec(
        image="<your_docker_pharmcat_image>",  # Insert your Docker image here (e.g. "docker.io/<username>/pharmcat-realm:latest")
        command=["sh", "-c"],
        args=[command_str]
    )


@kfp.dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["matplotlib", "numpy", "pandas", "scikit-learn", "seaborn", "shap"]
)
def explainability_analysis(
        project_files: Input[Model],
        input_data: Input[Dataset],
        pharmcat_results: Input[Dataset],
        results: Output[Dataset],
        sensitivity: float = 0.7,
        method: str = None,
        max_samples: int = -1,
        run_counterfactual: bool = True,
        run_rule_extraction: bool = True,
        top_k: int = 10
):
    import subprocess
    from pathlib import Path
    project_files_path = Path(project_files.path)
    input_data_path = Path(input_data.path)
    pharmcat_results_path = Path(pharmcat_results.path)
    results_path = Path(results.path)
    results_path.mkdir(parents=True, exist_ok=True)
    print(f"Ensured results directory exists: {results_path}")
    analyzer_script = project_files_path / "pgx_analyzer.py"
    if not analyzer_script.is_file(): raise Exception(f"Analyzer script not found at {analyzer_script}")
    print(f"Found analyzer script: {analyzer_script}")
    req_file = project_files_path / "requirements.txt"
    if req_file.is_file():
        print(f"Installing requirements from {req_file}...")
        try:
            subprocess.run(["pip", "install", "-r", str(req_file)], check=True, capture_output=True, text=True)
            print("Requirements installed.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing requirements: {e.stderr}")
            raise Exception("Failed to install requirements for analysis.")
    else:
        print("No requirements.txt found in project_files, skipping pip install.")
    phenotypes_csv = pharmcat_results_path / "phenotypes.csv"
    if not phenotypes_csv.is_file(): raise Exception(
        f"Phenotypes CSV not found at {phenotypes_csv}. Cannot run analysis.")
    print(f"Found phenotypes file: {phenotypes_csv}")

    # Build the command with all parameters
    command = [
        "python", str(analyzer_script),
        "--phenotypes_file", str(phenotypes_csv),
        "--output_dir", str(results_path),
        "--input_dir", str(input_data_path),
        "--convert_vcf",  # Convert VCF to CSV
        "--sensitivity", str(sensitivity),  # Add sensitivity parameter for fuzzy logic
        "--max_samples", str(max_samples),  # Add max_samples parameter
        "--top_k", str(top_k)  # Add top_k parameter for counterfactual analysis
    ]

    if method:
        command.extend(["--method", method])

    # Add optional flags
    if run_counterfactual:
        command.append("--run_counterfactual")

    if run_rule_extraction:
        command.append("--run_rule_extraction")

    print(
        f"\nRunning explanability analysis command with sensitivity={sensitivity}, max_samples={max_samples}: {' '.join(command)}")
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
    expected_json = results_path / "pgx_results.json"
    expected_preprocessed_dir = results_path / "preprocessed"
    if not expected_json.is_file():
        raise Exception("Explanability analysis failed: pgx_results.json not found")
    else:
        print(f"Found expected output file: {expected_json}")

    # Check for additional output files
    if run_counterfactual:
        counterfactual_json = results_path / "counterfactual_analysis.json"
        if not counterfactual_json.is_file():
            print(f"Warning: Expected 'counterfactual_analysis.json' not found: {counterfactual_json}")
        else:
            print(f"Found counterfactual analysis output: {counterfactual_json}")

    if run_rule_extraction:
        rule_json = results_path / "rule_extraction.json"
        if not rule_json.is_file():
            print(f"Warning: Expected 'rule_extraction.json' not found: {rule_json}")
        else:
            print(f"Found rule extraction output: {rule_json}")

        # Look for decision tree images
        tree_images = list(results_path.glob("*_decision_tree.png"))
        if not tree_images:
            print(f"Warning: No decision tree images found in results directory")
        else:
            print(f"Found {len(tree_images)} decision tree images")

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
def fairness_analysis(
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


@kfp.dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["matplotlib", "numpy", "pandas", "networkx", "seaborn"]
)
def visualization_analysis(
        project_files: Input[Model],
        analysis_results: Input[Dataset],
        visualization_results: Output[Dataset]
):
    import subprocess
    from pathlib import Path

    project_files_path = Path(project_files.path)
    analysis_results_path = Path(analysis_results.path)
    visualization_results_path = Path(visualization_results.path)
    visualization_results_path.mkdir(parents=True, exist_ok=True)

    print(f"Ensured Visualization results directory exists: {visualization_results_path}")

    visualizer_script = project_files_path / "pgx_visualizer.py"
    if not visualizer_script.is_file():
        raise Exception(f"Visualizer script not found at {visualizer_script}")

    print(f"Found Visualizer script: {visualizer_script}")

    # Find the required input files in the analysis results directory
    pgx_results_file = analysis_results_path / "pgx_results.json"
    counterfactual_file = analysis_results_path / "counterfactual_analysis.json"
    rules_file = analysis_results_path / "rule_extraction.json"
    decision_trees_file = analysis_results_path / "decision_trees.json"

    command = [
        "python", str(visualizer_script),
        "--output_dir", str(visualization_results_path)
    ]

    if pgx_results_file.is_file():
        command.extend(["--pgx_results", str(pgx_results_file)])
        print(f"Found PGx results file: {pgx_results_file}")
    else:
        print(f"Warning: PGx results file not found at {pgx_results_file}")

    if counterfactual_file.is_file():
        command.extend(["--counterfactual", str(counterfactual_file)])
        print(f"Found counterfactual analysis file: {counterfactual_file}")
    else:
        print(f"Warning: Counterfactual analysis file not found at {counterfactual_file}")

    if rules_file.is_file():
        command.extend(["--rules", str(rules_file)])
        print(f"Found rule extraction file: {rules_file}")
    else:
        print(f"Warning: Rule extraction file not found at {rules_file}")

    if decision_trees_file.is_file():
        command.extend(["--decision_trees", str(decision_trees_file)])
        print(f"Found decision trees file: {decision_trees_file}")
    else:
        print(f"Warning: Decision trees file not found at {decision_trees_file}")

    print(f"\nRunning Visualization analysis command: {' '.join(command)}")

    try:
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print("Visualization script executed successfully.")
    except subprocess.CalledProcessError as e:
        print("\n--- Error running Visualization script ---")
        print(f"Exit Code: {e.returncode}")
        print(f"stdout:\n{e.stdout}")
        print(f"stderr:\n{e.stderr}")
        raise Exception("Visualization script failed.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during visualization: {e}")
        raise

    print("\nVerifying visualization output...")
    expected_summary = visualization_results_path / "pgx_summary_dashboard.png"
    if not expected_summary.is_file():
        print(f"Warning: Expected summary dashboard not found at {expected_summary}")
    else:
        print(f"Found summary dashboard: {expected_summary}")

    dashboards = list(visualization_results_path.glob("dashboard_*.png"))
    print(f"Found {len(dashboards)} gene-specific dashboards")

    print("\nVisualization component completed successfully.")


@kfp.dsl.pipeline(
    name="PharmCAT PGx Analysis Pipeline (Dockerized)",
    description="Pipeline using pre-built Docker image for PharmCAT analysis"
)
def pharmcat_pipeline(
        github_repo_url: str,
        branch: str = "main",
        sensitivity: float = 0.7,  # Control the blend between explanation methods
        method: str = None,
        max_samples: int = -1,  # -1 means analyze all samples
        run_counterfactual: bool = True,  # Run counterfactual analysis
        run_rule_extraction: bool = True,  # Run rule extraction analysis
        top_k: int = 10  # Number of top features for counterfactual analysis
):
    download_task = download_project(
        github_repo_url=github_repo_url,
        branch=branch
    )
    download_task.set_caching_options(False)

    pharmcat_task = pharmcat_analysis_docker(
        input_folder=download_task.outputs["input_data"]
    )
    pharmcat_task.set_caching_options(False)
    pharmcat_task.set_cpu_request("2")
    pharmcat_task.set_cpu_limit("4")
    pharmcat_task.set_memory_request("4G")
    pharmcat_task.set_memory_limit("8G")

    analysis_task = explainability_analysis(
        project_files=download_task.outputs["project_files"],
        input_data=download_task.outputs["input_data"],
        pharmcat_results=pharmcat_task.outputs["result_folder"],
        sensitivity=sensitivity,
        method=method,
        max_samples=max_samples,
        run_counterfactual=run_counterfactual,
        run_rule_extraction=run_rule_extraction,
        top_k=top_k
    )
    analysis_task.set_caching_options(False)
    analysis_task.set_cpu_request("2")
    analysis_task.set_cpu_limit("4")
    analysis_task.set_memory_request("4G")
    analysis_task.set_memory_limit("8G")

    fairness_task = fairness_analysis(
        project_files=download_task.outputs["project_files"],
        demographic_data=download_task.outputs["demographic_data"],
        pharmcat_results=pharmcat_task.outputs["result_folder"]
    )
    fairness_task.set_caching_options(False)
    fairness_task.set_cpu_request("2")
    fairness_task.set_cpu_limit("4")
    fairness_task.set_memory_request("4G")
    fairness_task.set_memory_limit("8G")

    visualization_task = visualization_analysis(
        project_files=download_task.outputs["project_files"],
        analysis_results=analysis_task.outputs["results"]
    )
    visualization_task.set_caching_options(False)
    visualization_task.set_cpu_request("2")
    visualization_task.set_cpu_limit("4")
    visualization_task.set_memory_request("4G")
    visualization_task.set_memory_limit("8G")


if __name__ == "__main__":
    compiler = kfp.compiler.Compiler()
    compiler.compile(pipeline_func=pharmcat_pipeline, package_path="pharmcat_pipeline_.yaml")
