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
            "vcf_to_csv.py",
            "phenotype_mapper.py",
            "explainer.py",
            "explainer_visualizer.py",
            "fairness_bias_analyzer.py",
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
        image="<your_docker_pharmcat_image>",
        # Insert your Docker image here (e.g. "docker.io/<username>/pharmcat-realm:latest")
        command=["sh", "-c"],
        args=[command_str]
    )


@kfp.dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "numpy"]
)
def vcf_to_csv(
        project_files: Input[Model],
        input_data: Input[Dataset],
        csv_output: Output[Dataset]
):
    import subprocess
    import os
    from pathlib import Path

    project_files_path = Path(project_files.path)
    input_data_path = Path(input_data.path)
    csv_output_path = Path(csv_output.path)

    csv_output_path.mkdir(parents=True, exist_ok=True)
    print(f"Created CSV output directory: {csv_output_path}")

    vcf_to_csv_script = project_files_path / "vcf_to_csv.py"
    if not vcf_to_csv_script.is_file():
        raise Exception(f"VCF to CSV script not found at {vcf_to_csv_script}")

    print(f"Found VCF to CSV script: {vcf_to_csv_script}")

    output_csv_file = csv_output_path / "encoded.csv"

    command = [
        "python", str(vcf_to_csv_script),
        "--input_dir", str(input_data_path),
        "--output_csv", str(output_csv_file)
    ]

    print(f"\nRunning VCF to CSV conversion: {' '.join(command)}")

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print("VCF to CSV conversion completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during VCF to CSV conversion: {e.stdout}\n{e.stderr}")
        raise Exception("VCF to CSV conversion failed.")

    if not output_csv_file.is_file():
        raise Exception(f"Expected output CSV file not found at {output_csv_file}")

    print(f"VCF to CSV conversion successful. Output saved to {output_csv_file}")


@kfp.dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas"]
)
def phenotype_mapper(
        project_files: Input[Model],
        pharmcat_results: Input[Dataset],
        encoded_output: Output[Dataset]
):
    import subprocess
    import os
    from pathlib import Path

    project_files_path = Path(project_files.path)
    pharmcat_results_path = Path(pharmcat_results.path)
    encoded_output_path = Path(encoded_output.path)

    encoded_output_path.mkdir(parents=True, exist_ok=True)
    print(f"Created encoded phenotypes output directory: {encoded_output_path}")

    phenotype_mapper_script = project_files_path / "phenotype_mapper.py"
    if not phenotype_mapper_script.is_file():
        raise Exception(f"Phenotype mapper script not found at {phenotype_mapper_script}")

    print(f"Found phenotype mapper script: {phenotype_mapper_script}")

    phenotypes_file = pharmcat_results_path / "phenotypes.csv"
    if not phenotypes_file.is_file():
        # Try to find phenotypes.csv in subdirectories
        phenotypes_files = list(pharmcat_results_path.glob("**/phenotypes.csv"))
        if phenotypes_files:
            phenotypes_file = phenotypes_files[0]
            print(f"Found phenotypes.csv in subdirectory: {phenotypes_file}")
        else:
            raise Exception(f"Phenotypes CSV file not found at {phenotypes_file} or subdirectories")

    output_file = encoded_output_path / "phenotypes_encoded.csv"

    command = [
        "python", str(phenotype_mapper_script),
        "--input_csv", str(phenotypes_file),
        "--output_csv", str(output_file)
    ]

    print(f"\nRunning phenotype mapping: {' '.join(command)}")

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print("Phenotype mapping completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during phenotype mapping: {e.stdout}\n{e.stderr}")
        raise Exception("Phenotype mapping failed.")

    if not output_file.is_file():
        raise Exception(f"Expected encoded phenotypes file not found at {output_file}")

    print(f"Phenotype mapping successful. Output saved to {output_file}")


@kfp.dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "numpy", "scipy", "scikit-learn"]
)
def run_explainer(
        project_files: Input[Model],
        csv_data: Input[Dataset],
        encoded_phenotypes: Input[Dataset],
        explainer_results: Output[Dataset],
        sensitivity: float = 0.7
):
    import subprocess
    import os
    from pathlib import Path

    project_files_path = Path(project_files.path)
    csv_data_path = Path(csv_data.path)
    encoded_phenotypes_path = Path(encoded_phenotypes.path)
    explainer_results_path = Path(explainer_results.path)

    explainer_results_path.mkdir(parents=True, exist_ok=True)
    print(f"Created explainer results directory: {explainer_results_path}")

    explainer_script = project_files_path / "explainer.py"
    if not explainer_script.is_file():
        raise Exception(f"Explainer script not found at {explainer_script}")

    print(f"Found explainer script: {explainer_script}")

    input_file = csv_data_path / "encoded.csv"
    if not input_file.is_file():
        # Try to find encoded.csv in subdirectories
        input_files = list(csv_data_path.glob("**/encoded.csv"))
        if input_files:
            input_file = input_files[0]
            print(f"Found encoded.csv in subdirectory: {input_file}")
        else:
            raise Exception(f"Encoded CSV file not found at {input_file} or subdirectories")

    output_file = encoded_phenotypes_path / "phenotypes_encoded.csv"
    if not output_file.is_file():
        # Try to find phenotypes_encoded.csv in subdirectories
        output_files = list(encoded_phenotypes_path.glob("**/phenotypes_encoded.csv"))
        if output_files:
            output_file = output_files[0]
            print(f"Found phenotypes_encoded.csv in subdirectory: {output_file}")
        else:
            raise Exception(f"Encoded phenotypes file not found at {output_file} or subdirectories")

    results_dir = explainer_results_path / "explainer_results"

    command = [
        "python", str(explainer_script),
        "--input_file", str(input_file),
        "--output_file", str(output_file),
        "--results_dir", str(results_dir),
        "--sensitivity", str(sensitivity)
    ]

    print(f"\nRunning explainer analysis: {' '.join(command)}")

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print("Explainer analysis completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during explainer analysis: {e.stdout}\n{e.stderr}")
        raise Exception("Explainer analysis failed.")

    # Verify results
    json_files = list(results_dir.glob("*.json"))
    if not json_files:
        raise Exception(f"No result JSON files found in {results_dir}")

    print(f"Explainer analysis successful. Results saved to {results_dir}")


@kfp.dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "matplotlib", "seaborn"]
)
def visualize_results(
        project_files: Input[Model],
        explainer_results: Input[Dataset],
        visualization_output: Output[Dataset]
):
    import subprocess
    import os
    from pathlib import Path

    project_files_path = Path(project_files.path)
    explainer_results_path = Path(explainer_results.path)
    visualization_output_path = Path(visualization_output.path)

    visualization_output_path.mkdir(parents=True, exist_ok=True)
    print(f"Created visualization output directory: {visualization_output_path}")

    visualizer_script = project_files_path / "explainer_visualizer.py"
    if not visualizer_script.is_file():
        raise Exception(f"Visualizer script not found at {visualizer_script}")

    print(f"Found visualizer script: {visualizer_script}")

    # Find the result JSON files
    results_dir = explainer_results_path / "explainer_results"
    if not results_dir.is_dir():
        # Try to find explainer_results in parent directory
        results_dir = explainer_results_path

    # Look for either correlation or mutual information results
    json_files = list(results_dir.glob("*.json"))
    if not json_files:
        raise Exception(f"No result JSON files found in {results_dir}")

    input_file = json_files[0]  # Use the first JSON file found
    print(f"Using input file for visualization: {input_file}")

    command = [
        "python", str(visualizer_script),
        "--input_file", str(input_file),
        "--output_dir", str(visualization_output_path)
    ]

    print(f"\nRunning visualization: {' '.join(command)}")

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print("Visualization completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during visualization: {e.stdout}\n{e.stderr}")
        raise Exception("Visualization failed.")

    # Verify results
    image_files = list(visualization_output_path.glob("*.png"))
    if not image_files:
        raise Exception(f"No visualization images found in {visualization_output_path}")

    print(f"Visualization successful. Images saved to {visualization_output_path}")


@kfp.dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas"]
)
def fairness_bias_analyzer(
        project_files: Input[Model],
        demographic_data: Input[Dataset],
        pharmcat_results: Input[Dataset],
        fairness_results: Output[Dataset]
):
    import subprocess
    import os
    from pathlib import Path

    project_files_path = Path(project_files.path)
    demographic_data_path = Path(demographic_data.path)
    pharmcat_results_path = Path(pharmcat_results.path)
    fairness_results_path = Path(fairness_results.path)

    fairness_results_path.mkdir(parents=True, exist_ok=True)
    print(f"Created fairness results directory: {fairness_results_path}")

    fairness_script = project_files_path / "fairness_bias_analyzer.py"
    if not fairness_script.is_file():
        raise Exception(f"Fairness analyzer script not found at {fairness_script}")

    print(f"Found fairness analyzer script: {fairness_script}")

    # Find required input files
    population_codes_file = demographic_data_path / "population_codes.md"
    if not population_codes_file.is_file():
        population_codes_files = list(demographic_data_path.glob("**/population_codes.md"))
        if population_codes_files:
            population_codes_file = population_codes_files[0]
        else:
            raise Exception(f"Population codes file not found at {population_codes_file} or subdirectories")

    cohort_file = demographic_data_path / "pgx_cohort.csv"
    if not cohort_file.is_file():
        cohort_files = list(demographic_data_path.glob("**/pgx_cohort.csv"))
        if cohort_files:
            cohort_file = cohort_files[0]
        else:
            raise Exception(f"Cohort file not found at {cohort_file} or subdirectories")

    phenotypes_file = pharmcat_results_path / "phenotypes.csv"
    if not phenotypes_file.is_file():
        phenotypes_files = list(pharmcat_results_path.glob("**/phenotypes.csv"))
        if phenotypes_files:
            phenotypes_file = phenotypes_files[0]
        else:
            raise Exception(f"Phenotypes file not found at {phenotypes_file} or subdirectories")

    output_file = fairness_results_path / "fairness_analysis.json"

    command = [
        "python", str(fairness_script),
        "--population-codes", str(population_codes_file),
        "--cohort", str(cohort_file),
        "--phenotypes", str(phenotypes_file),
        "--output", str(output_file)
    ]

    print(f"\nRunning fairness analysis: {' '.join(command)}")

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print("Fairness analysis completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during fairness analysis: {e.stdout}\n{e.stderr}")
        raise Exception("Fairness analysis failed.")

    if not output_file.is_file():
        raise Exception(f"Expected fairness analysis file not found at {output_file}")

    print(f"Fairness analysis successful. Output saved to {output_file}")


@kfp.dsl.pipeline(
    name="PharmCAT PGx Analysis Pipeline",
    description="Pipeline for pharmacogenomic analysis using PharmCAT and SHAP"
)
def pharmcat_pipeline(
        github_repo_url: str,
        branch: str = "main",
        sensitivity: float = 0.7
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

    vcf_to_csv_task = vcf_to_csv(
        project_files=download_task.outputs["project_files"],
        input_data=download_task.outputs["input_data"]
    )
    vcf_to_csv_task.set_caching_options(False)
    vcf_to_csv_task.set_cpu_request("1")
    vcf_to_csv_task.set_cpu_limit("2")
    vcf_to_csv_task.set_memory_request("2G")
    vcf_to_csv_task.set_memory_limit("4G")

    phenotype_mapper_task = phenotype_mapper(
        project_files=download_task.outputs["project_files"],
        pharmcat_results=pharmcat_task.outputs["result_folder"]
    )
    phenotype_mapper_task.set_caching_options(False)
    phenotype_mapper_task.set_cpu_request("1")
    phenotype_mapper_task.set_cpu_limit("2")
    phenotype_mapper_task.set_memory_request("2G")
    phenotype_mapper_task.set_memory_limit("4G")

    explainer_task = run_explainer(
        project_files=download_task.outputs["project_files"],
        csv_data=vcf_to_csv_task.outputs["csv_output"],
        encoded_phenotypes=phenotype_mapper_task.outputs["encoded_output"],
        sensitivity=sensitivity
    )
    explainer_task.set_caching_options(False)
    explainer_task.set_cpu_request("2")
    explainer_task.set_cpu_limit("4")
    explainer_task.set_memory_request("4G")
    explainer_task.set_memory_limit("8G")

    visualize_task = visualize_results(
        project_files=download_task.outputs["project_files"],
        explainer_results=explainer_task.outputs["explainer_results"]
    )
    visualize_task.set_caching_options(False)
    visualize_task.set_cpu_request("1")
    visualize_task.set_cpu_limit("2")
    visualize_task.set_memory_request("2G")
    visualize_task.set_memory_limit("4G")

    fairness_task = fairness_bias_analyzer(
        project_files=download_task.outputs["project_files"],
        demographic_data=download_task.outputs["demographic_data"],
        pharmcat_results=pharmcat_task.outputs["result_folder"]
    )
    fairness_task.set_caching_options(False)
    fairness_task.set_cpu_request("1")
    fairness_task.set_cpu_limit("2")
    fairness_task.set_memory_request("2G")
    fairness_task.set_memory_limit("4G")


if __name__ == "__main__":
    compiler = kfp.compiler.Compiler()
    compiler.compile(pipeline_func=pharmcat_pipeline, package_path="pharmcat_pipeline.yaml")
