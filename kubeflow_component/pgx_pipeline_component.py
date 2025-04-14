import kfp
from kfp.dsl import Dataset, Input, Output, Model


# Component 1: Download project files
@kfp.dsl.component(
    base_image="python:3.10-slim",  # Use Python 3.10 instead of 3.12
    packages_to_install=["requests"]
)
def download_pharmcat_project(
        github_repo_url: str,
        project_files: Output[Model],
        input_data: Output[Dataset],
        demographic_data: Output[Dataset],
        branch: str = "main"  # Default branch if not in URL
):
    import os
    import subprocess
    import shutil
    import re
    import glob

    os.makedirs(project_files.path, exist_ok=True)
    os.makedirs(input_data.path, exist_ok=True)
    os.makedirs(os.path.join(demographic_data.path, "Demographics"), exist_ok=True)

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
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(["apt-get", "install", "-y", "git"], check=True)

    # Clone the repository
    temp_dir = "/tmp/pharmcat_repo"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    # Use the git clone command with explicit branch
    clone_cmd = ["git", "clone", "-b", branch, repo_url, temp_dir]
    print(f"Running: {' '.join(clone_cmd)}")

    try:
        subprocess.run(clone_cmd, check=True, capture_output=True, text=True)
        print(f"Successfully cloned branch '{branch}' to {temp_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Git clone failed with error: {e.stderr}")
        raise Exception(f"Failed to clone repository: {repo_url}, branch: {branch}")

    # Find key files recursively
    print("Searching for key files...")

    # Find pgx_shap_analyzer.py
    shap_files = glob.glob(os.path.join(temp_dir, "**", "pgx_shap_analyzer.py"), recursive=True)
    if shap_files:
        shutil.copy2(shap_files[0], os.path.join(project_files.path, "pgx_shap_analyzer.py"))
        print(f"Copied pgx_shap_analyzer.py from {shap_files[0]}")
    else:
        print("Warning: pgx_shap_analyzer.py not found in repository")

    # Find pgx_fairness_analyzer.py
    fairness_files = glob.glob(os.path.join(temp_dir, "**", "pgx_fairness_analyzer.py"), recursive=True)
    if fairness_files:
        shutil.copy2(fairness_files[0], os.path.join(project_files.path, "pgx_fairness_analyzer.py"))
        print(f"Copied pgx_fairness_analyzer.py from {fairness_files[0]}")
    else:
        print("Warning: pgx_fairness_analyzer.py not found in repository")

    # Find requirements.txt
    req_files = glob.glob(os.path.join(temp_dir, "**", "requirements.txt"), recursive=True)
    if req_files:
        shutil.copy2(req_files[0], os.path.join(project_files.path, "requirements.txt"))
        print(f"Copied requirements.txt from {req_files[0]}")
    else:
        print("Warning: requirements.txt not found in repository")

    # Find scripts directory
    scripts_dirs = []
    for root, dirs, _ in os.walk(temp_dir):
        for dir_name in dirs:
            if dir_name == "scripts":
                scripts_dirs.append(os.path.join(root, dir_name))

    if scripts_dirs:
        scripts_src = scripts_dirs[0]
        scripts_dst = os.path.join(project_files.path, "scripts")
        shutil.copytree(scripts_src, scripts_dst)
        print(f"Copied scripts directory from {scripts_src}")
    else:
        print("Warning: scripts directory not found in repository")

    # Find data directory with VCF files
    vcf_files = glob.glob(os.path.join(temp_dir, "**", "*.vcf"), recursive=True)

    if vcf_files:
        # Copy all VCF files to input_data
        for vcf_file in vcf_files:
            shutil.copy2(vcf_file, os.path.join(input_data.path, os.path.basename(vcf_file)))
        print(f"Copied {len(vcf_files)} VCF files to input_data")
    else:
        print("Warning: No VCF files found in repository")

    # Find Demographics directory or pgx_cohort.csv
    demo_files = glob.glob(os.path.join(temp_dir, "**", "pgx_cohort.csv"), recursive=True)
    if demo_files:
        # Copy the pgx_cohort.csv file
        demo_file = demo_files[0]
        demo_dir = os.path.dirname(demo_file)

        # If the file is in a Demographics directory, copy the entire directory
        if os.path.basename(demo_dir) == "Demographics":
            for item in os.listdir(demo_dir):
                src_item = os.path.join(demo_dir, item)
                dst_item = os.path.join(demographic_data.path, "Demographics", item)
                if os.path.isdir(src_item):
                    shutil.copytree(src_item, dst_item)
                else:
                    shutil.copy2(src_item, dst_item)
            print(f"Copied Demographics directory from {demo_dir}")
        else:
            # Just copy the pgx_cohort.csv file
            shutil.copy2(demo_file, os.path.join(demographic_data.path, "Demographics", "pgx_cohort.csv"))
            print(f"Copied pgx_cohort.csv from {demo_file}")
    else:
        print("Warning: Demographics data (pgx_cohort.csv) not found in repository")


@kfp.dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas"]
)
def run_pharmcat_analysis(
        project_files: Input[Model],
        input_data: Input[Dataset],
        pharmcat_results: Output[Dataset]
):
    import os
    import subprocess
    import shutil
    from pathlib import Path

    # Create results directory
    os.makedirs(pharmcat_results.path, exist_ok=True)

    # Install Java which is needed for PharmCAT
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(["apt-get", "install", "-y", "openjdk-17-jre-headless", "wget"], check=True)

    # Create PharmCAT directories
    os.makedirs("/pharmcat", exist_ok=True)
    os.makedirs("/tmp/pharmcat", exist_ok=True)

    # Download PharmCAT jar with the correct filename
    subprocess.run([
        "wget", "-O", "/pharmcat/pharmcat.jar",
        "https://github.com/PharmGKB/PharmCAT/releases/download/v2.15.1/pharmcat-2.15.1-all.jar"
    ], check=True)

    # Create PharmCAT wrapper script
    with open("/pharmcat/pharmcat_pipeline", "w") as f:
        f.write("#!/bin/bash\n")
        f.write('java -jar /pharmcat/pharmcat.jar "$@"\n')

    os.chmod("/pharmcat/pharmcat_pipeline", 0o755)

    # Copy the necessary scripts
    scripts_dir = os.path.join(project_files.path, "scripts")
    if not os.path.exists(scripts_dir):
        raise Exception(f"Scripts directory not found: {scripts_dir}")

    # Create scripts directory structure
    os.makedirs("/scripts", exist_ok=True)
    os.makedirs("/scripts/helper_scripts", exist_ok=True)

    # Copy processor script
    processor_script = os.path.join(scripts_dir, "pharmcat_folder_processor.py")
    if not os.path.exists(processor_script):
        raise Exception(f"PharmCAT processor script not found: {processor_script}")

    # First, copy the script
    shutil.copy2(processor_script, "/scripts/pharmcat_folder_processor.py")

    # Now modify it to fix the JAR path
    with open("/scripts/pharmcat_folder_processor.py", "r") as f:
        content = f.read()

    # Replace the JAR filename in the setup_pharmcat function
    modified_content = content.replace(
        "pharmcat-2.15.1.jar",
        "pharmcat-2.15.1-all.jar"
    )

    # Also modify the setup_pharmcat function to skip download if file exists
    modified_content = modified_content.replace(
        'subprocess.run([\n        "wget", "-O", "/pharmcat/pharmcat.jar",',
        'if not os.path.exists("/pharmcat/pharmcat.jar"):\n        subprocess.run([\n            "wget", "-O", "/pharmcat/pharmcat.jar",'
    )

    with open("/scripts/pharmcat_folder_processor.py", "w") as f:
        f.write(modified_content)

    # Copy helper scripts
    helper_scripts_dir = os.path.join(scripts_dir, "helper_scripts")
    if os.path.exists(helper_scripts_dir):
        for helper_script in os.listdir(helper_scripts_dir):
            src_path = os.path.join(helper_scripts_dir, helper_script)
            dst_path = os.path.join("/scripts/helper_scripts", helper_script)
            shutil.copy2(src_path, dst_path)

    # Run the PharmCAT processor script
    try:
        print("\nRunning PharmCAT processor...")
        result = subprocess.run([
            "python", "/scripts/pharmcat_folder_processor.py",
            "--input_folder", input_data.path,
            "--result_folder", pharmcat_results.path
        ], capture_output=True, text=True, check=True)

        print("\nPharmCAT Processor Output:")
        print(result.stdout)

        if result.stderr:
            print("\nPharmCAT Processor Warnings/Errors:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print("\nError running PharmCAT processor:")
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)
        raise Exception("Failed to process VCF files")

    # Verify results
    output_files = list(Path(pharmcat_results.path).glob("*"))
    print("\nOutput files generated:")
    for f in output_files:
        print(f"- {f}")

    phenotypes_csv = os.path.join(pharmcat_results.path, "phenotypes.csv")
    if not os.path.exists(phenotypes_csv):
        raise Exception("Expected phenotypes.csv not found in results")

    print("\nPharmCAT processing completed successfully")


# Component 3: Run SHAP analysis
@kfp.dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["matplotlib", "numpy", "pandas", "seaborn", "shap"]
)
def run_shap_analysis(
        project_files: Input[Model],
        input_data: Input[Dataset],
        pharmcat_results: Input[Dataset],
        shap_results: Output[Dataset]
):
    import os
    import subprocess

    os.makedirs(shap_results.path, exist_ok=True)

    # Check if analysis script exists
    shap_script = os.path.join(project_files.path, "pgx_shap_analyzer.py")
    if not os.path.exists(shap_script):
        raise Exception(f"SHAP analyzer script not found at {shap_script}")

    # Install requirements
    req_file = os.path.join(project_files.path, "requirements.txt")
    if os.path.exists(req_file):
        subprocess.run(["pip", "install", "-r", req_file], check=True)

    # Check if phenotypes file exists
    phenotypes_csv = os.path.join(pharmcat_results.path, "phenotypes.csv")
    if not os.path.exists(phenotypes_csv):
        raise Exception(f"Phenotypes CSV not found at {phenotypes_csv}")

    # Run SHAP analysis
    command = [
        "python", shap_script,
        "--input_dir", input_data.path,
        "--phenotypes_file", phenotypes_csv,
        "--output_dir", shap_results.path,
        "--convert_vcf"
    ]
    print(f"Running command: {' '.join(command)}")

    subprocess.run(command, check=True)

    if not os.path.exists(os.path.join(shap_results.path, "pgx_shap_results.json")):
        raise Exception("SHAP analysis failed: pgx_shap_results.json not found")

    print("SHAP analysis completed successfully")


# Component 4: Run fairness analysis
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
    import os
    import subprocess
    import glob

    os.makedirs(fairness_results.path, exist_ok=True)

    # Check if fairness script exists
    fairness_script = os.path.join(project_files.path, "pgx_fairness_analyzer.py")
    if not os.path.exists(fairness_script):
        raise Exception(f"Fairness analyzer script not found at {fairness_script}")

    # Install requirements
    req_file = os.path.join(project_files.path, "requirements.txt")
    if os.path.exists(req_file):
        subprocess.run(["pip", "install", "-r", req_file], check=True)

    # Find demographic file
    demo_file = os.path.join(demographic_data.path, "Demographics", "pgx_cohort.csv")
    if not os.path.exists(demo_file):
        # Try to find it recursively
        demo_files = glob.glob(os.path.join(demographic_data.path, "**", "pgx_cohort.csv"), recursive=True)
        if not demo_files:
            raise Exception(f"Demographic file pgx_cohort.csv not found in {demographic_data.path}")
        demo_file = demo_files[0]

    print(f"Using demographic file: {demo_file}")

    # Check if phenotypes file exists
    phenotypes_csv = os.path.join(pharmcat_results.path, "phenotypes.csv")
    if not os.path.exists(phenotypes_csv):
        raise Exception(f"Phenotypes CSV not found at {phenotypes_csv}")

    # Run fairness analysis
    command = [
        "python", fairness_script,
        "--demographic_file", demo_file,
        "--phenotypes_file", phenotypes_csv,
        "--output_dir", fairness_results.path
    ]
    print(f"Running command: {' '.join(command)}")

    subprocess.run(command, check=True)

    if not os.path.exists(os.path.join(fairness_results.path, "overall_fairness_report.json")):
        raise Exception("Fairness analysis failed: overall_fairness_report.json not found")

    print("Fairness analysis completed successfully")


# Pipeline to tie everything together
@kfp.dsl.pipeline(
    name="PharmCAT PGx Analysis Pipeline",
    description="Pipeline for pharmacogenomics analysis with PharmCAT, SHAP, and fairness evaluation"
)
def pharmcat_pipeline(
        github_repo_url: str,
        branch: str = "main"
):
    download_task = download_pharmcat_project(
        github_repo_url=github_repo_url,
        branch=branch
    )

    pharmcat_task = run_pharmcat_analysis(
        project_files=download_task.outputs["project_files"],
        input_data=download_task.outputs["input_data"]
    )
    pharmcat_task.set_cpu_request("2")
    pharmcat_task.set_cpu_limit("4")
    pharmcat_task.set_memory_request("4G")
    pharmcat_task.set_memory_limit("8G")

    shap_task = run_shap_analysis(
        project_files=download_task.outputs["project_files"],
        input_data=download_task.outputs["input_data"],
        pharmcat_results=pharmcat_task.outputs["pharmcat_results"]
    )
    shap_task.set_cpu_request("2")
    shap_task.set_cpu_limit("4")
    shap_task.set_memory_request("4G")
    shap_task.set_memory_limit("8G")

    fairness_task = run_fairness_analysis(
        project_files=download_task.outputs["project_files"],
        demographic_data=download_task.outputs["demographic_data"],
        pharmcat_results=pharmcat_task.outputs["pharmcat_results"]
    )
    fairness_task.set_cpu_request("2")
    fairness_task.set_cpu_limit("4")
    fairness_task.set_memory_request("4G")
    fairness_task.set_memory_limit("8G")


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=pharmcat_pipeline,
        package_path="pharmcat_pipeline.yaml"
    )
