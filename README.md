# realm_task_3_3_implementation_xai

## General Task Description

Components developed in Task 3.3 aim to implement agnostic XAI techniques on top of AI models that are used for various tasks such as classification or segmentation.
We aim to implement **two** XAI techniques per Use Case - that would be selected dynamically from the Fuzzy system based on User's Input (sensitivity value coming from the RIANA dashboard), implement bias and fairness metrics (as agreed [here](https://maastrichtuniversity.sharepoint.com/:w:/r/sites/FSE-REALM/_layouts/15/Doc.aspx?sourcedoc=%7B9EDAE561-2787-42D1-BBB8-C9320C0B1F25%7D&file=Report%20on%20Bias%20and%20Fairness%20Metrics%20%5BTask%203.3%5D.docx&action=default&mobileredirect=true)) based on model outputs and extract outputs in a digestible manner (images, metrics, etc.)

This component, no matter the Use Case, expects as input:
- Sensitivity value (RIANA dashboard)
- Trained model (AI Orchestrator)
- Compatible dataset (AI Orchestrator)

This component, no matter the Use Case, returns as output:
- XAI methodology output (depending on the Use Case - image or json file)
- Fairness and Bias results (depending on the Use Case - nothing if we are talking for images or json file)

## Pharmacogenomics (PGx) Analysis with Explainability (Use Case 2)

This project provides tools for analyzing pharmacogenomic phenotype predictions made by PharmCAT and provide explanations regarding its outputs. 
It uses categorical association analysis and mutual information techniques to identify which genetic variants contribute to phenotype predictions and includes fairness analysis to detect potential biases across demographic groups.

## Project Overview

PharmCAT is a tool that analyzes genetic data (VCF files) to predict pharmacogenomic phenotypes that influence drug metabolism and response. 
This project adds an explainability layer to understand why PharmCAT makes specific predictions and to identify potential biases.

**IMPORTANT:** For this use case, since we are using PharmCAT which is a rule based pgx model, and not an actual AI model, XAI is not directly applicable. 
Additionally, the underlying PharmCAT model is dockerized so we have no direct access to it in order to use XAI techniques like SHAP on the model's weights. 
Therefore, we perform mutual information analysis and categorical association analysis only on the provided input (vcf files) and output files (csv file).

Key components:
1. Converting VCF genetic data to analyzable CSV format using the `vcf_to_csv.py` script. This function extracts the essential information from VCF files:
     - Patient identifier
     - Chromosome, position, and variant ID (rsID)
     - Reference and alternate alleles
     - Gene information (from the PX field)
     - Genotype information 
     
   The function parses each non-header line in the VCF file, extracting the relevant fields and organizing them into a pandas DataFrame. This creates a clean, tabular representation of the genetic variants. More information can be seen in the [VCF to CSV Encoding](#VCF-to-CSV-Encoding) section
2. Mapping phenotypes to numeric values for analysis, using the `phenotype_mapping.py` script. The mapping can be seen in the [Numeric Encoding of Phenotypes](#Numeric-Encoding-of-Phenotypes) section 
3. Performing categorical association analysis and mutual information analysis to explain PharmCAT predictions using `explainer.py`
4. Analyzing potential demographic bias and fairness issues in predictions with the `fairness_bias_analyzer.py` script 
5. Visualizing results for better interpretation (Optional script provided. Not used in kubeflow pipeline)

The project focuses on the key pharmacogenes: **CYP2B6, CYP2C9, CYP2C19, CYP3A5, SLCO1B1, TPMT, and DPYD.**

## Getting Started

### Prerequisites

- Python 3.10
- Docker (for running PharmCAT)
- Required Python packages (install via `pip install -r requirements.txt`)

### Data Preparation

1. Place your VCF files in the `data/` directory
2. Ensure demographic data is in `Demographics/` (e.g. `pgx_cohort.csv` and `population_codes.md`)
3. Ground truth data should be placed in `Groundtruth/groundtruth_phenotype_filtered.csv`

### Running PharmCAT

Use Docker to run PharmCAT on your VCF files: `docker run -v /path/to/data/:/data -v /path/to/result/:/result pharmcat-realm --input_folder /data --result_folder /result`
This will generate phenotype predictions in the result directory.

### Execution Steps
1. **Download Dataset:** Download the provided dataset.
2. **Download PharmCAT Docker Components:** Download the necessary tools and scripts.
3. **Setup Execution Environment:**
   1. Create a folder called `pharmcat_execution` and unzip all files downloaded from step 2. into it. 
   2. Inside the `data` folder, place all `.vcf` files downloaded from step 1.
4. Navigate inside the `pharmcat_execution` directory and execute: `docker build -t pharmcat-realm .`
5. _(WINDOWS)_ Execute: `docker run -v <absolute_path_to_data_folder>:/data -v <absolute_path_to_result_folder>:/result pharmcat-realm --input_folder /data --result_folder /result`.
This will execute the pharmcat pipeline for all samples placed in the `data` folder. The `result` folder will contain `phenotype.json` files for each sample and a total `phenotypes.csv` file containing the output in the format:

| Sample ID | CYP2B6 | CYP2C19 | CYP2C9 | CYP3A5 | DPYD | SLCO1B1 |     TPMT      |
|:---------:|:------:|:-------:|:------:|:------:|:----:|:-------:|:-------------:|
|  HG00276  |   RM   |   NM    |   IM   |   PM   |  NM  |   DF    | INDETERMINATE |
|  HG00436  |   IM   |   NM    |   NM   |   PM   |  NM  |   NF    |      NM       |

6. Place the `result` folder in the root of the current project.
7. Execute `vcf_to_csv.py` to convert all files located in the `data` directory to a csv file. This file will be used in the next steps (e.g. `python vcf_to_csv.py --input_dir data/ --output_csv encoded.csv`)
8. Execute `phenotype_mapper.py` to convert the `result/phenotypes.csv` file to an encoded file that will be used later. (e.g. `python phenotype_mapper.py --input_csv result/phenotypes.csv --output_csv phenotypes_encoded.csv`)
9. Run Explainability Analysis using the `explainer.py` script. (e.g. `python explainer.py --input_file encoded.csv --output_file phenotypes_encoded.csv --results_dir explainer_results --sensitivity 0.7`)
   1. The `sensitivity` parameter (0.0-1.0) controls the analysis method:
      - Lower values (<0.5) use mutual information analysis (faster)
      - Higher values (≥0.5) use categorical association analysis (more precise)
10. Fairness and Bias Analysis: `python fairness_bias_analyzer.py --population-codes Demographics/population_codes.md --cohort Demographics/pgx_cohort.csv --phenotypes result/phenotypes.csv --groundtruth Groundtruth/groundtruth_phenotype_filtered.csv --output fairness_analysis.json`
11. Visualize Results (Optional): 
```
# Visualize explainability results
python explainer_visualizer.py --input_file explainer_results/categorical_association_analysis.json --output_dir explainer_visualizations

# Visualize fairness analysis
python pgx_fairness_visualizer.py --input_file fairness_analysis.json --output_dir fairness_visualizations
```
12. _Alternative Pipeline Execution_: Use the Kubeflow pipeline to automate the entire workflow: `python kubeflow_component/pgx_pipeline_component.py`. This will generate a `pharmcat_pipeline_.yaml` file that can be uploaded to a Kubeflow environment for execution. See more details in the [Kubeflow Pipeline Component](#kubeflow-pipeline-component) section. 
**Note:** Use `minikube start --cpus=8 --memory=16384` or a value greater than the default (`cpus=2` and `memory=2048`) in order to provide more computational resources to the container and speed up execution.

## JSON Output

The out of the `explainer.py` script will either be `categorical_association_analysis.json` or `mutual_information_analysis.json`, depending on sensitivity values:

### Categorical Association Analysis

The `categorical_association_analysis.json` file has the following structure:
```json
{
  "Gene": "CYP2C9",
  "Feature": "CYP2C19_rs28399504",
  "Association": 0.9926198253344827,
  "P_Value": 6.305116760146996e-16
}
```

### Mutual Information Analysis

The `mutual_information_analysis.json` file has the following structure:

```json
{
  "Gene": "CYP2B6",
  "Feature": "CYP2B6_rs3745274",
  "Importance": 0.7626478098149698
}
```

### Fairness Analysis

The `fairness_analysis.json` file produced by `fairness_bias_analyzer.py` has a bit more complex structure and will not be displayed here for brevity. The categories used for each of the 2 json objects are `Sex`, `Population` and `Superpopulation`, according to `pgx_cohort.csv` and `population_codes.md` files.
In short the output file `fairness_analysis.json` contains the objects like:

```json
"equalized_odds_metrics": {
    "CYP2B6": {
      "Sex": {
        "IM": {
          "error_rates_by_group": {
            "F": {
              "false_positive_rate": 0.07142857142857142
            },
            "M": {
              "false_positive_rate": 0.05263157894736842
            }
          }
        }
...
```
and

```json
"demographic_parity_metrics": {
    "CYP2B6": {
      "Sex": {
        "NM": {
          "prediction_rates_by_group": {
            "F": 0.38095238095238093,
            "M": 0.32142857142857145
          },
        }
...
```

Add more explanation here regarding fairness

## Understanding the Results

### Explainability Output

The analysis produces several types of outputs:

1. **Feature Importance**: Identifies which genetic variants most influence phenotype predictions
2. **Categorical Association Analysis / Mutual Information Analysis**: Shows relationships between genetic variants and phenotypes
3. **Visualizations**: Plots showing the relative importance of features for each gene

### Fairness Analysis Output

The fairness analysis produces:

1. **Equalized Odds Metrics**: Measures whether error rates are similar across demographic groups
2. **Demographic Parity Metrics**: Ensures prediction rates are similar across demographic groups
3. **Visualizations**: Plots showing visualizing the above results

### Phenotype Classifications and Encoding

The analyzed phenotypes include:
- **PM**: Poor Metabolizer
- **LPM**: Likely Poor Metabolizer
- **IM**: Intermediate Metabolizer
- **LIM**: Likely Intermediate Metabolizer
- **NM**: Normal Metabolizer
- **LNM**: Likely Normal Metabolizer
- **RM**: Rapid Metabolizer
- **LRM**: Likely Rapid Metabolizer
- **UM**: Ultrarapid Metabolizer
- **LUM**: Likely Ultrarapid Metabolizer
- **NF**: Normal Function
- **DF**: Decreased Function
- **PF**: Poor Function
- **PDF**: Possibly Decreased Function
- **IF**: Increased Function
- **INDETERMINATE**: Uncertain phenotype

#### VCF to CSV Encoding

The preprocessing function takes the CSV data and transforms it into a format suitable for analysis:
1. **Genotype Encoding:**
   1. Converts genotype strings (like "0/1") to numerical values:
      1. 0: Homozygous reference (0/0)
      2. 1: Heterozygous (0/1, 1/0, 0/2, 2/0)
      3. 2: Homozygous alternate or compound heterozygous (1/1, 1/2, 2/1, 2/2)
      4. -1: Missing or unknown genotype
2. **Patient-Centric Pivoting:**
   1. Reshapes the data so each row represents a patient
   2. Each column represents a specific gene-variant combination (e.g., "CYP2D6_rs3892097")
   3. Uses the first occurring value when duplicates exist
3. **Feature Engineering:**
   1. Creates "HAS_GENE_X" flags for important pharmacogenetic genes
   2. These flags indicate whether the patient has any known variants in that gene

This approach is useful because of:

1. **Dimensionality Reduction:** By encoding genotypes as 0, 1, 2, or -1, we simplify the data while preserving the biological meaning (homozygous reference, heterozygous, homozygous alternate).
2. **Handling Missing Data:** Using -1 for missing/unknown genotypes allows the model to distinguish between reference genotype (0) and missing data.
3. **Gene-Level Features:** The "HAS_GENE_X" flags provide a higher-level abstraction that can be useful for models.
4. **Patient-Centric View:** The pivoted format makes it easy to train models that predict metabolizer status for each patient.
5. **Sparse Data Handling:** By filling missing values with -1, we handle the sparse nature of genetic data where not all patients have data for all variants.

#### Numeric Encoding of Phenotypes

The `phenotype_mapper.py` script encodes these phenotypes as numeric values for analysis:

| Phenotype     | Encoded Value |
|---------------|---------------|
| NM            | 0             |
| LNM           | 1             |
| IM            | 2             |
| LIM           | 3             |
| PM            | 4             |
| LPM           | 5             |
| UM            | 6             |
| LUM           | 7             |
| RM            | 8             |
| LRM           | 9             |
| NF            | 10            |
| DF            | 11            |
| IF            | 12            |
| PF            | 13            |
| PDF           | 14            |
| INDETERMINATE | -1            |

### Feature Naming Conventions

Features in the output follow these patterns:
- `GENE_rsID`: Basic SNP identifier (e.g., "CYP2B6_rs8192709")
- `GENE_rsID_alt_ALLELE`: Alternative allele (e.g., "DPYD_rs1801160_alt_T")
- `GENE_rsID_ref_ALLELE`: Reference allele (e.g., "CYP2C9_rs28371686_ref_C")
- `GENE_rsID_gt_GENOTYPE`: Genotype (e.g., "CYP2C19_rs3758581_gt_1/1" for homozygous)

## Visualization Outputs (Optional)

The `explainer_visualizer.py` and `pgx_fairness_visualizer.py` scripts generate visualizations to help interpret analysis results.

### Explainer Visualizations (`explainer_visualizer.py`)

This script processes the output from `explainer.py` (either categorical association or mutual information analysis) and generates:

- **Top Features per Gene**: Bar charts showing the 10 most important features for each gene, sorted by:
  - Association magnitude (for categorical association analysis)
  - Importance score (for mutual information analysis)
- **Feature Importance Heatmap**: A heatmap visualization showing the relationship between top features and pharmacogenes, using:
  - "YlOrRd" color map for categorical association analysis
  - "viridis" color map for mutual information analysis
- **Automatic Analysis Type Detection**: The script automatically determines whether to visualize categorical association or mutual information data based on the input file structure

The visualizations are saved as PNG files in the specified output directory, with one bar chart per gene and a comprehensive heatmap.

### Fairness Visualizations (`pgx_fairness_visualizer.py`)

This script processes the output from `fairness_bias_analyzer.py` and generates:

- **Equalized Odds Visualizations**: Bar charts showing false positive rates by demographic groups for different genes and phenotypes
  - Each bar uses color coding based on the superpopulation (AFR, AMR, EAS, EUR, SAS)
  - Values are annotated on each bar for easy comparison
  
- **Demographic Parity Visualizations**: Bar charts depicting prediction rates across demographic groups
  - Consistent color coding matches the equalized odds visualizations
  - Values are displayed on each bar for direct comparison

- **Summary Dashboard**: A consolidated view containing:
  - A heatmap of average false positive rates by gene and superpopulation (using "YlOrRd" color map)
  - A heatmap of average prediction rates by gene and superpopulation (using "YlGnBu" color map)
  - Both heatmaps include annotation of values for detailed inspection

## Kubeflow Pipeline Component

The `pgx_pipeline_component.py` file defines a Kubeflow pipeline for automating the pharmacogenomics analysis workflow. This pipeline orchestrates the following components:

1. **Download Component** (`download_project`): Downloads project files, input data, demographic and ground truth data from a specified GitHub repository. The pipeline expects the repo to contain the `Demographics/`, `Groundtruth/` and `data/` folders at its root with all required files placed inside.
2. **PharmCAT Analysis** (`pharmcat_analysis_docker`): Executes the PharmCAT analysis in a Docker container, processing VCF files and generating phenotype predictions. The docker image is built using the provided Dockerfile and all required files and scripts by VITO, and is uploaded to an image hosting repository.
3. **VCF to CSV Conversion** (`vcf_to_csv`): Converts VCF genetic data files to analyzable CSV format by extracting essential genetic variant information.
4. **Phenotype Mapping** (`phenotype_mapper`): Maps PharmCAT phenotype predictions to numeric values for downstream analysis.
5. **Explainability Analysis** (`run_explainer`): Applies either categorical association analysis or mutual information analysis based on the input sensitivity value, to explain the PharmCAT predictions.
6. **Fairness Analysis** (`fairness_bias_analyzer`): Evaluates potential bias in the PharmCAT predictions across demographic groups.
7. **Explainer Visualization** (`explainer_visualizer`): Generates visual representations of the explainability analysis results, including feature importance plots and heatmaps.
8. **Fairness Visualization** (`fairness_visualizer`): Creates visual representations of fairness metrics, including equalized odds and demographic parity visualizations.

**IMPORTANT:** You need to modify the `pharmcat_analysis_docker` function in the `pgx_pipeline_component.py` file to specify your PharmCAT Docker image:

```python
# Replace this:
image="<your_docker_pharmcat_image>",

# With your actual Docker image, for example:
image="docker.io/username/pharmcat-realm:latest",
```

After configuring the Docker image, the pipeline can be compiled and deployed to a Kubeflow environment by executing `python .\kubeflow_component\pgx_pipeline_component.py` and then uploading the generated YAML file to the Kubeflow UI.
The Kubeflow UI expects 2 pipeline arguments when running: `github_repo_url` which contains data and python scripts and `sensitivity`, which is a [0,1] float number.

The pipeline structure can be seen in the image below:
![Kubeflow Pipeline](kubeflow_pipeline.png)

## Accessing the Generated Artifacts

The pipeline stores generated artifacts in MinIO object storage within the Kubeflow namespace. To access these artifacts:
1. Set up port forwarding to the MinIO service by running `kubectl port-forward -n kubeflow svc/minio-service 9000:9000` in a terminal window
2. Access the MinIO web interface at `http://localhost:9000`
3. Log in with the default credentials: **username:** `minio`, **password:** `minio123`
4. Navigate to the `mlpipeline` bucket, where you'll find the respective folders according to the automatically assigned uuid of the pipeline. (An example location could be: `http://localhost:9000/minio/mlpipeline/v2/artifacts/pharmcat-pgx-analysis-pipeline/50c16278-0dde-44f1-b018-5b859a3fadf2/`)

## 📜 License & Usage

All rights reserved by MetaMinds Innovations.
