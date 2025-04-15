**TODO**

## Pharmacogenomics (PGx) Analysis with Explainability

This project provides explainable AI tools for analyzing pharmacogenomic phenotype predictions made by PharmCAT. 
It uses SHAP analysis to explain which genetic variants contribute to phenotype predictions and includes a fairness analyzer to detect potential biases across demographic groups.

## Project Overview

PharmCAT is a software tool that analyzes genetic data (VCF files) to predict pharmacogenomic phenotypes that influence drug metabolism and response. 
This project adds an explainability layer to understand why PharmCAT makes specific predictions and to identify potential biases in those predictions.

Key components:
1. Converting VCF genetic data to analyzable csv format.
2. Providing explainability for PharmCAT predictions using SHAP analysis
3. Analyzing potential demographic bias and fairness issues in predictions

The project focuses on the key pharmacogenes: **CYP2B6, CYP2C9, CYP2C19, CYP3A5, SLCO1B1, TPMT, and DPYD.**

## Getting Started
### Prerequisites

- Python 3.10
- Docker (for running PharmCAT)
- Required Python packages (install via `pip install -r requirements.txt`)

### Execution Steps
1. **Download Dataset:** Download the provided dataset from [here](https://maastrichtuniversity.sharepoint.com/:f:/r/sites/FSE-REALM/Shared%20Documents/WP6/Open_Source_Datasets_per_UC/UC2_PGx2P_VITO/Data/V2?csf=1&web=1&e=1ReQnp).
2. **Download PharmCAT Docker Components:** Download the necessary tools and scripts from [here](https://maastrichtuniversity.sharepoint.com/:u:/r/sites/FSE-REALM/Shared%20Documents/WP6/Open_Source_Datasets_per_UC/UC2_PGx2P_VITO/Data/V2/Software/pharmcat-realm.zip?csf=1&web=1&e=NQFgXx)
3. **Setup Execution Environment:**
   1. Create a folder called `pharmcat_execution` and unzip all files downloaded from step 2. into it. 
   2. Inside the `data` folder, place all `.vcf` files downloaded from step 1.
4. Navigate inside the `pharmcat_execution` directory and execute: `docker build -t pharmcat-realm .`
5. _(WINDOWS)_ Execute: `docker run -v <absolute_path_to_data_folder>:/data -v <absolute_path_to_result_folder>:/result pharmcat-realm --input_folder /data --result_folder /result`. (e.g. `docker run -v C:/Users/gigak/Desktop/realm_uc2_vito_image/data/:/data -v C:/Users/gigak/Desktop/realm_uc2_vito_image/result/:/result pharmcat-realm --input_folder /data --result_folder /result`).
This will execute the pharmcat pipeline for all samples placed in the `data` folder. The `result` folder will contain `phenotype.json` files for each sample and a total `phenotypes.csv` file containing the output in the format:

| Sample ID | CYP2B6 | CYP2C19 | CYP2C9 | CYP3A5 | DPYD | SLCO1B1 |     TPMT      |
|:---------:|:------:|:-------:|:------:|:------:|:----:|:-------:|:-------------:|
|  HG00276  |   RM   |   NM    |   IM   |   PM   |  NM  |   DF    | INDETERMINATE |
|  HG00436  |   IM   |   NM    |   NM   |   PM   |  NM  |   NF    |      NM       |

6. Place the `result` folder in the root of the current project.
7. Execute the `pgx_shap_analyzer.py` script to explain PharmCAT predictions: `python pgx_shap_analyzer.py --input_dir <path_to_vcf_files> --phenotypes_file result/phenotypes.csv --output_dir pgx_shap_results --convert_vcf`. This will analyze the genetic variants that contribute to each phenotype prediction and generate a `pgx_shap_results.json` file with detailed explanations.
8. Execute the `pgx_fairness_analyzer.py` script: `python pgx_fairness_analyzer.py --demographic_file Demographics/pgx_cohort.csv --phenotypes_file result/phenotypes.csv --output_dir pgx_fairness_results`. This will generate individual fairness reports for each sample and an overall fairness report, highlighting any potential demographic bias in the predictions.
9. Examine the output files:
   1. `pgx_shap_results/pgx_shap_results.json`: Contains feature importance and sample-specific explanations
   2. `pgx_shap_results/pgx_shap_summary.txt`: Provides a human-readable summary of the SHAP analysis
   3. `pgx_fairness_results/overall_fairness_report.json`: Summarizes potential bias across demographic groups
   4. Individual sample reports in the respective output directories
10. _Alternative Pipeline Execution: Use the Kubeflow pipeline to automate the entire workflow: `python kubeflow_component/pgx_pipeline_component.py`. This will generate a `pharmcat_pipeline_.yaml` file that can be uploaded to a Kubeflow environment for execution. See more details in the [Kubeflow Pipeline Component](#kubeflow-pipeline-component) section. 

## SHAP Analyzer JSON Output

The output is organized into three main sections:

- `gene_explanations`
   
   This section contains model explanations for each gene (CYP2B6, CYP2C9, CYP2C19, CYP3A5, SLCO1B1, TPMT, DPYD). For each gene:
  - `prediction_distribution`: The count of different phenotype predictions across the analyzed samples:
    - PM (Poor Metabolizer)
    - IM (Intermediate Metabolizer)
    - NM (Normal Metabolizer)
    - RM (Rapid Metabolizer)
    - UM (Ultrarapid Metabolizer)
    - NF (Normal Function)
    - DF (Decreased Function)
    - PF (Poor Function)
    - IF (Increased Function)
    - INDETERMINATE (Uncertain phenotype)
  - `top_features_by_gene`: Organizes the most influential features by their source gene. For example, for CYP2B6, it shows which variants in DPYD, CYP2C19, CYP2C9, etc. had the greatest impact on CYP2B6 phenotype predictions. Each feature includes:
    - `feature`: The name of the genetic variant (format: `GENE_rsID_allele_type`)
    - `importance`: The average magnitude of impact this feature has across all samples
- `feature_importance`
   
   This provides a flattened view of global feature importance for each target gene. It lists the top 20 most important features that influence phenotype predictions for each gene, sorted by importance value.
- `sample_explanations`

   This contains specific explanations for individual samples, showing why particular phenotypes were predicted. For each sample-gene combination:
   - `sample_id` Identifier for the sample (e.g., **HG00276**)
   - `gene`: The gene being explained
   - `predicted_phenotype`: The predicted phenotype for this sample-gene pair
   - `top_contributions`: List of features that most influenced this prediction:
     - `feature`: Name of the genetic variant
     - `value`: The value of this feature for this sample (typically 0 or 1, where 1 means the variant is present)
     - `shap_value`: The SHAP value representing the impact on the prediction (positive values push toward the predicted phenotype, negative values push away)
   - `explanation`: A human-readable explanation of the prediction

### Feature Naming Conventions
Features in the output follow these patterns:
- `GENE_rsID`: Basic SNP identifier (e.g., "CYP2B6_rs8192709")
- `GENE_rsID_alt_ALLELE`: Alternative allele (e.g., "DPYD_rs1801160_alt_T")
- `GENE_rsID_ref_ALLELE`: Reference allele (e.g., "CYP2C9_rs28371686_ref_C")
- `GENE_rsID_gt_GENOTYPE`: Genotype (e.g., "CYP2C19_rs3758581_gt_1/1" for homozygous)
- `GENE_posNUMBER`: Position-based feature (e.g., "CYP3A5_pos99652770")

## Kubeflow Pipeline Component

The `pgx_pipeline_component.py` file defines a Kubeflow pipeline for automating the pharmacogenomics analysis workflow. This pipeline orchestrates the following components:

1. **Download Component**: Downloads project files, input data, and demographic data from a specified GitHub repository.
2. **PharmCAT Analysis**: Executes the PharmCAT analysis in a Docker container, processing VCF files and generating phenotype predictions.
3. **SHAP Analysis**: Applies SHAP (SHapley Additive exPlanations) analysis to explain the PharmCAT predictions.
4. **Fairness Analysis**: Evaluates potential bias in the PharmCAT predictions across demographic groups.

You need to specify the docker image containing the PharmCAT analysis code at **line 170** in the `pgx_pipeline_component.py` file. The pipeline can be compiled and deployed to a Kubeflow environment by first executing `python .\kubeflow_component\pgx_pipeline_component.py` and then uploading the generated YAML file to the Kubeflow UI.
The pipeline structure can be seen in the below image: ![Kubeflow Pipeline](kubeflow_pipeline.png)

## 📜 License & Usage

All rights reserved by MetaMinds Innovations.
