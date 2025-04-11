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

- Python 3.12+
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
7.

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


## 📜 License & Usage

All rights reserved by MetaMinds Innovations.
