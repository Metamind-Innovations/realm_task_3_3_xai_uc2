**TODO**

## Dataset

The PxG dataset used in this repository is located [here](https://maastrichtuniversity.sharepoint.com/:f:/r/sites/FSE-REALM/Shared%20Documents/WP6/Open_Source_Datasets_per_UC/UC2_PGx2P_VITO/Data/V2?csf=1&web=1&e=1ReQnp).

## Run Pharmcat docker image:

`docker run --rm -v C:/Users/gigak/PycharmProjects/realm_task_3_3_xai_uc2/Preprocessed/:/pharmcat/data pgkb/pharmcat pharmcat_pipeline --missing-to-ref -matcher -phenotyper -reporterJson /pharmcat/data/HG00276_freebayes.preprocessed.vcf`.

**(WINDOWS)** To run the above command for all the VCF files in the `Preprocessed` directory, execute the `process_all.bat` file. Modify the `C:/Users/gigak/PycharmProjects/realm_task_3_3_xai_uc2/Preprocessed/` path inside the `.bat` file accordingly.

All scripts expect the pharmcat preprocessing to already be done for all the samples. The file structure should look like:
```
├── pharmcat_processed
│   ├── HG00276
│   │   ├── HG00276_freebayes.preprocessed.match.json
│   │   ├── HG00276_freebayes.preprocessed.missing_pgx_var.vcf
│   │   ├── HG00276_freebayes.preprocessed.phenotype.json
│   │   ├── HG00276_freebayes.preprocessed.preprocessed.vcf.bgz
│   │   ├── HG00276_freebayes.preprocessed.vcf
│   │   ├── HG00276_freebayes.preprocessed.vcf.bgz
│   │   ├── HG00276_freebayes.preprocessed.vcf.bgz.csi
│   ├── HG00436
│   ├── HG00589
│   └── ...
└── ...
```

## 📜 License & Usage

All rights reserved by MetaMinds Innovations.
