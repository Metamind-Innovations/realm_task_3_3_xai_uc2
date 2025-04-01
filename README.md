**TODO**

How to run pharmcat docker image:

`docker run --rm -v C:/Users/gigak/PycharmProjects/realm_task_3_3_xai_uc2/Preprocessed/:/pharmcat/data pgkb/pharmcat pharmcat_pipeline --missing-to-ref -matcher -phenotyper -reporterJson /pharmcat/data/HG00276_freebayes.preprocessed.vcf`

**(WINDOWS)** To run the the above command for all the VCF files in the `Preprocessed` directory, execute the `process_all.bat` file. Modify the `C:/Users/gigak/PycharmProjects/realm_task_3_3_xai_uc2/Preprocessed/` path inside the `.bat` file accordingly

## 📜 License & Usage

All rights reserved by MetaMinds Innovations.
