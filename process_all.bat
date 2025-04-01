@echo off
for %%f in (C:\Users\gigak\PycharmProjects\realm_task_3_3_xai_uc2\Preprocessed\*.preprocessed.vcf) do (
  echo Processing %%f
  docker run --rm -v C:/Users/gigak/PycharmProjects/realm_task_3_3_xai_uc2/Preprocessed/:/pharmcat/data pgkb/pharmcat pharmcat_pipeline --missing-to-ref -matcher -phenotyper -reporterJson /pharmcat/data/%%~nxf
)
