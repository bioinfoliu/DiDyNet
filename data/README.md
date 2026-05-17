# Data Directory for DiDyNet 📂

## ⚠️ Note on Raw Data
Due to GitHub's file size limits and to ensure a lightweight repository, the massive original raw multi-omics datasets are **not** hosted here. 

If you need to access the raw, unfiltered data, please download it directly from the original reference publication:
> **Reference:** Sailani, M. R., et al. (2020). *Deep longitudinal multiomics profiling reveals two biological seasonal patterns in California.* Nature Communications, 11(1), 4933.

## 📊 Provided Cleaned Data
The files provided in this directory represent the **fully curated and preprocessed datasets** that are ready for immediate downstream analysis using the DiDyNet pipeline. 

To generate these clean datasets, the raw data underwent a rigorous preprocessing pipeline to ensure maximum data quality and consistency:
1. **Phenotypic Filtering:** We strictly retained only observations that possess clearly defined clinical phenotypic labels (e.g., Insulin Sensitive [IS] or Insulin Resistant [IR]).
2. **Mean Deduplication:** Any duplicate measurements recorded for the same subject at the exact same time point were averaged to yield a single, stable representative value.
3. **Strict Cohort Intersection:** We aligned the sample cohort across all multi-omics layers. The data here exclusively contains the shared subjects that are present in the Cytokine, Proteomics, and Transcriptomics datasets simultaneously.
4. **Feature Selection (K=100):** The dataset was finalized by applying our 2D Variance filtering strategy with the hyperparameter `K=100`. This isolates the most dynamic and informative features while eliminating massive background noise.

You can directly use these provided clean files to run the pipeline and reproduce the differential dynamic networks and hub results presented in our study.