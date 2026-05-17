# DiDyNet 🧬

**A Robust Framework for Differential Dynamic Network Inference from Longitudinal Multi-omics Data**

DiDyNet is a novel computational framework designed to decipher time-evolving molecular interactions underlying complex disease progression. Unlike traditional methods that rely on static snapshots, DiDyNet explicitly models temporal coordination between molecular trajectories using Dynamic Time Warping (DTW) and robust Linear Mixed Models (LMM).

## 📖 Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Pipeline Outputs](#pipeline-outputs)
- [Contact & Support](#contact--support)
- [Citation](#citation)

## 🌟 Overview
DiDyNet operates through a rigorous multi-step pipeline:
1. **2D Variance-based Filtering:** Prioritizes features with significant temporal or subject-level variance.
2. **DTW Alignment:** Quantifies dynamic coupling between feature trajectories, accurately accommodating biological asynchrony and time lags.
3. **Statistical Testing:** Identifies Differential Dynamic Couplings (DDCs) via Wilcoxon tests and FDR correction.
4. **LMM Post-hoc Refinement:** Distinguishes genuine coordinated dynamics (SCCs) from artifacts driven by marginal single-feature fluctuations.
5. **Network Construction:** Builds a highly robust, topologically stratified differential dynamic network.

## ⚙️ Installation

We recommend using a virtual environment (e.g., conda or venv). Clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone [https://github.com/bioinfoliu/DiDyNet.git](https://github.com/bioinfoliu/DiDyNet.git)
cd DiDyNet

# Install required dependencies
pip install -r requirements.txt

# Install the package locally
pip install -e .
```

## 📂 Data Preparation

Before running the pipeline, ensure your data directory is structured correctly. By default, the script expects a `data/raw_data/` folder inside your `base_path`. 

```text
Your_Base_Path/
└── data/
    └── raw_data/
        ├── cytokines.csv      # Cytokine omics data
        ├── proteomics.csv     # Proteomics data
        ├── RNA.csv            # Transcriptomics data
        └── IRIS_label.csv     # Subject phenotype labels (e.g., IS, IR)
```
*Note: The omics data CSVs should contain `SubjectID` and `Time` columns, followed by the molecular features.*

## 🚀 Quick Start

You can run the entire end-to-end DiDyNet pipeline using the provided `run_pipeline.py` script. The framework allows you to customize the analysis using command-line hyperparameters.

```bash
python run_pipeline.py \
  --base_path /path/to/your/workspace/ \
  --k 100 \
  --top_hubs 20 \
  --ers_threshold 0.90
```

### Arguments:
* `--base_path` : The absolute path to your working directory containing the `data/` folder.
* `--k` : The top K hyperparameter for the 2D Variance Filtering step (default: `100`).
* `--top_hubs` : The number of top hub nodes to extract per omics layer for network visualization (default: `20`).
* `--ers_threshold` : Edge Reliability Score threshold used for filtering highly robust consensus network edges (default: `0.90`).

## 📊 Pipeline Outputs

Once the pipeline finishes successfully, the results will be automatically saved in the `results/` folder within your `base_path`:
* **`data/cleaned/`**: The aligned and standardized cohort data shared across all omics layers.
* **`top_features/`**: Selected feature lists based on the 2D variance threshold.
* **`results/DTW/`**: Pairwise DTW distance matrices and Wilcoxon significance test results.
* **`results/PostHoc/`**: The LMM-refined Subtle Coordinated Couplings (SCCs).
* **`results/Robustness_All_SCC/`**: The final high-confidence network edge list and the generated topological network figures (PNG/PDF).

## 📬 Contact & Support

We welcome feedback, bug reports, and academic collaborations!

* **For Code & Technical Issues:** If you encounter any bugs, errors, or need help running the pipeline, please open an Issue on GitHub or contact **Zhe Liu** at `your_email@example.com`.
* **For Academic & Methodological Inquiries:** For questions regarding the experimental design, algorithms, biological insights, or collaboration opportunities, please contact the corresponding author, **Prof. Taesung Park** at `tspark@stats.snu.ac.kr`.

## 📝 Citation
If you use DiDyNet in your research, please cite our paper:
> *Liu, Z., Wu, K., & Park, T. "DiDyNet: A Robust Framework for Differential Dynamic Network Inference from Longitudinal Multi-omics Data." (In submission).*