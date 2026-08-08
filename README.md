# RAREGuard

**A Hybrid One-Class Anomaly Detection Framework with Fully Unsupervised Thresholding**

RAREGuard is a hybrid anomaly detection framework designed for fraud detection in highly imbalanced datasets. The framework combines three complementary one-class/anomaly detectors:

- Isolation Forest (IF)
- One-Class Support Vector Machine (OCSVM)
- Autoencoder (AE)

The normalized anomaly scores produced by these detectors are combined into a weighted ensemble:

\[
s_{\mathrm{RAREGuard}}
=
0.30s_{\mathrm{IF}}
+
0.30s_{\mathrm{OCSVM}}
+
0.40s_{\mathrm{AE}}
\]

The final deployment-oriented configuration uses **fully unsupervised thresholding**, allowing the decision threshold to be determined without using fraud labels during threshold calibration.

This repository contains the implementation, reproducibility notebook, experimental outputs, figures, and supporting documentation associated with the study:

> **RAREGuard: A Hybrid One-Class Anomaly Detection Framework with Fully Unsupervised Thresholding**

---

## Overview

Fraud detection datasets are typically characterized by extreme class imbalance, where fraudulent observations represent only a very small proportion of all transactions.

RAREGuard was developed to investigate whether complementary one-class anomaly detectors can be combined to provide robust fraud detection while reducing dependence on labelled fraud observations during deployment.

The experimental framework includes:

- stratified train/validation/test partitioning;
- K-nearest-neighbour imputation;
- min-max normalization;
- variance-based feature filtering;
- Laplacian Score feature selection;
- Principal Component Analysis (PCA);
- Isolation Forest;
- One-Class SVM;
- Autoencoder reconstruction-error scoring;
- normalized score-level fusion;
- fully unsupervised threshold calibration;
- label-assisted thresholding for comparative evaluation;
- ablation experiments;
- contamination-rate sensitivity analysis;
- computational-cost analysis; and
- scalability analysis.

Experiments are repeated across multiple random seeds to evaluate stability.

---

## Dataset

The experiments use the publicly available **Credit Card Fraud Detection** dataset.

The dataset contains:

- **284,807 transactions**
- **30 predictor variables**
- **492 fraudulent transactions**
- approximately **0.1727% fraud prevalence**
- a binary target variable named `Class`

`Class = 1` represents a fraudulent transaction and `Class = 0` represents a legitimate transaction.

The original dataset is **not redistributed in this repository**.

Download `creditcard.csv` from the public Kaggle Credit Card Fraud Detection dataset and place it at:

```text
data/raw/creditcard.csv
```

The reproducibility notebook checks for this file before starting the experiments.

---

## Repository Structure

```text
RAREGuard/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── data/
│   └── raw/
│       └── README.md
│
├── notebooks/
│   └── RAREGuard_Reproducibility_Notebook.ipynb
│
├── results/
│   ├── all_experimental_results.csv
│   ├── contamination_sensitivity_analysis.csv
│   ├── fully_unsupervised_results.csv
│   ├── label_assisted_results.csv
│   ├── mean_standard_deviation_table.csv
│   ├── representative_seed42_test_scores.csv
│   ├── scalability_analysis.csv
│   ├── summary_mean_results.csv
│   └── summary_standard_deviation.csv
│
├── figures/
│   ├── figure_2_precision_recall_curves.png
│   ├── figure_3_threshold_strategy_comparison.png
│   ├── figure_4_ablation_study.png
│   ├── figure_5_contamination_sensitivity.png
│   ├── figure_6_computational_cost_comparison.png
│   ├── figure_7_scalability_analysis.png
│   └── figure_8_performance_heatmap.png
│
├── docs/
│   └── reproducibility_guide.md
│
└── examples/
    └── expected_outputs.md
```

---

## RAREGuard Architecture

The RAREGuard experimental pipeline consists of the following main stages:

1. **Data preparation**
   - Load the Credit Card Fraud Detection dataset.
   - Separate predictors and the `Class` variable.
   - Create stratified training, validation, and test partitions.

2. **Preprocessing**
   - Apply KNN-based imputation.
   - Apply min-max feature scaling using training-set statistics.

3. **Feature selection**
   - Remove low-variance features.
   - Construct a neighbourhood graph from sampled training observations.
   - Calculate Laplacian Scores.
   - Retain the highest-ranked features.

4. **Dimensionality reduction**
   - Apply PCA while retaining 95% of the variance.

5. **Base anomaly detectors**
   - Isolation Forest
   - One-Class SVM
   - Autoencoder

6. **Score normalization**
   - Normalize detector anomaly scores to a common scale.

7. **Score fusion**
   - Combine the normalized detector scores using fixed ensemble weights.

8. **Threshold calibration**
   - Evaluate both label-assisted and fully unsupervised thresholding.
   - The deployment-oriented RAREGuard configuration uses fully unsupervised threshold calibration.

9. **Evaluation**
   - Precision
   - Recall
   - F1-score
   - PR-AUC
   - ROC-AUC
   - Matthews Correlation Coefficient (MCC)
   - training time
   - total execution time

---

## Experimental Configurations

The notebook evaluates the complete RAREGuard framework together with individual detectors and ablation configurations.

The principal configurations include:

- `RAREGuard_Full`
- `RAREGuard_No_FS`
- `RAREGuard_No_PCA`
- Isolation Forest
- One-Class SVM
- Autoencoder

Experiments are conducted under both:

- **Fully unsupervised thresholding**
- **Label-assisted thresholding**

The label-assisted experiments are retained as a comparative evaluation regime. The proposed deployment-oriented configuration uses fully unsupervised threshold calibration.

---

## Reproducibility

The principal experiments are repeated using the following random seeds:

```text
42
7
21
100
2026
```

Aggregate results are reported using the mean and standard deviation across these runs.

The repository also contains the underlying experimental outputs so that reported summaries can be independently inspected without rerunning the complete model-training process.

---

## Installation

### Option 1: pip

Clone the repository:

```bash
git clone https://github.com/<USERNAME>/RAREGuard.git
cd RAREGuard
```

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

On Windows:

```bash
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### Option 2: Conda

The repository also provides an environment specification:

```bash
conda env create -f environment.yml
conda activate rareguard
```

---

## Dataset Setup

Create the expected dataset location if it does not already exist:

```text
data/raw/
```

Download the original Credit Card Fraud Detection dataset and copy:

```text
creditcard.csv
```

to:

```text
RAREGuard/data/raw/creditcard.csv
```

The final structure should therefore include:

```text
RAREGuard/
└── data/
    └── raw/
        └── creditcard.csv
```

The dataset itself is excluded from version control.

---

## Running the Experiments

Launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
notebooks/RAREGuard_Reproducibility_Notebook.ipynb
```

Run the notebook cells sequentially from top to bottom.

The notebook automatically creates or uses the following output directories:

```text
results/
figures/
```

A complete run regenerates the experimental outputs and publication figures.

---

## Experimental Outputs

The `results/` directory contains both run-level and aggregate experimental outputs.

### `all_experimental_results.csv`

Master experimental output combining the label-assisted and fully unsupervised experimental runs.

### `fully_unsupervised_results.csv`

Results obtained using fully unsupervised threshold calibration.

### `label_assisted_results.csv`

Results obtained using label-assisted threshold optimization for comparative evaluation.

### `summary_mean_results.csv`

Mean performance statistics across repeated experimental runs.

### `summary_standard_deviation.csv`

Standard deviations corresponding to the multi-seed experimental results.

### `mean_standard_deviation_table.csv`

Publication-oriented representation of the principal results as mean ± standard deviation.

### `contamination_sensitivity_analysis.csv`

Results from the contamination-rate sensitivity experiment used to evaluate the robustness of unsupervised threshold calibration.

### `representative_seed42_test_scores.csv`

Test-set anomaly scores from the representative seed-42 run used to construct the precision-recall curves.

Multi-seed aggregate performance is reported separately in the summary result files.

### `scalability_analysis.csv`

Experimental results used to evaluate the behaviour of RAREGuard as dataset size increases.

---

## Figures

The `figures/` directory contains the publication-oriented figures generated from the experiments.

### Figure 2 — Precision-Recall Curves

Compares the precision-recall behaviour of the individual anomaly detectors and the RAREGuard ensemble.

### Figure 3 — Threshold Strategy Comparison

Compares performance under fully unsupervised and label-assisted thresholding strategies.

### Figure 4 — Ablation Study

Examines the contribution of feature selection and PCA to the complete RAREGuard framework.

### Figure 5 — Contamination-Rate Sensitivity

Evaluates the effect of different assumed contamination rates on Precision, Recall, F1-score, and MCC.

### Figure 6 — Computational Cost Comparison

Compares training and total execution time across the evaluated models.

### Figure 7 — Scalability Analysis

Examines model performance and computational behaviour as dataset size increases.

### Figure 8 — Performance Heatmap

Provides a compact comparison of key performance metrics across the evaluated models.

---

## Contamination-Rate Sensitivity

The fully unsupervised decision threshold depends on an assumed contamination rate.

The sensitivity analysis evaluates:

```text
0.001
0.0015
0.002
0.003
0.005
```

The principal experimental configuration uses:

```text
c = 0.002
```

The sensitivity experiment is included to examine how threshold assumptions affect Precision, Recall, F1-score, and MCC rather than relying on a single contamination setting without robustness assessment.

---

## Notes on Labels

RAREGuard is designed around one-class/anomaly-detection models and a fully unsupervised thresholding strategy.

Fraud labels are retained in the experimental dataset for:

- stratified dataset partitioning;
- post-hoc performance evaluation;
- calculation of Precision, Recall, F1, PR-AUC, ROC-AUC, and MCC; and
- the separate label-assisted thresholding experiments used for comparison.

The deployment-oriented fully unsupervised thresholding procedure does not use fraud labels to optimize the final decision threshold.

---

## Reproducing the Publication Results

To reproduce the study:

1. Clone or download this repository.
2. Install the required environment.
3. Download the original dataset.
4. Place `creditcard.csv` in `data/raw/`.
5. Open `notebooks/RAREGuard_Reproducibility_Notebook.ipynb`.
6. Run the notebook sequentially from top to bottom.
7. Compare the regenerated files in `results/` with the supplied reference outputs.
8. Compare the regenerated figures with those provided in `figures/`.

Additional instructions are provided in:

```text
docs/reproducibility_guide.md
```

---

## Software and Dependencies

The implementation is written in Python and uses libraries including:

- NumPy
- pandas
- scikit-learn
- PyTorch
- Matplotlib
- SciPy

Exact environment and dependency information is provided in:

```text
requirements.txt
environment.yml
```

---

## Citation

If you use RAREGuard, its code, or its experimental outputs in academic work, please cite the associated publication and software repository.

Citation metadata is provided in:

```text
CITATION.cff
```

GitHub can use this file to provide a **Cite this repository** option.

The full publication citation and DOI can be added after publication.

---

## Data Availability

The Credit Card Fraud Detection dataset used in this study is publicly available from Kaggle.

The original dataset is not redistributed as part of this repository. Users should obtain the dataset from its original source and place `creditcard.csv` in:

```text
data/raw/
```

The processed experimental outputs required to inspect the reported results are provided in the `results/` directory.

---

## Code Availability

The implementation, reproducibility notebook, experimental outputs, figures, environment specifications, and supporting documentation associated with RAREGuard are provided in this repository.

A permanently archived release and DOI may also be provided through Zenodo.

---

## License

The software in this repository is released under the **MIT License**.

See:

```text
LICENSE
```

for the complete license text.

---

## Author

**Mniko Simon Mniko**

RAREGuard was developed as part of research on unsupervised learning methods for fraud detection in highly imbalanced datasets.

---

## Disclaimer

This repository is provided for research and reproducibility purposes.

The fraud detection results reported in the associated study are based on the specified dataset, experimental design, preprocessing pipeline, hyperparameters, random seeds, and evaluation procedures. Performance on other datasets or operational environments may differ.