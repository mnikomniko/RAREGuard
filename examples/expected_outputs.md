# Expected Outputs

This document describes the files that should be generated when the RAREGuard reproducibility notebook is executed successfully.

The main notebook is:

```text
notebooks/RAREGuard_Reproducibility_Notebook.ipynb
```

After placing `creditcard.csv` in `data/raw/` and running the notebook sequentially from beginning to end, experimental outputs are written to the `results/` and `figures/` directories.

---

## 1. Dataset Verification

Before running the experiments, the notebook should successfully load the Credit Card Fraud Detection dataset.

The expected dataset characteristics are:

```text
Dataset shape: (284807, 31)
Fraud count: 492
Fraud percentage: 0.1727 %
```

If these values differ substantially, verify that the correct `creditcard.csv` dataset has been placed in:

```text
data/raw/creditcard.csv
```

---

## 2. Expected Results Files

A successful complete run should generate the following files in:

```text
results/
```

### 2.1 `label_assisted_results.csv`

Contains experimental results obtained using the label-assisted thresholding strategy.

This file records results for individual experimental runs and includes information such as:

- random seed;
- experimental configuration;
- model;
- thresholding mode;
- feature-selection setting;
- PCA setting;
- selected features;
- PCA dimensions;
- threshold;
- Precision;
- Recall;
- F1-score;
- ROC-AUC;
- PR-AUC;
- MCC;
- training time;
- total execution time; and
- confusion-matrix components.

The label-assisted experiments are included as a comparative evaluation regime.

---

### 2.2 `fully_unsupervised_results.csv`

Contains the experimental results obtained using fully unsupervised threshold calibration.

The decision threshold in this regime is determined without using fraud labels for threshold optimization.

This file provides the run-level results used to evaluate the deployment-oriented RAREGuard configuration.

---

### 2.3 `all_experimental_results.csv`

Contains the combined run-level results from:

```text
label_assisted_results.csv
```

and:

```text
fully_unsupervised_results.csv
```

This is the master experimental-results file and can be used to independently reproduce aggregate comparisons across models, configurations, seeds, and thresholding strategies.

---

### 2.4 `summary_mean_results.csv`

Contains the mean performance values calculated across repeated experimental runs.

The principal experiments use the random seeds:

```text
42
7
21
100
2026
```

This file provides the average values used in the aggregate performance analysis.

---

### 2.5 `summary_standard_deviation.csv`

Contains the standard deviation of the performance metrics across the repeated experimental runs.

It complements:

```text
summary_mean_results.csv
```

and provides information about the stability of each experimental configuration across random seeds.

---

### 2.6 `mean_standard_deviation_table.csv`

Contains the publication-oriented summary of the principal experimental results expressed as:

```text
mean ± standard deviation
```

This table is derived from the repeated experimental runs and is intended to facilitate comparison with the aggregate results reported in the associated study.

---

### 2.7 `contamination_sensitivity_analysis.csv`

Contains the results of the contamination-rate sensitivity experiment.

The evaluated contamination values are:

```text
0.001
0.0015
0.002
0.003
0.005
```

The analysis examines how the assumed contamination rate affects:

- Precision;
- Recall;
- F1-score; and
- MCC.

The principal RAREGuard configuration uses:

```text
c = 0.002
```

---

### 2.8 `representative_seed42_test_scores.csv`

Contains test-set anomaly scores generated from the representative experimental run using:

```text
seed = 42
```

The file contains the scores required to construct the precision-recall curves for:

- Isolation Forest;
- One-Class SVM;
- Autoencoder; and
- RAREGuard.

The seed-42 run is used for curve visualization. Aggregate performance comparisons are based on the repeated multi-seed experiments.

---

### 2.9 `scalability_analysis.csv`

Contains the results of the scalability experiment.

The scalability analysis evaluates RAREGuard at increasing dataset sizes and records performance and computational measures including:

- sample size;
- Precision;
- Recall;
- F1-score;
- PR-AUC;
- ROC-AUC;
- MCC;
- training time;
- total execution time;
- selected features; and
- PCA dimensions.

---

## 3. Expected Figures

A successful run should generate the following publication-oriented figures in:

```text
figures/
```

### Figure 2 — Precision-Recall Curves

Expected file:

```text
figure_2_precision_recall_curves.png
```

The figure compares precision-recall curves for:

- Isolation Forest;
- One-Class SVM;
- Autoencoder; and
- RAREGuard.

The curves are generated from the representative seed-42 test scores.

---

### Figure 3 — Threshold Strategy Comparison

Expected file:

```text
figure_3_threshold_strategy_comparison.png
```

This figure compares model performance under:

- fully unsupervised thresholding; and
- label-assisted thresholding.

It illustrates the effect of threshold-selection strategy on model performance.

---

### Figure 4 — Ablation Study

Expected file:

```text
figure_4_ablation_study.png
```

This figure compares the complete RAREGuard framework with ablated configurations, including:

- `RAREGuard_Full`;
- `RAREGuard_No_FS`; and
- `RAREGuard_No_PCA`.

The purpose is to examine the contribution of feature selection and PCA to the complete framework.

---

### Figure 5 — Contamination-Rate Sensitivity

Expected file:

```text
figure_5_contamination_sensitivity.png
```

This figure visualizes the sensitivity of:

- Precision;
- Recall;
- F1-score; and
- MCC

to different assumed contamination rates.

---

### Figure 6 — Computational Cost Comparison

Expected file:

```text
figure_6_computational_cost_comparison.png
```

This figure compares computational requirements across the evaluated models using measures such as:

- training time; and
- total execution time.

---

### Figure 7 — Scalability Analysis

Expected file:

```text
figure_7_scalability_analysis.png
```

This figure illustrates how RAREGuard behaves as the number of transactions used in the experiment increases.

It presents both predictive-performance and computational behaviour across dataset sizes.

---

### Figure 8 — Performance Heatmap

Expected file:

```text
figure_8_performance_heatmap.png
```

The heatmap provides a compact comparison of key performance metrics across the evaluated models.

The principal metrics displayed include:

- F1-score;
- MCC; and
- PR-AUC.

---

## 4. Expected Directory After a Successful Run

After successful execution, the relevant portion of the repository should resemble:

```text
RAREGuard/
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
└── figures/
    ├── figure_2_precision_recall_curves.png
    ├── figure_3_threshold_strategy_comparison.png
    ├── figure_4_ablation_study.png
    ├── figure_5_contamination_sensitivity.png
    ├── figure_6_computational_cost_comparison.png
    ├── figure_7_scalability_analysis.png
    └── figure_8_performance_heatmap.png
```

A complete run should therefore produce:

- **9 result CSV files**
- **7 publication figures**

---

## 5. Reproducibility Check

A reproduction should not be considered successful merely because the notebook executes without errors.

Users should also verify that:

1. the expected dataset dimensions are obtained;
2. all expected result files are generated;
3. all expected figures are generated;
4. the experimental configurations match those documented in the notebook;
5. the principal results are consistent with the supplied reference outputs; and
6. minor differences attributable to software versions, hardware, or numerical computation are distinguished from substantive differences in model performance.

The supplied files in `results/` and `figures/` serve as reference outputs from the reported RAREGuard experiments.

---

## 6. Notes on Runtime

Execution time can vary depending on:

- CPU and GPU availability;
- system memory;
- PyTorch version;
- scikit-learn version;
- hardware architecture; and
- whether the notebook is executed locally or in a cloud environment.

Computational-time measurements therefore may not reproduce exactly across machines.

Performance metrics should be used as the primary basis for validating experimental reproduction, while runtime values should be interpreted in the context of the execution environment.

---

## 7. Additional Instructions

For complete setup and reproduction instructions, see:

```text
docs/reproducibility_guide.md
```

For an overview of the RAREGuard framework, repository structure, dataset requirements, installation instructions, and citation information, see the main:

```text
README.md
```