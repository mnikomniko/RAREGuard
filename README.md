# RAREGuard 2.0.0

Reproducibility package for

> M. S. Mniko, W. Mwangi, and E. M. Karanja, "Hybrid One-Class Fraud Detection Under Extreme Class Imbalance: Label-Free Threshold Calibration and Sensitivity to Training-Reference Contamination," submitted to *IEEE Access*, 2026.

RAREGuard fuses min–max normalised scores from an isolation forest, a one-class SVM and an autoencoder, and sets the alert threshold from **unlabelled** validation scores (percentile, c = 0.002). The study measures how much each design choice matters when no fraud labels are used for calibration: fusion rule and weights, threshold estimator, assumed contamination, preprocessing, and fraud hidden in the legitimate training reference. It also tests transfer to PaySim.

This package contains everything needed to rerun the study, plus every result reported in the article, so the numbers can be checked without rerunning anything.


## Contents

```
code/rareguard_lib.py      all shared code: splits, preprocessing, detectors, fusion, thresholds, metrics, statistics, run logging
config.json                every setting used (seeds, model parameters, data paths)
notebooks/                 00-09, one per experiment group (Google Colab)
data/                      data sources and checksums (raw data not included), fixed splits
results/<experiment>/      per-split results, summaries, run_info.json (environment and file hashes)
results/stats/             paired tests with Holm correction for all experiments
logs/run_log.csv           one line per notebook run
manuscript/                figures (PDF), LaTeX tables and provenance.csv for the article
tools/verify_results.py    checks the archived results without data or GPU
```



## Check the results in two minutes (no data, no GPU)

```bash
pip install numpy pandas scipy scikit-learn
python tools/verify_results.py
```

The script confirms that the code is the version recorded in every run, that no per-split result file has changed since it was written, and that all summaries and paired tests can be recomputed from the per-split files.


## Reproduce the study

**Requirements:** a Google account (Colab with a GPU runtime), or a local Python 3.11+ environment with the packages in `requirements.txt`. The archived runs used Google Colab with an NVIDIA Tesla T4; exact versions are in every `results/*/run_info.json`.

1. Download the two datasets (see `data/README.md`) and put them on Google Drive.
2. Copy this folder to Google Drive. In the first code cell of each notebook, set `ROOT` to its location (default: `MyDrive/PhD/RAREGuard_v6`).
3. Edit the data paths in `config.json`.
4. In Colab, choose *Runtime* > *Change runtime type* > *T4 GPU*, then run the notebooks in order, each with *Run all*:

| Notebook | Experiment | Trains | Time on T4* | Writes |
|---|---|---|---|---|
| `00_setup_and_data` | data check, fixed splits | no | 1 min | `results/E0_data/`, `data/splits/` |
| `01_E1_components` | E1 components, ablation, label-free vs label-assisted threshold | yes | 12 min | `results/E1_components/` |
| `02_E2_E3_fusion` | E2 fusion baselines, E3 fusion weights | no | <1 min | `results/E2_E3_fusion/` |
| `03_E4_E5_thresholds` | E4 threshold estimators, E5 assumed contamination | no | <1 min | `results/E4_E5_thresholds/` |
| `04_E6_reference_contamination` | E6 fraud in the training reference | yes | 5 min | `results/E6_reference_contamination/` |
| `05_E7_early_stopping` | E7 autoencoder early stopping | yes | 3 min | `results/E7_early_stopping/` |
| `06_E8_cost_scalability` | E8 run time, scalability | yes | 3 min | `results/E8_cost/` |
| `07_E9_paysim` | E9 PaySim | yes | 9 min | `results/E9_paysim/` |
| `08_statistics` | paired t-tests, Holm correction | no | <1 min | `results/stats/` |
| `09_manuscript_figures_tables` | Figures 2-5, Tables 1-6 | no | <1 min | `manuscript/` |

\*From `logs/run_log.csv` for the archived runs.

Notebooks that train skip training if their results already exist. Set `FORCE_RERUN = True` to redo an experiment; the replaced files are moved to `_archive/`.

**What to expect when rerunning.** Splits, preprocessing, the isolation forest and the one-class SVM are deterministic given the seed and the package versions. Autoencoder training on a GPU is not bit-for-bit deterministic, so its scores, and all results that involve it, can differ in the third decimal. A different scikit-learn version can also change the isolation forest slightly.


## Where each result in the article comes from

| Article | Result file(s) |
|---|---|
| Table 1, Fig. 2 | `results/E1_components/summary_mean_sd.csv`, `scores/ulb_seed42.npz` |
| Table 2, Fig. 3 | `results/E2_E3_fusion/fusion_baselines_*.csv`, `results/stats/paired_tests.csv` |
| Fusion weights, detector contribution | `results/E2_E3_fusion/fusion_weights_*.csv`, `ae_contribution_by_seed.csv` |
| Table 3, Fig. 4 | `results/E4_E5_thresholds/*.csv` |
| Table 4, Fig. 5 | `results/E6_reference_contamination/*.csv` |
| Early stopping | `results/E7_early_stopping/*.csv` |
| Table 5, scalability | `results/E8_cost/*.csv` |
| Table 6, PaySim diagnostics | `results/E9_paysim/*.csv` |
| All p-values and confidence intervals | `results/stats/paired_tests.csv` |

`manuscript/provenance.csv` records, for each figure and table, the source file, the notebook run that produced it, the device and the MD5 hash of the source.


## Result files

* `*_by_seed.csv`: raw results, one row per split (seed) and condition. Everything else is derived from these.
* `*_summary.csv`: mean and SD over the five splits (`F1_mean`, `F1_sd`, …).
* `run_info.json`: start and end time, notebook, device, package versions, full configuration, and MD5 hash of every file written.
* `scores/*.npz`: saved anomaly scores, reused by later notebooks and by Figure 2.

Common columns: `Seed`, the condition column, `Precision`, `Recall`, `F1`, `MCC`, `PR_AUC`, `ROC_AUC`, `TP`, `FP`, `TN`, `FN`, `Threshold`, `Test_Alert_Rate`, `Alerts_per_10k`, `TP_per_10k`, `FP_per_10k`, and times in seconds (`*_s`).


## Citation

See `CITATION.cff`. Please cite the article and this software:

> M. S. Mniko, W. Mwangi, and E. M. Karanja, "RAREGuard: Hybrid one-class fraud detection with fraud-label-free threshold calibration (software, version 2.0.0)," Zenodo, 2026, doi: 10.5281/zenodo.24447609.

The first release (1.0.0) is at doi: 10.5281/zenodo.21852872. Changes are listed in `CHANGELOG.md`.


## Licence

Code: MIT (see `LICENSE`). Result files, figures and tables: CC BY 4.0. The datasets keep their original licences and are not included.
