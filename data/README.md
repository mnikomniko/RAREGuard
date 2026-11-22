# Data

The two datasets are publicly available but are **not redistributed** here. Download them from their original sources and check their licence terms there.

| Dataset | File name expected | Size | MD5 of the file used | Used in this study |
|---|---|---|---|---|
| ULB Credit Card Fraud Detection (Machine Learning Group, Université Libre de Bruxelles; Kaggle) | `creditcard.csv` | 143.8 MB | `e90efcb83d69faf99fcab8b0255024de` | all 284,807 rows, 492 frauds |
| PaySim synthetic mobile-money transactions (López-Rojas et al.; Kaggle) | `PS_20174392719_1491204439457_log.csv` | 470.7 MB | `e92a5f7447f43712f1dca473d0b0fa85` | stratified sample of 300,000 rows (seed 31415), 387 frauds |

Put the paths of both files in `config.json` (`data.ulb_csv`, `data.paysim_csv`). Notebook 00 checks the MD5 hashes and writes them to `results/E0_data/data_files.csv`. If your hash differs, you have a different version of the file and the saved splits will not load.

## Fixed splits (`data/splits/`)

| File | Contents |
|---|---|
| `ulb_seed{42,7,21,100,2026}.npz` | row indices of the 70/15/15 stratified train/validation/test split for each seed |
| `paysim_seed{...}.npz` | the same for the 300,000-row PaySim sample |
| `paysim_sample_rows.npy` | row numbers (in the original PaySim file) of the 300,000 sampled transactions |

Each `.npz` also stores the number of rows and frauds, so a split cannot be applied to the wrong file by accident. Notebook 00 recreates the files identically if they are missing (`scikit-learn` stratified splitting with the seeds above; PaySim sample with seed 31415).

Split sizes per seed: ULB 199,364 / 42,721 / 42,722 rows (344 / 74 / 74 frauds); PaySim sample 210,000 / 45,000 / 45,000 rows (271 / 58 / 58 frauds).

## Features used

* **ULB:** all 30 columns (`Time`, `V1`-`V28`, `Amount`); the label is `Class`.
* **PaySim, baseline:** one-hot `type`, `log1p(amount)`, `step`.
* **PaySim, enhanced:** baseline plus six features from the balances *before* the transaction (`oldbalanceOrg`, `oldbalanceDest`). Post-transaction balances, `isFlaggedFraud` and account identifiers are never read.
