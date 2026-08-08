# Dataset

The original dataset used in the RAREGuard experiments is **not redistributed with this repository**.

## Credit Card Fraud Detection Dataset

RAREGuard uses the publicly available Credit Card Fraud Detection dataset containing European cardholder transactions.

The dataset used in the experiments contains:

- **284,807 transactions**
- **30 predictor variables**
- **492 fraudulent transactions**
- **284,315 legitimate transactions**
- **Fraud prevalence:** approximately 0.1727%
- **Target variable:** `Class`
  - `0` = legitimate transaction
  - `1` = fraudulent transaction

## Download Instructions

Download the Credit Card Fraud Detection dataset from its original public Kaggle source.

The downloaded file required by the RAREGuard notebook is:

```text
creditcard.csv
```

Place the file in this directory so that the final structure is:

```text
RAREGuard/
└── data/
    └── raw/
        ├── README.md
        └── creditcard.csv
```

The reproducibility notebook expects the dataset at:

```text
data/raw/creditcard.csv
```

## Important

Do not rename `creditcard.csv`, because the reproducibility notebook checks for this exact filename.

The dataset is excluded from this repository through `.gitignore`. Therefore, each user must obtain the original dataset separately before running the experiments.

After placing the dataset in the correct directory, run:

```text
notebooks/RAREGuard_Reproducibility_Notebook.ipynb
```

from the beginning to reproduce the RAREGuard experiments.

## Data Availability

The dataset remains the property of its original providers and is subject to the terms and conditions of its source platform. This repository provides the RAREGuard implementation and experimental outputs but does not redistribute the original transaction dataset.