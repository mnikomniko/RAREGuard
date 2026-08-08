# Reproducibility Guide

The experiments reported in the manuscript can be reproduced by executing

RAREGuard_Reproducibility_Notebook.ipynb

Requirements

Python 3.11

Install dependencies

pip install -r requirements.txt

Download the Credit Card Fraud Detection dataset from Kaggle.

Place

creditcard.csv

inside

data/raw/

Run all notebook cells sequentially.

The notebook reproduces:

• Fully unsupervised experiments

• Label-assisted experiments

• Ablation study

• Contamination-rate sensitivity analysis

• Computational cost analysis

• Scalability analysis

The generated figures correspond directly to those reported in the manuscript.