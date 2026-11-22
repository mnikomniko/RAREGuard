"""
Check the archived results without the raw data or a GPU.

    python tools/verify_results.py

1. The code in code/rareguard_lib.py is the version recorded in every run_info.json.
2. Every *_by_seed.csv file still has the MD5 hash recorded when it was written.
3. Every *_summary.csv file can be recomputed from its *_by_seed.csv file.
4. The paired tests in results/stats/paired_tests.csv can be recomputed.

Exits with status 1 if anything does not match.
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
import rareguard_lib as rg  # noqa: E402

problems = []


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def check(ok, message):
    print(("  ok    " if ok else "  FAIL  ") + message)
    if not ok:
        problems.append(message)


# 1-2. code version and file hashes ------------------------------------------
print("Code version and file hashes")
lib_md5 = md5(os.path.join(ROOT, "code", "rareguard_lib.py"))
for info_path in sorted(glob.glob(os.path.join(ROOT, "results", "*", "run_info.json"))):
    info = json.load(open(info_path))
    exp = info["experiment"]
    check(info["environment"]["rareguard_lib_md5"] == lib_md5, f"{exp}: rareguard_lib.py matches the run")
    for f in info["files"]:
        path = os.path.join(ROOT, f["file"])
        if f["file"].endswith("_by_seed.csv") and os.path.exists(path):
            check(md5(path) == f["md5"], f"{f['file']} unchanged since the run")

# 3. summaries ---------------------------------------------------------------
print("\nSummaries recomputed from per-seed results")
SUMMARIES = [
    ("E1_components", "runs_by_seed.csv", "summary_mean_sd.csv", ["Threshold_Mode", "Experiment"]),
    ("E2_E3_fusion", "fusion_baselines_by_seed.csv", "fusion_baselines_summary.csv", ["Configuration"]),
    ("E2_E3_fusion", "fusion_weights_by_seed.csv", "fusion_weights_summary.csv", ["Configuration"]),
    ("E4_E5_thresholds", "threshold_methods_by_seed.csv", "threshold_methods_summary.csv", ["Method"]),
    ("E4_E5_thresholds", "assumed_contamination_by_seed.csv", "assumed_contamination_summary.csv", ["Contamination"]),
    ("E6_reference_contamination", "reference_contamination_by_seed.csv", "reference_contamination_summary.csv",
     ["Training_Reference_Contamination"]),
    ("E7_early_stopping", "early_stopping_by_seed.csv", "early_stopping_summary.csv", ["Protocol"]),
    ("E9_paysim", "representation_by_seed.csv", "representation_summary.csv", ["Representation"]),
    ("E9_paysim", "pca_ablation_by_seed.csv", "pca_ablation_summary.csv", ["PCA"]),
    ("E9_paysim", "contamination_by_seed.csv", "contamination_summary.csv", ["Contamination"]),
]
for exp, raw, summ, by in SUMMARIES:
    d = os.path.join(ROOT, "results", exp)
    stored = pd.read_csv(os.path.join(d, summ))
    cols = [c[:-5] for c in stored.columns if c.endswith("_mean")]
    again = rg.mean_sd(pd.read_csv(os.path.join(d, raw)), by, cols)
    merged = stored.merge(again, on=by, suffixes=("", "_new"))
    ok = len(merged) == len(stored) and all(
        np.allclose(merged[c], merged[c + "_new"], equal_nan=True, atol=1e-9)
        for c in stored.columns if c.endswith(("_mean", "_sd")))
    check(ok, f"{exp}/{summ}")

# 4. paired tests ------------------------------------------------------------
print("\nPaired tests recomputed")
tests = pd.read_csv(os.path.join(ROOT, "results", "stats", "paired_tests.csv"))
for (family, source), block in tests.groupby(["Family", "Source"], sort=False):
    df = pd.read_csv(os.path.join(ROOT, "results", source))
    if family == "E1 components":
        df = df[df.Threshold_Mode == "Fully unsupervised"]
    col = {"E1": "Experiment", "E2": "Configuration", "E3": "Configuration", "E4": "Method",
           "E6": "Training_Reference_Contamination", "E7": "Protocol"}.get(family[:2])
    if family.startswith("E9"):
        col = "Representation" if "representation" in source else "PCA"
    ref = block.Reference.iloc[0]
    others = list(dict.fromkeys(block.Condition))
    if col == "Training_Reference_Contamination":
        ref, others = float(ref), [float(o) for o in others]
    again = rg.compare_family(df, col, ref, others, list(dict.fromkeys(block.Metric)), family)
    ok = np.allclose(block.Delta.to_numpy(), again.Delta.to_numpy(), atol=1e-9) and \
        np.allclose(block.p_Holm.to_numpy(), again.p_Holm.to_numpy(), atol=1e-9)
    check(ok, family)

print("\nAll checks passed." if not problems else f"\n{len(problems)} check(s) failed.")
sys.exit(1 if problems else 0)
