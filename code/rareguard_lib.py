"""
rareguard_lib.py

Shared code for the RAREGuard experiments. Every notebook in notebooks/ imports
this file, so the data splits, preprocessing, detectors, thresholds and metrics
are defined once and used the same way everywhere.

If you change anything here, add a line to CHANGES at the bottom and bump
LIB_VERSION. The version and a hash of this file are written into every
run_info.json, so each result file can be traced back to the code that made it.
"""

import os
import json
import time
import math
import random
import shutil
import hashlib
import datetime
import platform

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score,
                             average_precision_score, matthews_corrcoef,
                             confusion_matrix)
from scipy import stats
from scipy.stats import genpareto

LIB_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Configuration and folders
# ---------------------------------------------------------------------------

# These are the settings used in the RAREGuard_v5 notebooks (Steps 1-5).
# Notebook 00 writes them to config.json the first time; after that, edit
# config.json rather than this dictionary.
DEFAULT_CONFIG = {
    "project": "RAREGuard",
    "data": {
        "ulb_csv": "/content/drive/MyDrive/PhD/creditcard.csv",
        "paysim_csv": "/content/drive/MyDrive/PhD/PS_20174392719_1491204439457_log.csv",
    },
    "seeds": [42, 7, 21, 100, 2026],
    "split": {"test_size": 0.30, "val_share_of_rest": 0.50},
    "features": {
        "knn_impute_k": 5,
        "variance_threshold": 1e-5,
        "laplacian_k": 10,
        "laplacian_sample": 5000,
        "n_features": 20,
        "pca_variance": 0.95,
    },
    "detectors": {
        "if_trees": 300,
        "ocsvm_nu": 0.002,
        "ocsvm_max_rows": 30000,
        "ae_lr": 0.001,
        "ae_batch": 1024,
        "ae_epochs": 30,
        "ae_patience": 5,
        "ae_heldout_share": 0.10,
    },
    "fusion_weights": {"IF": 0.30, "OCSVM": 0.30, "AE": 0.40},
    "contamination": 0.002,
    "label_assisted": {"pmin": 80, "pmax": 99.9, "ngrid": 200},
    "e5_contamination_grid": [0.001, 0.0015, 0.002, 0.003, 0.005],
    "e6": {"reference_size": 30000, "fraud_shares": [0.0, 0.001, 0.002, 0.005, 0.01]},
    "e8": {"sizes": [50000, 100000, 150000, 200000, 250000, "all"], "seed": 42},
    "paysim": {"max_rows": 300000, "sample_seed": 31415, "ae_early_stopping": "heldout"},
}

FOLDERS = ["code", "notebooks", "data/splits", "results", "logs",
           "manuscript/figures", "manuscript/tables"]

METRICS = ["Precision", "Recall", "F1", "MCC", "PR_AUC", "ROC_AUC"]
COUNTS = ["TP", "FP", "TN", "FN"]


def init_project(root):
    """Create the folder layout and config.json (only if missing)."""
    for f in FOLDERS:
        os.makedirs(os.path.join(root, f), exist_ok=True)
    cfg_path = os.path.join(root, "config.json")
    if not os.path.exists(cfg_path):
        with open(cfg_path, "w") as fh:
            json.dump(DEFAULT_CONFIG, fh, indent=2)
        print("Wrote a new config.json - check the data paths in it.")
    return load_config(root)


def load_config(root):
    cfg_path = os.path.join(root, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"{cfg_path} not found. Run notebook 00 first.")
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    cfg["root"] = root
    return cfg


def proj(cfg, *parts):
    return os.path.join(cfg["root"], *parts)


def results_dir(cfg, experiment):
    d = proj(cfg, "results", experiment)
    os.makedirs(d, exist_ok=True)
    return d


def file_md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def environment():
    env = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "scipy": __import__("scipy").__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "rareguard_lib": LIB_VERSION,
        "rareguard_lib_md5": file_md5(__file__),
    }
    try:
        import torch
        env["torch"] = torch.__version__
        env["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        env["torch"] = "not installed"
        env["device"] = "cpu"
    return env


def show_environment():
    env = environment()
    print(" | ".join(f"{k}={v}" for k, v in env.items()
                     if k in ("python", "scikit_learn", "torch", "device", "rareguard_lib")))
    return env


# ---------------------------------------------------------------------------
# Run logging: every saved result goes through a Run
# ---------------------------------------------------------------------------

class Run:
    """
    Keeps track of one notebook run.

        run = rg.Run(cfg, "E1_components", "01_E1_components.ipynb")
        run.save_csv(df, "runs_by_seed.csv")
        run.finish()

    If a file already exists it is moved to <experiment>/_archive/<timestamp>/
    before the new one is written, so nothing is silently overwritten.
    finish() writes run_info.json and adds one line to logs/run_log.csv.
    """

    def __init__(self, cfg, experiment, notebook, description=""):
        self.cfg = cfg
        self.experiment = experiment
        self.notebook = notebook
        self.description = description
        self.dir = results_dir(cfg, experiment)
        self.started = datetime.datetime.now()
        self.stamp = self.started.strftime("%Y%m%d_%H%M%S")
        self.files = []
        print(f"[{experiment}] started {self.started:%Y-%m-%d %H:%M}  ->  results/{experiment}/")

    def _archive(self, path):
        if os.path.exists(path):
            rel = os.path.relpath(path, self.dir)
            dest = os.path.join(self.dir, "_archive", self.stamp, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.move(path, dest)

    def _register(self, path, rows=None):
        rel = os.path.relpath(path, self.cfg["root"])
        self.files.append({"file": rel, "rows": rows, "md5": file_md5(path)})
        print(f"  saved {rel}" + (f"  ({rows} rows)" if rows is not None else ""))

    def save_csv(self, df, name, index=False):
        path = os.path.join(self.dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._archive(path)
        df.to_csv(path, index=index)
        self._register(path, len(df))
        return path

    def save_scores(self, name, **arrays):
        path = os.path.join(self.dir, "scores", name + ".npz")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._archive(path)
        packed = {k: (np.asarray(v, dtype=np.float32) if np.asarray(v).dtype.kind == "f" else np.asarray(v))
                  for k, v in arrays.items()}
        np.savez_compressed(path, **packed)
        self._register(path)
        return path

    def save_figure(self, fig, name, dpi=200):
        path = os.path.join(self.dir, "quicklook", name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._archive(path)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        self._register(path)
        return path

    def finish(self, notes=""):
        ended = datetime.datetime.now()
        minutes = (ended - self.started).total_seconds() / 60
        info = {
            "experiment": self.experiment,
            "notebook": self.notebook,
            "description": self.description,
            "started": self.started.isoformat(timespec="seconds"),
            "finished": ended.isoformat(timespec="seconds"),
            "minutes": round(minutes, 2),
            "seeds": self.cfg.get("seeds"),
            "notes": notes,
            "environment": environment(),
            "config": {k: v for k, v in self.cfg.items() if k != "root"},
            "files": self.files,
        }
        info_path = os.path.join(self.dir, "run_info.json")
        self._archive(info_path)
        with open(info_path, "w") as fh:
            json.dump(info, fh, indent=2, default=str)

        log_path = proj(self.cfg, "logs", "run_log.csv")
        line = pd.DataFrame([{
            "finished": info["finished"], "experiment": self.experiment,
            "notebook": self.notebook, "minutes": info["minutes"],
            "device": info["environment"]["device"], "lib_version": LIB_VERSION,
            "files": "; ".join(f["file"] for f in self.files), "notes": notes,
        }])
        line.to_csv(log_path, mode="a", index=False, header=not os.path.exists(log_path))
        print(f"[{self.experiment}] finished in {minutes:.1f} min. "
              f"run_info.json written, logs/run_log.csv updated.")
        return info


def already_done(cfg, experiment, files):
    d = proj(cfg, "results", experiment)
    return all(os.path.exists(os.path.join(d, f)) for f in files)


def read_result(cfg, experiment, name):
    return pd.read_csv(proj(cfg, "results", experiment, name))


def load_scores(cfg, experiment, name):
    path = proj(cfg, "results", experiment, "scores", name + ".npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} is missing. Run the notebook that produces {experiment} first.")
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_ulb(cfg):
    """ULB credit card data: 30 features (Time, V1-V28, Amount) and Class."""
    df = pd.read_csv(cfg["data"]["ulb_csv"])
    if "Class" not in df.columns:
        raise ValueError("Expected a 'Class' column in the ULB file.")
    feature_names = [c for c in df.columns if c != "Class"]
    X = df[feature_names].to_numpy(dtype=np.float32)
    y = df["Class"].to_numpy(dtype=np.int64)
    return X, y, feature_names


PAYSIM_COLUMNS = ["step", "type", "amount", "oldbalanceOrg", "oldbalanceDest", "isFraud"]


def load_paysim_sample(cfg):
    """
    Fixed stratified PaySim sample (300,000 rows, seed 31415), the same rows
    as in Steps 4-4E. Post-transaction balances, isFlaggedFraud and the IDs
    are never read.
    """
    df = pd.read_csv(cfg["data"]["paysim_csv"], usecols=PAYSIM_COLUMNS)
    n = cfg["paysim"]["max_rows"]
    if n and len(df) > n:
        df, _ = train_test_split(df, train_size=n, stratify=df["isFraud"],
                                 random_state=cfg["paysim"]["sample_seed"])
    df = df.rename_axis("source_row").reset_index()
    return df


def paysim_features(df, representation="baseline"):
    """
    baseline: one-hot type, log1p(amount), step (7 columns)
    enhanced: baseline + six features from balances known before the transaction
    """
    X = pd.get_dummies(df[["type"]], prefix="", prefix_sep="", dtype=np.float32)
    X["log1p_amount"] = np.log1p(df["amount"].clip(lower=0)).astype(np.float32)
    X["step"] = df["step"].astype(np.float32)
    if representation == "enhanced":
        amount = df["amount"].clip(lower=0).to_numpy(dtype=np.float64)
        orig = df["oldbalanceOrg"].clip(lower=0).to_numpy(dtype=np.float64)
        dest = df["oldbalanceDest"].clip(lower=0).to_numpy(dtype=np.float64)
        X["log1p_sender_balance_before"] = np.log1p(orig).astype(np.float32)
        X["log1p_recipient_balance_before"] = np.log1p(dest).astype(np.float32)
        X["amount_to_sender_balance_before"] = np.minimum(amount / (orig + 1.0), 100.0).astype(np.float32)
        X["amount_to_recipient_balance_before"] = np.minimum(amount / (dest + 1.0), 100.0).astype(np.float32)
        X["sender_balance_before_zero"] = (orig == 0).astype(np.float32)
        X["recipient_balance_before_zero"] = (dest == 0).astype(np.float32)
    elif representation != "baseline":
        raise ValueError("representation must be 'baseline' or 'enhanced'")
    return X.to_numpy(dtype=np.float32), list(X.columns)


def make_split(y, seed, cfg):
    """70/15/15 stratified split as row indices (identical to the v5 notebooks)."""
    idx = np.arange(len(y))
    train, rest = train_test_split(idx, test_size=cfg["split"]["test_size"],
                                   stratify=y, random_state=seed)
    val, test = train_test_split(rest, test_size=cfg["split"]["val_share_of_rest"],
                                 stratify=y[rest], random_state=seed)
    return train, val, test


def get_split(cfg, dataset, y, seed):
    """Load the saved split for (dataset, seed); create and save it the first time."""
    path = proj(cfg, "data", "splits", f"{dataset}_seed{seed}.npz")
    if os.path.exists(path):
        with np.load(path) as s:
            if int(s["n"]) != len(y) or int(s["n_fraud"]) != int(y.sum()):
                raise ValueError(f"{path} does not match the data (rows or frauds differ). "
                                 "Has the data file changed?")
            return s["train"], s["val"], s["test"]
    train, val, test = make_split(y, seed, cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, train=train, val=val, test=test, n=len(y), n_fraud=int(y.sum()))
    return train, val, test


def split_summary(y, splits):
    rows = []
    for seed, (tr, va, te) in splits.items():
        rows.append({"Seed": seed,
                     "Train_N": len(tr), "Train_Fraud": int(y[tr].sum()),
                     "Val_N": len(va), "Val_Fraud": int(y[va].sum()),
                     "Test_N": len(te), "Test_Fraud": int(y[te].sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Preprocessing and label-free feature selection
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def laplacian_score(X, k=10):
    """Laplacian score per feature (lower = better preserves local structure)."""
    W = kneighbors_graph(X, n_neighbors=k, mode="connectivity", include_self=True).tocsr()
    D = np.asarray(W.sum(axis=1)).ravel().astype(np.float32)
    d_sum = D.sum() + 1e-12
    scores = np.zeros(X.shape[1], dtype=np.float32)
    for j in range(X.shape[1]):
        f = X[:, j].astype(np.float32)
        f_tilde = f - (D @ f) / d_sum
        fDf = (D * f_tilde) @ f_tilde
        fWf = f_tilde @ (W @ f_tilde)
        scores[j] = (fDf - fWf) / (fDf + 1e-12)
    return scores


def prepare_features(X_train, X_val, X_test, feature_names, seed, cfg,
                     use_feature_selection=True, use_pca=True):
    """
    KNN imputation -> min-max scaling -> variance filter -> Laplacian top-k -> PCA.
    Everything is fitted on the training partition only and uses no labels.
    """
    t0 = time.time()
    fc = cfg["features"]

    imputer = KNNImputer(n_neighbors=fc["knn_impute_k"])
    tr = imputer.fit_transform(X_train).astype(np.float32)
    va = imputer.transform(X_val).astype(np.float32)
    te = imputer.transform(X_test).astype(np.float32)

    scaler = MinMaxScaler()
    tr = scaler.fit_transform(tr).astype(np.float32)
    va = scaler.transform(va).astype(np.float32)
    te = scaler.transform(te).astype(np.float32)

    selected = list(feature_names)
    if use_feature_selection:
        vt = VarianceThreshold(threshold=fc["variance_threshold"])
        tr = vt.fit_transform(tr).astype(np.float32)
        va = vt.transform(va).astype(np.float32)
        te = vt.transform(te).astype(np.float32)
        kept = [feature_names[i] for i in np.where(vt.get_support())[0]]

        n_sample = min(fc["laplacian_sample"], tr.shape[0])
        rng = np.random.RandomState(seed)
        sample_idx = rng.choice(tr.shape[0], size=n_sample, replace=False)
        lap = laplacian_score(tr[sample_idx], k=fc["laplacian_k"])
        top = np.argsort(lap)[:min(fc["n_features"], tr.shape[1])]

        selected = [kept[i] for i in top]
        tr, va, te = tr[:, top], va[:, top], te[:, top]

    if use_pca:
        pca = PCA(n_components=fc["pca_variance"], random_state=seed)
        tr = pca.fit_transform(tr).astype(np.float32)
        va = pca.transform(va).astype(np.float32)
        te = pca.transform(te).astype(np.float32)

    return {"train": tr, "val": va, "test": te, "selected": selected,
            "dims": int(tr.shape[1]), "seconds": time.time() - t0}


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

def minmax_from_train(train, val, test):
    lo, hi = np.min(train), np.max(train)
    den = hi - lo + 1e-12
    return (train - lo) / den, (val - lo) / den, (test - lo) / den


def fit_if(F, seed, cfg, fit_rows=None):
    set_seed(seed)
    t0 = time.time()
    X_fit = F["train"] if fit_rows is None else F["train"][fit_rows]
    model = IsolationForest(n_estimators=cfg["detectors"]["if_trees"], contamination="auto",
                            random_state=seed, n_jobs=-1)
    model.fit(X_fit)
    raw = [-model.decision_function(F[p]) for p in ("train", "val", "test")]
    return raw, time.time() - t0


def fit_ocsvm(F, y_train, seed, cfg, fit_rows=None):
    """Default reference: up to 30,000 randomly drawn legitimate training rows."""
    set_seed(seed)
    t0 = time.time()
    if fit_rows is None:
        normal = np.where(y_train == 0)[0]
        n = min(cfg["detectors"]["ocsvm_max_rows"], len(normal))
        fit_rows = np.random.RandomState(seed).choice(normal, size=n, replace=False)
    model = OneClassSVM(kernel="rbf", gamma="scale", nu=cfg["detectors"]["ocsvm_nu"])
    model.fit(F["train"][fit_rows])
    raw = [-model.decision_function(F[p]).ravel() for p in ("train", "val", "test")]
    return raw, time.time() - t0


def train_autoencoder(F, y_train, seed, cfg, fit_rows=None, early_stopping="validation"):
    """
    Autoencoder d-64-32-16-32-64-d trained on legitimate training rows.

    early_stopping='validation': monitor reconstruction loss on the validation
        partition (reference protocol, no validation labels used).
    early_stopping='heldout': hold out 10% of the legitimate training rows for
        stopping; those rows are never used for gradient updates.
    fit_rows: explicit training rows (used by E6 to inject fraud into the reference).
    Returns raw scores (train, val, test), seconds, and the number of epochs run.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    set_seed(seed)
    t0 = time.time()
    dc = cfg["detectors"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = F["train"].shape[1]

    model = nn.Sequential(
        nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, d),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=dc["ae_lr"])
    loss_fn = nn.MSELoss()

    if fit_rows is None:
        fit_rows = np.flatnonzero(np.asarray(y_train) == 0)
    if early_stopping == "heldout":
        fit_rows, stop_rows = train_test_split(fit_rows, test_size=dc["ae_heldout_share"],
                                               random_state=seed, shuffle=True)
        X_stop = torch.tensor(F["train"][stop_rows], dtype=torch.float32).to(device)
    elif early_stopping == "validation":
        X_stop = torch.tensor(F["val"], dtype=torch.float32).to(device)
    else:
        raise ValueError("early_stopping must be 'validation' or 'heldout'")

    loader = DataLoader(TensorDataset(torch.tensor(F["train"][fit_rows], dtype=torch.float32)),
                        batch_size=dc["ae_batch"], shuffle=True)

    best_loss, best_state, waited, epochs_run = math.inf, None, 0, 0
    for epoch in range(1, dc["ae_epochs"] + 1):
        epochs_run = epoch
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            stop_loss = loss_fn(model(X_stop), X_stop).item()
        if stop_loss < best_loss - 1e-7:
            best_loss, waited = stop_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
        if waited >= dc["ae_patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    raw = []
    with torch.no_grad():
        for p in ("train", "val", "test"):
            X = torch.tensor(F[p], dtype=torch.float32).to(device)
            raw.append(((model(X) - X) ** 2).mean(dim=1).cpu().numpy())
    return raw, time.time() - t0, epochs_run


def fit_detectors(F, y_train, seed, cfg, detectors=("IF", "OCSVM", "AE"),
                  ae_early_stopping="validation", fit_rows=None):
    """
    Fit the requested detectors and return min-max normalised scores.

    Returns (scores, seconds, info) where scores[name] = (train, val, test).
    Normalisation uses the minimum and maximum of each detector's scores on
    all training rows, as in the v5 notebooks.
    """
    fit_rows = fit_rows or {}
    scores, seconds, info = {}, {}, {}
    if "IF" in detectors:
        raw, seconds["IF"] = fit_if(F, seed, cfg, fit_rows.get("IF"))
        scores["IF"] = minmax_from_train(*raw)
    if "OCSVM" in detectors:
        raw, seconds["OCSVM"] = fit_ocsvm(F, y_train, seed, cfg, fit_rows.get("OCSVM"))
        scores["OCSVM"] = minmax_from_train(*raw)
    if "AE" in detectors:
        raw, seconds["AE"], info["AE_epochs"] = train_autoencoder(
            F, y_train, seed, cfg, fit_rows.get("AE"), ae_early_stopping)
        scores["AE"] = minmax_from_train(*raw)
    return scores, seconds, info


# ---------------------------------------------------------------------------
# Fusion rules (work on dicts of score arrays, e.g. {"IF": a, "OCSVM": b, "AE": c})
# ---------------------------------------------------------------------------

def weighted_fusion(parts, weights):
    return sum(w * parts[m] for m, w in weights.items() if w)


def median_fusion(parts, members=("IF", "OCSVM", "AE")):
    return np.median(np.vstack([parts[m] for m in members]), axis=0)


def rank_fusion(parts, train_parts, members=("IF", "OCSVM", "AE")):
    """
    Mean percentile rank. Each detector's score is mapped to its empirical CDF
    on the training scores, so validation and test use the same scale.
    """
    ranks = []
    for m in members:
        ref = np.sort(train_parts[m])
        ranks.append(np.searchsorted(ref, parts[m], side="right") / len(ref))
    return np.mean(ranks, axis=0)


def scores_from_npz(z, part):
    return {m: z[f"{m}_{part}"] for m in ("IF", "OCSVM", "AE") if f"{m}_{part}" in z}


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

def percentile_threshold(val_scores, contamination):
    return float(np.percentile(val_scores, 100 * (1 - contamination)))


def label_assisted_threshold(y_val, val_scores, cfg):
    """Reference only: picks the validation-F1-best percentile (uses validation labels)."""
    la = cfg["label_assisted"]
    best_f1, best_t = -1.0, None
    for t in np.percentile(val_scores, np.linspace(la["pmin"], la["pmax"], la["ngrid"])):
        _, _, f1, _ = precision_recall_fscore_support(y_val, (val_scores >= t).astype(int),
                                                      average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def label_free_thresholds(val_scores, contamination=0.002, mad_k=3.5, iqr_k=1.5, evt_tail=0.10):
    """Percentile, EVT (GPD peaks-over-threshold), median+MAD and IQR thresholds."""
    s = np.asarray(val_scores, dtype=float)
    s = s[np.isfinite(s)]
    med = np.median(s)
    mad = np.median(np.abs(s - med))
    q1, q3 = np.percentile(s, [25, 75])
    out = {
        "Percentile": float(np.quantile(s, 1 - contamination)),
        "EVT_GPD": np.nan,
        "MAD": float(med + mad_k * 1.4826 * mad),
        "IQR": float(q3 + iqr_k * (q3 - q1)),
    }
    u = float(np.quantile(s, 1 - evt_tail))
    excess = s[s > u] - u
    if len(excess) >= 30 and not np.allclose(excess, 0):
        try:
            shape, _, scale = genpareto.fit(excess, floc=0)
            survival = contamination / (len(excess) / len(s))
            out["EVT_GPD"] = float(u + genpareto.isf(survival, shape, loc=0, scale=scale))
        except (ValueError, FloatingPointError):
            pass
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate(y_true, scores, threshold, val_scores=None):
    y_pred = (scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    out = {
        "Threshold": float(threshold),
        "Precision": p, "Recall": r, "F1": f1,
        "MCC": matthews_corrcoef(y_true, y_pred),
        "PR_AUC": average_precision_score(y_true, scores),
        "ROC_AUC": roc_auc_score(y_true, scores),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "Test_N": n, "Test_Fraud": int(np.sum(y_true)),
        "Test_Alert_Rate": float(y_pred.mean()),
        "Alerts_per_10k": 1e4 * (tp + fp) / n,
        "TP_per_10k": 1e4 * tp / n,
        "FP_per_10k": 1e4 * fp / n,
    }
    if val_scores is not None:
        out["Val_Alert_Rate"] = float(np.mean(val_scores >= threshold))
    return out


def mean_sd(df, by, columns=None):
    """Mean and SD across seeds with flat column names, e.g. F1_mean, F1_sd."""
    columns = columns or [c for c in METRICS if c in df.columns]
    g = df.groupby(by, sort=False)[columns].agg(["mean", "std"])
    g.columns = [f"{c}_{'sd' if s == 'std' else s}" for c, s in g.columns]
    g.insert(0, "N_Seeds", df.groupby(by, sort=False).size())
    return g.reset_index()


# ---------------------------------------------------------------------------
# Paired statistics (used by notebooks 08 and 09)
# ---------------------------------------------------------------------------

def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def paired_difference(reference, other):
    """other - reference across matched seeds: mean, 95% t CI, paired t p, exact Wilcoxon p."""
    a, b = np.asarray(reference, float), np.asarray(other, float)
    d = b - a
    n = len(d)
    sd = d.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
    if sd > 1e-14:
        p_t = float(stats.ttest_rel(b, a).pvalue)
    else:
        p_t = 1.0 if abs(d.mean()) < 1e-14 else 0.0
    try:
        p_w = float(stats.wilcoxon(d, zero_method="wilcox", method="exact").pvalue)
    except ValueError:
        p_w = np.nan
    return {"N_Seeds": n, "Reference_Mean": a.mean(), "Other_Mean": b.mean(),
            "Delta": d.mean(), "Delta_SD": sd, "CI95_Low": d.mean() - half, "CI95_High": d.mean() + half,
            "Seeds_Higher": int((d > 0).sum()), "Seeds_Lower": int((d < 0).sum()),
            "p_t": p_t, "p_Wilcoxon": p_w}


def compare_family(df, condition_col, reference, others, metrics, family, seed_col="Seed"):
    """Compare each condition with the reference; Holm-adjust within the family, per metric."""
    rows = []
    for metric in metrics:
        wide = df.pivot_table(index=seed_col, columns=condition_col, values=metric, aggfunc="first")
        block = []
        for other in others:
            pair = wide[[reference, other]].dropna()
            res = paired_difference(pair[reference], pair[other])
            block.append({"Family": family, "Metric": metric, "Reference": reference,
                          "Condition": other, **res})
        adj = holm([r["p_t"] for r in block])
        for r, a in zip(block, adj):
            r["p_Holm"] = a
            r["Family_Size"] = len(block)
        rows.extend(block)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CHANGES
# 1.0  2026-11-22  First shared version. Settings copied from the RAREGuard_v5
#                  notebooks (Steps 1-5). Split, preprocessing, detectors and
#                  thresholds unchanged; feature_names is now passed explicitly
#                  instead of read from a global variable.
# ---------------------------------------------------------------------------
