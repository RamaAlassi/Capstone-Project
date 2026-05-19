from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibrationDisplay
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    pairwise_distances,
)
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Try to import StratifiedGroupKFold (available in newer sklearn versions)
try:
    from sklearn.model_selection import StratifiedGroupKFold
    STRATIFIED_GROUP_AVAILABLE = True
except Exception:
    StratifiedGroupKFold = None
    STRATIFIED_GROUP_AVAILABLE = False

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

# Input and output paths
BASE_DIR  = Path(r"C:\Users\admin\Desktop\Machine Learning For Capstone")
DATA_DIR  = BASE_DIR / "Data Split Outputs"

TRAIN_FILE = DATA_DIR / "classification_train_dataset.csv"
TEST_FILE  = DATA_DIR / "classification_test_dataset.csv"

OUTDIR = BASE_DIR / "Activity Classifier Output"

#  Column names that are metadata, not features
METADATA_COLS = ["Canonical_SMILES", "Scaffold", "Split"]
TARGET_COL    = "Activity_Class"

# Experiment settings
RANDOM_STATE = 42 # fixed seed for reproducibility
N_SPLITS     = 5 # for cross validation
N_JOBS       = 1 # CPU thing (one processor/core only)
ENABLE_XGBOOST = True 
ENABLE_XGBOOST_GPU = True # use gpu for XGboos becuase it was takeing time

N_ITER_SEARCH  = 60 # how many random hyperparameter combinations are tested
N_ITER_XGBOOST = 40 # how many random hyperparameter combinations are tested for xgboost
N_ITER_MULTISPACE = 60

TOP_N_FEATURES_TO_PLOT = 20 # nothing related to training

# Model selection rules
SEARCH_PRIMARY_METRIC = "mcc"
USE_ONE_STANDARD_ERROR_RULE = True
ONE_SE_MULTIPLIER = 1.0 # best CV MCC - (1 * standard error)

PRIMARY_SELECTION_METRIC = "cv_mcc_mean"
MODEL_LEVEL_CV_TOLERANCE = 0.07
MAX_ACCEPTABLE_TRAIN_CV_MCC_GAP = 0.18


# Creates a directory and all parent directories if they do not exist
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# Converts a label string to a safe filename-friendly format
def sanitize_name(text: str) -> str:
    return (
        str(text)
        .replace(" ", "_").replace("/", "_").replace("\\", "_")
        .replace("(", "").replace(")", "").replace("-", "_")
        .lower()
    )


# Returns a float from any value, or a default if conversion fails
def safe_metric_value(value, default=np.nan) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


# Returns the output size of dataset after a pipeline step
def _step_effective_dim(step, fallback_dim: int) -> float:
    if step is None or step == "passthrough":
        return float(fallback_dim)

    n_components = getattr(step, "n_components", None)
    if n_components is not None:
        return float(n_components)

    named_steps = getattr(step, "named_steps", None)
    if named_steps:
        kbest = named_steps.get("kbest")
        k = getattr(kbest, "k", None)
        if isinstance(k, (int, np.integer)):
            return float(k)

    return float(fallback_dim)


# model selection after RandomizedSearchCV
def param_complexity_score(model_name: str, params: dict, original_n_features: int) -> float:
    complexity = float(original_n_features) #The higher the score, the more complex the model is considered.

    if "multispace__k_desc" in params:
        complexity = float(
            params.get("multispace__k_desc", 0)
            + params.get("multispace__k_maccs", 0)
            + params.get("multispace__k_morgan", 0)
        )
    else: #complexity = original number of features
        complexity = min(
            complexity,
            _step_effective_dim(params.get("selector"), original_n_features),
            _step_effective_dim(params.get("pca"), original_n_features),
        )
    # assign a penalty for each model based on the hyperparameters
    if model_name.startswith("LogisticRegression"):
        C = max(float(params.get("model__C", 0.1)), 1e-12)
        l1_ratio = float(params.get("model__l1_ratio", 0.5))
        complexity += (8.0 * math.log10(1.0 + C)) + (2.0 * (1.0 - l1_ratio))
        # when C is larger / L1 ia lower => penalty
    elif model_name.startswith("SVM_RBF"):
        C = max(float(params.get("model__C", 1.0)), 1e-12)
        gamma = params.get("model__gamma", "scale")
        gamma_penalty = 1.0 if gamma == "scale" else (1000.0 * float(gamma))
        complexity += (10.0 * math.log10(1.0 + C)) + (25.0 * gamma_penalty)
        # larger c / larger gamma => penalty
    elif model_name.startswith("RandomForest"):
        n_estimators = float(params.get("model__n_estimators", 200))
        max_depth_raw = params.get("model__max_depth", 4)
        max_depth = 25.0 if max_depth_raw is None else float(max_depth_raw)
        min_leaf = float(params.get("model__min_samples_leaf", 1))
        complexity += (0.02 * n_estimators) + (5.0 * max_depth) + (2.0 / max(min_leaf, 1.0))

    elif model_name.startswith("XGBoost"):
        n_estimators = float(params.get("model__n_estimators", 200))
        max_depth = float(params.get("model__max_depth", 3))
        learning_rate = max(float(params.get("model__learning_rate", 0.05)), 1e-12)
        min_child_weight = float(params.get("model__min_child_weight", 1))
        complexity += (
            (0.02 * n_estimators)
            + (6.0 * max_depth)
            + (1.0 / learning_rate)
            + (1.0 / max(min_child_weight, 1.0))
        )
    #more trees, deeper trees,small learning rate, small weight
    return float(complexity)


# Picks the best hyperparameter index from a search using the 1-SE rule
def choose_cv_candidate_index(
    search: RandomizedSearchCV,
    model_name: str,
    n_features_input: int,
) -> tuple[int, dict]:
    results = search.cv_results_ #cross validation scores

    primary_scores = np.asarray(results[f"mean_test_{SEARCH_PRIMARY_METRIC}"], dtype=float)
    if not np.isfinite(primary_scores).any():
        raise RuntimeError(f"No valid CV scores for {SEARCH_PRIMARY_METRIC}.")

    best_idx_by_mean = int(np.nanargmax(primary_scores)) # find the highest
    best_mean = float(primary_scores[best_idx_by_mean])

    selection_note = {
        "primary_metric": SEARCH_PRIMARY_METRIC,
        "one_standard_error_rule": USE_ONE_STANDARD_ERROR_RULE,
        "best_mean_cv_score": best_mean,
    }

    # Collect all parameter sets within one standard error of the best score
    if USE_ONE_STANDARD_ERROR_RULE:
        primary_std = np.asarray(
            results.get(f"std_test_{SEARCH_PRIMARY_METRIC}", np.zeros_like(primary_scores)),
            dtype=float,
        )
        best_std = safe_metric_value(primary_std[best_idx_by_mean], default=0.0)
        best_se = best_std / math.sqrt(max(1, N_SPLITS))
        cutoff = best_mean - (ONE_SE_MULTIPLIER * best_se)
        candidate_indices = [
            i for i, score in enumerate(primary_scores)
            if np.isfinite(score) and float(score) >= cutoff
        ]
        selection_note.update({
            "best_std": best_std,
            "best_se": best_se,
            "one_se_cutoff": cutoff,
            "n_candidates_within_cutoff": len(candidate_indices),
        })
    else:
        candidate_indices = [best_idx_by_mean]

    train_scores = np.asarray(
        results.get(f"mean_train_{SEARCH_PRIMARY_METRIC}", np.full_like(primary_scores, np.nan)),
        dtype=float,
    )
    train_cv_gap = np.abs(train_scores - primary_scores)
    roc_scores = np.asarray(results.get("mean_test_roc_auc", np.full_like(primary_scores, np.nan)), dtype=float)
    ba_scores = np.asarray(
        results.get("mean_test_balanced_accuracy", np.full_like(primary_scores, np.nan)),
        dtype=float,
    )
    f1_scores = np.asarray(results.get("mean_test_f1_macro", np.full_like(primary_scores, np.nan)), dtype=float)
    param_list = results["params"]

    # Prefer candidates whose train-CV gap does not exceed the tolerance (keep the ones less than 0.18)
    acceptable_gap_indices = [
        i for i in candidate_indices
        if safe_metric_value(train_cv_gap[i], default=np.inf) <= MAX_ACCEPTABLE_TRAIN_CV_MCC_GAP
    ]
    ranked_pool = acceptable_gap_indices or candidate_indices

    # Sort candidates: smallest gap first, then simplest model, then best CV MCC
    ranked_candidates = sorted(
        ranked_pool,
        key=lambda i: (
            safe_metric_value(train_cv_gap[i], default=1e12),
            param_complexity_score(
                model_name=model_name,
                params=param_list[i],
                original_n_features=n_features_input,
            ),
            -safe_metric_value(primary_scores[i], default=-1e12),
            -safe_metric_value(roc_scores[i], default=-1e12),
            -safe_metric_value(ba_scores[i], default=-1e12),
            -safe_metric_value(f1_scores[i], default=-1e12),
        ), # rank them
    )
    # choose the best
    chosen_idx = int(ranked_candidates[0])
    selection_note.update({
        "chosen_idx": chosen_idx,
        "chosen_train_cv_gap": safe_metric_value(train_cv_gap[chosen_idx], default=np.nan),
        "chosen_complexity_score": param_complexity_score(
            model_name=model_name,
            params=param_list[chosen_idx],
            original_n_features=n_features_input,
        ),
        "used_gap_filter": bool(acceptable_gap_indices),
    })
    return chosen_idx, selection_note


# Checks that train and test files have the expected columns and classes
def validate_inputs(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    required_cols = set(METADATA_COLS + [TARGET_COL])
    missing_train = required_cols - set(train_df.columns)
    missing_test  = required_cols - set(test_df.columns)
    if missing_train:
        raise ValueError(f"Missing columns in training file: {sorted(missing_train)}")
    if missing_test:
        raise ValueError(f"Missing columns in test file: {sorted(missing_test)}")
    ft = [c for c in train_df.columns if c not in METADATA_COLS + [TARGET_COL]]
    fv = [c for c in test_df.columns  if c not in METADATA_COLS + [TARGET_COL]]
    if ft != fv:
        raise ValueError("Train and test feature columns do not match.")
    if train_df[ft].isna().any().any():
        raise ValueError("Training features contain missing values.")
    if test_df[fv].isna().any().any():
        raise ValueError("Test features contain missing values.")
    tc = sorted(pd.Series(train_df[TARGET_COL]).dropna().unique().tolist())
    vc = sorted(pd.Series(test_df[TARGET_COL]).dropna().unique().tolist())
    if tc != [0, 1]: raise ValueError(f"Training classes must be [0,1]. Found: {tc}")
    if vc != [0, 1]: raise ValueError(f"Test classes must be [0,1]. Found: {vc}")


#  Separates all feature columns into descriptor, MACCS, and Morgan groups 
def get_feature_columns(df: pd.DataFrame) -> tuple[list, list, list]:
    all_feats   = [c for c in df.columns if c not in METADATA_COLS + [TARGET_COL]]
    maccs_cols  = [c for c in all_feats if c.startswith("MACCS_")]
    morgan_cols = [c for c in all_feats if c.startswith("Morgan_")]
    desc_cols   = [c for c in all_feats if c not in maccs_cols + morgan_cols]
    return desc_cols, maccs_cols, morgan_cols


# Returns a cross-validation splitter respecting scaffold groupings
def get_cv(groups: pd.Series):
    if groups.nunique() < N_SPLITS:
        raise ValueError(f"Need ≥{N_SPLITS} unique scaffolds for {N_SPLITS}-fold CV, got {groups.nunique()}")
    if STRATIFIED_GROUP_AVAILABLE:
        return StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return GroupKFold(n_splits=N_SPLITS)


#  Computes the class imbalance ratio (negatives / positives)
def get_scale_pos_weight(y: np.ndarray) -> float:
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0: raise ValueError("Positive class count is zero.")
    return neg / pos


def get_xgboost_acceleration_params() -> dict:
    if not ENABLE_XGBOOST_GPU or not XGBOOST_AVAILABLE:
        return {"tree_method": "hist"}

    try:
        supported_params = XGBClassifier().get_params()
    except Exception:
        supported_params = {}

    if "device" in supported_params:
        return {"tree_method": "hist", "device": "cuda"}
    return {"tree_method": "gpu_hist", "predictor": "gpu_predictor"}


#Recursively converts numpy types to plain Python types for JSON export
def make_json_safe(value):
    if isinstance(value, dict):          return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [make_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):    return value.tolist()
    if isinstance(value, np.integer):    return int(value)
    if isinstance(value, np.floating):   return float(value)
    if isinstance(value, np.bool_):      return bool(value)
    if isinstance(value, Path):          return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None: return value
    return str(value)


# Selects top-k features independently from each chemical feature space
class MultiSpaceSelector(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        desc_cols:   list[str] | None = None,
        maccs_cols:  list[str] | None = None,
        morgan_cols: list[str] | None = None,
        k_desc:   int = 75,
        k_maccs:  int = 40,
        k_morgan: int = 50,
    ):
        self.desc_cols   = desc_cols   or []
        self.maccs_cols  = maccs_cols  or []
        self.morgan_cols = morgan_cols or []
        self.k_desc      = k_desc
        self.k_maccs     = k_maccs
        self.k_morgan    = k_morgan

    # convert train Kbest for Multispace
    def fit(self, X, y=None):
        X = self._to_df(X)
        self._sel_desc_   = self._fit_sel(X, y, self.desc_cols,   self.k_desc)
        self._sel_maccs_  = self._fit_sel(X, y, self.maccs_cols,  self.k_maccs)
        self._sel_morgan_ = self._fit_sel(X, y, self.morgan_cols, self.k_morgan)
        return self

    # Applies each fitted selector and stacks the results horizontally
    def transform(self, X):
        X = self._to_df(X)
        parts = []
        for sel, cols in [
            (self._sel_desc_,   self.desc_cols),
            (self._sel_maccs_,  self.maccs_cols),
            (self._sel_morgan_, self.morgan_cols),
        ]:
            if sel is None:
                continue
            arr = X[cols].values[:, sel._var_mask]
            parts.append(sel.transform(arr))
        return np.hstack(parts) if parts else np.zeros((len(X), 0))

    @staticmethod
    def _to_df(X):
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    @staticmethod
    def _fit_sel(X_df, y, cols, k):
        if not cols: return None
        X_arr = X_df[cols].values
        # Remove constant columns before scoring to avoid divide-by-zero
        var_mask = X_arr.var(axis=0) > 0
        live_cols = [c for c, keep in zip(cols, var_mask) if keep]
        if not live_cols: return None
        k_eff = min(int(k), len(live_cols))
        if k_eff <= 0: return None
        sel = SelectKBest(f_classif, k=k_eff)
        sel.fit(X_arr[:, var_mask], y)
        sel._col_names = live_cols
        sel._orig_cols = cols
        sel._var_mask  = var_mask
        return sel

    # Returns the names of all features that were selected across all groups
    def get_selected_feature_names(self) -> list[str]:
        names = []
        for sel, _ in [
            (self._sel_desc_,   self.desc_cols),
            (self._sel_maccs_,  self.maccs_cols),
            (self._sel_morgan_, self.morgan_cols),
        ]:
            if sel is not None:
                mask = sel.get_support()
                names.extend([c for c, keep in zip(sel._col_names, mask) if keep])
        return names


# Computes specificity (true negative rate) from a confusion matrix
def specificity_score(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan


# Computes all classification metrics at once and returns them as a dict 
def compute_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    return {
        "roc_auc":            roc_auc_score(y_true, y_score),
        "pr_auc":             average_precision_score(y_true, y_score),
        "mcc":                matthews_corrcoef(y_true, y_pred),
        "balanced_accuracy":  balanced_accuracy_score(y_true, y_pred),
        "accuracy":           accuracy_score(y_true, y_pred),
        "precision":          precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity":        specificity_score(y_true, y_pred),
        "f1_score":           f1_score(y_true, y_pred, zero_division=0),
        "f1_macro":           f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


#  Returns the dict of scoring functions used during cross-validation
def get_scoring_dict() -> dict[str, object]:
    return {
        "roc_auc":           "roc_auc",
        "pr_auc":            "average_precision",
        "mcc":               make_scorer(matthews_corrcoef),
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro":          make_scorer(f1_score, average="macro", zero_division=0),
        "f1_weighted":       make_scorer(f1_score, average="weighted", zero_division=0),
        "specificity":       make_scorer(specificity_score),
        "precision":         make_scorer(precision_score, zero_division=0),
        "recall":            make_scorer(recall_score, zero_division=0),
    }


# Gets probability scores or decision scores depending on what the model has
def get_proba_or_decision(estimator, X) -> tuple[np.ndarray, str]:
    if hasattr(estimator, "predict_proba"):
        p = estimator.predict_proba(X)
        if p.ndim == 2 and p.shape[1] >= 2:
            return np.asarray(p[:, 1], dtype=float).ravel(), "predict_proba_class1"
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X), dtype=float).ravel(), "decision_function"
    raise ValueError("Estimator exposes neither predict_proba nor decision_function.")


#  Searches all possible thresholds and picks the one that maximises MCC
def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    scores = np.asarray(y_score, dtype=float)
    scores = scores[np.isfinite(scores)]
    unique = np.unique(scores)
    candidates = unique if len(unique) <= 201 else np.quantile(unique, np.linspace(0.01, 0.99, 201))
    best = None
    for thr in np.unique(candidates):
        y_pred  = (y_score >= thr).astype(int)
        mcc     = matthews_corrcoef(y_true, y_pred)
        ba      = balanced_accuracy_score(y_true, y_pred)
        if best is None or mcc > best["mcc"] or (np.isclose(mcc, best["mcc"]) and ba > best["ba"]):
            best = {"threshold": float(thr), "mcc": mcc, "ba": ba}
    return best


#  Plots the ROC curve with AUC annotation 
def plot_roc_curve(y_true, y_score, title, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, color="#2E86AB", label=f"ROC-AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=10); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(save_path, dpi=300, bbox_inches="tight"); plt.close(fig)


#  Plots the precision-recall curve with average precision annotation
def plot_pr_curve(y_true, y_score, title, save_path):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(rec, prec, lw=2, color="#A23B72", label=f"PR-AUC = {ap:.3f}")
    ax.set_xlabel("Recall", fontsize=11); ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=10); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(save_path, dpi=300, bbox_inches="tight"); plt.close(fig)


#  Plots a colour-coded confusion matrix with count labels
def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Inactive", "Active"], fontsize=10)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Inactive", "Active"], fontsize=10)
    ax.set_xlabel("Predicted", fontsize=10); ax.set_ylabel("True", fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(save_path, dpi=300, bbox_inches="tight"); plt.close(fig)


# Runs CV per fold and draws a bar chart of MCC per fold with mean and SD (for reference onlya)
def plot_cv_mcc_per_fold(
    estimator,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    cv,
    model_label: str,
    save_path: Path,
) -> dict:
    cv_res = cross_validate(
        estimator, X, y,
        cv=cv, groups=groups,
        scoring=make_scorer(matthews_corrcoef),
        return_train_score=True,
        n_jobs=N_JOBS,
    )
    fold_val   = cv_res["test_score"]
    fold_train = cv_res["train_score"]
    mean_val   = float(np.mean(fold_val))
    std_val    = float(np.std(fold_val, ddof=1))
    n_folds    = len(fold_val)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8F9FA")

    x   = np.arange(n_folds)
  
    bar_colors = ["#2B7A78" if s >= mean_val else "#C0392B" for s in fold_val]

    bars = ax.bar(x, fold_val, width=0.55, color=bar_colors,
                  alpha=0.90, zorder=3, linewidth=0, label="Fold MCC")


    ax.scatter(x, fold_train, marker="D", s=55, color="#444444",
               zorder=5, label="Train MCC (within fold)", linewidths=0)

    ax.axhline(mean_val, color="#1A1A2E", lw=2.0, ls="--", zorder=4,
               label=f"CV Mean MCC = {mean_val:.3f}")

    ax.axhspan(mean_val - std_val, mean_val + std_val,
               color="#1A1A2E", alpha=0.07, zorder=2,
               label=f"±1 SD  (SD = {std_val:.3f})")

    for bar, val in zip(bars, fold_val):
        offset = 0.012 if val >= 0 else -0.012
        va = "bottom" if val >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                f"{val:.3f}", ha="center", va=va,
                fontsize=9.5, fontweight="600", color="#1A1A2E")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i+1}" for i in range(n_folds)],
                       fontsize=11, fontfamily="DejaVu Sans")
    ax.set_ylabel("Matthews Correlation Coefficient (MCC)",
                  fontsize=12, fontfamily="DejaVu Sans", labelpad=8)
    ax.set_xlabel("Cross-Validation Fold  (Scaffold-stratified)",
                  fontsize=12, fontfamily="DejaVu Sans", labelpad=8)
    ax.set_title(
        f"Per-Fold MCC — {model_label}\n"
        f"CV Mean ± SD = {mean_val:.3f} ± {std_val:.3f}",
        fontsize=12.5, fontfamily="DejaVu Sans", fontweight="bold", pad=14,
    )

    y_lo = min(0.0, float(np.min(fold_val)) - 0.10)
    y_hi = min(1.02, max(float(np.max(fold_val)), float(np.max(fold_train))) + 0.14)
    ax.set_ylim(y_lo, y_hi)

    ax.yaxis.grid(True, ls=":", lw=0.8, color="#CCCCCC", zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#BBBBBB")

    ax.legend(fontsize=9.5, framealpha=0.90, loc="lower right",
              frameon=True, edgecolor="#CCCCCC")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)

    return {
        "fold_val_scores":   fold_val.tolist(),
        "fold_train_scores": fold_train.tolist(),
        "mean": mean_val,
        "std":  std_val,
    }


# Plots a horizontal bar chart of the top-N most important features
def plot_feature_importance(importances_df, title, save_path, top_n=20):
    top = importances_df.head(top_n).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"], top["importance"], color="#2B7A78")
    ax.set_xlabel("Importance", fontsize=11); ax.set_ylabel("Feature", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(save_path, dpi=300, bbox_inches="tight"); plt.close(fig)


# Extracts the feature representation after the selector or PCA step
def get_representation_space(estimator, X: pd.DataFrame) -> np.ndarray:
    Xs = X.copy()
    if hasattr(estimator, "named_steps"):
        for step_name in ["multispace", "selector", "pca"]:
            step = estimator.named_steps.get(step_name)
            if step is not None and step != "passthrough":
                Xs = step.transform(Xs)
    Xs = np.asarray(Xs, dtype=float)
    return Xs.reshape(-1, 1) if Xs.ndim == 1 else Xs


# Builds a kNN applicability domain by computing distance thresholds
def build_knn_ad_artifacts(X_space: np.ndarray, k: int = 5,
                            percentile: float = 95.0) -> dict:
    X_space = np.asarray(X_space, dtype=float)
    n = X_space.shape[0]
    if n == 0: raise ValueError("Empty training matrix for AD.")
    if n == 1:
        return {"method": "knn_mean_distance", "k": 0,
                "threshold": float("inf"), "reference": X_space.astype(np.float32),
                "space_dim": int(X_space.shape[1]),
                "train_mean_knn_distance_summary": {"min": 0.0, "median": 0.0, "max": 0.0}}
    k_eff   = min(k, n - 1)
    dm      = pairwise_distances(X_space, metric="euclidean")
    np.fill_diagonal(dm, np.inf)
    dists   = np.sort(dm, axis=1)[:, :k_eff].mean(axis=1)
    return {
        "method": "knn_mean_distance", "k": k_eff,
        "percentile": float(percentile),
        "threshold": float(np.quantile(dists, percentile / 100)),
        "reference": X_space.astype(np.float32),
        "space_dim": int(X_space.shape[1]),
        "train_mean_knn_distance_summary": {
            "min": float(dists.min()), "median": float(np.median(dists)),
            "max": float(dists.max()),
        },
    }


# Defines the standard single-space model configurations and hyperparameter grids 
def get_standard_model_specs(scale_pos_weight: float) -> dict:
    specs = {}

    specs["LogisticRegression"] = {
        "estimator": LogisticRegression(
            penalty="elasticnet", solver="saga", class_weight="balanced",
            max_iter=5000, random_state=RANDOM_STATE, n_jobs=N_JOBS,
        ),
        "params": {
            "model__C":        [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 1.0, 3.0],
            "model__l1_ratio": [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0],
        },
    }

    specs["SVM_RBF"] = {
        "estimator": SVC(
            kernel="rbf", class_weight="balanced",
            probability=True, random_state=RANDOM_STATE,
        ),
        "params": {
            "model__C":     [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0],
            "model__gamma": ["scale", 1e-3, 5e-3, 1e-2],
        },
    }

    specs["RandomForest"] = {
        "estimator": RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=N_JOBS,
        ),
        "params": {
            "model__n_estimators":     [200, 400, 600],
            "model__max_depth":        [3, 5, 7, None],
            "model__min_samples_split":[5, 10, 20],
            "model__min_samples_leaf": [2, 4, 8, 16],
            "model__max_features":     ["sqrt", "log2", 0.1, 0.2, 0.3],
            "model__max_samples":      [0.6, 0.7, 0.8, None],
        },
    }

    if XGBOOST_AVAILABLE and ENABLE_XGBOOST:
        specs["XGBoost"] = {
            "estimator": XGBClassifier(
                objective="binary:logistic", eval_metric="logloss",
                **get_xgboost_acceleration_params(),
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS, verbosity=0,
                scale_pos_weight=scale_pos_weight,
            ),
            "params": {
                "model__n_estimators":     [50, 100, 200, 300],
                "model__max_depth":        [2, 3, 4],
                "model__learning_rate":    [0.01, 0.02, 0.05],
                "model__subsample":        [0.5, 0.6, 0.7, 0.8],
                "model__colsample_bytree": [0.3, 0.4, 0.5, 0.6],
                "model__min_child_weight": [3, 5, 10, 20],
                "model__gamma":            [0.5, 1.0, 2.0, 5.0],
                "model__reg_alpha":        [0.5, 1.0, 5.0, 10.0, 20.0],
                "model__reg_lambda":       [5.0, 10.0, 20.0, 50.0, 100.0],
            },
        }

    return specs


# Builds a list of PCA dimensionality options for descriptor-heavy spaces
def build_pca_candidates(n_samples: int, n_features: int) -> list[object]:
    max_comp = max(1, min(n_samples - 1, n_features))
    return [
        PCA(n_components=nc, svd_solver="randomized", random_state=RANDOM_STATE)
        for nc in [5, 10, 20, 30, 50, 75, 100]
        if 2 <= nc <= max_comp
    ]


# Builds a list of SelectKBest options tuned to each representation space
def build_selector_candidates(rep_name: str, n_features: int) -> list[object]:
    if rep_name == "descriptors_only":
        k_values = [10, 20, 30, 40, 50, 75, 100, 150, 200]
    elif rep_name == "maccs_only":
        k_values = [16, 32, 48, 64, 96]
    elif rep_name in {"morgan_only", "fingerprints_only"}:
        k_values = [32, 64, 128, 256]
    elif rep_name == "whole_dataset":
        k_values = [20, 30, 50, 75, 100, 150, 200, 300]
    else:
        k_values = [15, 25, 50, 75, 100]

    k_values = sorted({k for k in k_values if 1 <= k <= n_features})
    return [
        Pipeline([
            ("vt", VarianceThreshold(0.0)),
            ("kbest", SelectKBest(f_classif, k=k)),
        ])
        for k in k_values
    ]


# Assembles a single-space pipeline and returns it with its search space
def build_standard_pipeline(rep_name: str, base_estimator, base_params,
                             n_samples: int, n_features: int) -> tuple:
    pipe = Pipeline([
        ("selector", "passthrough"),
        ("pca",      "passthrough"),
        ("model",    clone(base_estimator)),
    ])

    selector_candidates = build_selector_candidates(rep_name=rep_name, n_features=n_features)
    pca_candidates = (
        build_pca_candidates(n_samples=n_samples, n_features=n_features)
        if rep_name in {"descriptors_only", "whole_dataset"}
        else []
    )

    spaces = []

    #Space 1: no dimensionality reduction
    base = dict(base_params)
    base["selector"] = ["passthrough"]
    base["pca"]      = ["passthrough"]
    spaces.append(base)

    # Space 2: feature selection
    if selector_candidates:
        s = dict(base_params)
        s["selector"] = selector_candidates
        s["pca"]      = ["passthrough"]
        spaces.append(s)

    # Space 3: PCA (only for descriptor-heavy representations) 
    if pca_candidates:
        s = dict(base_params)
        s["selector"] = ["passthrough"]
        s["pca"]      = pca_candidates
        spaces.append(s)

    return pipe, (spaces[0] if len(spaces) == 1 else spaces)


# Computes the number of search iterations based on the grid size
def _get_n_iter(param_distributions) -> int:
    def _size(space):
        lengths = [len(v) for v in space.values() if isinstance(v, list)]
        return max(1, math.prod(lengths)) if lengths else N_ITER_SEARCH
    if isinstance(param_distributions, list):
        total = sum(_size(s) for s in param_distributions)
        is_tree = any("model__n_estimators" in s for s in param_distributions)
        return min(N_ITER_XGBOOST if is_tree else N_ITER_SEARCH, max(1, total))
    total   = _size(param_distributions)
    is_tree = isinstance(param_distributions, dict) and "model__n_estimators" in param_distributions
    return min(N_ITER_XGBOOST if is_tree else N_ITER_SEARCH, total)


# Tunes a single-space model, selects the best parameters, evaluates it
def fit_standard_model(rep_name, model_name, model_cfg,
                       train_df, test_df, feature_cols, cv):
    X_train = train_df[feature_cols]
    X_test  = test_df[feature_cols]
    y_train = train_df[TARGET_COL].astype(int).to_numpy()
    y_test  = test_df[TARGET_COL].astype(int).to_numpy()
    groups  = train_df["Scaffold"].astype(str)

    pipe, param_dist = build_standard_pipeline(
        rep_name,
        model_cfg["estimator"], model_cfg["params"],
        n_samples=X_train.shape[0], n_features=X_train.shape[1],
    )

    # Run randomized hyperparameter search using scaffold-aware CV
    search = RandomizedSearchCV(
        estimator=pipe, param_distributions=param_dist,
        n_iter=_get_n_iter(param_dist),
        scoring=get_scoring_dict(), refit=False,
        cv=cv, random_state=RANDOM_STATE, n_jobs=N_JOBS,
        verbose=0, return_train_score=True, error_score="raise",
    )
    search.fit(X_train, y_train, groups=groups)
    chosen_idx, selection_note = choose_cv_candidate_index(
        search=search, model_name=model_name, n_features_input=len(feature_cols),
    )
    chosen_params = search.cv_results_["params"][chosen_idx]
    best_est = clone(pipe).set_params(**chosen_params)
    best_est.fit(X_train, y_train)

    # Tune the classification threshold using out-of-fold predictions
    thr_method = "predict_proba" if hasattr(best_est, "predict_proba") else "decision_function"
    oof = cross_val_predict(best_est, X_train, y_train, cv=cv, groups=groups,
                            method=thr_method, n_jobs=N_JOBS)
    if oof.ndim == 2: oof = oof[:, 1]
    oof = np.asarray(oof, dtype=float).ravel()
    thr_info = find_best_threshold(y_train, oof)
    best_thr = thr_info["threshold"]

    # Evaluate the fitted model on both train and test sets (ahter threshold selection)
    tr_score, _  = get_proba_or_decision(best_est, X_train)
    te_score, st = get_proba_or_decision(best_est, X_test)
    tr_metrics   = compute_metrics(y_train, (tr_score >= best_thr).astype(int), tr_score)
    te_metrics   = compute_metrics(y_test,  (te_score >= best_thr).astype(int), te_score)

    idx = chosen_idx
    best_params_json = json.dumps(make_json_safe(chosen_params))
    cv_row = {
        "representation": rep_name, "model": model_name,
        "n_features_input": len(feature_cols),
        "cv_roc_auc_mean":           float(search.cv_results_["mean_test_roc_auc"][idx]),
        "cv_mcc_mean":               float(search.cv_results_["mean_test_mcc"][idx]),
        "cv_balanced_accuracy_mean": float(search.cv_results_["mean_test_balanced_accuracy"][idx]),
        "cv_f1_macro_mean":          float(search.cv_results_["mean_test_f1_macro"][idx]),
        "threshold": best_thr,
        "best_params": best_params_json,
        "selection_note": json.dumps(make_json_safe(selection_note)),
    }
    test_row = {
        "representation": rep_name, "model": model_name,
        "n_features_input": len(feature_cols),
        "threshold": best_thr,
        **{f"train_{k}": float(v) for k, v in tr_metrics.items()},
        **{f"test_{k}":  float(v) for k, v in te_metrics.items()},
        "best_params": best_params_json,
        "selection_note": json.dumps(make_json_safe(selection_note)),
    }
    preds_df = pd.DataFrame({
        "representation": rep_name, "model": model_name,
        "true_label": y_test, "predicted_label": (te_score >= best_thr).astype(int),
        "score": te_score, "threshold": best_thr,
    })
    bundle = {
        "representation": rep_name, "model_name": model_name,
        "feature_columns": feature_cols, "best_estimator": best_est,
        "threshold": best_thr,
        "ad_artifacts": build_knn_ad_artifacts(get_representation_space(best_est, X_train)),
        "best_params": make_json_safe(chosen_params),
        "selection_note": make_json_safe(selection_note),
    }
    return cv_row, test_row, preds_df, bundle


# Tunes the Multi-Space SVM where each feature group is selected independently
def fit_multispace_svm(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    desc_cols: list[str],
    maccs_cols: list[str],
    morgan_cols: list[str],
    cv,
) -> tuple[dict, dict, pd.DataFrame, dict]:
    all_cols = desc_cols + maccs_cols + morgan_cols
    X_train  = train_df[all_cols]
    X_test   = test_df[all_cols]
    y_train  = train_df[TARGET_COL].astype(int).to_numpy()
    y_test   = test_df[TARGET_COL].astype(int).to_numpy()
    groups   = train_df["Scaffold"].astype(str)

    # Pipeline: MultiSpaceSelector selects top-k from each space, then SVM trains on the union ─
    pipe = Pipeline([
        ("multispace", MultiSpaceSelector(
            desc_cols=desc_cols, maccs_cols=maccs_cols, morgan_cols=morgan_cols)),
        ("model", SVC(kernel="rbf", class_weight="balanced",
                      probability=True, random_state=RANDOM_STATE)),
    ])

    # Search grid: number of features per space and SVM regularisation 
    param_grid = {
        "multispace__k_desc":   [15, 20, 30, 40, 50],
        "multispace__k_maccs":  [12, 16, 20, 25, 30, 40],
        "multispace__k_morgan": [20, 30, 40, 50, 60],
        "model__C":     [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0],
        "model__gamma": ["scale", 0.001, 0.005, 0.01],
    }

    search = RandomizedSearchCV(
        estimator=pipe, param_distributions=param_grid,
        n_iter=N_ITER_MULTISPACE,
        scoring=get_scoring_dict(), refit=False,
        cv=cv, random_state=RANDOM_STATE, n_jobs=N_JOBS,
        verbose=0, return_train_score=True, error_score="raise",
    )
    search.fit(X_train, y_train, groups=groups)
    model_name = "SVM_RBF_MultiSpace"
    chosen_idx, selection_note = choose_cv_candidate_index(
        search=search, model_name=model_name, n_features_input=len(all_cols),
    )
    chosen_params = search.cv_results_["params"][chosen_idx]
    best_est = clone(pipe).set_params(**chosen_params)
    best_est.fit(X_train, y_train)

    # Tune the decision threshold using out-of-fold probability predictions
    oof = cross_val_predict(best_est, X_train, y_train, cv=cv, groups=groups,
                            method="predict_proba", n_jobs=N_JOBS)[:, 1]
    oof = np.asarray(oof, dtype=float).ravel()
    thr_info = find_best_threshold(y_train, oof)
    best_thr = thr_info["threshold"]

    tr_score, _ = get_proba_or_decision(best_est, X_train)
    te_score, _ = get_proba_or_decision(best_est, X_test)
    tr_metrics  = compute_metrics(y_train, (tr_score >= best_thr).astype(int), tr_score)
    te_metrics  = compute_metrics(y_test,  (te_score >= best_thr).astype(int), te_score)

    idx = chosen_idx
    rep_name   = "multi_space"
    best_params_json = json.dumps(make_json_safe(chosen_params))

    cv_row = {
        "representation": rep_name, "model": model_name,
        "n_features_input": len(all_cols),
        "cv_roc_auc_mean":           float(search.cv_results_["mean_test_roc_auc"][idx]),
        "cv_mcc_mean":               float(search.cv_results_["mean_test_mcc"][idx]),
        "cv_balanced_accuracy_mean": float(search.cv_results_["mean_test_balanced_accuracy"][idx]),
        "cv_f1_macro_mean":          float(search.cv_results_["mean_test_f1_macro"][idx]),
        "threshold": best_thr,
        "best_params": best_params_json,
        "selection_note": json.dumps(make_json_safe(selection_note)),
    }
    test_row = {
        "representation": rep_name, "model": model_name,
        "n_features_input": len(all_cols), "threshold": best_thr,
        **{f"train_{k}": float(v) for k, v in tr_metrics.items()},
        **{f"test_{k}":  float(v) for k, v in te_metrics.items()},
        "best_params": best_params_json,
        "selection_note": json.dumps(make_json_safe(selection_note)),
    }
    preds_df = pd.DataFrame({
        "representation": rep_name, "model": model_name,
        "true_label": y_test, "predicted_label": (te_score >= best_thr).astype(int),
        "score": te_score, "threshold": best_thr,
    })
    # Extract the names of the selected features for interpretability
    ms_step = best_est.named_steps["multispace"]
    selected_names = ms_step.get_selected_feature_names()
    bundle = {
        "representation": rep_name, "model_name": model_name,
        "feature_columns": all_cols,
        "selected_feature_names": selected_names,
        "best_estimator": best_est,
        "threshold": best_thr,
        "ad_artifacts": build_knn_ad_artifacts(
            get_representation_space(best_est, X_train)),
        "best_params": make_json_safe(chosen_params),
        "selection_note": make_json_safe(selection_note),
    }
    return cv_row, test_row, preds_df, bundle


# Averages probabilities across the best model per representation
def fit_multispace_model(
    model_name: str,
    model_cfg: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    desc_cols: list[str],
    maccs_cols: list[str],
    morgan_cols: list[str],
    cv,
) -> tuple[dict, dict, pd.DataFrame, dict]:
    all_cols = desc_cols + maccs_cols + morgan_cols
    X_train = train_df[all_cols]
    X_test = test_df[all_cols]
    y_train = train_df[TARGET_COL].astype(int).to_numpy()
    y_test = test_df[TARGET_COL].astype(int).to_numpy()
    groups = train_df["Scaffold"].astype(str)

    multispace_name = f"{model_name}_MultiSpace"
    pipe = Pipeline([
        ("multispace", MultiSpaceSelector(
            desc_cols=desc_cols,
            maccs_cols=maccs_cols,
            morgan_cols=morgan_cols,
        )),
        ("model", clone(model_cfg["estimator"])),
    ])

    param_grid = {
        "multispace__k_desc": [15, 20, 30, 40, 50],
        "multispace__k_maccs": [12, 16, 20, 25, 30, 40],
        "multispace__k_morgan": [20, 30, 40, 50, 60],
        **model_cfg["params"],
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid,
        n_iter=N_ITER_MULTISPACE,
        scoring=get_scoring_dict(),
        refit=False,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=0,
        return_train_score=True,
        error_score="raise",
    )
    search.fit(X_train, y_train, groups=groups)
    chosen_idx, selection_note = choose_cv_candidate_index(
        search=search,
        model_name=multispace_name,
        n_features_input=len(all_cols),
    )
    chosen_params = search.cv_results_["params"][chosen_idx]
    best_est = clone(pipe).set_params(**chosen_params)
    best_est.fit(X_train, y_train)

    thr_method = "predict_proba" if hasattr(best_est, "predict_proba") else "decision_function"
    oof = cross_val_predict(
        best_est,
        X_train,
        y_train,
        cv=cv,
        groups=groups,
        method=thr_method,
        n_jobs=N_JOBS,
    )
    if oof.ndim == 2:
        oof = oof[:, 1]
    oof = np.asarray(oof, dtype=float).ravel()
    best_thr = find_best_threshold(y_train, oof)["threshold"]

    tr_score, _ = get_proba_or_decision(best_est, X_train)
    te_score, _ = get_proba_or_decision(best_est, X_test)
    tr_metrics = compute_metrics(y_train, (tr_score >= best_thr).astype(int), tr_score)
    te_metrics = compute_metrics(y_test, (te_score >= best_thr).astype(int), te_score)

    rep_name = "multi_space"
    best_params_json = json.dumps(make_json_safe(chosen_params))
    cv_row = {
        "representation": rep_name,
        "model": multispace_name,
        "n_features_input": len(all_cols),
        "cv_roc_auc_mean": float(search.cv_results_["mean_test_roc_auc"][chosen_idx]),
        "cv_mcc_mean": float(search.cv_results_["mean_test_mcc"][chosen_idx]),
        "cv_balanced_accuracy_mean": float(search.cv_results_["mean_test_balanced_accuracy"][chosen_idx]),
        "cv_f1_macro_mean": float(search.cv_results_["mean_test_f1_macro"][chosen_idx]),
        "threshold": best_thr,
        "best_params": best_params_json,
        "selection_note": json.dumps(make_json_safe(selection_note)),
    }
    test_row = {
        "representation": rep_name,
        "model": multispace_name,
        "n_features_input": len(all_cols),
        "threshold": best_thr,
        **{f"train_{k}": float(v) for k, v in tr_metrics.items()},
        **{f"test_{k}": float(v) for k, v in te_metrics.items()},
        "best_params": best_params_json,
        "selection_note": json.dumps(make_json_safe(selection_note)),
    }
    preds_df = pd.DataFrame({
        "representation": rep_name,
        "model": multispace_name,
        "true_label": y_test,
        "predicted_label": (te_score >= best_thr).astype(int),
        "score": te_score,
        "threshold": best_thr,
    })

    selected_names = best_est.named_steps["multispace"].get_selected_feature_names()
    bundle = {
        "representation": rep_name,
        "model_name": multispace_name,
        "feature_columns": all_cols,
        "selected_feature_names": selected_names,
        "best_estimator": best_est,
        "threshold": best_thr,
        "ad_artifacts": build_knn_ad_artifacts(get_representation_space(best_est, X_train)),
        "best_params": make_json_safe(chosen_params),
        "selection_note": make_json_safe(selection_note),
    }
    return cv_row, test_row, preds_df, bundle


def build_voting_ensemble(best_bundles, train_df, test_df, y_train, y_test,
                           preds_dir, roc_dir, cm_dir):
    entries = [
        (b["best_estimator"], b["feature_columns"],
         f"{sanitize_name(b['representation'])}__{sanitize_name(b['model_name'])}")
        for b in best_bundles
        if hasattr(b["best_estimator"], "predict_proba")
    ]
    if len(entries) < 2:
        print("[INFO] < 2 proba-capable estimators; skipping ensemble.")
        return None

    # Get class-1 probabilities from each model and stack them 
    tr_ps = np.column_stack([e.predict_proba(train_df[fc])[:, 1] for e, fc, _ in entries])
    te_ps = np.column_stack([e.predict_proba(test_df[fc])[:, 1]  for e, fc, _ in entries])
    tr_s  = tr_ps.mean(axis=1)
    te_s  = te_ps.mean(axis=1)

    # Tune the threshold on the train average score and evaluate
    thr  = find_best_threshold(y_train, tr_s)["threshold"]
    tr_m = compute_metrics(y_train, (tr_s >= thr).astype(int), tr_s)
    te_m = compute_metrics(y_test,  (te_s >= thr).astype(int), te_s)
    print(f"  [Ensemble] train_MCC={tr_m['mcc']:.4f} | test_MCC={te_m['mcc']:.4f} | "
          f"test_ROC={te_m['roc_auc']:.4f}")

    pd.DataFrame({
        "representation": "ensemble_soft_vote", "model": "SoftVoting",
        "true_label": y_test, "predicted_label": (te_s >= thr).astype(int),
        "score": te_s, "threshold": thr,
    }).to_csv(preds_dir / "ensemble_soft_vote__predictions.csv", index=False)

    plot_roc_curve(y_test, te_s, "Ensemble (soft-vote) | ROC",
                   roc_dir / "ensemble_soft_vote__roc.png")
    plot_confusion_matrix(y_test, (te_s >= thr).astype(int),
                          f"Ensemble (soft-vote) | thr={thr:.2f}",
                          cm_dir / "ensemble_soft_vote__cm.png")
    return {"train_metrics": tr_m, "test_metrics": te_m, "threshold": thr}


def main():
    # Create all output subdirectories 
    tables_dir = OUTDIR / "tables";          ensure_dir(tables_dir)
    plots_dir  = OUTDIR / "plots";           ensure_dir(plots_dir)
    preds_dir  = OUTDIR / "predictions";     ensure_dir(preds_dir)
    models_dir = OUTDIR / "models";          ensure_dir(models_dir)
    roc_dir    = plots_dir / "roc_curves";   ensure_dir(roc_dir)
    pr_dir     = plots_dir / "pr_curves";    ensure_dir(pr_dir)
    cm_dir     = plots_dir / "conf_matrices"; ensure_dir(cm_dir)
    calib_dir  = plots_dir / "calibration";  ensure_dir(calib_dir)
    fi_dir     = plots_dir / "feat_import";  ensure_dir(fi_dir)

    # Load the processed train and test splits
    print("[INFO] Loading data...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df  = pd.read_csv(TEST_FILE)
    validate_inputs(train_df, test_df)
    train_df[TARGET_COL] = train_df[TARGET_COL].astype(int)
    test_df[TARGET_COL]  = test_df[TARGET_COL].astype(int)

    # Identify the three feature groups and define the five representation spaces
    desc_cols, maccs_cols, morgan_cols = get_feature_columns(train_df)
    feature_sets = {
        "descriptors_only":  desc_cols,
        "maccs_only":        maccs_cols,
        "morgan_only":       morgan_cols,
        "fingerprints_only": maccs_cols + morgan_cols,
        "whole_dataset":     desc_cols + maccs_cols + morgan_cols,
    }
    pd.DataFrame(
        [{"representation": k, "n_features": len(v)} for k, v in feature_sets.items()]
    ).to_csv(tables_dir / "feature_inventory.csv", index=False)

    y_train = train_df[TARGET_COL].to_numpy()
    y_test  = test_df[TARGET_COL].to_numpy()
    spw     = get_scale_pos_weight(y_train)
    cv      = get_cv(train_df["Scaffold"].astype(str))

    model_specs = get_standard_model_specs(scale_pos_weight=spw)

    cv_rows     = []
    test_rows   = []
    pred_frames = []
    best_bundle_per_rep: dict[str, dict]  = {}
    best_cv_mcc_per_rep: dict[str, float] = {}

    # Loop: fit every model on every representation space
    for rep_name, cols in feature_sets.items():
        if not cols:
            print(f"[WARNING] Skipping empty representation: {rep_name}")
            continue
        print(f"\n[INFO] Representation: {rep_name} ({len(cols)} features)")
        for model_name, cfg in model_specs.items():
            print(f"  -> Tuning {model_name}")
            try:
                cv_row, test_row, preds, bundle = fit_standard_model(
                    rep_name, model_name, cfg,
                    train_df, test_df, cols, cv,
                )
                cv_rows.append(cv_row); test_rows.append(test_row)
                pred_frames.append(preds)
                joblib.dump(bundle, models_dir / f"{sanitize_name(rep_name)}__{sanitize_name(model_name)}.joblib")

                # Save per-model evaluation plots
                plot_roc_curve(preds["true_label"].to_numpy(), preds["score"].to_numpy(),
                               f"{rep_name} | {model_name}",
                               roc_dir / f"{sanitize_name(rep_name)}__{sanitize_name(model_name)}__roc.png")
                plot_pr_curve(preds["true_label"].to_numpy(), preds["score"].to_numpy(),
                              f"{rep_name} | {model_name}",
                              pr_dir / f"{sanitize_name(rep_name)}__{sanitize_name(model_name)}__pr.png")
                plot_confusion_matrix(preds["true_label"].to_numpy(),
                                      preds["predicted_label"].to_numpy(),
                                      f"{rep_name} | {model_name} | thr={test_row['threshold']:.2f}",
                                      cm_dir / f"{sanitize_name(rep_name)}__{sanitize_name(model_name)}__cm.png")
                preds.to_csv(preds_dir / f"{sanitize_name(rep_name)}__{sanitize_name(model_name)}__preds.csv",
                             index=False)

                # Track the best CV MCC per representation for the ensemble
                cv_mcc = cv_row["cv_mcc_mean"]
                if rep_name not in best_cv_mcc_per_rep or cv_mcc > best_cv_mcc_per_rep[rep_name]:
                    best_cv_mcc_per_rep[rep_name] = cv_mcc
                    best_bundle_per_rep[rep_name]  = bundle

            except Exception as exc:
                print(f"  [WARNING] {rep_name} | {model_name} failed: {exc}")

    #  Fit the Multi-Space SVM
    print(f"\n[INFO] Fitting additional Multi-Space models (n_iter={N_ITER_MULTISPACE})...")
    for model_name, cfg in model_specs.items():
        if model_name == "SVM_RBF":
            continue
        ms_model_name = f"{model_name}_MultiSpace"
        print(f"  -> Tuning {ms_model_name}")
        try:
            ms_cv, ms_test, ms_preds, ms_bundle = fit_multispace_model(
                model_name, cfg, train_df, test_df, desc_cols, maccs_cols, morgan_cols, cv,
            )
            cv_rows.append(ms_cv); test_rows.append(ms_test)
            pred_frames.append(ms_preds)
            joblib.dump(ms_bundle, models_dir / f"multi_space__{sanitize_name(ms_model_name)}.joblib")

            plot_roc_curve(ms_preds["true_label"].to_numpy(), ms_preds["score"].to_numpy(),
                           f"Multi-Space | {model_name}",
                           roc_dir / f"multi_space__{sanitize_name(ms_model_name)}__roc.png")
            plot_pr_curve(ms_preds["true_label"].to_numpy(), ms_preds["score"].to_numpy(),
                          f"Multi-Space | {model_name}",
                          pr_dir / f"multi_space__{sanitize_name(ms_model_name)}__pr.png")
            plot_confusion_matrix(ms_preds["true_label"].to_numpy(), ms_preds["predicted_label"].to_numpy(),
                                  f"Multi-Space | {model_name} | thr={ms_test['threshold']:.2f}",
                                  cm_dir / f"multi_space__{sanitize_name(ms_model_name)}__cm.png")
            ms_preds.to_csv(
                preds_dir / f"multi_space__{sanitize_name(ms_model_name)}__preds.csv",
                index=False,
            )

            cv_mcc = ms_cv["cv_mcc_mean"]
            if "multi_space" not in best_cv_mcc_per_rep or cv_mcc > best_cv_mcc_per_rep["multi_space"]:
                best_cv_mcc_per_rep["multi_space"] = cv_mcc
                best_bundle_per_rep["multi_space"] = ms_bundle

            print(f"  {ms_model_name}: CV MCC={ms_cv['cv_mcc_mean']:.4f} | "
                  f"train MCC={ms_test['train_mcc']:.4f} | "
                  f"test  MCC={ms_test['test_mcc']:.4f}")
        except Exception as exc:
            print(f"  [WARNING] {ms_model_name} failed: {exc}")

    print(f"\n[INFO] Fitting Multi-Space SVM (n_iter={N_ITER_MULTISPACE})...")
    try:
        ms_cv, ms_test, ms_preds, ms_bundle = fit_multispace_svm(
            train_df, test_df, desc_cols, maccs_cols, morgan_cols, cv,
        )
        cv_rows.append(ms_cv); test_rows.append(ms_test)
        pred_frames.append(ms_preds)
        joblib.dump(ms_bundle, models_dir / "multi_space__svm_rbf_multispace.joblib")

        plot_roc_curve(ms_preds["true_label"].to_numpy(), ms_preds["score"].to_numpy(),
                       "Multi-Space SVM | ROC", roc_dir / "multi_space__svm_rbf__roc.png")
        plot_pr_curve(ms_preds["true_label"].to_numpy(), ms_preds["score"].to_numpy(),
                      "Multi-Space SVM | PR",  pr_dir / "multi_space__svm_rbf__pr.png")
        plot_confusion_matrix(ms_preds["true_label"].to_numpy(), ms_preds["predicted_label"].to_numpy(),
                              f"Multi-Space SVM | thr={ms_test['threshold']:.2f}",
                              cm_dir / "multi_space__svm_rbf__cm.png")
        ms_preds.to_csv(preds_dir / "multi_space__svm_rbf__preds.csv", index=False)

        cv_mcc = ms_cv["cv_mcc_mean"]
        if "multi_space" not in best_cv_mcc_per_rep or cv_mcc > best_cv_mcc_per_rep["multi_space"]:
            best_cv_mcc_per_rep["multi_space"] = cv_mcc
            best_bundle_per_rep["multi_space"]  = ms_bundle

        print(f"  Multi-Space SVM: CV MCC={ms_cv['cv_mcc_mean']:.4f} | "
              f"train MCC={ms_test['train_mcc']:.4f} | "
              f"test  MCC={ms_test['test_mcc']:.4f}")
    except Exception as exc:
        print(f"  [WARNING] Multi-Space SVM failed: {exc}")

    if not test_rows:
        raise RuntimeError("No models completed successfully.")

    # Collect all results and compute the train-CV gap for each model
    cv_df   = pd.DataFrame(cv_rows).sort_values(["representation", "model"]).reset_index(drop=True)
    test_df_sum = pd.DataFrame(test_rows).sort_values(["representation", "model"]).reset_index(drop=True)
    preds_all   = pd.concat(pred_frames, ignore_index=True)

    full_summary = test_df_sum.merge(
        cv_df,
        on=["representation", "model", "n_features_input", "threshold", "best_params", "selection_note"],
        how="left",
    )
    full_summary["abs_train_cv_gap"] = (full_summary["train_mcc"] - full_summary["cv_mcc_mean"]).abs()

    comparison_ready = full_summary.sort_values(
        ["cv_mcc_mean", "abs_train_cv_gap", "test_mcc"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    #Save all summary tables to disk
    cv_df.to_csv(tables_dir / "cv_summary.csv", index=False)
    test_df_sum.to_csv(tables_dir / "test_summary.csv", index=False)
    full_summary.to_csv(tables_dir / "full_classification_summary.csv", index=False)
    comparison_ready.to_csv(tables_dir / "comparison_ready_summary.csv", index=False)
    preds_all.to_csv(preds_dir / "all_models_predictions.csv", index=False)
    preds_all.to_csv(preds_dir / "test_predictions_all_models.csv", index=False)

    # Find the best model per representation by CV MCC
    best_by_cv = (
        full_summary.sort_values(
            ["representation", "cv_mcc_mean", "abs_train_cv_gap", "cv_roc_auc_mean"],
            ascending=[True, False, True, False],
        )
        .groupby("representation", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    # Find the best model per representation by test MCC
    best_by_test = (
        full_summary.sort_values(
            ["representation", "test_mcc", "test_roc_auc", "abs_train_cv_gap"],
            ascending=[True, False, False, True],
        )
        .groupby("representation", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    # Find the single winner for each metric
    metric_winners = pd.concat([
        full_summary.nlargest(1, "cv_mcc_mean").assign(metric="cv_mcc_mean"),
        full_summary.nlargest(1, "test_mcc").assign(metric="test_mcc"),
        full_summary.nlargest(1, "test_roc_auc").assign(metric="test_roc_auc"),
        full_summary.nlargest(1, "test_pr_auc").assign(metric="test_pr_auc"),
        full_summary.nlargest(1, "test_balanced_accuracy").assign(metric="test_balanced_accuracy"),
    ], ignore_index=True)
    best_by_cv.to_csv(tables_dir / "best_model_per_representation_by_cv.csv", index=False)
    best_by_test.to_csv(tables_dir / "best_model_per_representation_by_test.csv", index=False)
    metric_winners.to_csv(tables_dir / "metric_winners.csv", index=False)

    # Build and evaluate the soft-voting ensemble
    print("\n[INFO] Building soft-voting ensemble...")
    ens = build_voting_ensemble(
        list(best_bundle_per_rep.values()),
        train_df, test_df, y_train, y_test,
        preds_dir, roc_dir, cm_dir,
    )
    if ens:
        joblib.dump(ens, models_dir / "ensemble_soft_vote.joblib")

    # Select the overall winner by the primary CV metric.
    # Generalization gap is kept as a tie-breaker/diagnostic, not the main selector.
    best_cv = float(full_summary["cv_mcc_mean"].max())
    selection_pool_df = full_summary.sort_values(
        ["cv_mcc_mean", "cv_balanced_accuracy_mean", "cv_roc_auc_mean", "abs_train_cv_gap", "n_features_input"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    winner_df = selection_pool_df.head(1).reset_index(drop=True)
    final_winner_selection = selection_pool_df.copy()
    final_winner_selection.insert(0, "best_cv_mcc", best_cv)
    final_winner_selection.insert(1, "model_level_cv_tolerance", MODEL_LEVEL_CV_TOLERANCE)
    final_winner_selection.insert(2, "max_acceptable_train_cv_gap", MAX_ACCEPTABLE_TRAIN_CV_MCC_GAP)
    final_winner_selection.to_csv(tables_dir / "final_winner_selection.csv", index=False)
    winner_df.to_csv(tables_dir / "final_winner.csv", index=False)

    #Write the path to the winning model file so Prediction.py can find it
    if not winner_df.empty:
        w = winner_df.iloc[0]
        wp = models_dir / f"{sanitize_name(w['representation'])}__{sanitize_name(w['model'])}.joblib"
        if wp.exists():
            (models_dir / "overall_best_model_path.txt").write_text(str(wp), encoding="utf-8")

    # Generate the per-fold MCC plot for the model with the best test MCC
    best_test_row = full_summary.sort_values("test_mcc", ascending=False).iloc[0]
    bt_rep   = best_test_row["representation"]
    bt_model = best_test_row["model"]
    bt_path  = models_dir / f"{sanitize_name(bt_rep)}__{sanitize_name(bt_model)}.joblib"
    if bt_path.exists():
        print(f"\n[INFO] Per-fold MCC plot: {bt_rep} | {bt_model}")
        bt_bundle = joblib.load(bt_path)
        fcols = bt_bundle["feature_columns"]
        cv_fold_info = plot_cv_mcc_per_fold(
            estimator=bt_bundle["best_estimator"],
            X=train_df[fcols],
            y=y_train,
            groups=train_df["Scaffold"].astype(str),
            cv=cv,
            model_label=f"{bt_rep} | {bt_model}",
            save_path=plots_dir / "best_test_mcc__cv_per_fold.png",
        )
        pd.DataFrame({
            "fold":        [f"Fold {i+1}" for i in range(len(cv_fold_info["fold_val_scores"]))],
            "val_mcc":     cv_fold_info["fold_val_scores"],
            "train_mcc":   cv_fold_info["fold_train_scores"],
            "cv_mean_mcc": cv_fold_info["mean"],
            "cv_std_mcc":  cv_fold_info["std"],
            "representation": bt_rep, "model": bt_model,
        }).to_csv(tables_dir / "best_test_mcc__cv_per_fold.csv", index=False)
        print(f"  Val  folds: {[f'{s:.3f}' for s in cv_fold_info['fold_val_scores']]} "
              f"-> mean={cv_fold_info['mean']:.3f} std={cv_fold_info['std']:.3f}")
        print(f"  Train folds: {[f'{s:.3f}' for s in cv_fold_info['fold_train_scores']]} "
              f"-> mean={np.mean(cv_fold_info['fold_train_scores']):.3f}")

    #  Generate calibration curve and feature importance for the overall winner
    if not winner_df.empty:
        w = winner_df.iloc[0]
        bp = models_dir / f"{sanitize_name(w['representation'])}__{sanitize_name(w['model'])}.joblib"
        if bp.exists():
            b = joblib.load(bp)
            est  = b["best_estimator"]
            fcols = b["feature_columns"]
            Xtr  = train_df[fcols]
            Xte  = test_df[fcols]

            try:
                fig, ax = plt.subplots(figsize=(6, 6))
                CalibrationDisplay.from_estimator(est, Xte, y_test, n_bins=10, ax=ax)
                ax.set_title(f"Calibration | {w['representation']} | {w['model']}", fontweight="bold")
                fig.tight_layout()
                fig.savefig(calib_dir / "winner_calibration.png", dpi=300, bbox_inches="tight")
                plt.close(fig)
            except Exception:
                pass

            final_est = est.named_steps.get("model") or est.named_steps.get("multispace")
            # ── For MultiSpace pipelines, get the names of the selected features ─
            if "multispace" in est.named_steps:
                feat_names = est.named_steps["multispace"].get_selected_feature_names()
                final_est  = est.named_steps["model"]
            else:
                feat_names = fcols

            # Use built-in feature importances if available, else permutation importance ─
            if hasattr(final_est, "feature_importances_"):
                imp_df = pd.DataFrame({
                    "feature": feat_names,
                    "importance": final_est.feature_importances_,
                }).sort_values("importance", ascending=False)
                imp_df.to_csv(tables_dir / "winner_feature_importance.csv", index=False)
                plot_feature_importance(imp_df,
                                        f"Feature Importance | {w['representation']} | {w['model']}",
                                        fi_dir / "winner_feature_importance.png",
                                        top_n=TOP_N_FEATURES_TO_PLOT)
            else:
                try:
                    perm = permutation_importance(
                        est, Xte, y_test, n_repeats=10,
                        random_state=RANDOM_STATE, n_jobs=N_JOBS,
                        scoring="balanced_accuracy",
                    )
                    imp_df = pd.DataFrame({
                        "feature": fcols, "importance": perm.importances_mean,
                    }).sort_values("importance", ascending=False)
                    imp_df.to_csv(tables_dir / "winner_feature_importance.csv", index=False)
                    plot_feature_importance(imp_df,
                                            f"Permutation Importance | {w['representation']} | {w['model']}",
                                            fi_dir / "winner_feature_importance.png",
                                            top_n=TOP_N_FEATURES_TO_PLOT)
                except Exception:
                    pass

    #  Write a plain-text summary report 
    with open(OUTDIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write("QSAR Classification — Multi-Space Anti-Overfitting Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Training rows : {len(train_df)}\n")
        f.write(f"Test rows     : {len(test_df)}\n")
        f.write(f"Descriptors   : {len(desc_cols)}\n")
        f.write(f"MACCS bits    : {len(maccs_cols)}\n")
        f.write(f"Morgan bits   : {len(morgan_cols)}\n")
        f.write(f"CV strategy   : {'StratifiedGroupKFold' if STRATIFIED_GROUP_AVAILABLE else 'GroupKFold'}(n_splits={N_SPLITS})\n")
        f.write(f"N_ITER_SEARCH : {N_ITER_SEARCH}  |  N_ITER_XGBOOST: {N_ITER_XGBOOST}  |  N_ITER_MULTISPACE: {N_ITER_MULTISPACE}\n\n")
        f.write("MODEL-SELECTION RULES\n")
        f.write(f"  Search metric              = {SEARCH_PRIMARY_METRIC}\n")
        f.write(f"  1-SE rule                  = {USE_ONE_STANDARD_ERROR_RULE}\n")
        f.write(f"  Winner selection           = highest CV MCC\n")
        f.write(f"  Gap use                    = tie-breaker/diagnostic only\n\n")
        f.write("KEY ARCHITECTURAL CHANGE\n")
        f.write("  MultiSpaceSelector: select top-k independently from descriptors,\n")
        f.write("  MACCS, and Morgan, then concatenate — avoids scale-mixing bias.\n\n")

        if not winner_df.empty:
            row = winner_df.iloc[0]
            f.write("Overall winner (highest CV MCC)\n")
            f.write(f"  {row['representation']} | {row['model']}\n")
            f.write(f"  CV MCC    = {row['cv_mcc_mean']:.4f}\n")
            f.write(f"  CV ROC    = {row['cv_roc_auc_mean']:.4f}\n")
            f.write(f"  Test MCC  = {row['test_mcc']:.4f}\n")
            f.write(f"  Test ROC  = {row['test_roc_auc']:.4f}\n")
            f.write(f"  Train MCC = {row['train_mcc']:.4f}  (train-CV gap = {row['train_mcc']-row['cv_mcc_mean']:.4f})\n\n")

        if "multi_space__svm_rbf_multispace.joblib" and (models_dir / "multi_space__svm_rbf_multispace.joblib").exists():
            f.write("Multi-Space SVM result\n")
            ms_row = full_summary[full_summary["model"] == "SVM_RBF_MultiSpace"]
            if not ms_row.empty:
                r = ms_row.iloc[0]
                f.write(f"  CV MCC   = {r['cv_mcc_mean']:.4f}\n")
                f.write(f"  Test MCC = {r['test_mcc']:.4f}\n")
                f.write(f"  Test ROC = {r['test_roc_auc']:.4f}\n\n")

        if ens:
            f.write("Soft-voting ensemble (best per representation)\n")
            f.write(f"  train MCC = {ens['train_metrics']['mcc']:.4f}\n")
            f.write(f"  test  MCC = {ens['test_metrics']['mcc']:.4f}\n")
            f.write(f"  test  ROC = {ens['test_metrics']['roc_auc']:.4f}\n\n")

        best_test = full_summary.sort_values("test_mcc", ascending=False).iloc[0]
        f.write("Best observed on test (reference only)\n")
        f.write(f"  {best_test['representation']} | {best_test['model']}\n")
        f.write(f"  Test MCC = {best_test['test_mcc']:.4f}  |  Test ROC = {best_test['test_roc_auc']:.4f}\n\n")
        f.write("NOTE ON RESIDUAL TRAIN-CV GAP\n")
        f.write("  With scaffold-stratified 5-fold CV and ~23 actives per validation\n")
        f.write("  fold from entirely novel scaffolds, some train-CV gap is irreducible.\n")
        f.write("  A gap < 0.25 on MCC is expected and scientifically honest to report.\n")
        f.write("  This run now prefers near-best CV models with smaller train-CV gaps.\n")

    print(f"\n[INFO] All results saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
