from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from scipy.stats import kendalltau, spearmanr
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    ndcg_score,
    r2_score,
    pairwise_distances,
)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

# This section defines settings for the regression run
BASE_DIR  = Path(r"C:\Users\admin\Desktop\Machine Learning For Capstone")
SPLIT_DIR = BASE_DIR / "Data Split Outputs"

TRAIN_PATH = SPLIT_DIR / "regression_train_dataset.csv"
TEST_PATH  = SPLIT_DIR / "regression_test_dataset.csv"
TARGET_COL = "pChEMBL Value"
META_COLS  = ["Canonical_SMILES", "Scaffold", "Split"]

OUTDIR = BASE_DIR / "Regression Output"

RANDOM_STATE = 42
N_SPLITS     = 5
N_JOBS       = 1
RANKING_EVAL_K = 5


N_ITER_SEARCH  = 60
N_ITER_XGBOOST = 40

USE_ALL_REPRESENTATIONS = True

# This section controls model selection behavior:
# Use Spearman rank correlation as the primary metric, apply the 1-SE rule,
# and reject models whose training-CV gap exceeds the threshold
PRIMARY_SELECTION_METRIC  = "Spearman"
USE_ONE_STANDARD_ERROR_RULE = True
ONE_SE_MULTIPLIER           = 1.0

MODEL_LEVEL_CV_TOLERANCE          = 0.03
MAX_ACCEPTABLE_TRAIN_CV_SPEARMAN_GAP = 0.15


# This section contains small utility functions used throughout the script
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_name(text: str) -> str:
    # This function converts a string into a safe filename by replacing special characters
    return (
        str(text)
        .replace(" ", "_").replace("/", "_").replace("\\", "_")
        .replace("(", "").replace(")", "").replace("-", "_")
        .lower()
    )


def rmse(y_true, y_pred) -> float:
    # This function computes the root mean squared error between true and predicted values
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def pearson_corr(y_true, y_pred) -> float:
    # This function computes the Pearson correlation coefficient between two arrays
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def concordance_correlation_coefficient(y_true, y_pred) -> float:
    # This function computes the concordance correlation coefficient (CCC),
    # which measures both precision and accuracy of the predictions together
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mean_true = np.mean(y_true); mean_pred = np.mean(y_pred)
    var_true  = np.var(y_true);  var_pred  = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    return float((2 * cov) / denom) if denom != 0 else np.nan


def safe_metric_value(value, default: float = 0.0) -> float:
    # This function converts a metric value to float and replaces infinite values with a default
    try:
        value = float(value)
    except Exception:
        return float(default)
    return float(value) if np.isfinite(value) else float(default)


def descending_rank(values: np.ndarray) -> np.ndarray:
    # This function assigns ranks to values where the highest value gets rank 1
    return pd.Series(np.asarray(values, dtype=float)).rank(
        ascending=False, method="min"
    ).astype(int).to_numpy()


def pairwise_accuracy_score(y_true, y_pred) -> float:
    # This function measures how often the model correctly orders pairs of compounds by potency
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    n = len(y_true)
    if n < 2:
        return 0.0
    correct = 0.0; comparable = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            delta_true = y_true[i] - y_true[j]
            if np.isclose(delta_true, 0.0):
                continue
            delta_pred = y_pred[i] - y_pred[j]
            comparable += 1
            if np.isclose(delta_pred, 0.0):
                correct += 0.5
            elif delta_true * delta_pred > 0:
                correct += 1.0
    return float(correct / comparable) if comparable > 0 else 0.0


def spearman_rank_score(y_true, y_pred) -> float:
    # This function computes Spearman rank correlation
    stat = spearmanr(np.asarray(y_true, dtype=float).ravel(),
                     np.asarray(y_pred, dtype=float).ravel()).statistic
    return safe_metric_value(stat, default=0.0)


def kendall_tau_score(y_true, y_pred) -> float:
    # This function computes Kendall's tau, another rank agreement metric
    stat = kendalltau(np.asarray(y_true, dtype=float).ravel(),
                      np.asarray(y_pred, dtype=float).ravel()).statistic
    return safe_metric_value(stat, default=0.0)


def ndcg_at_k_score(y_true, y_pred, k: int = RANKING_EVAL_K) -> float:
    # This function computes NDCG@k, which rewards correctly placing the highest-potency
    # compounds at the top of the predicted ranking
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if len(y_true) == 0:
        return 0.0
    gains = y_true.copy()
    min_gain = np.min(gains)
    if min_gain < 0:
        gains = gains - min_gain
    if np.allclose(gains, gains[0]):
        return 0.0
    k_eff = min(int(k), len(gains))
    return safe_metric_value(
        ndcg_score(gains.reshape(1, -1), y_pred.reshape(1, -1), k=k_eff), default=0.0
    )


def top_k_overlap_fraction(y_true, y_pred, k: int = RANKING_EVAL_K) -> float:
    # This function measures what fraction of the true top-k compounds also appear in the predicted top-k
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if len(y_true) == 0:
        return 0.0
    k_eff = min(int(k), len(y_true))
    return float(
        len(set(np.argsort(-y_true)[:k_eff]) & set(np.argsort(-y_pred)[:k_eff])) / k_eff
    )


def ndcg_at_5_scorer(y_true, y_pred)  -> float: return ndcg_at_k_score(y_true, y_pred, k=RANKING_EVAL_K)
def top5_overlap_scorer(y_true, y_pred) -> float: return top_k_overlap_fraction(y_true, y_pred, k=RANKING_EVAL_K)


def get_ranking_scorers() -> dict:
    # This function returns a dictionary of all ranking metrics to evaluate during cross-validation
    return {
        "NDCG_at_5":       make_scorer(ndcg_at_5_scorer,      greater_is_better=True),
        "PairwiseAccuracy":make_scorer(pairwise_accuracy_score,greater_is_better=True),
        "Spearman":        make_scorer(spearman_rank_score,    greater_is_better=True),
        "KendallTau":      make_scorer(kendall_tau_score,      greater_is_better=True),
        "Top5_Overlap":    make_scorer(top5_overlap_scorer,    greater_is_better=True),
        "RMSE": "neg_root_mean_squared_error",
        "MAE":  "neg_mean_absolute_error",
        "R2":   "r2",
    }


def stringify_params(params: dict) -> dict:
    # This function converts scikit-learn objects in a params dict to readable strings for saving
    out = {}
    for key, value in params.items():
        cname = value.__class__.__name__ if hasattr(value, "__class__") else ""
        if cname == "PCA":
            out[key] = f"PCA(n_components={getattr(value, 'n_components', None)})"
        elif cname == "SelectKBest":
            out[key] = f"SelectKBest(k={getattr(value, 'k', None)})"
        else:
            out[key] = str(value)
    return out


# This section contains functions for building the list of feature reduction options to search over

def build_pca_candidates(n_samples: int, n_features: int) -> list[object]:
    # This function returns a list of PCA options to try
    max_allowed = max(1, min(n_samples - 1, n_features))
    candidates: list[object] = ["passthrough"]
    for n_components in [5, 10, 20, 30]:
        if 2 <= n_components <= max_allowed:
            solver = "randomized" if n_components < max_allowed else "full"
            candidates.append(
                PCA(n_components=n_components, svd_solver=solver, random_state=RANDOM_STATE)
            )
    return candidates


def build_selector_candidates(rep_name: str, n_features: int, n_samples: int) -> list[object]:
    # This function builds a list of SelectKBest options tailored to each feature space
    candidates: list[object] = ["passthrough"]

    if rep_name == "descriptors_only":
        k_values = [10, 20, 30, 40, 50, 75, 100, 150, 200]
    elif rep_name == "maccs_only":
        k_values = [16, 32, 48, 64, 96]
    elif rep_name == "morgan_only":
        k_values = [32, 64, 128, 256]
    elif rep_name == "fingerprints_only":
        k_values = [32, 64, 128, 256]
    elif rep_name == "whole_dataset":
        k_values = [20, 30, 50, 75, 100, 150]
    else:
        k_values = [20, 50, 100]

    k_values = sorted({k for k in k_values if 1 <= k <= n_features})

    for k in k_values:
        candidates.append(SelectKBest(score_func=f_regression, k=k))
    return candidates


def _space_size(space: dict) -> int:
    # This function estimates how many combinations exist in a single hyperparameter space dict
    lengths = []
    for values in space.values():
        if isinstance(values, list):
            lengths.append(len(values))
        else:
            return N_ITER_SEARCH
    return max(1, math.prod(lengths)) if lengths else 1


def get_search_n_iter(param_distributions) -> int:
    # This function chooses how many random iterations to run based on the size of the search space
    if isinstance(param_distributions, list):
        total    = sum(_space_size(space) for space in param_distributions)
        has_xgb  = any(isinstance(s, dict) and "model__n_estimators" in s for s in param_distributions)
        has_rf   = any(isinstance(s, dict) and "model__n_estimators" in s for s in param_distributions)
        use_tree = has_xgb or has_rf
        return min(N_ITER_XGBOOST if use_tree else N_ITER_SEARCH, max(1, total))
    total   = _space_size(param_distributions)
    is_tree = isinstance(param_distributions, dict) and "model__n_estimators" in param_distributions
    return min(N_ITER_XGBOOST if is_tree else N_ITER_SEARCH, total)


def get_cv_and_groups(train_df: pd.DataFrame):
    # This function sets up scaffold-based cross-validation using the Scaffold column as groups
    groups = train_df["Scaffold"].astype(str)
    unique_groups = groups.nunique(dropna=False)
    if unique_groups < N_SPLITS:
        raise ValueError(f"Need at least {N_SPLITS} unique scaffolds, found {unique_groups}")
    return GroupKFold(n_splits=N_SPLITS), groups


# This section contains the data loading function

def load_data(train_path: Path, test_path: Path):
    # This function loads the train and test sets and splits the columns into descriptor,
    # fingerprint, and combined feature groups
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    if TARGET_COL not in train_df.columns or TARGET_COL not in test_df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' is missing.")

    maccs_cols  = [c for c in train_df.columns if c.startswith("MACCS_")]
    morgan_cols = [c for c in train_df.columns if c.startswith("Morgan_")]
    fp_cols     = maccs_cols + morgan_cols

    exclude_cols = [TARGET_COL] + [c for c in META_COLS if c in train_df.columns]
    desc_cols    = [c for c in train_df.columns if c not in exclude_cols and c not in fp_cols]

    if USE_ALL_REPRESENTATIONS:
        feature_sets = {
            "descriptors_only":  desc_cols,
            "maccs_only":        maccs_cols,
            "morgan_only":       morgan_cols,
            "fingerprints_only": fp_cols,
            "whole_dataset":     desc_cols + fp_cols,
        }
    else:
        feature_sets = {
            "descriptors_only":  desc_cols,
            "fingerprints_only": fp_cols,
            "whole_dataset":     desc_cols + fp_cols,
        }

    return train_df, test_df, feature_sets, desc_cols, fp_cols


# This section defines all regression models and their hyperparameter search grids

def get_model_spaces():
    # This function returns a dictionary of model configurations with expanded regularisation grids
    # designed to find well-generalising models rather than the best-fitting training curve
    models = {
        "ElasticNet": {
            "estimator": ElasticNet(max_iter=10000, random_state=RANDOM_STATE),
            "params": {
                "model__alpha":    [1e-3, 1e-2, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
                "model__l1_ratio": [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
            },
        },
        "PLS": {
            "estimator": PLSRegression(),
            "params": {
                "model__n_components": [2, 3, 5, 8, 10, 15],
            },
        },
        "SVR_RBF": {
            "estimator": SVR(kernel="rbf"),
            "params": {
                "model__C":       [0.01, 0.05, 0.1, 0.3, 1, 3, 10, 30],
                "model__gamma":   ["scale", 1e-4, 1e-3, 1e-2],
                "model__epsilon": [0.05, 0.1, 0.2, 0.3, 0.5],
            },
        },
        "RandomForest": {
            "estimator": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS),
            "params": {
                "model__n_estimators":     [200, 400, 600, 800],
                "model__max_depth":        [3, 5, 7, 10, None],
                "model__min_samples_split":[5, 10, 20],
                "model__min_samples_leaf": [2, 4, 8, 16],
                "model__max_features":     ["sqrt", "log2", 0.1, 0.2, 0.3],
                "model__max_samples":      [0.6, 0.7, 0.8, None],
            },
        },
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = {
            "estimator": XGBRegressor(
                objective="reg:squarederror", tree_method="hist",
                random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0,
            ),
            "params": {
                "model__n_estimators":     [50, 100, 200, 300],
                "model__max_depth":        [2, 3, 4],
                "model__learning_rate":    [0.01, 0.02, 0.05],
                "model__subsample":        [0.6, 0.7, 0.8],
                "model__colsample_bytree": [0.4, 0.5, 0.6, 0.7],
                "model__min_child_weight": [3, 5, 8, 15],
                "model__gamma":            [0.1, 0.5, 1.0, 2.0],
                "model__reg_alpha":        [0.1, 0.5, 1.0, 5.0, 10.0],
                "model__reg_lambda":       [5.0, 10.0, 20.0, 50.0],
            },
        }

    return models


def filter_model_params_for_data(model_name: str, params, n_samples: int, n_features: int):
    # This function caps the number of PLS components to what is statistically valid
    # given the training data size and number of features
    def _filter_one(space: dict) -> dict:
        out = dict(space)
        if model_name == "PLS" and "model__n_components" in out:
            max_allowed = min(n_samples - 1, n_features)
            sel = out.get("selector", "passthrough")
            if hasattr(sel, "get_params"):
                sp = sel.get_params()
                k  = sp.get("k") or sp.get("kbest__k")
                if k is not None:
                    max_allowed = min(max_allowed, int(k))
            pca = out.get("pca", "passthrough")
            if hasattr(pca, "n_components") and getattr(pca, "n_components", None) is not None:
                max_allowed = min(max_allowed, int(pca.n_components))
            valid = [n for n in [2, 5, 10, 20, 30, 50] if n <= max_allowed]
            out["model__n_components"] = sorted(set(valid)) if valid else [max(1, int(max_allowed))]
        return out

    if isinstance(params, list):
        filtered = [_filter_one(s) for s in params]
        return [s for s in filtered if all(
            not isinstance(v, list) or len(v) > 0 for v in s.values()
        )]
    return _filter_one(params)


def build_estimator_and_params(model_name, rep_name, base_estimator, base_params,
                                n_samples, n_features):
    # This function builds a scikit-learn Pipeline with a selector, PCA, and model step,
    # and assembles the list of parameter spaces to search over
    estimator = Pipeline(steps=[
        ("selector", "passthrough"),
        ("pca",      "passthrough"),
        ("model",    clone(base_estimator)),
    ])

    param_spaces: list[dict] = []

    # This space uses no dimensionality reduction
    base_space = dict(base_params)
    base_space["selector"] = ["passthrough"]
    base_space["pca"]      = ["passthrough"]
    param_spaces.append(base_space)

    # This space tries SelectKBest with various k values (not used for PLS)
    if model_name != "PLS":
        sel_candidates = [
            c for c in build_selector_candidates(rep_name=rep_name, n_features=n_features,
                                                  n_samples=n_samples)
            if c != "passthrough"
        ]
        if sel_candidates:
            sel_space = dict(base_params)
            sel_space["selector"] = sel_candidates
            sel_space["pca"]      = ["passthrough"]
            param_spaces.append(sel_space)

        # This space tries PCA only for descriptor-heavy representations
        if rep_name in {"descriptors_only", "whole_dataset"}:
            pca_candidates = [
                c for c in build_pca_candidates(n_samples=n_samples, n_features=n_features)
                if c != "passthrough"
            ]
            if pca_candidates:
                pca_space = dict(base_params)
                pca_space["selector"] = ["passthrough"]
                pca_space["pca"]      = pca_candidates
                param_spaces.append(pca_space)

    if len(param_spaces) == 1:
        return estimator, filter_model_params_for_data(model_name, param_spaces[0],
                                                        n_samples, n_features)
    return estimator, filter_model_params_for_data(model_name, param_spaces,
                                                    n_samples, n_features)


# This section contains the model selection logic using the 1-standard-error rule

def param_complexity_score(model_name: str, params: dict, original_n_features: int) -> float:
    # This function assigns a complexity score to a set of hyperparameters so that simpler models
    # within the 1-SE window can be preferred over more complex ones
    sel = params.get("selector", "passthrough")
    pca = params.get("pca",      "passthrough")

    def _sel_dim(s, default):
        if hasattr(s, "get_params"):
            sp = s.get_params()
            k  = sp.get("k") or sp.get("kbest__k")
            if k is not None:
                return int(k)
        return int(default)

    def _pca_dim(p, default):
        nc = getattr(p, "n_components", None)
        return int(nc) if nc is not None else int(default)

    effective_dim = float(_sel_dim(sel, original_n_features))
    if pca != "passthrough":
        effective_dim = min(effective_dim, float(_pca_dim(pca, effective_dim)))

    complexity = effective_dim

    if model_name == "PLS":
        complexity += 5.0 * float(params.get("model__n_components", 2))
    elif model_name == "ElasticNet":
        alpha    = max(float(params.get("model__alpha", 1e-4)), 1e-12)
        l1_ratio = float(params.get("model__l1_ratio", 0.5))
        complexity += (2.0 / alpha) + l1_ratio
    elif model_name == "SVR_RBF":
        C       = max(float(params.get("model__C",       1.0)), 1e-12)
        epsilon = max(float(params.get("model__epsilon", 0.1)), 1e-12)
        gamma_v = params.get("model__gamma", "scale")
        gamma_n = 0.01 if gamma_v == "scale" else (0.1 if gamma_v == "auto" else
                  safe_metric_value(gamma_v, 0.01))
        complexity += (10.0 * np.log10(1.0 + C)) + (2.0 / epsilon) + (100.0 * gamma_n)
    elif model_name == "XGBoost":
        n_estimators = float(params.get("model__n_estimators", 200))
        max_depth_raw = params.get("model__max_depth", 4)
        max_depth = 50.0 if max_depth_raw is None else float(max_depth_raw)
        lr = float(params.get("model__learning_rate", 0.05))
        complexity += (0.02 * n_estimators) + (5.0 * max_depth) + (1.0 / max(lr, 1e-12))

    elif model_name == "RandomForest":
        n_estimators = float(params.get("model__n_estimators", 200))
        max_depth_raw = params.get("model__max_depth", 4)
        max_depth = 50.0 if max_depth_raw is None else float(max_depth_raw)
        complexity += (0.02 * n_estimators) + (5.0 * max_depth)

    return float(complexity)


def choose_cv_candidate_index(search: RandomizedSearchCV, model_name: str,
                               n_features_input: int) -> tuple[int, dict]:
    # This function applies the 1-SE rule to pick the simplest model that is statistically
    # indistinguishable from the best CV score, preferring small train-CV gap
    results = search.cv_results_

    primary_mean_col = f"mean_test_{PRIMARY_SELECTION_METRIC}"
    primary_std_col  = f"std_test_{PRIMARY_SELECTION_METRIC}"

    primary_scores = np.asarray(results[primary_mean_col], dtype=float)
    if not np.isfinite(primary_scores).any():
        raise RuntimeError(f"No valid CV scores for {PRIMARY_SELECTION_METRIC}")

    best_idx_by_mean = int(np.nanargmax(primary_scores))
    best_mean        = float(primary_scores[best_idx_by_mean])

    selection_note = {
        "primary_metric": PRIMARY_SELECTION_METRIC,
        "one_standard_error_rule": USE_ONE_STANDARD_ERROR_RULE,
        "best_mean_cv_score": best_mean,
    }

    if USE_ONE_STANDARD_ERROR_RULE:
        primary_std = np.asarray(results.get(primary_std_col, np.zeros_like(primary_scores)), dtype=float)
        best_std    = safe_metric_value(primary_std[best_idx_by_mean], default=0.0)
        best_se     = best_std / math.sqrt(max(1, N_SPLITS))
        cutoff      = best_mean - (ONE_SE_MULTIPLIER * best_se)
        candidate_indices = [
            i for i, s in enumerate(primary_scores)
            if np.isfinite(s) and float(s) >= cutoff
        ]
        selection_note.update({
            "best_std": best_std, "best_se": best_se,
            "one_se_cutoff": cutoff, "n_candidates_within_cutoff": len(candidate_indices),
        })
    else:
        candidate_indices = [best_idx_by_mean]

    pairwise_scores     = np.asarray(results["mean_test_PairwiseAccuracy"], dtype=float)
    ndcg_scores         = np.asarray(results["mean_test_NDCG_at_5"],        dtype=float)
    rmse_scores         = np.asarray(results["mean_test_RMSE"],             dtype=float)
    train_primary_scores = np.asarray(
        results.get(f"mean_train_{PRIMARY_SELECTION_METRIC}", np.full_like(primary_scores, np.nan)),
        dtype=float,
    )
    train_cv_gap = np.abs(train_primary_scores - primary_scores)
    param_list   = results["params"]

    # This sorts the candidate models so the one with the smallest train-CV gap and lowest complexity
    # gets chosen first
    ranked_candidates = sorted(
        candidate_indices,
        key=lambda i: (
            safe_metric_value(train_cv_gap[i],       default=1e12),
            param_complexity_score(model_name=model_name, params=param_list[i],
                                   original_n_features=n_features_input),
            -safe_metric_value(primary_scores[i],    default=-1e12),
            -safe_metric_value(pairwise_scores[i],   default=-1e12),
            -safe_metric_value(ndcg_scores[i],       default=-1e12),
            -safe_metric_value(rmse_scores[i],       default=-1e12),
        ),
    )

    chosen_idx = int(ranked_candidates[0])
    selection_note["chosen_idx"]              = chosen_idx
    selection_note["chosen_train_cv_gap"]     = safe_metric_value(train_cv_gap[chosen_idx], default=np.nan)
    selection_note["chosen_complexity_score"] = param_complexity_score(
        model_name=model_name, params=param_list[chosen_idx],
        original_n_features=n_features_input,
    )
    return chosen_idx, selection_note


# This section contains functions for generating diagnostic plots

def plot_cv_spearman_per_fold(
    estimator,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    cv,
    model_label: str,
    save_path: Path,
) -> dict:
    # This function runs cross-validation and plots the Spearman correlation for each fold,
    # showing the mean line and ±1 SD band
    from sklearn.model_selection import cross_validate

    cv_results = cross_validate(
        estimator, X, y,
        cv=cv,
        groups=groups,
        scoring=make_scorer(spearman_rank_score, greater_is_better=True),
        return_train_score=False,
        n_jobs=N_JOBS,
    )
    fold_scores = cv_results["test_score"]
    mean_sp     = float(np.mean(fold_scores))
    std_sp      = float(np.std(fold_scores, ddof=1))
    n_folds     = len(fold_scores)

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    fold_labels = [f"Fold {i + 1}" for i in range(n_folds)]
    x_pos = np.arange(n_folds)

    # This part colours bars: green if above the mean, red if below
    bar_colors = ["#2E7D6E" if s >= mean_sp else "#C0392B" for s in fold_scores]
    bars = ax.bar(x_pos, fold_scores, color=bar_colors, width=0.55,
                  zorder=3, linewidth=0, alpha=0.88)

    ax.axhline(mean_sp, color="#1A1A2E", linewidth=1.8, linestyle="--",
               zorder=4, label=f"Mean ρ = {mean_sp:.3f}")

    ax.axhspan(mean_sp - std_sp, mean_sp + std_sp,
               color="#1A1A2E", alpha=0.08, zorder=2,
               label=f"±1 SD  (SD = {std_sp:.3f})")

    for bar, score in zip(bars, fold_scores):
        va = "bottom" if score >= 0 else "top"
        offset = 0.005 if score >= 0 else -0.005
        ax.text(bar.get_x() + bar.get_width() / 2,
                score + offset, f"{score:.3f}",
                ha="center", va=va, fontsize=9,
                fontfamily="DejaVu Sans", color="#1A1A2E", fontweight="500")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(fold_labels, fontsize=10, fontfamily="DejaVu Sans")
    ax.set_ylabel("Spearman Rank Correlation (ρ)", fontsize=11,
                  fontfamily="DejaVu Sans", labelpad=8)
    ax.set_xlabel("Cross-Validation Fold (Scaffold-based)", fontsize=11,
                  fontfamily="DejaVu Sans", labelpad=8)
    ax.set_title(
        f"Per-fold Spearman ρ — {model_label}\n"
        f"Mean ± SD = {mean_sp:.3f} ± {std_sp:.3f}",
        fontsize=12, fontfamily="DejaVu Sans", fontweight="bold", pad=12,
    )

    y_min = min(0, np.min(fold_scores) - 0.08)
    y_max = min(1.0, np.max(fold_scores) + 0.12)
    ax.set_ylim(y_min, y_max)

    ax.yaxis.grid(True, linestyle=":", linewidth=0.8, color="#CCCCCC", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")

    ax.legend(fontsize=9, framealpha=0.85, loc="lower right",
              frameon=True, edgecolor="#CCCCCC")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()

    return {"fold_scores": fold_scores.tolist(), "mean": mean_sp, "std": std_sp}


def save_scatter_plot(y_true, y_score, title, path):
    # This function draws a scatter plot of observed vs predicted pChEMBL values
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_score, alpha=0.75)
    lo = min(np.min(y_true), np.min(y_score))
    hi = max(np.max(y_true), np.max(y_score))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Observed pChEMBL"); plt.ylabel("Model ranking score")
    plt.title(title); plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


def save_rank_residual_plot(y_true, y_score, title, path):
    # This function draws a residual plot (observed minus predicted) against the predicted score
    residuals = np.asarray(y_true) - np.asarray(y_score)
    plt.figure(figsize=(6, 4.5))
    plt.scatter(y_score, residuals, alpha=0.75)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Model ranking score"); plt.ylabel("Observed - score")
    plt.title(title); plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()


# This section contains the applicability domain functions

def get_representation_space(best_estimator, X: pd.DataFrame) -> np.ndarray:
    # This function projects data into the feature space the model actually sees
    # by running the pipeline up to (but not including) the final model step
    X_space = X.copy()
    if hasattr(best_estimator, "named_steps"):
        sel = best_estimator.named_steps.get("selector")
        if sel is not None and sel != "passthrough":
            X_space = sel.transform(X_space)
        pca = best_estimator.named_steps.get("pca")
        if pca is not None and pca != "passthrough":
            X_space = pca.transform(X_space)
    X_space = np.asarray(X_space, dtype=float)
    if X_space.ndim == 1:
        X_space = X_space.reshape(-1, 1)
    return X_space


def build_knn_ad_artifacts(X_space: np.ndarray, k: int = 5, percentile: float = 95.0) -> dict:
    # This function builds the k-NN applicability domain by computing pairwise distances
    # between training compounds and setting the threshold at the given percentile
    X_space   = np.asarray(X_space, dtype=float)
    n_samples = int(X_space.shape[0])
    if n_samples == 0:
        raise ValueError("Cannot build AD artifacts from an empty training matrix.")
    if n_samples == 1:
        return {
            "method": "knn_mean_distance", "distance_metric": "euclidean", "k": 0,
            "percentile": float(percentile), "threshold": float("inf"),
            "reference": X_space.astype(np.float32), "space_dim": int(X_space.shape[1]),
            "train_mean_knn_distance_summary": {"min": 0.0, "median": 0.0, "max": 0.0},
        }
    k_eff = min(int(k), max(1, n_samples - 1))
    dist_matrix = pairwise_distances(X_space, X_space, metric="euclidean")
    np.fill_diagonal(dist_matrix, np.inf)
    mean_knn_dist = np.sort(dist_matrix, axis=1)[:, :k_eff].mean(axis=1)
    threshold = float(np.quantile(mean_knn_dist, percentile / 100.0))
    return {
        "method": "knn_mean_distance", "distance_metric": "euclidean",
        "k": int(k_eff), "percentile": float(percentile), "threshold": threshold,
        "reference": X_space.astype(np.float32), "space_dim": int(X_space.shape[1]),
        "train_mean_knn_distance_summary": {
            "min": float(np.min(mean_knn_dist)),
            "median": float(np.median(mean_knn_dist)),
            "max": float(np.max(mean_knn_dist)),
        },
    }


# This section contains the fitting logic for one model + representation combination

def build_prediction_frame(rep_name, model_name, test_df, y_test, y_score):
    # This function assembles a DataFrame of test predictions with observed and predicted ranks
    pred_df = pd.DataFrame({
        "representation": rep_name, "model": model_name,
        "observed_pChEMBL": y_test, "ranking_score": y_score,
    })
    for col in META_COLS:
        if col in test_df.columns:
            pred_df[col] = test_df[col].values
    pred_df["observed_rank"]   = descending_rank(y_test)
    pred_df["predicted_rank"]  = descending_rank(y_score)
    pred_df["score_minus_observed"] = y_score - y_test
    pred_df["abs_rank_gap"]    = np.abs(pred_df["observed_rank"] - pred_df["predicted_rank"])
    return pred_df.sort_values(["predicted_rank", "ranking_score", "observed_pChEMBL"],
                                ascending=[True, False, False]).reset_index(drop=True)


def fit_one_combination(model_name, model_cfg, rep_name, train_df, test_df,
                         feature_cols, cv, groups):
    # This function runs hyperparameter search for one model + representation combination,
    # selects the best configuration using the 1-SE rule, retrains on the full training set,
    # evaluates on the test set, and builds the model bundle for saving
    X_train = train_df[feature_cols].copy()
    X_test  = test_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].astype(float).to_numpy()
    y_test  = test_df[TARGET_COL].astype(float).to_numpy()

    estimator, param_distributions = build_estimator_and_params(
        model_name=model_name, rep_name=rep_name,
        base_estimator=model_cfg["estimator"], base_params=model_cfg["params"],
        n_samples=X_train.shape[0], n_features=X_train.shape[1],
    )

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=get_search_n_iter(param_distributions),
        scoring=get_ranking_scorers(),
        refit=False,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=0,
        return_train_score=True,
    )
    search.fit(X_train, y_train, groups=groups)

    chosen_idx, selection_note = choose_cv_candidate_index(
        search=search, model_name=model_name, n_features_input=len(feature_cols),
    )

    # This part retrains the chosen configuration on the full training set
    chosen_params  = search.cv_results_["params"][chosen_idx]
    best_estimator = clone(estimator).set_params(**chosen_params)
    best_estimator.fit(X_train, y_train)

    y_score_train = np.asarray(best_estimator.predict(X_train), dtype=float).ravel()
    y_score_test  = np.asarray(best_estimator.predict(X_test),  dtype=float).ravel()

    train_space  = get_representation_space(best_estimator, X_train)
    ad_artifacts = build_knn_ad_artifacts(train_space)

    cv_spearman   = float(search.cv_results_["mean_test_Spearman"][chosen_idx])
    cv_pairwise   = float(search.cv_results_["mean_test_PairwiseAccuracy"][chosen_idx])
    cv_ndcg       = float(search.cv_results_["mean_test_NDCG_at_5"][chosen_idx])
    cv_kendall    = float(search.cv_results_["mean_test_KendallTau"][chosen_idx])
    cv_top5       = float(search.cv_results_["mean_test_Top5_Overlap"][chosen_idx])
    cv_rmse       = -float(search.cv_results_["mean_test_RMSE"][chosen_idx])
    cv_mae        = -float(search.cv_results_["mean_test_MAE"][chosen_idx])
    cv_r2         = float(search.cv_results_["mean_test_R2"][chosen_idx])

    train_row = {
        "representation": rep_name, "model": model_name,
        "n_features_input": len(feature_cols),
        "train_NDCG_at_5":        ndcg_at_k_score(y_train, y_score_train, k=RANKING_EVAL_K),
        "train_PairwiseAccuracy":  pairwise_accuracy_score(y_train, y_score_train),
        "train_Spearman":          spearman_rank_score(y_train, y_score_train),
        "train_KendallTau":        kendall_tau_score(y_train, y_score_train),
        "train_Top5_Overlap":      top_k_overlap_fraction(y_train, y_score_train, k=RANKING_EVAL_K),
        "train_RMSE":              rmse(y_train, y_score_train),
        "train_MAE":               float(mean_absolute_error(y_train, y_score_train)),
        "train_R2":                float(r2_score(y_train, y_score_train)),
        "train_PCC":               pearson_corr(y_train, y_score_train),
        "train_CCC":               concordance_correlation_coefficient(y_train, y_score_train),
        "best_params":             json.dumps(stringify_params(chosen_params)),
    }
    cv_row = {
        "representation": rep_name, "model": model_name,
        "n_features_input":        len(feature_cols),
        "cv_NDCG_at_5_mean":       cv_ndcg,
        "cv_PairwiseAccuracy_mean":cv_pairwise,
        "cv_Spearman_mean":        cv_spearman,
        "cv_KendallTau_mean":      cv_kendall,
        "cv_Top5_Overlap_mean":    cv_top5,
        "cv_RMSE_mean":            cv_rmse,
        "cv_MAE_mean":             cv_mae,
        "cv_R2_mean":              cv_r2,
        "best_params":             json.dumps(stringify_params(chosen_params)),
    }
    test_row = {
        "representation": rep_name, "model": model_name,
        "n_features_input": len(feature_cols),
        "test_NDCG_at_5":        ndcg_at_k_score(y_test, y_score_test, k=RANKING_EVAL_K),
        "test_PairwiseAccuracy":  pairwise_accuracy_score(y_test, y_score_test),
        "test_Spearman":          spearman_rank_score(y_test, y_score_test),
        "test_KendallTau":        kendall_tau_score(y_test, y_score_test),
        "test_Top5_Overlap":      top_k_overlap_fraction(y_test, y_score_test, k=RANKING_EVAL_K),
        "test_RMSE":              rmse(y_test, y_score_test),
        "test_MAE":               float(mean_absolute_error(y_test, y_score_test)),
        "test_R2":                float(r2_score(y_test, y_score_test)),
        "test_PCC":               pearson_corr(y_test, y_score_test),
        "test_CCC":               concordance_correlation_coefficient(y_test, y_score_test),
        "best_params":            json.dumps(stringify_params(chosen_params)),
    }
    pred_df = build_prediction_frame(rep_name, model_name, test_df, y_test, y_score_test)
    model_bundle = {
        "best_estimator":       best_estimator,
        "feature_columns":      feature_cols,
        "target_column":        TARGET_COL,
        "representation":       rep_name,
        "model_name":           model_name,
        "best_params":          chosen_params,
        "selection_metric":     PRIMARY_SELECTION_METRIC,
        "selection_note":       selection_note,
        "ranking_eval_k":       RANKING_EVAL_K,
        "ad_artifacts":         ad_artifacts,
        "output_interpretation":"Use model output as a ranking score. Higher = higher priority.",
    }
    return train_row, cv_row, test_row, pred_df, model_bundle


# This section contains the soft-voting ensemble function

def build_regression_ensemble(best_bundles, train_df, test_df, y_train, y_test,
                                preds_dir, scatter_dir, residual_dir):
    # This function averages the predictions of the best model from each representation
    # to build a soft-voting ensemble and evaluates it on the test set
    bundles_info = [
        (b["best_estimator"], b["feature_columns"],
         sanitize_name(b["representation"]) + "__" + sanitize_name(b["model_name"]))
        for b in best_bundles
    ]
    if len(bundles_info) < 2:
        print("[INFO] Fewer than 2 models for ensemble; skipping.")
        return None

    train_preds = np.column_stack([
        np.asarray(est.predict(train_df[fc].values), dtype=float)
        for est, fc, _ in bundles_info
    ])
    test_preds = np.column_stack([
        np.asarray(est.predict(test_df[fc].values), dtype=float)
        for est, fc, _ in bundles_info
    ])

    ensemble_train = train_preds.mean(axis=1)
    ensemble_test  = test_preds.mean(axis=1)

    train_spearman = spearman_rank_score(y_train, ensemble_train)
    test_spearman  = spearman_rank_score(y_test,  ensemble_test)
    test_r2        = float(r2_score(y_test, ensemble_test))

    print(f"  [Ensemble] train_Spearman={train_spearman:.4f} | "
          f"test_Spearman={test_spearman:.4f} | "
          f"test_R2={test_r2:.4f}")

    ens_pred_df = pd.DataFrame({
        "representation": "ensemble_avg",
        "model":          "AveragePredictions",
        "observed_pChEMBL": y_test,
        "ranking_score":    ensemble_test,
        "observed_rank":    descending_rank(y_test),
        "predicted_rank":   descending_rank(ensemble_test),
    })
    ens_pred_df.to_csv(preds_dir / "ensemble_avg__test_predictions.csv", index=False)
    save_scatter_plot(y_test, ensemble_test,
                      title="Ensemble (avg) | Observed vs Ranking Score",
                      path=scatter_dir / "ensemble_avg__scatter.png")
    save_rank_residual_plot(y_test, ensemble_test,
                            title="Ensemble (avg) | Score Residuals",
                            path=residual_dir / "ensemble_avg__residual.png")

    return {
        "representation":  "ensemble_avg",
        "model_name":      "AveragePredictions",
        "train_Spearman":  train_spearman,
        "test_Spearman":   test_spearman,
        "test_R2":         test_r2,
        "test_PairwiseAccuracy": pairwise_accuracy_score(y_test, ensemble_test),
        "test_NDCG_at_5":  ndcg_at_k_score(y_test, ensemble_test, k=RANKING_EVAL_K),
        "test_RMSE":       rmse(y_test, ensemble_test),
    }


# This is the main function that orchestrates the full regression training and evaluation pipeline
def main():
    ensure_dir(OUTDIR)
    tables_dir   = OUTDIR / "tables";    ensure_dir(tables_dir)
    plots_dir    = OUTDIR / "plots";     ensure_dir(plots_dir)
    scatter_dir  = plots_dir / "scatter"; ensure_dir(scatter_dir)
    residual_dir = plots_dir / "residual";ensure_dir(residual_dir)
    preds_dir    = OUTDIR / "predictions";ensure_dir(preds_dir)
    models_dir   = OUTDIR / "models";     ensure_dir(models_dir)

    print("[INFO] Loading ranking data...")
    train_df, test_df, feature_sets, descriptor_cols, fingerprint_cols = load_data(TRAIN_PATH, TEST_PATH)
    cv, groups = get_cv_and_groups(train_df)
    model_spaces = get_model_spaces()

    y_train = train_df[TARGET_COL].astype(float).to_numpy()
    y_test  = test_df[TARGET_COL].astype(float).to_numpy()

    train_rows  = []
    cv_rows     = []
    test_rows   = []
    pred_frames = []
    best_bundle_per_rep: dict[str, dict]  = {}
    best_cv_spearman_per_rep: dict[str, float] = {}

    # This outer loop iterates over each feature representation (descriptors, fingerprints, etc.)
    for rep_name, cols in feature_sets.items():
        if len(cols) == 0:
            print(f"[WARNING] Skipping empty representation: {rep_name}")
            continue

        print(f"\n[INFO] Representation: {rep_name} ({len(cols)} features)")
        # This inner loop tries each model algorithm for the current representation
        for model_name, model_cfg in model_spaces.items():
            print(f"  -> Tuning {model_name}")
            try:
                train_row, cv_row, test_row, pred_df, model_bundle = fit_one_combination(
                    model_name=model_name, model_cfg=model_cfg, rep_name=rep_name,
                    train_df=train_df, test_df=test_df,
                    feature_cols=cols, cv=cv, groups=groups,
                )
                train_rows.append(train_row)
                cv_rows.append(cv_row)
                test_rows.append(test_row)
                pred_frames.append(pred_df)

                save_scatter_plot(pred_df["observed_pChEMBL"].to_numpy(),
                                  pred_df["ranking_score"].to_numpy(),
                                  title=f"{rep_name} | {model_name} | Observed vs Ranking Score",
                                  path=scatter_dir / f"{rep_name}__{model_name}__scatter.png")
                save_rank_residual_plot(pred_df["observed_pChEMBL"].to_numpy(),
                                        pred_df["ranking_score"].to_numpy(),
                                        title=f"{rep_name} | {model_name} | Score Residuals",
                                        path=residual_dir / f"{rep_name}__{model_name}__residual.png")
                pred_df.to_csv(preds_dir / f"{rep_name}__{model_name}__test_predictions.csv", index=False)

                bundle_name = f"{sanitize_name(rep_name)}__{sanitize_name(model_name)}.joblib"
                joblib.dump(model_bundle, models_dir / bundle_name)

                # This part keeps track of the best model per representation for the ensemble
                cv_spearman = cv_row["cv_Spearman_mean"]
                if rep_name not in best_cv_spearman_per_rep or cv_spearman > best_cv_spearman_per_rep[rep_name]:
                    best_cv_spearman_per_rep[rep_name] = cv_spearman
                    best_bundle_per_rep[rep_name]       = model_bundle

            except Exception as exc:
                print(f"  [WARNING] {rep_name} | {model_name} failed: {exc}")

    if not test_rows:
        raise RuntimeError("No ranking models finished successfully.")

    # This part compiles all results into summary tables and computes train-test gap columns
    train_df_sum = pd.DataFrame(train_rows).sort_values(["representation", "model"]).reset_index(drop=True)
    cv_df        = pd.DataFrame(cv_rows).sort_values(["representation", "model"]).reset_index(drop=True)
    test_df_sum  = pd.DataFrame(test_rows).sort_values(["representation", "model"]).reset_index(drop=True)
    preds_all    = pd.concat(pred_frames, ignore_index=True)

    full_summary = (
        train_df_sum
        .merge(cv_df,       on=["representation", "model", "n_features_input", "best_params"], how="outer")
        .merge(test_df_sum, on=["representation", "model", "n_features_input", "best_params"], how="outer")
    )
    full_summary["NDCG_at_5_gap_train_minus_test"]         = full_summary["train_NDCG_at_5"]        - full_summary["test_NDCG_at_5"]
    full_summary["PairwiseAccuracy_gap_train_minus_test"]   = full_summary["train_PairwiseAccuracy"] - full_summary["test_PairwiseAccuracy"]
    full_summary["Spearman_gap_train_minus_test"]           = full_summary["train_Spearman"]         - full_summary["test_Spearman"]
    full_summary["Spearman_gap_train_minus_cv"]             = full_summary["train_Spearman"]         - full_summary["cv_Spearman_mean"]
    full_summary["PairwiseAccuracy_gap_train_minus_cv"]     = full_summary["train_PairwiseAccuracy"] - full_summary["cv_PairwiseAccuracy_mean"]
    full_summary["RMSE_gap_test_minus_train"]               = full_summary["test_RMSE"]              - full_summary["train_RMSE"]
    full_summary["R2_gap_train_minus_test"]                 = full_summary["train_R2"]               - full_summary["test_R2"]

    # This part sorts the models by train-CV gap (ascending) so well-generalising models appear first
    ranking_overview = full_summary.copy()
    ranking_overview["abs_Spearman_gap_train_minus_test"] = ranking_overview["Spearman_gap_train_minus_test"].abs()
    ranking_overview["abs_Spearman_gap_train_minus_cv"]   = ranking_overview["Spearman_gap_train_minus_cv"].abs()
    ranking_overview["abs_PA_gap_train_minus_test"]       = ranking_overview["PairwiseAccuracy_gap_train_minus_test"].abs()
    ranking_overview = ranking_overview.sort_values(
        ["abs_Spearman_gap_train_minus_cv", "cv_Spearman_mean",
         "cv_PairwiseAccuracy_mean", "cv_NDCG_at_5_mean",
         "abs_Spearman_gap_train_minus_test", "test_Spearman", "test_PairwiseAccuracy"],
        ascending=[True, False, False, False, True, False, False],
    ).reset_index(drop=True)

    train_df_sum.to_csv(tables_dir / "train_summary.csv", index=False)
    cv_df.to_csv(tables_dir / "cv_summary.csv", index=False)
    test_df_sum.to_csv(tables_dir / "test_summary.csv", index=False)
    full_summary.to_csv(tables_dir / "full_performance_summary.csv", index=False)
    ranking_overview.to_csv(tables_dir / "model_ranking_overview.csv", index=False)
    preds_all.to_csv(preds_dir / "test_predictions_all_models.csv", index=False)

    # This part saves a dedicated overfitting diagnostic table showing train, CV, and test Spearman scores
    gap_cols = ["representation", "model", "train_Spearman", "cv_Spearman_mean",
                "test_Spearman", "Spearman_gap_train_minus_cv",
                "Spearman_gap_train_minus_test", "train_R2", "test_R2", "R2_gap_train_minus_test"]
    full_summary[gap_cols].sort_values("abs_Spearman_gap_train_minus_cv" if "abs_Spearman_gap_train_minus_cv" in full_summary.columns
                                        else "Spearman_gap_train_minus_cv").to_csv(
        tables_dir / "overfitting_gap_diagnostics.csv", index=False)

    # This part selects the best model per representation by CV and by test Spearman
    cv_sort_cols   = ["cv_Spearman_mean", "cv_PairwiseAccuracy_mean", "cv_NDCG_at_5_mean"]
    test_sort_cols = ["test_Spearman", "test_PairwiseAccuracy", "test_NDCG_at_5"]

    best_by_cv_df = (
        cv_df.sort_values(cv_sort_cols, ascending=[False, False, False])
        .groupby("representation", as_index=False).first()
        .sort_values(cv_sort_cols, ascending=[False, False, False])
        .reset_index(drop=True)
    )
    best_by_cv_df.to_csv(tables_dir / "best_model_per_representation_by_cv.csv", index=False)

    best_by_test_df = (
        test_df_sum.sort_values(test_sort_cols, ascending=[False, False, False])
        .groupby("representation", as_index=False).first()
        .sort_values(test_sort_cols, ascending=[False, False, False])
        .reset_index(drop=True)
    )
    best_by_test_df.to_csv(tables_dir / "best_model_per_representation_by_test.csv", index=False)

    # This part finds the metric winners — one row per metric showing the best model for that metric
    metric_winners = pd.concat([
        cv_df.nlargest(1, "cv_Spearman_mean").assign(metric="cv_Spearman_mean"),
        cv_df.nlargest(1, "cv_PairwiseAccuracy_mean").assign(metric="cv_PairwiseAccuracy_mean"),
        cv_df.nlargest(1, "cv_NDCG_at_5_mean").assign(metric="cv_NDCG_at_5_mean"),
        test_df_sum.nlargest(1, "test_Spearman").assign(metric="test_Spearman"),
        test_df_sum.nlargest(1, "test_PairwiseAccuracy").assign(metric="test_PairwiseAccuracy"),
        test_df_sum.nlargest(1, "test_NDCG_at_5").assign(metric="test_NDCG_at_5"),
        test_df_sum.nsmallest(1, "test_RMSE").assign(metric="test_RMSE_reference"),
        test_df_sum.nlargest(1, "test_R2").assign(metric="test_R2_reference"),
    ], ignore_index=True)
    metric_winners.to_csv(tables_dir / "metric_winners.csv", index=False)

    # This part selects the overall winner by the primary CV metric.
    # Generalization gap is kept as a tie-breaker/diagnostic, not the main selector.
    best_cv_spearman = float(ranking_overview["cv_Spearman_mean"].max())
    selection_pool_df = ranking_overview.sort_values(
        ["cv_Spearman_mean", "cv_PairwiseAccuracy_mean", "cv_NDCG_at_5_mean",
         "abs_Spearman_gap_train_minus_cv",
         "abs_Spearman_gap_train_minus_test"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    overall_winner_df = selection_pool_df.head(1).reset_index(drop=True)

    if not overall_winner_df.empty:
        winner = overall_winner_df.iloc[0]
        wp = models_dir / f"{sanitize_name(winner['representation'])}__{sanitize_name(winner['model'])}.joblib"
        if wp.exists():
            (models_dir / "overall_best_model_path.txt").write_text(str(wp), encoding="utf-8")

    # This part generates the per-fold Spearman plot for the best test-Spearman model
    best_test_sp_row   = test_df_sum.sort_values("test_Spearman", ascending=False).iloc[0]
    best_test_rep_sp   = best_test_sp_row["representation"]
    best_test_model_sp = best_test_sp_row["model"]
    best_test_sp_bundle_path = (
        models_dir / f"{sanitize_name(best_test_rep_sp)}__{sanitize_name(best_test_model_sp)}.joblib"
    )
    if best_test_sp_bundle_path.exists():
        print(f"\n[INFO] Plotting per-fold Spearman for best test-Spearman model: "
              f"{best_test_rep_sp} | {best_test_model_sp}")
        best_sp_bundle = joblib.load(best_test_sp_bundle_path)
        cv_sp_results  = plot_cv_spearman_per_fold(
            estimator=best_sp_bundle["best_estimator"],
            X=train_df[best_sp_bundle["feature_columns"]],
            y=y_train,
            groups=train_df["Scaffold"].astype(str),
            cv=cv,
            model_label=f"{best_test_rep_sp} | {best_test_model_sp}",
            save_path=plots_dir / "best_test_spearman_model__cv_spearman_per_fold.png",
        )
        fold_sp_df = pd.DataFrame({
            "fold":         [f"Fold {i + 1}" for i in range(len(cv_sp_results["fold_scores"]))],
            "spearman":     cv_sp_results["fold_scores"],
            "spearman_mean":cv_sp_results["mean"],
            "spearman_std": cv_sp_results["std"],
            "representation": best_test_rep_sp,
            "model":        best_test_model_sp,
        })
        fold_sp_df.to_csv(tables_dir / "best_test_spearman_model__cv_spearman_per_fold.csv", index=False)
        print(f"  -> Fold Spearman: {[f'{s:.3f}' for s in cv_sp_results['fold_scores']]}  "
              f"Mean={cv_sp_results['mean']:.3f}  SD={cv_sp_results['std']:.3f}")

    # This part builds the average-prediction ensemble from the best model per representation
    print("\n[INFO] Building average-prediction ensemble...")
    ensemble_result = build_regression_ensemble(
        best_bundles=list(best_bundle_per_rep.values()),
        train_df=train_df, test_df=test_df,
        y_train=y_train, y_test=y_test,
        preds_dir=preds_dir, scatter_dir=scatter_dir, residual_dir=residual_dir,
    )
    if ensemble_result is not None:
        joblib.dump(ensemble_result, models_dir / "ensemble_avg.joblib")

    # This part writes a plain-text summary report of the overall winner and ensemble performance
    with open(OUTDIR / "ranking_report.txt", "w", encoding="utf-8") as f:
        f.write("QSAR Ranking — Improved Anti-Overfitting Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Training rows : {len(train_df)}\n")
        f.write(f"Test rows     : {len(test_df)}\n")
        f.write(f"Representations: {list(feature_sets.keys())}\n")
        f.write(f"N_ITER_SEARCH : {N_ITER_SEARCH}\n")
        f.write(f"N_ITER_XGBOOST: {N_ITER_XGBOOST}\n")
        f.write(f"RandomForest added as a new model family.\n")
        f.write(f"Primary selection metric          : {PRIMARY_SELECTION_METRIC}\n")
        f.write(f"1-SE rule                         : {USE_ONE_STANDARD_ERROR_RULE}\n")
        f.write(f"Winner selection                  : highest CV Spearman\n")
        f.write(f"Gap use                           : tie-breaker/diagnostic only\n\n")

        if not overall_winner_df.empty:
            row = overall_winner_df.iloc[0]
            f.write("Overall winner (highest CV Spearman)\n")
            f.write(
                f"  {row['representation']} | {row['model']} | "
                f"train_Spearman={row['train_Spearman']:.4f} | "
                f"cv_Spearman={row['cv_Spearman_mean']:.4f} | "
                f"test_Spearman={row['test_Spearman']:.4f} | "
                f"|train-cv gap|={abs(row['Spearman_gap_train_minus_cv']):.4f}\n"
            )

        if ensemble_result is not None:
            f.write("\nAverage-prediction ensemble\n")
            f.write(
                f"  train_Spearman={ensemble_result['train_Spearman']:.4f} | "
                f"test_Spearman={ensemble_result['test_Spearman']:.4f} | "
                f"test_R2={ensemble_result['test_R2']:.4f}\n"
            )

        if not best_by_test_df.empty:
            row = best_by_test_df.iloc[0]
            f.write("\nBest observed on test (reference only)\n")
            f.write(
                f"  {row['representation']} | {row['model']} | "
                f"test_Spearman={row['test_Spearman']:.4f} | "
                f"test_PairwiseAccuracy={row['test_PairwiseAccuracy']:.4f} | "
                f"test_NDCG_at_{RANKING_EVAL_K}={row['test_NDCG_at_5']:.4f}\n"
            )

    print(f"\n[INFO] Ranking results saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
