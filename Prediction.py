
from __future__ import annotations

from pathlib import Path
import json
import re
import warnings

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# This class must be defined here so that joblib can load the MultiSpace SVM
# model bundle that was saved in QSAR Activity Classification.py.
# It is an exact copy of the class used during training.
class MultiSpaceSelector(BaseEstimator, TransformerMixin):

    def __init__(self, desc_cols=None, maccs_cols=None, morgan_cols=None,
                 k_desc=75, k_maccs=40, k_morgan=50):
        self.desc_cols   = desc_cols   or []
        self.maccs_cols  = maccs_cols  or []
        self.morgan_cols = morgan_cols or []
        self.k_desc      = k_desc
        self.k_maccs     = k_maccs
        self.k_morgan    = k_morgan

    def fit(self, X, y=None):
        # This method fits a separate feature selector for each of the three chemical spaces
        X = self._to_df(X)
        self._sel_desc_   = self._fit_sel(X, y, self.desc_cols,   self.k_desc)
        self._sel_maccs_  = self._fit_sel(X, y, self.maccs_cols,  self.k_maccs)
        self._sel_morgan_ = self._fit_sel(X, y, self.morgan_cols, self.k_morgan)
        return self

    def transform(self, X):
        # This method applies each fitted selector to its corresponding feature group and stacks the results
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
        # This helper converts numpy arrays to DataFrames so column names can be used
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    @staticmethod
    def _fit_sel(X_df, y, cols, k):
        # This method removes zero-variance columns then fits SelectKBest on whatever columns remain
        if not cols:
            return None
        X_arr = X_df[cols].values
        var_mask = X_arr.var(axis=0) > 0
        live_cols = [c for c, keep in zip(cols, var_mask) if keep]
        if not live_cols:
            return None
        k_eff = min(int(k), len(live_cols))
        if k_eff <= 0:
            return None
        sel = SelectKBest(f_classif, k=k_eff)
        sel.fit(X_arr[:, var_mask], y)
        sel._col_names = live_cols
        sel._orig_cols = cols
        sel._var_mask  = var_mask
        return sel

    def get_selected_feature_names(self) -> list:
        # This method returns the names of all features that were selected across all three spaces
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


# This section defines all file paths and user-configurable settings for the prediction run
ROOT = Path(r"C:\Users\admin\Desktop\Machine Learning For Capstone")

PREDICTION_FEATURE_FILE = ROOT / "Prediction Feature Outputs" / "chembl_prediction_model_ready.csv"
RAW_PREDICTION_METADATA_FILE = ROOT / "Prediction Dataset.csv"
MODEL_DIR = ROOT / "Prediction Models"

# Fallback model paths used only when the official winner cannot be located.
# The official winner is read from Activity Classifier Output/tables/final_winner_selection.csv
# and may now be any model family (LR, RF, XGBoost, SVM, or any *_MultiSpace variant).
PRIMARY_MODEL = MODEL_DIR / "multi_space__svm_rbf_multispace.joblib"
REGRESSION_MODEL = MODEL_DIR / "whole_dataset__svr_rbf.joblib"

# When set to True, the script always uses the official winning model from the training output
# rather than any manually deployed file in the Prediction Models directory.
# This must be True now that multi-space models exist for all algorithms, not just SVM.
PREFER_OFFICIAL_WINNERS = True

OUTDIR = ROOT / "Prediction Outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

CANONICAL_SMILES_COL = "Canonical_SMILES"
MORGAN_PREFIX = "Morgan_"
MACCS_PREFIX = "MACCS_"

MAX_MISSING_FRACTION_BEFORE_ERROR = 0.25
DESCRIPTOR_MISSING_THRESHOLD = 0.20
DESCRIPTOR_CORRELATION_THRESHOLD = 0.95
EXPORT_FLOAT_DECIMALS = 4

PRIMARY_THRESHOLD_OVERRIDE = 0.24
EXPLORATORY_RANGE_LOWER = 0.20

TOP_N_REPURPOSING = 20

# These are fallback locations to search for the prediction feature file if the primary path fails
FEATURE_FILE_CANDIDATES = [
    ROOT / "Prediction Feature Outputs" / "chembl_prediction_model_ready.csv",
    ROOT / "chembl_prediction_feature_outputs" / "chembl_prediction_model_ready.csv",
    ROOT / "Prediction Feature Outputs" / "chembl_prediction_model_ready",
    ROOT / "chembl_prediction_feature_outputs" / "chembl_prediction_model_ready",
]

# These are all directories to search when auto-discovering model files
MODEL_SEARCH_DIRS = [
    MODEL_DIR,
    ROOT / "Activity Classifier Output" / "models",
    ROOT / "Regression Output" / "models",
]

CLASSIFICATION_TABLE_PATH = ROOT / "Activity Classifier Output" / "tables" / "final_winner_selection.csv"
REGRESSION_BEST_MODEL_PATH_FILE = ROOT / "Regression Output" / "models" / "overall_best_model_path.txt"

CLASSIFICATION_RAW_PATH = ROOT / "Feature Generation Outputs" / "Classification_dataset.csv"
REGRESSION_RAW_PATH = ROOT / "Feature Generation Outputs" / "Regression_dataset.csv"
CLASSIFICATION_TRAIN_PROCESSED_PATH = ROOT / "Data Split Outputs" / "classification_train_dataset.csv"
REGRESSION_TRAIN_PROCESSED_PATH = ROOT / "Data Split Outputs" / "regression_train_dataset.csv"


# This section contains small utility functions used throughout the script
def log(msg: str):
    print(msg, flush=True)


def sanitize_name(text: str) -> str:
    # This function converts a string into a safe filename by replacing special characters
    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .lower()
    )


def normalize_name(text: str) -> str:
    # This function strips non-alphanumeric characters for consistent name matching
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def detect_separator(path: Path) -> str:
    # This function reads the first line of a file to decide if it uses commas or semicolons
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    return ";" if first_line.count(";") > first_line.count(",") else ","


def resolve_table(base: Path) -> Path:
    # This function finds the actual file even if the extension is missing from the given path
    if base.exists():
        return base

    parent = base.parent
    stem = base.stem if base.suffix else base.name

    if parent.exists():
        for ext in [".csv", ".txt", ".tsv", ".xlsx", ".parquet"]:
            p = parent / f"{stem}{ext}"
            if p.exists():
                return p

        stem_low = stem.lower()
        for p in parent.iterdir():
            if p.is_file() and p.stem.lower() == stem_low and p.suffix.lower() in {".csv", ".txt", ".tsv", ".xlsx", ".parquet"}:
                return p

    raise FileNotFoundError(f"Could not find file for: {base}")


def read_table(path: Path) -> pd.DataFrame:
    # This function loads a table from disk, choosing the right reader based on file extension
    path = resolve_table(path)
    suffix = path.suffix.lower()

    if suffix in {".csv", ".txt"}:
        sep = detect_separator(path)
        return pd.read_csv(path, sep=sep, low_memory=False)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)
    if suffix == ".xlsx":
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file format: {path.suffix}")


def standardize_smiles(smiles: str):
    # This function converts a SMILES string into its canonical RDKit form
    if pd.isna(smiles):
        return None
    smiles = str(smiles).strip()
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # This function converts all columns in a DataFrame to numeric, replacing non-parseable values with NaN
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    # This function returns the first column name from the list that actually exists in the DataFrame
    for col in candidates:
        if col in df.columns:
            return col
    return None


def coalesce_columns(df: pd.DataFrame, candidates: list[str], index=None) -> pd.Series:
    # This function returns the first non-empty value across several candidate columns for each row
    idx = df.index if index is None else index
    out = pd.Series([pd.NA] * len(idx), index=idx, dtype="object")
    for col in candidates:
        if col in df.columns:
            values = df[col]
            out = out.where(out.notna() & (out.astype(str).str.strip() != ""), values)
    return out


def round_clean_numeric_columns(df: pd.DataFrame, decimals: int = EXPORT_FLOAT_DECIMALS) -> pd.DataFrame:
    # This function rounds all float columns to a fixed number of decimal places before saving
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(decimals)
    return out


# This section contains functions for detecting and retrieving drug metadata columns
def infer_metadata_columns(df: pd.DataFrame) -> list[str]:
    # This function identifies which columns contain drug metadata rather than molecular features
    keyword_whitelist = {
        "canonical_smiles", "raw_smiles", "smiles", "parent_molecule", "parent",
        "name", "synonyms", "molecule_chembl_id", "chembl_id", "pref_name",
        "phase", "drug_applicants", "first_approval", "atc_codes",
        "availability_type", "withdrawn_flag", "withdrawn", "orphan",
        "molecule_type", "max_phase", "drug_type", "usan_stem", "usan_substem",
        "usan_year", "first_in_class", "chirality", "prodrug", "natural_product",
        "black_box_warning", "black_box", "indication_class", "approval_status", "source",
        "oral", "parenteral", "topical", "passes_rule_of_five"
    }

    keep = []
    for col in df.columns:
        if str(col).startswith(MACCS_PREFIX) or str(col).startswith(MORGAN_PREFIX):
            continue

        ncol = normalize_name(col)
        if ncol in keyword_whitelist or col == CANONICAL_SMILES_COL:
            keep.append(col)
            continue

        series = df[col]
        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        ):
            keep.append(col)

    seen = set()
    ordered = []
    for col in keep:
        if col not in seen:
            ordered.append(col)
            seen.add(col)
    return ordered


def merge_readable_metadata(pred_df: pd.DataFrame) -> pd.DataFrame:
    # This function joins human-readable drug information (names, approval status, etc.)
    # onto the prediction table using SMILES as the merge key
    if not RAW_PREDICTION_METADATA_FILE.exists():
        log("[WARN] Raw prediction metadata file not found. Continuing without metadata merge.")
        return pred_df

    raw = read_table(RAW_PREDICTION_METADATA_FILE)
    smiles_col = None
    for cand in ["Smiles", "SMILES", "smiles", "Canonical_SMILES", "canonical_smiles"]:
        if cand in raw.columns:
            smiles_col = cand
            break
    if smiles_col is None:
        log("[WARN] Could not find a SMILES column in the raw prediction metadata file.")
        return pred_df

    raw = raw.copy()
    raw[CANONICAL_SMILES_COL] = raw[smiles_col].apply(standardize_smiles)

    preferred = [
        CANONICAL_SMILES_COL, "Parent Molecule", "Name", "Synonyms", "Molecule ChEMBL ID",
        "Pref Name", "Phase", "Drug Type", "Availability Type", "Withdrawn Flag", "Orphan",
        "Passes Rule of Five", "Oral", "Parenteral", "Topical", "Black Box",
        "First Approval", "ATC Codes"
    ]
    keep = [c for c in preferred if c in raw.columns]
    if CANONICAL_SMILES_COL not in keep:
        keep = [CANONICAL_SMILES_COL] + keep

    raw_meta = raw[keep].dropna(subset=[CANONICAL_SMILES_COL]).drop_duplicates(CANONICAL_SMILES_COL)
    merge_cols = [c for c in raw_meta.columns if c == CANONICAL_SMILES_COL or c not in pred_df.columns]
    return pred_df.merge(raw_meta[merge_cols], on=CANONICAL_SMILES_COL, how="left")


# This section contains functions for locating and loading saved model files
def load_model(path: Path):
    # This function loads a joblib model bundle from disk
    if not path.exists():
        raise FileNotFoundError(
            f"Saved model not found: {path}\n"
            "You need the fitted pipeline object, not just best parameters."
        )
    return joblib.load(path)


def unwrap_model_object(obj):
    # This function unpacks a model bundle dictionary into its individual components
    if isinstance(obj, dict) and "best_estimator" in obj:
        model = obj["best_estimator"]
        feature_cols = obj.get("feature_columns")
        threshold = obj.get("threshold")
        return model, feature_cols, threshold, obj
    return obj, None, None, None


def model_role_from_bundle(obj) -> str | None:
    # This function determines whether a loaded model is a classifier or a regressor
    model, _, threshold, bundle = unwrap_model_object(obj)

    if isinstance(bundle, dict):
        if threshold is not None or "threshold_score_type" in bundle:
            return "primary"
        if bundle.get("target_column") == "pChEMBL Value":
            return "regression"
        if "output_interpretation" in bundle:
            return "regression"
        model_name = normalize_name(bundle.get("model_name", ""))
        if any(x in model_name for x in ["svr", "pls", "elasticnet", "xgbregressor", "regressor"]):
            return "regression"
        if any(x in model_name for x in ["logistic", "svc", "svm", "randomforest", "xgboost", "classifier"]):
            return "primary"

    cls_name = normalize_name(getattr(model, "__class__", type(model)).__name__)
    if any(x in cls_name for x in ["svr", "pls", "elasticnet", "regressor"]):
        return "regression"
    if any(x in cls_name for x in ["svc", "svm", "logistic", "classifier"]):
        return "primary"
    return None


def read_classification_best_model_path() -> Path | None:
    # This function reads the saved winner table to find the path to the best classification model
    if not CLASSIFICATION_TABLE_PATH.exists():
        return None
    try:
        df = pd.read_csv(CLASSIFICATION_TABLE_PATH)
        if df.empty:
            return None
        row = df.iloc[0]
        rep_name = row.get("representation")
        model_name = row.get("model")
        if pd.isna(rep_name) or pd.isna(model_name):
            return None
        return ROOT / "Activity Classifier Output" / "models" / f"{sanitize_name(rep_name)}__{sanitize_name(model_name)}.joblib"
    except Exception:
        return None


def read_regression_best_model_path() -> Path | None:
    # This function reads a plain text file that stores the path to the best regression model
    if not REGRESSION_BEST_MODEL_PATH_FILE.exists():
        return None
    try:
        text = REGRESSION_BEST_MODEL_PATH_FILE.read_text(encoding="utf-8").strip()
        return Path(text) if text else None
    except Exception:
        return None


def iter_candidate_model_paths() -> list[Path]:
    # This function builds a list of all model files to consider, starting with explicitly named ones
    candidates: list[Path] = []

    for p in [PRIMARY_MODEL, REGRESSION_MODEL]:
        if p.exists() and p not in candidates:
            candidates.append(p)

    for p in [read_classification_best_model_path(), read_regression_best_model_path()]:
        if p is not None and p.exists() and p not in candidates:
            candidates.append(p)

    for d in MODEL_SEARCH_DIRS:
        if d.exists() and d.is_dir():
            for p in sorted(d.glob("*.joblib")):
                if p not in candidates:
                    candidates.append(p)

    return candidates


def discover_model_paths() -> tuple[Path, Path]:
    # This function resolves the final classifier and regression model paths to use for inference.
    # It checks both explicitly deployed models and the official winners from the training output.
    official_primary_path = read_classification_best_model_path()
    official_regression_path = read_regression_best_model_path()

    legacy_primary_path = PRIMARY_MODEL if PRIMARY_MODEL.exists() else None
    legacy_regression_path = REGRESSION_MODEL if REGRESSION_MODEL.exists() else None

    if legacy_primary_path is None and PRIMARY_MODEL.name.startswith("Activity_"):
        alt = PRIMARY_MODEL.with_name(PRIMARY_MODEL.name.replace("Activity_", "", 1))
        if alt.exists():
            legacy_primary_path = alt

    if legacy_regression_path is None:
        for old_prefix in ["Potency_", "Regression_"]:
            if REGRESSION_MODEL.name.startswith(old_prefix):
                alt = REGRESSION_MODEL.with_name(REGRESSION_MODEL.name.replace(old_prefix, "", 1))
                if alt.exists():
                    legacy_regression_path = alt
                    break

    primary_path = None
    regression_path = None

    if PREFER_OFFICIAL_WINNERS:
        if official_primary_path is not None and official_primary_path.exists():
            primary_path = official_primary_path
        else:
            primary_path = legacy_primary_path

        if official_regression_path is not None and official_regression_path.exists():
            regression_path = official_regression_path
        else:
            regression_path = legacy_regression_path
    else:
        if legacy_primary_path is not None:
            primary_path = legacy_primary_path
        elif official_primary_path is not None and official_primary_path.exists():
            primary_path = official_primary_path

        if legacy_regression_path is not None:
            regression_path = legacy_regression_path
        elif official_regression_path is not None and official_regression_path.exists():
            regression_path = official_regression_path

    if primary_path is not None and official_primary_path is not None and official_primary_path.exists():
        if primary_path.resolve() != official_primary_path.resolve():
            if not PREFER_OFFICIAL_WINNERS and primary_path.parent.resolve() == MODEL_DIR.resolve():
                log(f"[INFO] Using deployed primary model override from Prediction Models: {primary_path}")
                log(f"[INFO] Official primary winner remains available at: {official_primary_path}")
            else:
                log(f"[WARN] Using non-official primary model: {primary_path}")
                log(f"[WARN] Official primary winner is: {official_primary_path}")

    if regression_path is not None and official_regression_path is not None and official_regression_path.exists():
        if regression_path.resolve() != official_regression_path.resolve():
            if not PREFER_OFFICIAL_WINNERS and regression_path.parent.resolve() == MODEL_DIR.resolve():
                log(f"[INFO] Using deployed regression model override from Prediction Models: {regression_path}")
                log(f"[INFO] Official regression winner remains available at: {official_regression_path}")
            else:
                log(f"[WARN] Using non-official regression model: {regression_path}")
                log(f"[WARN] Official regression winner is: {official_regression_path}")

    if primary_path is not None and regression_path is not None:
        return primary_path, regression_path

    # If either model is still missing, scan all candidate files and classify each one by role
    discovered_primary = []
    discovered_regression = []
    for path in iter_candidate_model_paths():
        try:
            obj = load_model(path)
            role = model_role_from_bundle(obj)
        except Exception as exc:
            log(f"[WARN] Skipping unreadable model bundle: {path} ({exc})")
            continue

        if role == "primary":
            discovered_primary.append(path)
        elif role == "regression":
            discovered_regression.append(path)

    if primary_path is None:
        if len(discovered_primary) == 1:
            primary_path = discovered_primary[0]
        elif len(discovered_primary) > 1:
            raise FileNotFoundError(
                "Could not choose a unique primary classifier automatically. "
                f"Candidates found: {[str(p) for p in discovered_primary]}"
            )

    if regression_path is None:
        if len(discovered_regression) == 1:
            regression_path = discovered_regression[0]
        elif len(discovered_regression) > 1:
            raise FileNotFoundError(
                "Could not choose a unique regression model automatically. "
                f"Candidates found: {[str(p) for p in discovered_regression]}"
            )

    if primary_path is None:
        raise FileNotFoundError(
            "Could not locate the primary classifier bundle. "
            "Place the activity model joblib in 'Prediction Models' or keep the original 'Activity Classifier Output/models' folder."
        )
    if regression_path is None:
        raise FileNotFoundError(
            "Could not locate the regression model bundle. "
            "Place the regression model joblib in 'Prediction Models' or keep the original 'Regression Output/models' folder."
        )

    return primary_path, regression_path

# This section contains functions that rebuild the preprocessing pipeline
# used during training, so that prediction data is transformed in exactly the same way
def get_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    exclude_numeric_cols: list[str] | None = None,
):
    # This function separates numeric columns into descriptor columns and fingerprint columns
    exclude_numeric_cols = set(exclude_numeric_cols or [])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target_col and c not in exclude_numeric_cols]
    fp_cols = [c for c in feature_cols if c.startswith("MACCS_") or c.startswith("Morgan_")]
    desc_cols = [c for c in feature_cols if c not in fp_cols]
    return desc_cols, fp_cols


def get_correlated_drop_columns(df: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    # This function identifies descriptor columns that are too highly correlated with another column
    if df.shape[1] <= 1:
        return []
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    return [column for column in upper.columns if (upper[column] > threshold).any()]


def fit_reconstructed_preprocessor(train_raw_df: pd.DataFrame, target_col: str) -> dict:
    # This function replicates the exact preprocessing steps from Datasplit.py on the training set
    # and returns all fitted objects needed to later transform prediction data the same way
    desc_cols, fp_cols = get_feature_columns(train_raw_df, target_col=target_col)

    # This part processes descriptor columns: drop high-missing, impute with median,
    # remove zero-variance, remove highly correlated, then standardize
    train_desc = train_raw_df[desc_cols].replace([np.inf, -np.inf], np.nan).copy()
    keep_desc_cols = train_desc.columns[train_desc.isna().mean() <= DESCRIPTOR_MISSING_THRESHOLD].tolist()
    train_desc = train_desc[keep_desc_cols].copy()

    descriptor_medians = train_desc.median(numeric_only=True)
    train_desc = train_desc.fillna(descriptor_medians)

    desc_variance_selector = VarianceThreshold(threshold=0.0)
    train_desc_vt = desc_variance_selector.fit_transform(train_desc)
    desc_after_variance_cols = train_desc.columns[desc_variance_selector.get_support()].tolist()

    train_desc_vt_df = pd.DataFrame(train_desc_vt, columns=desc_after_variance_cols, index=train_raw_df.index)
    desc_corr_drop_cols = get_correlated_drop_columns(
        train_desc_vt_df,
        threshold=DESCRIPTOR_CORRELATION_THRESHOLD,
    )
    train_desc_final = train_desc_vt_df.drop(columns=desc_corr_drop_cols, errors="ignore")

    desc_scaler = StandardScaler()
    train_desc_scaled = pd.DataFrame(
        desc_scaler.fit_transform(train_desc_final),
        columns=train_desc_final.columns,
        index=train_raw_df.index,
    )

    # This part processes fingerprint columns: fill missing with 0 and remove zero-variance bits
    train_fp = train_raw_df[fp_cols].copy().fillna(0).astype(np.int8)
    fp_variance_selector = VarianceThreshold(threshold=0.0)
    train_fp_vt = fp_variance_selector.fit_transform(train_fp)
    fp_after_variance_cols = train_fp.columns[fp_variance_selector.get_support()].tolist()
    train_fp_final = pd.DataFrame(train_fp_vt, columns=fp_after_variance_cols, index=train_raw_df.index).astype(np.int8)

    train_processed = pd.concat([train_desc_scaled, train_fp_final], axis=1)

    return {
        "target_col": target_col,
        "descriptor_columns_all": desc_cols,
        "fingerprint_columns_all": fp_cols,
        "descriptor_columns_kept_after_missing": keep_desc_cols,
        "descriptor_medians": descriptor_medians.to_dict(),
        "descriptor_columns_after_variance": desc_after_variance_cols,
        "descriptor_correlation_drop_columns": desc_corr_drop_cols,
        "final_descriptor_columns": train_desc_scaled.columns.tolist(),
        "final_fingerprint_columns": train_fp_final.columns.tolist(),
        "final_feature_columns": train_processed.columns.tolist(),
        "descriptor_variance_selector": desc_variance_selector,
        "fingerprint_variance_selector": fp_variance_selector,
        "descriptor_scaler": desc_scaler,
    }


def transform_with_reconstructed_preprocessor(raw_df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    # This function applies the preprocessing artifacts (fitted on training data) to any new data
    desc_all = list(artifacts["descriptor_columns_all"])
    desc_keep = list(artifacts["descriptor_columns_kept_after_missing"])
    desc_medians = pd.Series(artifacts["descriptor_medians"], dtype=float)
    desc_after_variance_cols = list(artifacts["descriptor_columns_after_variance"])
    desc_corr_drop_cols = list(artifacts["descriptor_correlation_drop_columns"])
    desc_scaler = artifacts["descriptor_scaler"]
    desc_variance_selector = artifacts["descriptor_variance_selector"]

    # This part aligns and transforms the descriptor columns using training-fitted objects
    desc_input = pd.DataFrame(index=raw_df.index)
    for col in desc_all:
        if col in raw_df.columns:
            desc_input[col] = pd.to_numeric(raw_df[col], errors="coerce")
        else:
            desc_input[col] = np.nan

    desc_input = desc_input[desc_keep].replace([np.inf, -np.inf], np.nan)
    desc_input = desc_input.fillna(desc_medians.reindex(desc_keep))
    desc_vt = desc_variance_selector.transform(desc_input)
    desc_vt_df = pd.DataFrame(desc_vt, columns=desc_after_variance_cols, index=raw_df.index)
    desc_final = desc_vt_df.drop(columns=desc_corr_drop_cols, errors="ignore")
    desc_scaled = pd.DataFrame(
        desc_scaler.transform(desc_final),
        columns=desc_final.columns,
        index=raw_df.index,
    )

    # This part aligns and transforms the fingerprint columns using training-fitted objects
    fp_all = list(artifacts["fingerprint_columns_all"])
    fp_after = list(artifacts["final_fingerprint_columns"])
    fp_variance_selector = artifacts["fingerprint_variance_selector"]

    fp_input = pd.DataFrame(index=raw_df.index)
    for col in fp_all:
        if col in raw_df.columns:
            fp_input[col] = pd.to_numeric(raw_df[col], errors="coerce")
        else:
            fp_input[col] = 0
    fp_input = fp_input.fillna(0).astype(np.int8)
    fp_vt = fp_variance_selector.transform(fp_input)
    fp_final = pd.DataFrame(fp_vt, columns=fp_after, index=raw_df.index).astype(np.int8)

    processed = pd.concat([desc_scaled, fp_final], axis=1)
    processed = processed[artifacts["final_feature_columns"]]
    return processed


def get_processed_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    # This function returns only the feature columns, excluding SMILES, scaffold, split, and target
    meta = {CANONICAL_SMILES_COL, "Scaffold", "Split", target_col}
    return [c for c in df.columns if c not in meta]


def build_raw_train_subset(raw_dataset_path: Path, processed_train_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    # This function loads the raw and processed training tables and aligns them by SMILES
    raw_df = read_table(raw_dataset_path)
    processed_train_df = read_table(processed_train_path)

    if CANONICAL_SMILES_COL not in raw_df.columns or CANONICAL_SMILES_COL not in processed_train_df.columns:
        raise ValueError("Both raw and processed training tables must contain Canonical_SMILES.")

    raw_df = raw_df.drop_duplicates(subset=[CANONICAL_SMILES_COL]).copy()
    processed_train_df = processed_train_df.drop_duplicates(subset=[CANONICAL_SMILES_COL]).copy()

    train_smiles = processed_train_df[CANONICAL_SMILES_COL].astype(str)
    raw_train_df = raw_df[raw_df[CANONICAL_SMILES_COL].astype(str).isin(set(train_smiles))].copy()

    missing_train_smiles = set(train_smiles) - set(raw_train_df[CANONICAL_SMILES_COL].astype(str))
    if missing_train_smiles:
        raise ValueError(
            f"Could not reconstruct the raw training subset exactly. Missing {len(missing_train_smiles)} training molecules from the raw endpoint dataset."
        )

    raw_train_df = raw_train_df.set_index(CANONICAL_SMILES_COL).loc[train_smiles].reset_index()
    return raw_train_df, processed_train_df


def reconstruct_endpoint_matrix(
    endpoint_name: str,
    prediction_raw_df: pd.DataFrame,
    raw_dataset_path: Path,
    processed_train_path: Path,
    target_col: str,
) -> tuple[pd.DataFrame, dict]:
    # This function rebuilds the preprocessing pipeline from the saved training split,
    # applies it to the prediction data, and verifies it matches the saved processed training set
    log(f"[INFO] Reconstructing {endpoint_name} preprocessing from the saved training split...")

    raw_train_df, processed_train_df = build_raw_train_subset(raw_dataset_path, processed_train_path)
    artifacts = fit_reconstructed_preprocessor(raw_train_df, target_col=target_col)

    # This part re-transforms the training set and checks that it matches the saved processed version
    recon_train = transform_with_reconstructed_preprocessor(raw_train_df, artifacts)
    processed_feature_cols = get_processed_feature_columns(processed_train_df, target_col=target_col)
    saved_train_X = ensure_numeric(processed_train_df[processed_feature_cols])
    recon_train = recon_train[processed_feature_cols]

    diff = np.abs(recon_train.to_numpy(dtype=float) - saved_train_X.to_numpy(dtype=float))
    max_abs_diff = float(np.nanmax(diff)) if diff.size else 0.0
    mean_abs_diff = float(np.nanmean(diff)) if diff.size else 0.0
    log(
        f"[INFO] {endpoint_name} reconstruction check | features={len(processed_feature_cols)} | "
        f"max_abs_diff={max_abs_diff:.6g} | mean_abs_diff={mean_abs_diff:.6g}"
    )
    if max_abs_diff > 1e-4:
        raise ValueError(
            f"Preprocessing reconstruction failed for '{endpoint_name}': "
            f"max_abs_diff={max_abs_diff:.6g} exceeds tolerance 1e-4. "
            "The prediction feature file may have been generated with different pipeline settings. "
            "Re-run Prediction Dataset Feature Generation.py to regenerate features."
        )

    prediction_processed = transform_with_reconstructed_preprocessor(prediction_raw_df, artifacts)

    diagnostics = {
        "endpoint": endpoint_name,
        "n_train_rows_reconstructed": int(len(raw_train_df)),
        "n_processed_features": int(len(processed_feature_cols)),
        "reconstruction_max_abs_diff": max_abs_diff,
        "reconstruction_mean_abs_diff": mean_abs_diff,
    }
    return prediction_processed, diagnostics


# This section contains functions for aligning prediction features to the model's expected inputs
# and for running the actual predictions
def infer_expected_columns(model, fallback_cols):
    # This function tries to read the expected feature names directly from the model object
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "named_steps"):
        for _, step in reversed(list(model.named_steps.items())):
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    return list(fallback_cols)


def align_to_expected_columns(X: pd.DataFrame, expected_cols):
    # This function adds any missing columns (filled with 0) and drops any extra columns
    # so the feature matrix exactly matches what the model was trained on
    X = X.copy()

    missing = [c for c in expected_cols if c not in X.columns]
    extra = [c for c in X.columns if c not in expected_cols]

    if missing:
        log(f"[WARN] Missing {len(missing)} expected columns. Filling them with 0.")
        for c in missing:
            X[c] = 0.0

    if extra:
        log(f"[INFO] Dropping {len(extra)} extra columns not used by the model.")
        X = X.drop(columns=extra)

    X = X[expected_cols]
    X = ensure_numeric(X)
    return X, missing, extra


def check_feature_coverage(missing_cols: list[str], expected_cols: list[str], model_label: str):
    # This function raises an error if too many features are missing, which indicates a pipeline mismatch
    expected_n = max(1, len(expected_cols))
    missing_n = len(missing_cols)
    missing_fraction = missing_n / expected_n
    log(
        f"[INFO] {model_label} feature coverage: present={expected_n - missing_n} / {expected_n}, "
        f"missing={missing_n} ({missing_fraction:.1%})"
    )
    if missing_fraction > MAX_MISSING_FRACTION_BEFORE_ERROR:
        raise ValueError(
            f"{model_label} prediction features are badly mismatched with the deployed model. "
            f"Missing {missing_n} of {expected_n} expected columns ({missing_fraction:.1%}). "
            "This usually means the prediction file was built in a different feature space than the trained model."
        )
    return missing_fraction


def fill_remaining_nans(X: pd.DataFrame, label: str) -> tuple[pd.DataFrame, int]:
    # This function fills any remaining NaN values with 0 as a safety measure before passing to the model
    X = X.copy()
    n_nan = int(X.isna().sum().sum())
    if n_nan > 0:
        log(f"[WARN] {label} still contains {n_nan} NaN values after preprocessing reconstruction. Filling with 0 as a safety fallback.")
        X = X.fillna(0)
    return X, n_nan


def get_threshold_scores(model, X: pd.DataFrame) -> tuple[np.ndarray, str]:
    # This function extracts the raw probability or decision score needed to apply a custom threshold
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if getattr(proba, "ndim", 1) == 2 and proba.shape[1] >= 2:
            return np.asarray(proba[:, 1], dtype=float).ravel(), "predict_proba_class1"

    if hasattr(model, "decision_function"):
        score = model.decision_function(X)
        return np.asarray(score, dtype=float).ravel(), "decision_function"

    raise ValueError("Estimator does not expose predict_proba or decision_function for thresholding.")


def predict_binary_with_optional_threshold(model, X: pd.DataFrame, prefix: str, threshold=None):
    # This function runs the classifier and applies either a custom threshold or the model default
    results = pd.DataFrame(index=X.index)

    if threshold is not None:
        threshold_scores, score_type = get_threshold_scores(model, X)
        y_pred = (threshold_scores >= float(threshold)).astype(int)
        results[f"{prefix}_pred"] = y_pred
        results[f"{prefix}_threshold_used"] = float(threshold)
        results[f"{prefix}_threshold_score"] = threshold_scores
        results[f"{prefix}_threshold_score_type"] = score_type
        if score_type == "predict_proba_class1":
            results[f"{prefix}_proba_class1"] = threshold_scores
        return results

    y_pred = model.predict(X)
    results[f"{prefix}_pred"] = y_pred

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if getattr(proba, "ndim", 1) == 2:
            if proba.shape[1] == 2:
                results[f"{prefix}_proba_class1"] = proba[:, 1]
            else:
                for i in range(proba.shape[1]):
                    results[f"{prefix}_proba_class{i}"] = proba[:, i]
    elif hasattr(model, "decision_function"):
        results[f"{prefix}_threshold_score"] = np.asarray(model.decision_function(X), dtype=float).ravel()
        results[f"{prefix}_threshold_score_type"] = "decision_function"
    return results


def get_representation_space(best_estimator, X: pd.DataFrame) -> np.ndarray:
    # This function projects the input into the feature space the model actually trained on
    # by running the pipeline steps up to (but not including) the final estimator
    X_space = X.copy()

    if hasattr(best_estimator, "named_steps"):
        # MultiSpaceSelector pipelines use "multispace"; single-space pipelines use "selector"
        multispace = best_estimator.named_steps.get("multispace")
        selector   = best_estimator.named_steps.get("selector")

        if multispace is not None and multispace != "passthrough":
            X_space = multispace.transform(X_space)
        elif selector is not None and selector != "passthrough":
            X_space = selector.transform(X_space)

        pca = best_estimator.named_steps.get("pca")
        if pca is not None and pca != "passthrough":
            X_space = pca.transform(X_space)

    X_space = np.asarray(X_space, dtype=float)
    if X_space.ndim == 1:
        X_space = X_space.reshape(-1, 1)
    return X_space


def apply_ad_from_bundle(bundle_obj, X: pd.DataFrame, prefix: str) -> pd.DataFrame:
    # This function checks each compound's applicability domain by measuring its k-NN distance
    # to the training set in the model's feature space
    results = pd.DataFrame(index=X.index)
    if not isinstance(bundle_obj, dict):
        return results

    ad_artifacts = bundle_obj.get("ad_artifacts")
    if not isinstance(ad_artifacts, dict):
        return results

    reference = ad_artifacts.get("reference")
    threshold = ad_artifacts.get("threshold")
    k = int(ad_artifacts.get("k", 0) or 0)
    if reference is None or threshold is None:
        return results

    X_space = get_representation_space(bundle_obj["best_estimator"], X)
    ref = np.asarray(reference, dtype=float)
    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)
    if X_space.ndim == 1:
        X_space = X_space.reshape(-1, 1)

    if ref.shape[1] != X_space.shape[1]:
        raise ValueError(
            f"Applicability-domain space mismatch for {prefix}: training dim={ref.shape[1]}, prediction dim={X_space.shape[1]}"
        )

    if k <= 0 or ref.shape[0] == 0:
        mean_knn_distance = np.zeros(X_space.shape[0], dtype=float)
    else:
        k_eff = min(k, ref.shape[0])
        dist = pairwise_distances(X_space, ref, metric="euclidean")
        mean_knn_distance = np.sort(dist, axis=1)[:, :k_eff].mean(axis=1)

    results[f"{prefix}_ad_mean_knn_distance"] = mean_knn_distance
    results[f"{prefix}_ad_threshold"] = float(threshold)
    results[f"{prefix}_in_ad"] = mean_knn_distance <= float(threshold)
    results[f"{prefix}_ad_method"] = str(ad_artifacts.get("method", "knn_mean_distance"))
    return results


# This section contains functions for building the final clean output tables
def add_primary_reliability(primary_df: pd.DataFrame) -> pd.DataFrame:
    # This function assigns a reliability label (high / medium / low / out_of_domain)
    # based on how far each prediction score is from the classification threshold
    out = primary_df.copy()

    margin = None
    if "primary_threshold_score" in out.columns and "primary_threshold_used" in out.columns:
        margin = np.abs(out["primary_threshold_score"].astype(float) - out["primary_threshold_used"].astype(float))
    elif "primary_proba_class1" in out.columns:
        margin = np.abs(out["primary_proba_class1"].astype(float) - 0.5)

    if margin is None:
        return out

    out["primary_threshold_margin"] = margin

    if len(margin) > 1:
        q50 = float(np.quantile(margin, 0.50))
        q75 = float(np.quantile(margin, 0.75))
        labels = np.where(margin >= q75, "high", np.where(margin >= q50, "medium", "low"))
    else:
        labels = np.array(["medium"] * len(out), dtype=object)

    out["primary_reliability"] = labels

    if "primary_in_ad" in out.columns:
        out.loc[~out["primary_in_ad"].fillna(False), "primary_reliability"] = "out_of_domain"
    return out


def sort_all_results(df: pd.DataFrame) -> pd.DataFrame:
    # This function sorts all results by predicted active first, then by ranking score, then by probability
    sort_cols = []
    ascending = []

    if "predicted_active" in df.columns:
        sort_cols.append("predicted_active")
        ascending.append(False)

    if "ranking_score" in df.columns:
        sort_cols.append("ranking_score")
        ascending.append(False)

    if "activity_probability" in df.columns:
        sort_cols.append("activity_probability")
        ascending.append(False)

    if not sort_cols:
        return df
    return df.sort_values(by=sort_cols, ascending=ascending, na_position="last").reset_index(drop=True)


def sort_active_ranked_results(df: pd.DataFrame) -> pd.DataFrame:
    # This function sorts the active-only subset by ranking score first, then by probability
    sort_cols = []
    ascending = []

    if "ranking_score" in df.columns:
        sort_cols.append("ranking_score")
        ascending.append(False)

    if "activity_probability" in df.columns:
        sort_cols.append("activity_probability")
        ascending.append(False)

    if not sort_cols:
        return df.reset_index(drop=True)
    return df.sort_values(by=sort_cols, ascending=ascending, na_position="last").reset_index(drop=True)


def clean_column_types(df: pd.DataFrame) -> pd.DataFrame:
    # This function converts rank and predicted_active to integers and boolean-like columns to booleans
    out = df.copy()

    int_like_cols = ["rank", "predicted_active"]
    bool_like_cols = ["activity_in_ad", "regression_in_ad"]

    for col in int_like_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    for col in bool_like_cols:
        if col in out.columns:
            vals = out[col]
            if pd.api.types.is_bool_dtype(vals):
                out[col] = vals
            else:
                out[col] = vals.map(lambda x: pd.NA if pd.isna(x) else bool(x)).astype("boolean")

    return out


def build_clean_prediction_table(results_df: pd.DataFrame) -> pd.DataFrame:
    # This function assembles the final export table with renamed, reordered, and type-cleaned columns
    clean = pd.DataFrame(index=results_df.index)

    clean["chembl_id"] = coalesce_columns(
        results_df,
        ["chembl_id", "Molecule ChEMBL ID", "Parent Molecule", "molecule_chembl_id", "parent_molecule"],
        index=results_df.index,
    )
    clean["name"] = coalesce_columns(
        results_df,
        ["name", "Name", "Pref Name", "pref_name", "Synonyms"],
        index=results_df.index,
    )
    if CANONICAL_SMILES_COL in results_df.columns:
        clean["smiles"] = results_df[CANONICAL_SMILES_COL]
    else:
        clean["smiles"] = coalesce_columns(results_df, ["smiles", "SMILES", "Canonical_SMILES"], index=results_df.index)

    clean["predicted_active"] = pd.to_numeric(results_df.get("primary_pred"), errors="coerce")

    if "primary_proba_class1" in results_df.columns:
        clean["activity_probability"] = pd.to_numeric(results_df["primary_proba_class1"], errors="coerce")
    elif (
        "primary_threshold_score" in results_df.columns
        and "primary_threshold_score_type" in results_df.columns
        and results_df["primary_threshold_score_type"].astype(str).eq("predict_proba_class1").any()
    ):
        clean["activity_probability"] = pd.to_numeric(results_df["primary_threshold_score"], errors="coerce")
    else:
        clean["activity_probability"] = np.nan

    clean["activity_threshold_margin"] = pd.to_numeric(results_df.get("primary_threshold_margin"), errors="coerce")
    clean["activity_in_ad"] = results_df.get("primary_in_ad")
    clean["activity_reliability"] = results_df.get("primary_reliability")

    clean["ranking_score"] = pd.to_numeric(results_df.get("regression_score"), errors="coerce")
    clean["regression_in_ad"] = results_df.get("regression_in_ad")

    ordered_cols = [
        "chembl_id",
        "name",
        "smiles",
        "predicted_active",
        "activity_probability",
        "activity_threshold_margin",
        "activity_in_ad",
        "activity_reliability",
        "ranking_score",
        "regression_in_ad",
    ]
    clean = clean[ordered_cols]
    clean = clean_column_types(clean)
    clean = round_clean_numeric_columns(clean, decimals=EXPORT_FLOAT_DECIMALS)
    return clean


def discover_prediction_feature_file() -> Path:
    # This function tries each candidate path in order and returns the first one that resolves
    for candidate in FEATURE_FILE_CANDIDATES:
        try:
            resolved = resolve_table(candidate)
            log(f"[INFO] Using prediction feature file: {resolved}")
            return resolved
        except Exception:
            continue
    raise FileNotFoundError(
        "Could not find the model-ready prediction feature file. Run the prediction feature-generation script first."
    )


# This is the main function that orchestrates the full prediction pipeline
def main():
    prediction_feature_path = discover_prediction_feature_file()

    log("[INFO] Loading raw-aligned prediction features...")
    df = read_table(prediction_feature_path)
    if CANONICAL_SMILES_COL not in df.columns:
        raise ValueError(f"The prediction file must contain '{CANONICAL_SMILES_COL}'.")

    # This part extracts metadata columns and joins drug information from the raw prediction dataset
    metadata_cols = infer_metadata_columns(df)
    if CANONICAL_SMILES_COL not in metadata_cols:
        metadata_cols = [CANONICAL_SMILES_COL] + metadata_cols
    metadata = merge_readable_metadata(df[metadata_cols].copy())

    log("[INFO] Resolving saved fitted models...")
    primary_model_path, regression_model_path = discover_model_paths()
    log(f"[INFO] Primary model bundle: {primary_model_path}")
    log(f"[INFO] Regression model bundle: {regression_model_path}")

    primary_loaded = load_model(primary_model_path)
    regression_loaded = load_model(regression_model_path)

    primary_model, primary_feature_cols, primary_threshold, primary_bundle_obj = unwrap_model_object(primary_loaded)
    regression_model, regression_feature_cols, _, regression_bundle_obj = unwrap_model_object(regression_loaded)

    # This part rebuilds the preprocessing steps for both endpoints using the saved training split,
    # then transforms the prediction data through the same pipeline
    X_primary_full, primary_recon_diag = reconstruct_endpoint_matrix(
        endpoint_name="primary_classification",
        prediction_raw_df=df,
        raw_dataset_path=CLASSIFICATION_RAW_PATH,
        processed_train_path=CLASSIFICATION_TRAIN_PROCESSED_PATH,
        target_col="Activity_Class",
    )
    X_regression_full, regression_recon_diag = reconstruct_endpoint_matrix(
        endpoint_name="regression",
        prediction_raw_df=df,
        raw_dataset_path=REGRESSION_RAW_PATH,
        processed_train_path=REGRESSION_TRAIN_PROCESSED_PATH,
        target_col="pChEMBL Value",
    )

    primary_expected = primary_feature_cols if primary_feature_cols is not None else infer_expected_columns(primary_model, X_primary_full.columns)
    regression_expected = regression_feature_cols if regression_feature_cols is not None else infer_expected_columns(regression_model, X_regression_full.columns)

    X_primary, primary_missing, primary_extra = align_to_expected_columns(X_primary_full, primary_expected)
    X_regression, regression_missing, regression_extra = align_to_expected_columns(X_regression_full, regression_expected)

    primary_missing_fraction = check_feature_coverage(primary_missing, primary_expected, "Primary model")
    regression_missing_fraction = check_feature_coverage(regression_missing, regression_expected, "Regression model")

    X_primary, primary_remaining_nan = fill_remaining_nans(X_primary, "Primary matrix")
    X_regression, regression_remaining_nan = fill_remaining_nans(X_regression, "Regression matrix")

    # This part applies the threshold override if one is set, otherwise uses the bundle's OOF threshold
    if PRIMARY_THRESHOLD_OVERRIDE is not None:
        log(f"[INFO] PRIMARY_THRESHOLD_OVERRIDE applied: {PRIMARY_THRESHOLD_OVERRIDE} "
            f"(bundle OOF threshold was {primary_threshold:.4f})")
        primary_threshold = float(PRIMARY_THRESHOLD_OVERRIDE)

    log("[INFO] Running primary classification...")
    if primary_threshold is not None:
        log(f"[INFO] Using threshold: {primary_threshold}")
    else:
        log("[WARN] No saved primary threshold found. Falling back to model.predict().")

    primary_out = predict_binary_with_optional_threshold(
        primary_model,
        X_primary,
        prefix="primary",
        threshold=primary_threshold,
    )

    if "primary_pred" not in primary_out.columns:
        raise RuntimeError("Primary classifier did not return predictions.")

    active_mask = primary_out["primary_pred"] == 1
    n_active = int(active_mask.sum())
    log(f"[INFO] Primary predicted actives: {n_active} / {len(active_mask)}")

    # Build exploratory mask: compounds below the active threshold but within [EXPLORATORY_RANGE_LOWER, primary_threshold)
    _prob_col = None
    if "primary_threshold_score" in primary_out.columns:
        _prob_col = "primary_threshold_score"
    elif "primary_proba_class1" in primary_out.columns:
        _prob_col = "primary_proba_class1"

    if _prob_col is not None and primary_threshold is not None:
        _prob_scores = primary_out[_prob_col].astype(float)
        exploratory_mask = (~active_mask) & (_prob_scores >= EXPLORATORY_RANGE_LOWER) & (_prob_scores < float(primary_threshold))
    else:
        exploratory_mask = pd.Series(False, index=active_mask.index)

    n_exploratory = int(exploratory_mask.sum())
    log(f"[INFO] Exploratory range compounds ({EXPLORATORY_RANGE_LOWER}–{primary_threshold}): {n_exploratory} / {len(exploratory_mask)}")

    regression_full = pd.DataFrame(index=df.index)
    regression_full["regression_score"] = np.nan

    primary_ad = apply_ad_from_bundle(primary_bundle_obj, X_primary, prefix="primary")
    primary_out = pd.concat([primary_out, primary_ad], axis=1)
    primary_out = add_primary_reliability(primary_out)

    regression_ad_full = pd.DataFrame(index=df.index)
    regression_output_note = None
    if isinstance(regression_bundle_obj, dict):
        regression_output_note = regression_bundle_obj.get(
            "output_interpretation",
            "Use model output as a ranking score among predicted actives.",
        )

    # This part runs the regression model on predicted actives and exploratory-range compounds
    combined_reg_mask = active_mask | exploratory_mask
    n_combined = int(combined_reg_mask.sum())
    if n_combined > 0:
        log(f"[INFO] Running regression on {n_active} predicted active(s) and {n_exploratory} exploratory-range compound(s)...")
        reg_pred = np.asarray(regression_model.predict(X_regression.loc[combined_reg_mask]), dtype=float).reshape(-1)
        regression_full.loc[combined_reg_mask, "regression_score"] = reg_pred
        regression_ad_combined = apply_ad_from_bundle(regression_bundle_obj, X_regression.loc[combined_reg_mask], prefix="regression")
        regression_ad_full = regression_ad_full.join(regression_ad_combined, how="left")
    else:
        log("[WARN] No compounds were predicted active or in the exploratory range. Ranked output will be empty.")

    results_full = pd.concat([metadata, primary_out, regression_full, regression_ad_full], axis=1)
    results_clean = build_clean_prediction_table(results_full)
    results_clean = sort_all_results(results_clean)

    # This part assigns each compound a confidence tier:
    # high_confidence (>= threshold), exploratory (EXPLORATORY_RANGE_LOWER to threshold), below_threshold (< EXPLORATORY_RANGE_LOWER)
    if "activity_probability" in results_clean.columns:
        prob = results_clean["activity_probability"].astype(float)
        _thr = float(primary_threshold) if primary_threshold is not None else float("inf")
        tier = np.where(
            results_clean["predicted_active"].fillna(0) == 1,
            "high_confidence",
            np.where(
                (prob >= EXPLORATORY_RANGE_LOWER) & (prob < _thr),
                "exploratory",
                "below_threshold",
            ),
        )
        insert_after = "activity_reliability"
        if insert_after in results_clean.columns:
            results_clean.insert(results_clean.columns.get_loc(insert_after) + 1, "confidence_tier", tier)
        else:
            results_clean["confidence_tier"] = tier

    # This part extracts only the predicted-active compounds and ranks them by potency score
    active_ranked = results_clean.loc[results_clean["predicted_active"].fillna(0) == 1].copy()
    active_ranked = sort_active_ranked_results(active_ranked)
    if not active_ranked.empty:
        active_ranked.insert(0, "rank", pd.Series(np.arange(1, len(active_ranked) + 1), dtype="Int64"))

    # This part extracts predicted-active AND exploratory-range compounds and ranks them together
    if "confidence_tier" in results_clean.columns:
        _exploratory_tiers = results_clean["confidence_tier"].isin(["high_confidence", "exploratory"])
    else:
        _exploratory_tiers = results_clean["predicted_active"].fillna(0) == 1
    active_and_exploratory_ranked = results_clean.loc[_exploratory_tiers].copy()
    active_and_exploratory_ranked = sort_active_ranked_results(active_and_exploratory_ranked)
    if not active_and_exploratory_ranked.empty:
        active_and_exploratory_ranked.insert(0, "rank", pd.Series(np.arange(1, len(active_and_exploratory_ranked) + 1), dtype="Int64"))

    out_all_csv = OUTDIR / "mac1_predictions_all_clean.csv"
    out_active_csv = OUTDIR / "mac1_predictions_active_ranked_clean.csv"
    out_exploratory_csv = OUTDIR / "mac1_predictions_active_and_exploratory_ranked_clean.csv"

    results_clean.to_csv(out_all_csv, index=False)
    active_ranked.to_csv(out_active_csv, index=False)
    active_and_exploratory_ranked.to_csv(out_exploratory_csv, index=False)

    # This part builds a top-N shortlist ranked by activity probability regardless of the binary threshold
    out_top_n_csv = OUTDIR / f"mac1_repurposing_top{TOP_N_REPURPOSING}.csv"
    top_n_shortlist = (
        results_clean
        .sort_values("activity_probability", ascending=False, na_position="last")
        .head(TOP_N_REPURPOSING)
        .reset_index(drop=True)
        .copy()
    )
    top_n_shortlist.insert(0, "repurposing_rank", pd.Series(np.arange(1, len(top_n_shortlist) + 1), dtype="Int64"))
    top_n_shortlist.to_csv(out_top_n_csv, index=False)
    log(f"[INFO] Top-{TOP_N_REPURPOSING} repurposing shortlist saved to: {out_top_n_csv}")

    # This part assembles and saves a JSON summary of all prediction run settings and statistics
    summary = {
        "workflow": "primary_activity_classifier_then_regression_on_actives_and_exploratory",
        "n_total_candidates": int(len(df)),
        "n_primary_predicted_active": n_active,
        "n_exploratory_range": n_exploratory,
        "exploratory_range": [EXPLORATORY_RANGE_LOWER, float(primary_threshold)] if primary_threshold is not None else None,
        "n_ranked_active_saved": int(len(active_ranked)),
        "n_active_and_exploratory_saved": int(len(active_and_exploratory_ranked)),
        "primary_missing_fraction": float(primary_missing_fraction),
        "regression_missing_fraction": float(regression_missing_fraction),
        "primary_remaining_nan_filled_with_zero": int(primary_remaining_nan),
        "regression_remaining_nan_filled_with_zero": int(regression_remaining_nan),
        "primary_model_path": str(primary_model_path),
        "regression_model_path": str(regression_model_path),
        "prediction_feature_file": str(prediction_feature_path),
        "raw_prediction_metadata_file": str(RAW_PREDICTION_METADATA_FILE) if RAW_PREDICTION_METADATA_FILE.exists() else None,
        "primary_threshold_used": None if primary_threshold is None else float(primary_threshold),
        "primary_threshold_score_type": (
            None if "primary_threshold_score_type" not in primary_out.columns
            else str(primary_out["primary_threshold_score_type"].dropna().iloc[0])
            if not primary_out["primary_threshold_score_type"].dropna().empty else None
        ),
        "primary_reconstruction_diagnostics": primary_recon_diag,
        "regression_reconstruction_diagnostics": regression_recon_diag,
        "regression_model_role": "ranking_regressor",
        "regression_output_interpretation": regression_output_note,
        "regression_output_column_in_csv": "ranking_score",
        "n_primary_in_ad": int(results_clean["activity_in_ad"].fillna(False).sum()) if "activity_in_ad" in results_clean.columns else None,
        "n_active_with_regression_in_ad": int(active_ranked["regression_in_ad"].fillna(False).sum()) if "regression_in_ad" in active_ranked.columns else None,
        "n_exploratory_with_regression_in_ad": int(active_and_exploratory_ranked["regression_in_ad"].fillna(False).sum()) if "regression_in_ad" in active_and_exploratory_ranked.columns else None,
        "top_n_repurposing": TOP_N_REPURPOSING,
        "all_results_file": str(out_all_csv),
        "active_ranked_results_file": str(out_active_csv),
        "active_and_exploratory_ranked_results_file": str(out_exploratory_csv),
        "repurposing_shortlist_file": str(out_top_n_csv),
    }

    summary_path = OUTDIR / "prediction_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log("[INFO] Done.")
    log(f"[INFO] Clean all-results file saved to: {out_all_csv}")
    log(f"[INFO] Clean active-ranked file saved to: {out_active_csv}")
    log(f"[INFO] Active and exploratory ranked file saved to: {out_exploratory_csv}")
    log(f"[INFO] Prediction summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
