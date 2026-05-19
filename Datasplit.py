import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Paths: input feature tables and output directory
BASE_DIR = Path(r"C:\Users\admin\Desktop\Machine Learning For Capstone")
FEATURE_OUTPUT_DIR = BASE_DIR / "Feature Generation Outputs"
CLASSIFICATION_PATH = FEATURE_OUTPUT_DIR / "Classification_dataset.csv"
REGRESSION_PATH = FEATURE_OUTPUT_DIR / "Regression_dataset.csv"

OUT_DIR = BASE_DIR / "Data Split Outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGRESSION_TRAIN_OUTPUT = OUT_DIR / "regression_train_dataset.csv"
CLASSIFICATION_TRAIN_OUTPUT = OUT_DIR / "classification_train_dataset.csv"
REGRESSION_TEST_OUTPUT = OUT_DIR / "regression_test_dataset.csv"
CLASSIFICATION_TEST_OUTPUT = OUT_DIR / "classification_test_dataset.csv"
SPLIT_SUMMARY_OUTPUT = OUT_DIR / "data_split_detailed_summary.txt"
REQUESTED_OUTPUT_NAMES = {
    REGRESSION_TRAIN_OUTPUT.name,
    CLASSIFICATION_TRAIN_OUTPUT.name,
    REGRESSION_TEST_OUTPUT.name,
    CLASSIFICATION_TEST_OUTPUT.name,
    SPLIT_SUMMARY_OUTPUT.name,
}

# Split and preprocessing settings 
SEED = 42
TEST_SIZE = 0.20
N_SPLIT_ATTEMPTS = 1500

DESCRIPTOR_MISSING_THRESHOLD = 0.20
DESCRIPTOR_CORRELATION_THRESHOLD = 0.95
TOP_N_SCAFFOLDS_TO_SAVE = 15


# Extracts the Murcko scaffold (core ring system) from a SMILES string
def murcko_from_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return "INVALID_SMILES"
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)
        return scaffold_smiles if scaffold_smiles else "NO_SCAFFOLD"
    except Exception:
        return "NO_SCAFFOLD"


# Separates feature columns into descriptors and fingerprints
def get_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    exclude_numeric_cols: list[str] | None = None,
):
    exclude_numeric_cols = set(exclude_numeric_cols or [])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target_col and c not in exclude_numeric_cols]
    fp_cols = [c for c in feature_cols if c.startswith("MACCS_") or c.startswith("Morgan_")]
    desc_cols = [c for c in feature_cols if c not in fp_cols]
    return desc_cols, fp_cols


# Finds descriptor columns that are highly correlated with another column
def get_correlated_drop_columns(df: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    if df.shape[1] <= 1:
        return []
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    return [column for column in upper.columns if (upper[column] > threshold).any()]


# Splits a dataframe into train/test based on which scaffolds go to test
def finalize_split(df: pd.DataFrame, test_scaffolds: set[str]):
    test_df = df[df["Scaffold"].isin(test_scaffolds)].copy()
    train_df = df[~df["Scaffold"].isin(test_scaffolds)].copy()
    return train_df, test_df


# Returns the fraction of each class in a split as a dict
def safe_class_distribution(df: pd.DataFrame, target_col: str) -> dict[int, float]:
    counts = df[target_col].value_counts(normalize=True).sort_index()
    return {int(k): float(v) for k, v in counts.items()}


# Scores a candidate binary split by penalizing size mismatch, class imbalance, and low scaffold diversity in the test set 
def score_binary_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    target_test_size: int,
    global_positive_ratio: float,
) -> float:
    if len(train_df) == 0 or len(test_df) == 0:
        return np.inf

    train_classes = set(train_df[target_col].unique())
    test_classes = set(test_df[target_col].unique())
    if train_classes != {0, 1} or test_classes != {0, 1}:
        return np.inf

    size_penalty = abs(len(test_df) - target_test_size) / max(target_test_size, 1)
    test_ratio = float(test_df[target_col].mean())
    ratio_penalty = abs(test_ratio - global_positive_ratio)
    scaffold_count_penalty = 1 / max(test_df["Scaffold"].nunique(), 1)

    return (2.0 * size_penalty) + (3.0 * ratio_penalty) + (0.25 * scaffold_count_penalty)


# Searches over many random scaffold groupings to find the best binary split ─
def scaffold_train_test_split_binary(
    df: pd.DataFrame,
    target_col: str = "Activity_Class",
    test_size: float = 0.20,
    seed: int = 42,
    n_attempts: int = 300,
):
    # Group scaffolds and count molecules and actives per scaffold
    grouped = (
        df.groupby("Scaffold")[target_col]
        .agg(n_molecules="size", positives="sum")
        .reset_index()
    )
    grouped["negatives"] = grouped["n_molecules"] - grouped["positives"]

    rows = grouped.to_dict("records")
    total_n = len(df)
    target_test_n = max(1, int(round(total_n * test_size)))
    global_positive_ratio = float(df[target_col].mean())

    best = None

    # Each attempt shuffles the scaffold list and greedily fills the test set
    for attempt in range(n_attempts):
        rng = random.Random(seed + attempt)
        shuffled = rows.copy()
        rng.shuffle(shuffled)

        chosen = []
        chosen_set = set()
        current_n = 0

        # Add a scaffold to the test set if it brings us closer to the target size
        for row in shuffled:
            scaffold = row["Scaffold"]
            scaffold_n = int(row["n_molecules"])

            if scaffold in chosen_set:
                continue

            include_gap = abs((current_n + scaffold_n) - target_test_n)
            skip_gap = abs(current_n - target_test_n)

            if include_gap < skip_gap or (include_gap == skip_gap and rng.random() < 0.5):
                chosen.append(scaffold)
                chosen_set.add(scaffold)
                current_n += scaffold_n

            if current_n >= target_test_n:
                break

        # If still short, add remaining scaffolds sorted by how close they get us
        if current_n < target_test_n:
            remaining_rows = [r for r in shuffled if r["Scaffold"] not in chosen_set]
            remaining_rows = sorted(
                remaining_rows,
                key=lambda r: abs((current_n + int(r["n_molecules"])) - target_test_n)
            )

            for row in remaining_rows:
                scaffold = row["Scaffold"]
                scaffold_n = int(row["n_molecules"])
                chosen.append(scaffold)
                chosen_set.add(scaffold)
                current_n += scaffold_n
                if current_n >= target_test_n:
                    break

        if not chosen and shuffled:
            chosen = [shuffled[0]["Scaffold"]]

        train_df, test_df = finalize_split(df, set(chosen))
        score = score_binary_split(
            train_df=train_df,
            test_df=test_df,
            target_col=target_col,
            target_test_size=target_test_n,
            global_positive_ratio=global_positive_ratio,
        )

        # Keep this split if it beats the current best score
        if best is None or score < best["score"]:
            best = {
                "score": score,
                "train_df": train_df,
                "test_df": test_df,
                "test_scaffolds": set(chosen),
            }

    if best is None:
        raise ValueError("Could not build a valid scaffold split for binary classification.")

    return best["train_df"], best["test_df"]


# Scores a regression split by checking size, mean, std, and scaffold count
def score_regression_split(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    target_test_size: int,
) -> float:
    if len(train_df) == 0 or len(test_df) == 0:
        return np.inf

    size_penalty = abs(len(test_df) - target_test_size) / max(target_test_size, 1)
    mean_penalty = abs(test_df[target_col].mean() - full_df[target_col].mean())
    std_penalty = abs(test_df[target_col].std() - full_df[target_col].std())
    scaffold_count_penalty = 1 / max(test_df["Scaffold"].nunique(), 1)

    return (2.0 * size_penalty) + (1.5 * mean_penalty) + (1.0 * std_penalty) + (0.25 * scaffold_count_penalty)


# Same greedy scaffold search but for continuous regression targets
def scaffold_train_test_split_regression(
    df: pd.DataFrame,
    target_col: str = "pChEMBL Value",
    test_size: float = 0.20,
    seed: int = 42,
    n_attempts: int = 300,
):
    grouped = (
        df.groupby("Scaffold")
        .agg(
            n_molecules=(target_col, "size"),
            target_mean=(target_col, "mean"),
            target_std=(target_col, "std"),
        )
        .reset_index()
    )
    rows = grouped.to_dict("records")
    total_n = len(df)
    target_test_n = max(1, int(round(total_n * test_size)))

    best = None

    for attempt in range(n_attempts):
        rng = random.Random(seed + attempt)
        shuffled = rows.copy()
        rng.shuffle(shuffled)

        chosen = []
        chosen_set = set()
        current_n = 0

        for row in shuffled:
            scaffold = row["Scaffold"]
            scaffold_n = int(row["n_molecules"])

            if scaffold in chosen_set:
                continue

            include_gap = abs((current_n + scaffold_n) - target_test_n)
            skip_gap = abs(current_n - target_test_n)

            if include_gap < skip_gap or (include_gap == skip_gap and rng.random() < 0.5):
                chosen.append(scaffold)
                chosen_set.add(scaffold)
                current_n += scaffold_n

            if current_n >= target_test_n:
                break

        if current_n < target_test_n:
            remaining_rows = [r for r in shuffled if r["Scaffold"] not in chosen_set]
            remaining_rows = sorted(
                remaining_rows,
                key=lambda r: abs((current_n + int(r["n_molecules"])) - target_test_n)
            )

            for row in remaining_rows:
                scaffold = row["Scaffold"]
                scaffold_n = int(row["n_molecules"])
                chosen.append(scaffold)
                chosen_set.add(scaffold)
                current_n += scaffold_n
                if current_n >= target_test_n:
                    break

        if not chosen and shuffled:
            chosen = [shuffled[0]["Scaffold"]]

        train_df, test_df = finalize_split(df, set(chosen))
        score = score_regression_split(
            full_df=df,
            train_df=train_df,
            test_df=test_df,
            target_col=target_col,
            target_test_size=target_test_n,
        )

        if best is None or score < best["score"]:
            best = {
                "score": score,
                "train_df": train_df,
                "test_df": test_df,
                "test_scaffolds": set(chosen),
            }

    if best is None:
        raise ValueError("Could not build a valid scaffold split for regression.")

    return best["train_df"], best["test_df"]


# Fits all preprocessing steps on train only, then applies them to test 
def fit_preprocessor(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    exclude_numeric_cols: list[str] | None = None,
):
    desc_cols, fp_cols = get_feature_columns(
        train_df,
        target_col=target_col,
        exclude_numeric_cols=exclude_numeric_cols,
    )

    # Descriptor pipeline: missing filter → imputation → variance filter → correlation filter → scaling
    train_desc = train_df[desc_cols].replace([np.inf, -np.inf], np.nan).copy()
    test_desc = test_df[desc_cols].replace([np.inf, -np.inf], np.nan).copy()

    # Drop descriptors that are missing more than the allowed fraction
    keep_desc_cols = train_desc.columns[train_desc.isna().mean() <= DESCRIPTOR_MISSING_THRESHOLD].tolist()
    train_desc = train_desc[keep_desc_cols].copy()
    test_desc = test_desc[keep_desc_cols].copy()

    # Fill remaining missing values with the training median
    descriptor_medians = train_desc.median()
    train_desc = train_desc.fillna(descriptor_medians)
    test_desc = test_desc.fillna(descriptor_medians)

    # Remove zero-variance descriptors
    desc_variance_selector = VarianceThreshold(threshold=0.0)
    train_desc_vt = desc_variance_selector.fit_transform(train_desc)
    test_desc_vt = desc_variance_selector.transform(test_desc)
    desc_after_variance_cols = train_desc.columns[desc_variance_selector.get_support()].tolist()

    train_desc_vt_df = pd.DataFrame(train_desc_vt, columns=desc_after_variance_cols, index=train_df.index)
    test_desc_vt_df = pd.DataFrame(test_desc_vt, columns=desc_after_variance_cols, index=test_df.index)

    # Remove descriptors that are highly correlated with another
    desc_corr_drop_cols = get_correlated_drop_columns(
        train_desc_vt_df,
        threshold=DESCRIPTOR_CORRELATION_THRESHOLD,
    )
    train_desc_final = train_desc_vt_df.drop(columns=desc_corr_drop_cols, errors="ignore")
    test_desc_final = test_desc_vt_df.drop(columns=desc_corr_drop_cols, errors="ignore")

    # Standardize descriptor values to zero mean and unit variance
    desc_scaler = StandardScaler()
    train_desc_scaled = pd.DataFrame(
        desc_scaler.fit_transform(train_desc_final),
        columns=train_desc_final.columns,
        index=train_df.index,
    )
    test_desc_scaled = pd.DataFrame(
        desc_scaler.transform(test_desc_final),
        columns=train_desc_final.columns,
        index=test_df.index,
    )

    # Fingerprint pipeline: fill missing with 0, remove zero-variance bits
    train_fp = train_df[fp_cols].copy().fillna(0).astype(np.int8)
    test_fp = test_df[fp_cols].copy().fillna(0).astype(np.int8)

    fp_variance_selector = VarianceThreshold(threshold=0.0)
    train_fp_vt = fp_variance_selector.fit_transform(train_fp)
    test_fp_vt = fp_variance_selector.transform(test_fp)
    fp_after_variance_cols = train_fp.columns[fp_variance_selector.get_support()].tolist()

    # Keep fingerprints as binary integers (NO STANDARDIZATION)
    train_fp_final = pd.DataFrame(train_fp_vt, columns=fp_after_variance_cols, index=train_df.index).astype(np.int8)
    test_fp_final = pd.DataFrame(test_fp_vt, columns=fp_after_variance_cols, index=test_df.index).astype(np.int8)

    # Combine processed descriptors and fingerprints side by side
    train_processed = pd.concat([train_desc_scaled, train_fp_final], axis=1)
    test_processed = pd.concat([test_desc_scaled, test_fp_final], axis=1)

    assert train_processed.columns.tolist() == test_processed.columns.tolist(), (
        f"Feature mismatch after preprocessing: "
        f"{len(train_processed.columns)} train cols vs {len(test_processed.columns)} test cols. "
        "Check that variance/correlation filters produce identical column sets for train and test."
    )

    # Bundle all fitted objects so they can be saved and reused later
    artifacts = {
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

    return train_processed, test_processed, artifacts


# Glues the metadata columns back onto the processed feature matrix
def make_modeling_table(
    original_df: pd.DataFrame,
    processed_X: pd.DataFrame,
    target_col: str,
    extra_cols: list[str] | None = None,
):
    base_cols = ["Canonical_SMILES", target_col, "Scaffold", "Split"]
    extra_cols = extra_cols or []
    ordered_cols = [c for c in base_cols + extra_cols if c in original_df.columns]
    base = original_df[ordered_cols].copy()
    return pd.concat([base.reset_index(drop=True), processed_X.reset_index(drop=True)], axis=1)


# Counts actives, inactives, and scaffolds per train/test split
def summarize_binary_classification(df: pd.DataFrame, target_col: str = "Activity_Class") -> pd.DataFrame:
    rows = []
    for split_name in ["train", "test"]:
        sub = df[df["Split"] == split_name]
        rows.append(
            {
                "split": split_name,
                "n_molecules": len(sub),
                "actives": int((sub[target_col] == 1).sum()),
                "inactives": int((sub[target_col] == 0).sum()),
                "active_ratio": float(sub[target_col].mean()) if len(sub) else np.nan,
                "unique_scaffolds": sub["Scaffold"].nunique(),
            }
        )
    return pd.DataFrame(rows)


# Reports the pChEMBL mean, std, min, max per split for regression
def summarize_regression(df: pd.DataFrame, target_col: str = "pChEMBL Value") -> pd.DataFrame:
    rows = []
    for split_name in ["train", "test"]:
        sub = df[df["Split"] == split_name]
        rows.append(
            {
                "split": split_name,
                "n_molecules": len(sub),
                "target_mean": float(sub[target_col].mean()) if len(sub) else np.nan,
                "target_std": float(sub[target_col].std()) if len(sub) else np.nan,
                "target_min": float(sub[target_col].min()) if len(sub) else np.nan,
                "target_max": float(sub[target_col].max()) if len(sub) else np.nan,
                "unique_scaffolds": sub["Scaffold"].nunique(),
            }
        )
    return pd.DataFrame(rows)


# Returns the most frequent scaffolds and how many molecules they contain
def top_scaffold_table(df: pd.DataFrame, top_n: int = 10):
    return (
        df.groupby("Scaffold")
        .size()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index(name="n_molecules")
    )


def clean_data_split_outputs():
    for path in OUT_DIR.iterdir():
        if path.is_file() and path.name not in REQUESTED_OUTPUT_NAMES:
            path.unlink()


# Checks whether any scaffold appears in both train and test
def scaffold_overlap_audit(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_scaffolds = set(train_df["Scaffold"].astype(str))
    test_scaffolds = set(test_df["Scaffold"].astype(str))
    overlap = train_scaffolds & test_scaffolds
    return pd.DataFrame(
        [
            {
                "train_unique_scaffolds": len(train_scaffolds),
                "test_unique_scaffolds": len(test_scaffolds),
                "shared_scaffolds_between_train_test": len(overlap),
                "scaffold_split_is_clean": len(overlap) == 0,
            }
        ]
    )


def main():
    #  Load the feature tables generated by Feature Engineering.py
    print("Loading classification and regression feature tables...")
    clf_df = pd.read_csv(CLASSIFICATION_PATH)
    reg_df = pd.read_csv(REGRESSION_PATH)

    # Compute a Murcko scaffold for every molecule
    clf_df["Scaffold"] = clf_df["Canonical_SMILES"].apply(murcko_from_smiles)
    reg_df["Scaffold"] = reg_df["Canonical_SMILES"].apply(murcko_from_smiles)

    # Search for the best scaffold-based split for classification
    print("Searching for balanced scaffold split for binary classification...")
    clf_train, clf_test = scaffold_train_test_split_binary(
        clf_df,
        target_col="Activity_Class",
        test_size=TEST_SIZE,
        seed=SEED,
        n_attempts=N_SPLIT_ATTEMPTS,
    )

    # Search for the best scaffold-based split for regression
    print("[INFO] Searching for representative scaffold split for regression...")
    reg_train, reg_test = scaffold_train_test_split_regression(
        reg_df,
        target_col="pChEMBL Value",
        test_size=TEST_SIZE,
        seed=SEED,
        n_attempts=N_SPLIT_ATTEMPTS,
    )

    # Add a Split label column to each subset
    clf_train = clf_train.copy()
    clf_test = clf_test.copy()
    reg_train = reg_train.copy()
    reg_test = reg_test.copy()

    clf_train["Split"] = "train"
    clf_test["Split"] = "test"
    reg_train["Split"] = "train"
    reg_test["Split"] = "test"

    clf_split = pd.concat([clf_train, clf_test], axis=0, ignore_index=True)
    reg_split = pd.concat([reg_train, reg_test], axis=0, ignore_index=True)

    # Apply preprocessing: fit on train, transform both train and test
    print("[INFO] Applying leakage-safe preprocessing to activity and regression datasets...")
    clf_X_train, clf_X_test, clf_artifacts = fit_preprocessor(clf_train, clf_test, target_col="Activity_Class")
    reg_X_train, reg_X_test, reg_artifacts = fit_preprocessor(reg_train, reg_test, target_col="pChEMBL Value")

    # Re-attach metadata (SMILES, scaffold, split) to the processed feature tables
    clf_train_processed = make_modeling_table(clf_train, clf_X_train, target_col="Activity_Class")
    clf_test_processed = make_modeling_table(clf_test, clf_X_test, target_col="Activity_Class")
    reg_train_processed = make_modeling_table(reg_train, reg_X_train, target_col="pChEMBL Value")
    reg_test_processed = make_modeling_table(reg_test, reg_X_test, target_col="pChEMBL Value")

    # Save only the requested four datasets and one detailed summary file.
    clean_data_split_outputs()

    reg_train_processed.to_csv(REGRESSION_TRAIN_OUTPUT, index=False)
    clf_train_processed.to_csv(CLASSIFICATION_TRAIN_OUTPUT, index=False)
    reg_test_processed.to_csv(REGRESSION_TEST_OUTPUT, index=False)
    clf_test_processed.to_csv(CLASSIFICATION_TEST_OUTPUT, index=False)

    clf_summary = summarize_binary_classification(clf_split)
    reg_summary = summarize_regression(reg_split)
    clf_overlap = scaffold_overlap_audit(clf_train, clf_test)
    reg_overlap = scaffold_overlap_audit(reg_train, reg_test)
    clf_top_overall = top_scaffold_table(clf_split, top_n=TOP_N_SCAFFOLDS_TO_SAVE)
    reg_top_overall = top_scaffold_table(reg_split, top_n=TOP_N_SCAFFOLDS_TO_SAVE)
    clf_top_train = top_scaffold_table(clf_train, top_n=TOP_N_SCAFFOLDS_TO_SAVE)
    clf_top_test = top_scaffold_table(clf_test, top_n=TOP_N_SCAFFOLDS_TO_SAVE)
    reg_top_train = top_scaffold_table(reg_train, top_n=TOP_N_SCAFFOLDS_TO_SAVE)
    reg_top_test = top_scaffold_table(reg_test, top_n=TOP_N_SCAFFOLDS_TO_SAVE)

    with open(SPLIT_SUMMARY_OUTPUT, "w", encoding="utf-8") as summary_file:
        summary_file.write("Data Split Detailed Summary\n")
        summary_file.write("=" * 27 + "\n\n")
        summary_file.write("Saved outputs\n")
        summary_file.write(f"- Regression train dataset: {REGRESSION_TRAIN_OUTPUT}\n")
        summary_file.write(f"- Classification train dataset: {CLASSIFICATION_TRAIN_OUTPUT}\n")
        summary_file.write(f"- Regression test dataset: {REGRESSION_TEST_OUTPUT}\n")
        summary_file.write(f"- Classification test dataset: {CLASSIFICATION_TEST_OUTPUT}\n")
        summary_file.write(f"- Detailed summary: {SPLIT_SUMMARY_OUTPUT}\n\n")
        summary_file.write("Settings\n")
        summary_file.write(f"- Seed: {SEED}\n")
        summary_file.write(f"- Test size: {TEST_SIZE}\n")
        summary_file.write(f"- Split attempts: {N_SPLIT_ATTEMPTS}\n")
        summary_file.write(f"- Descriptor missing threshold: {DESCRIPTOR_MISSING_THRESHOLD}\n")
        summary_file.write(f"- Descriptor correlation threshold: {DESCRIPTOR_CORRELATION_THRESHOLD}\n\n")
        summary_file.write("Classification split summary\n")
        summary_file.write(clf_summary.to_string(index=False))
        summary_file.write("\n\nRegression split summary\n")
        summary_file.write(reg_summary.to_string(index=False))
        summary_file.write("\n\nClassification scaffold overlap audit\n")
        summary_file.write(clf_overlap.to_string(index=False))
        summary_file.write("\n\nRegression scaffold overlap audit\n")
        summary_file.write(reg_overlap.to_string(index=False))
        summary_file.write("\n\nClassification preprocessing details\n")
        summary_file.write(f"- Original descriptor columns: {len(clf_artifacts['descriptor_columns_all'])}\n")
        summary_file.write(f"- Original fingerprint columns: {len(clf_artifacts['fingerprint_columns_all'])}\n")
        summary_file.write(f"- Descriptors kept after missing filter: {len(clf_artifacts['descriptor_columns_kept_after_missing'])}\n")
        summary_file.write(f"- Descriptors after variance filter: {len(clf_artifacts['descriptor_columns_after_variance'])}\n")
        summary_file.write(f"- Correlated descriptors dropped: {len(clf_artifacts['descriptor_correlation_drop_columns'])}\n")
        summary_file.write(f"- Final descriptor columns: {len(clf_artifacts['final_descriptor_columns'])}\n")
        summary_file.write(f"- Final fingerprint columns: {len(clf_artifacts['final_fingerprint_columns'])}\n")
        summary_file.write(f"- Final feature columns: {len(clf_artifacts['final_feature_columns'])}\n")
        summary_file.write("\nRegression preprocessing details\n")
        summary_file.write(f"- Original descriptor columns: {len(reg_artifacts['descriptor_columns_all'])}\n")
        summary_file.write(f"- Original fingerprint columns: {len(reg_artifacts['fingerprint_columns_all'])}\n")
        summary_file.write(f"- Descriptors kept after missing filter: {len(reg_artifacts['descriptor_columns_kept_after_missing'])}\n")
        summary_file.write(f"- Descriptors after variance filter: {len(reg_artifacts['descriptor_columns_after_variance'])}\n")
        summary_file.write(f"- Correlated descriptors dropped: {len(reg_artifacts['descriptor_correlation_drop_columns'])}\n")
        summary_file.write(f"- Final descriptor columns: {len(reg_artifacts['final_descriptor_columns'])}\n")
        summary_file.write(f"- Final fingerprint columns: {len(reg_artifacts['final_fingerprint_columns'])}\n")
        summary_file.write(f"- Final feature columns: {len(reg_artifacts['final_feature_columns'])}\n")
        summary_file.write("\nTop classification scaffolds overall\n")
        summary_file.write(clf_top_overall.to_string(index=False))
        summary_file.write("\n\nTop regression scaffolds overall\n")
        summary_file.write(reg_top_overall.to_string(index=False))
        summary_file.write("\n\nTop classification train scaffolds\n")
        summary_file.write(clf_top_train.to_string(index=False))
        summary_file.write("\n\nTop classification test scaffolds\n")
        summary_file.write(clf_top_test.to_string(index=False))
        summary_file.write("\n\nTop regression train scaffolds\n")
        summary_file.write(reg_top_train.to_string(index=False))
        summary_file.write("\n\nTop regression test scaffolds\n")
        summary_file.write(reg_top_test.to_string(index=False))
        summary_file.write("\n")

    clean_data_split_outputs()



    print(f"\nAll outputs saved to: {OUT_DIR}")
    print("Saved requested files only:")
    print(f" - {REGRESSION_TRAIN_OUTPUT.name}")
    print(f" - {CLASSIFICATION_TRAIN_OUTPUT.name}")
    print(f" - {REGRESSION_TEST_OUTPUT.name}")
    print(f" - {CLASSIFICATION_TEST_OUTPUT.name}")
    print(f" - {SPLIT_SUMMARY_OUTPUT.name}")

if __name__ == "__main__":
    main()
