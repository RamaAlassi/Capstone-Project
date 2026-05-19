from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

from mordred import Calculator, descriptors

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# ── Paths: where to find the master training dataset and prediction input ─────
ROOT = Path(r"C:\Users\admin\Desktop\Machine Learning For Capstone")
MASTER_DIR = ROOT / "Feature Generation Outputs"
MASTER_STEM = "Master_feature_dataset"
PREDICTION_FILE = ROOT / "Prediction Dataset.csv"
OUTDIR = ROOT / "Prediction Feature Outputs"

# ── List of column names that might contain SMILES in different datasets ──────
SMILES_CANDIDATES = [
    "Smiles", "SMILES", "smiles", "canonical_smiles", "Canonical SMILES",
    "cleaned_smiles", "standardized_smiles", "mol_smiles"
]

# ── Columns in the master dataset that are metadata, not features ─────────────
MASTER_NON_FEATURE_COLUMNS = {
    "Canonical_SMILES",
    "Molecule ChEMBL ID",
    "Standard Relation",
    "R_squared",
    "pChEMBL Value",
    "Activity_Class",
    "Standard_Value_nM",
}

# ── Priority order for metadata columns to include in the review output ───────
REVIEW_METADATA_PRIORITY = [
    "Parent Molecule", "Name", "Synonyms", "Phase", "Drug Applicants",
    "First Approval", "ATC Codes", "Availability Type", "Withdrawn Flag",
    "Orphan", "Drug Type", "Passes Rule of Five", "Smiles", "Inchi"
]


# ── Prints a message immediately (flush ensures it shows in real-time) ─────────
def log(msg: str):
    print(msg, flush=True)


# ── Converts a string to lowercase with only alphanumeric characters ──────────
def normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


# ── Detects whether a CSV file uses semicolons or commas as separators ────────
def detect_separator(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    return ";" if first_line.count(";") > first_line.count(",") else ","


# ── Finds the actual file path, trying different extensions if needed ─────────
def resolve_table(base: Path) -> Path:
    if base.exists():
        return base

    parent = base.parent
    stem = base.name

    for ext in [".csv", ".txt", ".tsv", ".xlsx", ".parquet"]:
        p = parent / f"{stem}{ext}"
        if p.exists():
            return p

    stem_low = stem.lower()
    for p in parent.iterdir():
        if p.is_file() and p.stem.lower() == stem_low and p.suffix.lower() in {".csv", ".txt", ".tsv", ".xlsx", ".parquet"}:
            return p

    raise FileNotFoundError(f"Could not find file for: {base}")


# ── Reads a table from disk, auto-detecting the format and separator ──────────
def read_table(path: Path) -> pd.DataFrame:
    path = resolve_table(path)
    suffix = path.suffix.lower()
    log(f"[INFO] Reading: {path}")

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


# ── Finds the first column name in df that matches any of the candidates ──────
def find_column(df: pd.DataFrame, candidates):
    norm_map = {normalize_name(c): c for c in df.columns}
    for cand in candidates:
        if normalize_name(cand) in norm_map:
            return norm_map[normalize_name(cand)]
    return None


# ── RDKit tools for molecule standardization (shared across all SMILES) ───────
remover = SaltRemover.SaltRemover()
largest_fragment_chooser = rdMolStandardize.LargestFragmentChooser()
uncharger = rdMolStandardize.Uncharger()


# ── Standardizes a SMILES: removes salts, keeps main fragment, removes charges ─
def standardize_smiles(smiles):
    if pd.isna(smiles):
        return np.nan

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.nan

    try:
        mol = remover.StripMol(mol)
        mol = largest_fragment_chooser.choose(mol)
        mol = uncharger.uncharge(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return np.nan


# ── Converts a fingerprint bit vector to a list of integers ───────────────────
def bitvect_to_int_list(fp):
    return [int(bit) for bit in list(fp)]


# ── Identifies which columns in the master dataset are numeric features ───────
def infer_master_feature_columns(master: pd.DataFrame):
    feature_cols = []
    for col in master.columns:
        if col in MASTER_NON_FEATURE_COLUMNS:
            continue
        numeric_series = pd.to_numeric(master[col], errors="coerce")
        if numeric_series.notna().mean() >= 0.95:
            feature_cols.append(col)
    if not feature_cols:
        raise ValueError("No numeric feature columns were found in the master feature dataset.")
    return feature_cols


# ── Aligns candidate features to the exact column order of the training schema ─
def align_to_master(candidate_features: pd.DataFrame, master: pd.DataFrame, feature_cols):
    aligned = pd.DataFrame(index=candidate_features.index)
    missing_cols = []

    for col in feature_cols:
        if col in candidate_features.columns:
            aligned[col] = pd.to_numeric(candidate_features[col], errors="coerce")
        else:
            # ── If a feature column is missing, fill it with the master dataset median ─
            missing_cols.append(col)
            master_num = pd.to_numeric(master[col], errors="coerce")
            fill_value = 0.0 if master_num.notna().sum() == 0 else float(master_num.median())
            aligned[col] = fill_value

    return aligned, missing_cols


# ── Computes the same raw features that were computed during training ──────────
def generate_raw_features(canonical_smiles_series: pd.Series):
    log("[INFO] Building RDKit molecule objects...")
    mols = [Chem.MolFromSmiles(smi) for smi in canonical_smiles_series]
    valid_mask = [mol is not None for mol in mols]

    if not all(valid_mask):
        canonical_smiles_series = canonical_smiles_series.loc[valid_mask].reset_index(drop=True)
        mols = [mol for mol in mols if mol is not None]

    log(f"[INFO] Valid molecules for feature generation: {len(mols)}")

    # ── Compute every RDKit 2D descriptor for each valid molecule ─────────────
    log("[INFO] Computing RDKit descriptors...")
    rdkit_names = [name for name, _ in Descriptors.descList]
    rdkit_rows = []
    for mol in mols:
        row = []
        for _, func in Descriptors.descList:
            try:
                row.append(func(mol))
            except Exception:
                row.append(np.nan)
        rdkit_rows.append(row)
    rdkit_df = pd.DataFrame(rdkit_rows, columns=rdkit_names)

    # ── Compute Mordred 2D descriptors ────────────────────────────────────────
    log("[INFO] Computing Mordred descriptors...")
    calc = Calculator(descriptors, ignore_3D=True)
    mordred_df = calc.pandas(mols, nproc=1)
    mordred_df = mordred_df.apply(pd.to_numeric, errors="coerce")

    # ── Compute MACCS keys (167 bits) and Morgan fingerprints (2048 bits) ─────
    log("[INFO] Computing MACCS and Morgan fingerprints...")
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    maccs_rows = []
    morgan_rows = []
    for mol in mols:
        maccs_rows.append(bitvect_to_int_list(MACCSkeys.GenMACCSKeys(mol)))
        morgan_rows.append(bitvect_to_int_list(morgan_generator.GetFingerprint(mol)))

    maccs_df = pd.DataFrame(maccs_rows, columns=[f"MACCS_{i}" for i in range(167)], dtype=np.int8)
    morgan_df = pd.DataFrame(morgan_rows, columns=[f"Morgan_{i}" for i in range(2048)], dtype=np.int8)

    # ── Merge features exactly as done during training ────────────────────────
    log("[INFO] Assembling raw feature matrix...")
    descriptor_df = pd.concat([rdkit_df, mordred_df], axis=1)
    descriptor_df = descriptor_df.loc[:, ~descriptor_df.columns.duplicated()]
    descriptor_df = descriptor_df.replace([np.inf, -np.inf], np.nan)

    fingerprint_df = pd.concat([maccs_df, morgan_df], axis=1)
    fingerprint_df = fingerprint_df.loc[:, ~fingerprint_df.columns.duplicated()]
    fingerprint_df = fingerprint_df.astype(np.int8)

    features_df = pd.concat([descriptor_df, fingerprint_df], axis=1)
    features_df = features_df.loc[:, ~features_df.columns.duplicated()]

    log(f"[INFO] RDKit descriptor matrix: {rdkit_df.shape}")
    log(f"[INFO] Mordred descriptor matrix: {mordred_df.shape}")
    log(f"[INFO] MACCS fingerprint matrix: {maccs_df.shape}")
    log(f"[INFO] Morgan fingerprint matrix: {morgan_df.shape}")
    log(f"[INFO] Final merged feature matrix: {features_df.shape}")

    return canonical_smiles_series.reset_index(drop=True), features_df.reset_index(drop=True)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ── Load the master training feature dataset and the prediction input file ─
    master_path = resolve_table(MASTER_DIR / MASTER_STEM)
    candidate_path = resolve_table(PREDICTION_FILE)

    master = read_table(master_path)
    cand_raw = read_table(candidate_path)

    # ── Find the SMILES column in both datasets ───────────────────────────────
    master_smiles_col = find_column(master, SMILES_CANDIDATES)
    if master_smiles_col is None:
        raise ValueError(f"Could not find a SMILES column in the master dataset. Columns: {list(master.columns)}")

    cand_smiles_col = find_column(cand_raw, SMILES_CANDIDATES)
    if cand_smiles_col is None:
        raise ValueError(f"Could not find a SMILES column in the prediction dataset. Columns: {list(cand_raw.columns)}")

    log(f"[INFO] Master SMILES column: {master_smiles_col}")
    log(f"[INFO] Prediction SMILES column: {cand_smiles_col}")

    # ── Identify the exact feature columns the trained models expect ──────────
    feature_cols = infer_master_feature_columns(master)
    log(f"[INFO] Master feature columns kept for modeling: {len(feature_cols)}")

    # ── Standardize master SMILES to detect overlap with candidates ───────────
    log("[INFO] Standardizing master dataset SMILES for overlap detection...")
    master_std = master[[master_smiles_col]].copy()
    master_std["Canonical_SMILES"] = master_std[master_smiles_col].apply(standardize_smiles)
    master_smiles_set = set(master_std["Canonical_SMILES"].dropna().astype(str))

    # ── Standardize candidate SMILES ─────────────────────────────────────────
    cand = cand_raw.copy()
    cand["Raw_SMILES"] = cand[cand_smiles_col]
    cand["Canonical_SMILES"] = cand["Raw_SMILES"].apply(standardize_smiles)

    # ── Remove rows with invalid SMILES ───────────────────────────────────────
    n_total = len(cand)
    n_invalid = cand["Canonical_SMILES"].isna().sum()
    cand = cand.dropna(subset=["Canonical_SMILES"]).copy()

    # ── Remove duplicate candidate molecules ──────────────────────────────────
    n_before_dups = len(cand)
    cand = cand.drop_duplicates(subset=["Canonical_SMILES"], keep="first").copy()
    n_duplicate_smiles = n_before_dups - len(cand)

    # ── Remove any candidate that already exists in the training data ─────────
    n_before_overlap = len(cand)
    cand = cand[~cand["Canonical_SMILES"].isin(master_smiles_set)].copy()
    n_overlap_removed = n_before_overlap - len(cand)

    if cand.empty:
        raise ValueError("No molecules remain after removing invalid SMILES, duplicates, and overlaps with the master dataset.")

    log(f"[INFO] Raw candidate rows: {n_total}")
    log(f"[INFO] Removed invalid / missing SMILES: {n_invalid}")
    log(f"[INFO] Removed duplicate standardized molecules: {n_duplicate_smiles}")
    log(f"[INFO] Removed overlaps with master dataset: {n_overlap_removed}")
    log(f"[INFO] Final candidate molecules to featurize: {len(cand)}")

    # ── Generate raw features for the cleaned candidates ──────────────────────
    canonical_smiles_clean, raw_feature_pool = generate_raw_features(cand["Canonical_SMILES"].reset_index(drop=True))

    # ── Keep candidate rows in sync if any molecule became invalid during featurization ─
    cand = cand.iloc[:len(canonical_smiles_clean)].reset_index(drop=True).copy()
    cand["Canonical_SMILES"] = canonical_smiles_clean

    # ── Align generated features to the exact column schema expected by the models ─
    aligned_features, missing_cols = align_to_master(raw_feature_pool, master, feature_cols)

    # ── Model-ready output: only the feature columns plus SMILES ─────────────
    model_ready_df = pd.concat(
        [cand[["Canonical_SMILES"]].reset_index(drop=True), aligned_features.reset_index(drop=True)],
        axis=1,
    )

    # ── Review output: priority metadata columns plus the feature columns ─────
    review_meta = [c for c in REVIEW_METADATA_PRIORITY if c in cand.columns]
    review_df = pd.concat(
        [cand[review_meta + ["Canonical_SMILES"]].reset_index(drop=True), aligned_features.reset_index(drop=True)],
        axis=1,
    )

    # ── Save all outputs ──────────────────────────────────────────────────────
    cleaned_candidates_path = OUTDIR / "chembl_candidates_no_overlap.csv"
    model_ready_path = OUTDIR / "chembl_prediction_model_ready.csv"
    review_path = OUTDIR / "chembl_prediction_with_metadata.csv"
    report_path = OUTDIR / "chembl_prediction_feature_report.txt"
    missing_cols_path = OUTDIR / "missing_master_feature_columns.txt"

    cand.to_csv(cleaned_candidates_path, index=False)
    model_ready_df.to_csv(model_ready_path, index=False)
    review_df.to_csv(review_path, index=False)

    # ── Write a human-readable report summarizing what was done ───────────────
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ChEMBL prediction feature generation report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Master dataset: {master_path}\n")
        f.write(f"Prediction dataset: {candidate_path}\n")
        f.write(f"Master SMILES column: {master_smiles_col}\n")
        f.write(f"Prediction SMILES column: {cand_smiles_col}\n")
        f.write(f"Raw candidate rows: {n_total}\n")
        f.write(f"Removed invalid/missing SMILES: {n_invalid}\n")
        f.write(f"Removed duplicate standardized molecules: {n_duplicate_smiles}\n")
        f.write(f"Removed overlaps with master: {n_overlap_removed}\n")
        f.write(f"Final candidate rows: {len(cand)}\n")
        f.write(f"Master feature columns kept: {len(feature_cols)}\n")
        f.write(f"Missing master feature columns backfilled from master medians/defaults: {len(missing_cols)}\n")
        f.write(f"Model-ready output shape: {model_ready_df.shape}\n")
        f.write(f"Review output shape: {review_df.shape}\n")
        f.write("\nImportant note:\n")
        f.write("- This output is raw-feature aligned and model-ready in schema.\n")
        f.write("- Train-dependent preprocessing must still come from the training pipeline.\n")

    # ── Save the list of any feature columns that were missing and backfilled ─
    with open(missing_cols_path, "w", encoding="utf-8") as f:
        for col in missing_cols:
            f.write(str(col) + "\n")

    log("[INFO] Done.")
    log(f"[INFO] Cleaned candidates saved to: {cleaned_candidates_path}")
    log(f"[INFO] Model-ready dataset saved to: {model_ready_path}")
    log(f"[INFO] Metadata review dataset saved to: {review_path}")
    log(f"[INFO] Report saved to: {report_path}")
    log(f"[INFO] Missing-column list saved to: {missing_cols_path}")


if __name__ == "__main__":
    main()
