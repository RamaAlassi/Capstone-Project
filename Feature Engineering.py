import os
import warnings
from multiprocessing import freeze_support

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

from mordred import Calculator, descriptors

warnings.filterwarnings("ignore")

# File paths and output folder
BASE_DIR = r"C:\Users\admin\Desktop\Machine Learning For Capstone"
DATASET_PATH = os.path.join(BASE_DIR, "Dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Feature Generation Outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Filtering thresholds and column name constants
QUALITY_R2_MIN = 0.90
CLASSIFICATION_THRESHOLD_PCHEMBL = 6.0
CLASSIFICATION_GREY_ZONE_MARGIN = 0.2

SMILES_COL = "Smiles"
VALUE_COL = "Standard Value"
UNIT_COL = "Standard Units"
REL_COL = "Standard Relation"
ID_COL = "Molecule ChEMBL ID"
PCHEMBL_COL = "pChEMBL Value"
R2_COL = "R_squared"


# Strips stray quotes and whitespace from a measurement relation string
def normalize_relation(x):
    if pd.isna(x):
        return np.nan
    return str(x).replace("'", "").replace('"', "").strip()


# Converts any concentration unit to nanomolar using a lookup table 
def convert_to_nm(value, unit):
    try:
        value = float(value)
    except (ValueError, TypeError):
        return np.nan

    unit_map = {
        "nm": 1,
        "um": 1000,
        "µm": 1000,
        "μm": 1000,
        "mm": 1_000_000,
        "m": 1_000_000_000,
        "pm": 0.001,
    }

    unit_key = str(unit).lower().strip() if pd.notna(unit) else ""
    factor = unit_map.get(unit_key, np.nan)
    return value * factor if pd.notna(factor) else np.nan


# Converts nM IC50 to pChEMBL using the -log10 formula
def calculate_pchembl(value_nm):
    if pd.notna(value_nm) and value_nm > 0:
        return -np.log10(value_nm * 1e-9)
    return np.nan


# RDKit tools for molecule standardization (initialized once at module level)
remover = SaltRemover.SaltRemover()
largest_fragment_chooser = rdMolStandardize.LargestFragmentChooser()
uncharger = rdMolStandardize.Uncharger()


# Cleans a SMILES string (removes salts, keeps the main fragment, removes charges)
def standardize_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.nan

    mol = remover.StripMol(mol)
    mol = largest_fragment_chooser.choose(mol)
    mol = uncharger.uncharge(mol)

    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return np.nan


# Converts a fingerprint bit vector into a plain Python list of integers
def bitvect_to_int_list(fp):
    return [int(bit) for bit in list(fp)]


def main():
    # Load the raw CSV and drop any row that is missing a SMILES string
    print("Loading and cleaning source dataset...")
    df = pd.read_csv(DATASET_PATH)

    df = df.dropna(subset=[SMILES_COL]).copy()
    df[REL_COL] = df[REL_COL].apply(normalize_relation)
    df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors="coerce")
    df[R2_COL] = pd.to_numeric(df[R2_COL], errors="coerce")

    # Convert all activity measurements to nM, then calculate pChEMBL
    df["Standard_Value_nM"] = df.apply(
        lambda row: convert_to_nm(row[VALUE_COL], row[UNIT_COL]),
        axis=1,
    )

    df[PCHEMBL_COL] = df.apply(
        lambda row: calculate_pchembl(row["Standard_Value_nM"])
        if pd.isna(row.get(PCHEMBL_COL, np.nan))
        else pd.to_numeric(row[PCHEMBL_COL], errors="coerce"),
        axis=1,
    )

    # Remove rows without a valid pChEMBL, then standardize the SMILES
    df = df.dropna(subset=[PCHEMBL_COL]).copy()
    df["Canonical_SMILES"] = df[SMILES_COL].apply(standardize_smiles)
    df = df.dropna(subset=["Canonical_SMILES"]).copy()

    # Remove duplicate molecules, keeping only the row with the highest potency
    df = df.sort_values(by=PCHEMBL_COL, ascending=False)
    df = df.drop_duplicates(subset=["Canonical_SMILES"], keep="first").reset_index(drop=True)

    print(f"Remaining molecules after cleaning: {len(df)}")

    # Apply the shared quality (only rows with "=" relation and R² ≥ 0.9 allowed to continue)
    df = df[
        df[R2_COL].notna() &
        (df[R2_COL] >= QUALITY_R2_MIN) &
        (df[REL_COL] == "=")
    ].reset_index(drop=True)

    print(f"Molecules after quality (R² ≥ {QUALITY_R2_MIN}) and relation ('=') filter: {len(df)}")

    # Parse each SMILES into an RDKit molecule object
    print("Building RDKit molecule objects...")
    mols = [Chem.MolFromSmiles(smi) for smi in df["Canonical_SMILES"]]
    valid_mask = [mol is not None for mol in mols]

    if not all(valid_mask):
        df = df.loc[valid_mask].reset_index(drop=True)
        mols = [mol for mol in mols if mol is not None]

    print(f"Valid molecules available for feature generation: {len(mols)}")

    # This loop computes every RDKit 2D descriptor for each molecule
    print("Computing RDKit descriptors...")
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

    # Mordred computes a broader set of 2D physicochemical descriptors
    print("Computing Mordred descriptors...")
    calc = Calculator(descriptors, ignore_3D=True)
    mordred_df = calc.pandas(mols, nproc=1)
    mordred_df = mordred_df.apply(pd.to_numeric, errors="coerce")

    # MACCS keys encode 167 structural fragments; Morgan encodes circular substructures
    print("Computing MACCS and Morgan fingerprints...")
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    maccs_rows = []
    morgan_rows = []

    for mol in mols:
        maccs_rows.append(bitvect_to_int_list(MACCSkeys.GenMACCSKeys(mol)))
        morgan_rows.append(bitvect_to_int_list(morgan_generator.GetFingerprint(mol)))

    maccs_df = pd.DataFrame(maccs_rows, columns=[f"MACCS_{i}" for i in range(167)], dtype=np.int8)
    morgan_df = pd.DataFrame(morgan_rows, columns=[f"Morgan_{i}" for i in range(2048)], dtype=np.int8)

    print(f"RDKit descriptor matrix: {rdkit_df.shape}")
    print(f"Mordred descriptor matrix: {mordred_df.shape}")
    print(f"MACCS fingerprint matrix: {maccs_df.shape}")
    print(f"Morgan fingerprint matrix: {morgan_df.shape}")

    # Merge RDKit and Mordred side by side, drop any duplicated column names
    print("Assembling raw feature matrix...")
    descriptor_df = pd.concat([rdkit_df, mordred_df], axis=1)
    descriptor_df = descriptor_df.loc[:, ~descriptor_df.columns.duplicated()]
    descriptor_df = descriptor_df.replace([np.inf, -np.inf], np.nan)

    # Merge MACCS and Morgan into one fingerprint table; keep values as 0/1
    fingerprint_df = pd.concat([maccs_df, morgan_df], axis=1)
    fingerprint_df = fingerprint_df.loc[:, ~fingerprint_df.columns.duplicated()]
    fingerprint_df = fingerprint_df.astype(np.int8)

    # Stack descriptors and fingerprints side by side into the full feature matrix
    features_df = pd.concat([descriptor_df, fingerprint_df], axis=1)
    features_df = features_df.loc[:, ~features_df.columns.duplicated()]

    print(f"Raw descriptor matrix: {descriptor_df.shape}")
    print(f"Binary fingerprint matrix: {fingerprint_df.shape}")
    print(f"Final merged feature matrix: {features_df.shape}")

    # Attach SMILES and target columns to the feature matrix
    print("Building final modeling tables...")
    filtered_df = pd.concat(
        [df[["Canonical_SMILES", ID_COL, REL_COL, R2_COL, PCHEMBL_COL]], features_df],
        axis=1,
    )

    # Regression dataset: continuous pChEMBL target, metadata columns dropped
    regression_df = filtered_df.drop(columns=[REL_COL, R2_COL, ID_COL]).copy()

    # Classification dataset: remove grey-zone molecules near the activity threshold,
    # then assign binary label (1 = active, 0 = inactive)
    pre_grey_zone_df = filtered_df.copy()
    if CLASSIFICATION_GREY_ZONE_MARGIN > 0:
        lower = CLASSIFICATION_THRESHOLD_PCHEMBL - CLASSIFICATION_GREY_ZONE_MARGIN
        upper = CLASSIFICATION_THRESHOLD_PCHEMBL + CLASSIFICATION_GREY_ZONE_MARGIN
        classification_df = filtered_df[
            (filtered_df[PCHEMBL_COL] < lower) | (filtered_df[PCHEMBL_COL] > upper)
        ].copy()
    else:
        classification_df = filtered_df.copy()

    classification_df["Activity_Class"] = np.where(
        classification_df[PCHEMBL_COL] >= CLASSIFICATION_THRESHOLD_PCHEMBL,
        1,
        0,
    )
    classification_df = classification_df.drop(columns=[REL_COL, R2_COL, PCHEMBL_COL, ID_COL])

    # Reorder columns so SMILES and label appear first
    classification_cols = ["Canonical_SMILES", "Activity_Class"] + [
        c for c in classification_df.columns
        if c not in ["Canonical_SMILES", "Activity_Class"]
    ]
    classification_df = classification_df[classification_cols]

    # Save the two modeling datasets
    regression_path = os.path.join(OUTPUT_DIR, "Regression_dataset.csv")
    classification_path = os.path.join(OUTPUT_DIR, "Classification_dataset.csv")

    regression_df.to_csv(regression_path, index=False)
    classification_df.to_csv(classification_path, index=False)

    # Summary table: row counts, feature counts, and class distribution
    class_counts = classification_df["Activity_Class"].value_counts().sort_index().to_dict()
    summary_df = pd.DataFrame(
        [
            {
                "dataset": "regression",
                "n_rows": len(regression_df),
                "n_columns": regression_df.shape[1],
                "target_mean": regression_df[PCHEMBL_COL].mean() if not regression_df.empty else np.nan,
                "target_std": regression_df[PCHEMBL_COL].std() if not regression_df.empty else np.nan,
                "quality_r2_min": QUALITY_R2_MIN,
            },
            {
                "dataset": "classification",
                "n_rows": len(classification_df),
                "n_columns": classification_df.shape[1],
                "inactive_count": class_counts.get(0, 0),
                "active_count": class_counts.get(1, 0),
                "rows_before_grey_zone_filter": len(pre_grey_zone_df),
                "grey_zone_margin": CLASSIFICATION_GREY_ZONE_MARGIN,
                "quality_r2_min": QUALITY_R2_MIN,
            },
        ]
    )
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "feature_generation_summary.csv"), index=False)

    # Plot the pChEMBL distribution and the active/inactive class balance
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(df[PCHEMBL_COL], bins=30, kde=True)
    plt.axvline(
        x=CLASSIFICATION_THRESHOLD_PCHEMBL,
        color="red",
        linestyle="--",
        label=f"Threshold = {CLASSIFICATION_THRESHOLD_PCHEMBL}",
    )
    plt.title("pChEMBL distribution")
    plt.xlabel("pChEMBL")
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.countplot(x=classification_df["Activity_Class"])
    plt.title("Classification class balance")
    plt.xticks([0, 1], ["Inactive", "Active"])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_generation_overview.png"), dpi=300)
    plt.close()

    print("\nFeature generation complete.")
    print(f"Saved: {regression_path}")
    print(f"Saved: {classification_path}")


if __name__ == "__main__":
    freeze_support()
    main()
