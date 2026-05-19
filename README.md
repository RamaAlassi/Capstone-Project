# QSAR Machine Learning for SARS-CoV-2 NSP3 Mac1 Drug Repurposing

This repository contains the machine-learning component of a capstone project focused on **QSAR-based drug repurposing for SARS-CoV-2 NSP3 Mac1**. The goal of this workflow is to use ligand-based machine learning to classify compounds as potentially active or inactive against Mac1 and to rank predicted candidates according to estimated potency.

The machine-learning workflow was developed as part of the project:

**QSAR-Based Drug Repurposing and Ensemble Docking Validation Targeting SARS-CoV-2 NSP3 Mac1**

## Project Overview

SARS-CoV-2 NSP3 Mac1 is a macrodomain involved in reversing ADP-ribosylation, which helps the virus interfere with host antiviral responses. Because of this role, Mac1 is considered a relevant target for antiviral drug discovery.

This machine-learning pipeline was designed to:

- Collect and curate Mac1 bioactivity data
- Generate molecular descriptors and fingerprints
- Train QSAR classification models to identify potentially active compounds
- Train regression models to rank compounds by predicted potency
- Apply trained models to approved ChEMBL compounds for drug-repurposing screening
- Evaluate whether predictions fall inside or outside the models' applicability domains

## Data Source

Bioactivity data were collected from the **ChEMBL biochemical assay CHEMBL5441457**. The initial dataset contained **1,443 bioactivity records**. To confirm target relevance, the assay sequence was aligned against a SARS-CoV-2 NSP3 Mac1 reference sequence using BLAST.

After cleaning and preprocessing, the curated dataset contained:

- **433 unique molecules** for regression modeling
- **3,823 generated molecular features**
- **357 molecules** for classification after removing the grey zone around the activity threshold
  - 107 active compounds
  - 250 inactive compounds

## Data Preprocessing

The raw ChEMBL dataset was cleaned before feature generation using the following steps:

1. Removed records with missing SMILES.
2. Converted standard activity values to a common unit, **nM**.
3. Calculated missing pChEMBL values from IC50 values when possible.
4. Standardized chemical structures using RDKit.
5. Converted valid molecules to canonical SMILES.
6. Removed molecules that failed canonicalization.
7. Removed duplicate molecules by keeping the record with the highest pChEMBL value.
8. Kept only high-quality activity records with:
   - Standard relation = `"="`
   - R² ≥ 0.9

For classification, molecules within a grey zone of **±0.2 pChEMBL units** around the activity threshold were removed. Therefore, molecules with pChEMBL between **5.8 and 6.2** were excluded to reduce label uncertainty.

Classification labels were assigned as:

- **Active:** pChEMBL ≥ 6.0
- **Inactive:** pChEMBL < 6.0

## Molecular Feature Generation

Molecular features were generated from canonical SMILES using:

- **RDKit 2D descriptors**
- **Mordred 2D descriptors**
- **MACCS fingerprints** with 167 bits
- **Morgan fingerprints** with radius = 2 and 2,048 bits

After feature generation:

- Duplicate descriptor columns were removed.
- Infinite descriptor values were converted to missing values.
- Descriptor-based and fingerprint-based features were handled separately during preprocessing.

## Scaffold-Based Data Splitting

Bemis-Murcko scaffolds were generated using RDKit. A scaffold-aware 80/20 train-test split was applied using a fixed random seed of 42.

This split was used to reduce structural leakage by ensuring that molecules sharing the same core scaffold were not split between training and testing sets.

### Classification Split

- Training set: 286 molecules
  - 86 active
  - 200 inactive
  - 145 unique scaffolds
- Test set: 71 molecules
  - 21 active
  - 50 inactive
  - 57 unique scaffolds

### Regression Split

- Training set: 346 molecules
- Test set: 87 molecules
- Regression pChEMBL range: 4.01-7.89
- Full dataset mean pChEMBL: 5.58
- Full dataset standard deviation: 0.92

## Leakage-Safe Preprocessing

All preprocessing transformations were fitted only on the training set and then applied to the test set.

### Descriptor Features

For RDKit and Mordred descriptors:

- Removed features with more than 20% missing values.
- Imputed remaining missing values using training-set medians.
- Removed zero-variance descriptors.
- Removed highly correlated descriptors with correlation ≥ 0.95.
- Standardized descriptors using `StandardScaler` fitted on the training set only.

### Fingerprint Features

For MACCS and Morgan fingerprints:

- Filled missing values with 0.
- Kept features as binary 0/1 values.
- Removed only zero-variance fingerprint bits based on the training set.

## Machine Learning Tasks

Two QSAR modeling tasks were developed.

### 1. Activity Classification

The classification model was trained to distinguish active from inactive compounds.

Algorithms tested:

- Logistic Regression
- Support Vector Machine with RBF kernel
- Random Forest
- XGBoost

The main evaluation metric for model selection was **Matthews Correlation Coefficient (MCC)** because it is suitable for imbalanced classification datasets.

### 2. Regression-Based Ranking

The regression model was trained to rank compounds according to predicted pChEMBL values.

Algorithms tested:

- ElasticNet
- Partial Least Squares Regression
- Support Vector Regression with RBF kernel
- Random Forest
- XGBoost

The main evaluation metric for model selection was **Spearman's rank correlation coefficient**, because the goal was candidate prioritization rather than exact pChEMBL prediction.

## Feature Representations Tested

Several molecular representation spaces were evaluated:

- Descriptors only
- MACCS fingerprints only
- Morgan fingerprints only
- Combined fingerprints
- Full feature set containing descriptors and fingerprints

For classification, a multi-space feature-selection pipeline was also tested. In this approach, informative features were selected separately from:

- RDKit/Mordred descriptors
- MACCS fingerprints
- Morgan fingerprints

The selected features were then combined into one reduced feature set.

## Hyperparameter Tuning

Hyperparameter tuning was performed using randomized search with **5-fold scaffold-based cross-validation**.

Dimensionality-reduction and feature-selection approaches included:

- `SelectKBest` for selecting the most informative features
- PCA for descriptor-based features
- SelectKBest for fingerprint features

PCA was not applied to fingerprint features because fingerprints are binary data.

## Final Classification Model

The selected classification model was:

**Multi-Space SVM-RBF**

Final selected feature set:

- 30 Morgan fingerprint bits
- 30 MACCS fingerprint bits
- 30 RDKit/Mordred descriptor features

Optimized hyperparameters:

- C = 5
- gamma = `"scale"`

Performance:

| Metric | Value |
|---|---:|
| Mean CV MCC | 0.639 |
| Train MCC | 0.766 |
| Test MCC | 0.711 |
| ROC-AUC | 0.928 |
| Balanced Accuracy | 0.869 |
| F1-macro | ~0.85 |
| Sensitivity | 85.7% |
| Specificity | 88.0% |

Confusion matrix summary on the test set:

- True positives: 18
- True negatives: 44
- False negatives: 3
- False positives: 6

These results showed that the classifier was able to distinguish active from inactive compounds with good generalization performance.

## Final Regression Model

The selected regression model was:

**SVR-RBF using the full feature set**

Feature-selection and hyperparameters:

- SelectKBest selected 50 features out of 1,996 columns
- gamma = 0.01
- epsilon = 0.05
- C = 1

Performance:

| Metric | Value |
|---|---:|
| CV Spearman rₛ | 0.674 |
| Test Spearman rₛ | 0.647 |
| Pairwise Accuracy | 0.740 |
| R² | 0.414 |
| RMSE | 0.703 pChEMBL |
| MAE | 0.524 pChEMBL |

The regression model was used mainly for **ranking compounds**, not for predicting exact potency values. The ranking metrics supported its use for candidate prioritization.

## Applicability Domain

The applicability domain was evaluated using a **k-nearest neighbor distance-based method** with k = 5.

For each training compound:

1. The mean Euclidean distance to its 5 nearest training neighbors was calculated.
2. The 95th percentile of these mean distances was used as the applicability-domain threshold.
3. A prediction was considered inside the applicability domain if its mean distance to the 5 nearest training neighbors was below this threshold.

This step was used to determine whether external compounds were chemically similar enough to the training set for predictions to be considered reliable.

## External Screening

The trained classification and regression models were applied to an external prediction dataset of **1,991 approved ChEMBL compounds**.

The activity classifier was used first as the primary screening step. A relaxed probability threshold of **0.24** was used to reduce the risk of missing potentially active compounds.

### Prioritized Candidates

The final prioritized compounds were:

| Compound | ChEMBL ID | Classification Result | Predicted Probability | Regression Score |
|---|---|---|---:|---:|
| Elvitegravir | CHEMBL204656 | Predicted active | 0.28 | 5.95 |
| Trelagliptin | CHEMBL1650443 | Predicted active | 0.25 | 5.16 |
| Linagliptin | Not specified here | Near-threshold exploratory hit | 0.20-0.24 | Higher than Trelagliptin |
| Tafenoquine | Not specified here | Near-threshold exploratory hit | 0.20-0.24 | Higher than Trelagliptin |

Elvitegravir and Trelagliptin were classified as predicted active compounds. Linagliptin and Tafenoquine were retained as exploratory near-threshold candidates.

However, all four prioritized candidates were outside both the classification and regression applicability domains. Therefore, they should be interpreted as exploratory repurposing candidates rather than confirmed computational hits.

## Main Limitations

- The QSAR models were trained on a relatively small and assay-specific dataset.
- The number of active compounds was limited.
- The regression model was more suitable for ranking than exact pChEMBL prediction.
- The prioritized external candidates were outside the models' applicability domains.
- Experimental validation is required before any compound can be considered a reliable Mac1 inhibitor.

## Software and Libraries

The machine-learning workflow used Python-based cheminformatics and machine-learning tools, including:

- Python
- RDKit
- Mordred
- scikit-learn
- XGBoost
- Pandas
- NumPy
- SciPy
- Matplotlib
- Seaborn
- Joblib


## Conclusion

This machine-learning workflow provides a QSAR-based screening and ranking approach for identifying potential SARS-CoV-2 NSP3 Mac1 inhibitors. The classification model showed strong performance in distinguishing active from inactive compounds, while the regression model provided useful ranking ability for candidate prioritization.

The external screening identified Elvitegravir and Trelagliptin as predicted active candidates, while Linagliptin and Tafenoquine were retained as exploratory near-threshold hits. Since all candidates were outside the applicability domain, they require further docking, rescoring, and experimental validation before being considered reliable Mac1 inhibitors.
