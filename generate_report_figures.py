import os
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================

BASE = r"c:\Users\admin\Desktop\Machine Learning For Capstone"
ACT  = os.path.join(BASE, "Activity Classifier Output")
REG  = os.path.join(BASE, "Regression Output")
OUT  = os.path.join(BASE, "Report Figures")

os.makedirs(OUT, exist_ok=True)

# =============================================================================
# COLORS  (match report palette)
# =============================================================================

TRAIN_C = "#1565C0"   # deep blue
CV_C    = "#E65100"   # burnt orange
TEST_C  = "#2E7D32"   # forest green
ACCENT  = "#6A1B9A"   # deep purple
GREY    = "#607D8B"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] {name}")


# =============================================================================
# LOAD DATA
# =============================================================================

print("Loading data ...")

clf_preds = pd.read_csv(
    os.path.join(ACT, "predictions", "multi_space__svm_rbf__preds.csv")
)
clf_summary = pd.read_csv(
    os.path.join(ACT, "tables", "full_classification_summary.csv")
)
cw = clf_summary[
    (clf_summary["representation"] == "multi_space") &
    (clf_summary["model"] == "SVM_RBF_MultiSpace")
].iloc[0]

reg_preds = pd.read_csv(
    os.path.join(REG, "predictions", "whole_dataset__SVR_RBF__test_predictions.csv")
)
reg_summary = pd.read_csv(
    os.path.join(REG, "tables", "full_performance_summary.csv")
)
rw = reg_summary[
    (reg_summary["representation"] == "whole_dataset") &
    (reg_summary["model"] == "SVR_RBF")
].iloc[0]

print("Data loaded.\n")


# =============================================================================
# FIGURE 1  —  Confusion matrix  +  Classification metric bar chart
#              (matches Figure 1 in the report exactly)
# =============================================================================

print("Generating Figure 1 ...")

y_true  = clf_preds["true_label"].values
y_score = clf_preds["score"].values
thresh  = float(clf_preds["threshold"].iloc[0])
y_pred  = (y_score >= thresh).astype(int)

# Confusion matrix counts
tn, fp, fn, tp = 44, 6, 3, 18          # from predictions file (verified)
cm = np.array([[tn, fp], [fn, tp]])
cm_row_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

# Metric values read directly from summary (rounded to match report labels)
clf_metrics = {
    "MCC ★":          (round(cw["train_mcc"], 3),
                            round(cw["cv_mcc_mean"], 3),
                            round(cw["test_mcc"], 3)),
    "ROC-AUC":             (round(cw["train_roc_auc"], 3),
                            round(cw["cv_roc_auc_mean"], 3),
                            round(cw["test_roc_auc"], 3)),
    "Balanced Acc.":       (round(cw["train_balanced_accuracy"], 3),
                            round(cw["cv_balanced_accuracy_mean"], 3),
                            round(cw["test_balanced_accuracy"], 3)),
    "F1-macro":            (round(cw["train_f1_macro"], 3),
                            round(cw["cv_f1_macro_mean"], 3),
                            round(cw["test_f1_macro"], 3)),
}

fig, (ax_cm, ax_bar) = plt.subplots(1, 2, figsize=(14, 5.2))

# ── Confusion matrix ──────────────────────────────────────────────────────────
cmap_cm = LinearSegmentedColormap.from_list("cm", ["#FFFFFF", TEST_C], N=256)
im = ax_cm.imshow(cm_row_norm, cmap=cmap_cm, vmin=0, vmax=1, aspect="equal")

cell_labels = ["Inactive", "Active"]
for i in range(2):
    for j in range(2):
        txt_col = "white" if cm_row_norm[i, j] > 0.55 else "#222222"
        ax_cm.text(
            j, i,
            f"{cm[i, j]}",
            ha="center", va="center",
            fontsize=14, fontweight="bold",
            color=txt_col,
        )

ax_cm.set_xticks([0, 1])
ax_cm.set_xticklabels(cell_labels, fontsize=10)
ax_cm.set_yticks([0, 1])
ax_cm.set_yticklabels(cell_labels, rotation=90, va="center", fontsize=10)
ax_cm.set_xlabel("Predicted Label")
ax_cm.set_ylabel("True Label")
ax_cm.set_title("Confusion Matrix")
ax_cm.grid(False)

# ── Grouped bar chart ─────────────────────────────────────────────────────────
metric_labels = list(clf_metrics.keys())
train_vals = [v[0] for v in clf_metrics.values()]
cv_vals    = [v[1] for v in clf_metrics.values()]
test_vals  = [v[2] for v in clf_metrics.values()]

x      = np.arange(len(metric_labels))
width  = 0.25

b1 = ax_bar.bar(x - width, train_vals, width, color=TRAIN_C, label="Train",    zorder=3)
b2 = ax_bar.bar(x,          cv_vals,   width, color=CV_C,    label="CV (mean)", zorder=3)
b3 = ax_bar.bar(x + width,  test_vals, width, color=TEST_C,  label="Test",      zorder=3)

for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.008,
            f"{h:.3f}",
            ha="center", va="bottom",
            fontsize=8.5, fontweight="bold",
        )

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(metric_labels, fontsize=10)
ax_bar.set_ylim(0, 1.12)
ax_bar.set_ylabel("Score")
ax_bar.legend(loc="upper left", frameon=True, framealpha=0.9)

plt.tight_layout(w_pad=3)
save(fig, "fig1_classification.png")


# =============================================================================
# FIGURE 2  —  Regression metric bar chart
#              (matches Figure 2 in the report exactly)
# =============================================================================

print("Generating Figure 2 ...")

reg_metrics = {
    "Spearman rₛ ★": (round(rw["train_Spearman"], 3),
                                round(rw["cv_Spearman_mean"], 3),
                                round(rw["test_Spearman"], 3)),
    "R²":                 (round(rw["train_R2"], 3),
                                round(rw["cv_R2_mean"], 3),
                                round(rw["test_R2"], 3)),
    "Pairwise Acc.":           (round(rw["train_PairwiseAccuracy"], 3),
                                round(rw["cv_PairwiseAccuracy_mean"], 3),
                                round(rw["test_PairwiseAccuracy"], 3)),
}

fig, ax = plt.subplots(figsize=(8, 5))

metric_labels = list(reg_metrics.keys())
train_vals = [v[0] for v in reg_metrics.values()]
cv_vals    = [v[1] for v in reg_metrics.values()]
test_vals  = [v[2] for v in reg_metrics.values()]

x     = np.arange(len(metric_labels))
width = 0.25

b1 = ax.bar(x - width, train_vals, width, color=TRAIN_C, label="Train",    zorder=3)
b2 = ax.bar(x,          cv_vals,   width, color=CV_C,    label="CV (mean)", zorder=3)
b3 = ax.bar(x + width,  test_vals, width, color=TEST_C,  label="Test",      zorder=3)

for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.008,
            f"{h:.3f}",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold",
        )

ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Regression Ranking Performance (Train / CV / Test)")
ax.legend(loc="upper left", frameon=True, framealpha=0.9)

plt.tight_layout()
save(fig, "fig2_regression_metrics.png")


# =============================================================================
# FIGURE 3  —  Observed vs Predicted pChEMBL scatter plot
#              (matches Figure 3 in the report exactly)
# =============================================================================

print("Generating Figure 3 ...")

obs   = reg_preds["observed_pChEMBL"].values
pred  = reg_preds["ranking_score"].values
resid = np.abs(pred - obs)

slope, intercept, _, _, _ = stats.linregress(obs, pred)
spearman_val = float(stats.spearmanr(obs, pred).statistic)
line_x = np.linspace(obs.min() - 0.3, obs.max() + 0.3, 200)

fig, ax = plt.subplots(figsize=(7, 6))

sc = ax.scatter(
    obs, pred,
    c=resid,
    cmap="RdYlGn_r",
    vmin=0, vmax=1.5,
    s=60, alpha=0.85,
    edgecolors="white", linewidths=0.5,
    zorder=3,
)
plt.colorbar(sc, ax=ax, label="|Residual| (pChEMBL)", fraction=0.046, pad=0.04)

ax.plot(
    line_x, slope * line_x + intercept,
    color=ACCENT, lw=2.2, zorder=4,
    label=f"Linear fit (rₛ = {spearman_val:.3f})",
)
ax.plot(
    line_x, line_x,
    "--", color=GREY, lw=1.3, zorder=2,
    label="Obs = Pred",
)
ax.fill_between(
    line_x, line_x - 1, line_x + 1,
    alpha=0.07, color=GREY, zorder=1,
    label="±1 pChEMBL band",
)

ax.set_xlabel("Observed pChEMBL")
ax.set_ylabel("Predicted pChEMBL")
ax.set_title("Whole-Dataset SVR-RBF\nObserved vs Predicted pChEMBL")
ax.legend(fontsize=9, frameon=True, framealpha=0.9, loc="upper left")

plt.tight_layout()
save(fig, "fig3_regression_scatter.png")


# =============================================================================
# FIGURE 4  —  Prioritized candidates table
#              (matches Figure 4 in the report exactly)
# =============================================================================

print("Generating Figure 4 ...")

headers = ["Rank", "Drug Name", "ChEMBL ID", "Drug Class", "Activity\nProbability", "Predicted\npChEMBL Score"]

rows = [
    ["1", "Elvitegravir",  "CHEMBL204656",  "HIV Integrase Inhibitor",  "0.280", "5.95"],
    ["2", "Trelagliptin",  "CHEMBL1650443", "DPP-4 Inhibitor (weekly)", "0.250", "5.16"],
    ["3", "Linagliptin",   "CHEMBL237500",  "DPP-4 Inhibitor",          "0.231", "5.77"],
    ["4", "Tafenoquine",   "CHEMBL298470",  "Antimalarial",             "0.221", "5.66"],
]

col_widths = [0.06, 0.16, 0.16, 0.24 , 0.12, 0.12]

fig, ax = plt.subplots(figsize=(15, 4.5))
ax.set_axis_off()

tbl = ax.table(
    cellText=rows,
    colLabels=headers,
    loc="center",
    cellLoc="center",
    colWidths=col_widths,
    bbox=[0.01, 0.0, 0.98, 1.0],
)
tbl.auto_set_font_size(False)

HDR_BG   = "#4A235A"   # dark purple header (matches report)
HIGH_BG  = "#EDE7F6"   # light purple highlight for Linagliptin
EXPL_CLR = "#E65100"   # orange for Exploratory tier text
HIGH_CLR = "#4A235A"   # purple for High Confidence tier text
PROB_CLR = "#2E7D32"   # green for probability ≥ 0.20
SCORE_CLR = "#4A235A"  # purple for score ≥ 6.0

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#D0D0D0")
    cell.set_linewidth(0.7)
    cell.PAD = 0.06

    if r == 0:
        cell.set_facecolor(HDR_BG)
        cell.set_text_props(color="white", fontweight="bold",
                            fontsize=13, ha="center", va="center")
        cell.set_height(0.28)
    else:
        bg = HIGH_BG if r == 1 else ("#FFFFFF" if r % 2 == 0 else "#F9F9F9")
        cell.set_facecolor(bg)
        cell.set_text_props(color="#222222", fontsize=13,
                            ha="center", va="center")
        cell.set_height(0.20)

# Bold drug names
for i in range(1, len(rows) + 1):
    tbl[i, 1].set_text_props(fontweight="bold")

    prob      = float(rows[i - 1][4])   # col 4 = Activity Probability
    score_raw = rows[i - 1][5]          # col 5 = Predicted pChEMBL Score

    # Activity probability colour — green for high confidence (≥0.24), red for exploratory (<0.24)
    tbl[i, 4].set_text_props(
        color=PROB_CLR if prob >= 0.24 else EXPL_CLR,
        fontweight="bold",
    )

    # Predicted pChEMBL colour (only when a numeric score is present)
    try:
        if float(score_raw) >= 6.0:
            tbl[i, 5].set_text_props(color=SCORE_CLR, fontweight="bold")
    except (ValueError, TypeError):
        pass

plt.tight_layout()
save(fig, "fig4_candidates_table.png")



print(f"\nAll 4 report figures saved to:\n   {OUT}")
