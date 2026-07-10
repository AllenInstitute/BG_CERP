"""
Macaque enhancer scoring — panels A, B, C, D.

    A  Univariate AUC per feature (10-fold CV, mean ± SD).
    B  ROC — ATAC alone vs 10-fold OOF LR on the top-K features from A.
    C  On/off boxplots (ATAC vs CREsted) with Mann-Whitney p-values.
    D  Dumbbell of mean(on)/mean(off) specificity per cell-type target.

Required inputs (see INPUT_DIR below):
    macaque_209_features.tsv
    macaque_209_extra_features.tsv
    on_target_specificity_macaque_none_split_base99_per_enhancer.csv
    macaque_209_phylop_conserved.csv
    contribution_scores/contribution_scores_99_mean.csv
    on_off_specificity_macaque_summary.csv

Outputs (OUTPUT_DIR): panels_ABCD.{pdf,svg,png}.
Existing outputs are not overwritten unless OVERWRITE = True.
"""

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------- config
HERE = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR  = "/allen/programs/celltypes/workgroups/hct/sarojaS/260604_grammar_flex"
OUTPUT_DIR = HERE
OVERWRITE  = False

N_FOLDS  = 10
RNG_SEED = 42
TOP_K    = 5  # panel B uses the TOP_K best features (by mean CV AUC) from A

FEATURES_TSV    = os.path.join(INPUT_DIR, "macaque_209_features.tsv")
EXTRA_FEAT_TSV  = os.path.join(INPUT_DIR, "macaque_209_extra_features.tsv")
PRED_SPEC_CSV   = os.path.join(INPUT_DIR, "on_target_specificity_macaque_none_split_base99_per_enhancer.csv")
PHYLOP_FRAC_CSV = os.path.join(INPUT_DIR, "macaque_209_phylop_conserved.csv")
CONTRIB_CSV     = os.path.join(INPUT_DIR, "contribution_scores", "contribution_scores_99_mean.csv")
ON_OFF_CSV      = os.path.join(INPUT_DIR, "on_off_specificity_macaque_summary.csv")

OUT_PDF = os.path.join(OUTPUT_DIR, "panels_ABCD.pdf")
OUT_SVG = os.path.join(OUTPUT_DIR, "panels_ABCD.svg")
OUT_PNG = os.path.join(OUTPUT_DIR, "panels_ABCD.png")

# Illustrator-friendly vector text (TrueType, editable text in SVG/PDF).
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"]  = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.family"]  = "DejaVu Sans"
mpl.rcParams["axes.linewidth"]    = 0.7
mpl.rcParams["axes.edgecolor"]    = "#333333"
mpl.rcParams["axes.titleweight"]  = "bold"
mpl.rcParams["axes.titlepad"]     = 8
mpl.rcParams["xtick.major.width"] = 0.7
mpl.rcParams["ytick.major.width"] = 0.7
mpl.rcParams["legend.frameon"]    = False

FEATS_PANEL_A = [
    "atac", "crested", "phylop", "contrib",
    "gc", "skew", "kurt", "bimod", "tss",
]

FEAT_LABELS = {
    "atac":    "ATAC specificity",
    "crested": "CREsted 99 spec.",
    "phylop":  "PhyloP > 2 frac",
    "contrib": "Model motif contribution score",
    "gc":      "GC content",
    "skew":    "Skewness (target)",
    "kurt":    "Kurtosis (target)",
    "bimod":   "Bimodality (target)",
    "tss":     "log10 TSS dist.",
}
FEAT_COLORS = {
    "atac":    "#4878CF",
    "crested": "#5BA053",
    "phylop":  "#E08A2C",
    "contrib": "#C44E52",
    "gc":      "#8172B2",
    "skew":    "#937860",
    "kurt":    "#DA8BC3",
    "bimod":   "#8C8C8C",
    "tss":     "#4EA1A1",
}
BLUE, RED, GREY, LIGHT_GREY = "#3B7BB8", "#C44E52", "#7F7F7F", "#D9D9D9"


# ----------------------------------------------------------------- data load
def load_lr_df() -> pd.DataFrame:
    feats = pd.read_csv(FEATURES_TSV, sep="\t").rename(
        columns={"Enhancer ID": "enhancer_id"})
    feats["enhancer_id"] = feats["enhancer_id"].astype(str)
    feats = feats[["enhancer_id", "On Target", "own_group_proportion"]].copy()
    feats["On Target"] = feats["On Target"].str.lower()
    feats = feats[feats["On Target"].isin(["high", "medium", "low", "none"])].copy()
    feats["label"] = (feats["On Target"] != "none").astype(int)
    feats = feats.rename(columns={"own_group_proportion": "atac"})

    pred = pd.read_csv(PRED_SPEC_CSV)[["enhancer_id", "specificity"]].rename(
        columns={"specificity": "crested"})
    pred["enhancer_id"] = pred["enhancer_id"].astype(str)

    phy = pd.read_csv(PHYLOP_FRAC_CSV)[
        ["enhancer_id", "phyloP_conserved_frac"]].rename(
        columns={"phyloP_conserved_frac": "phylop"})
    phy["enhancer_id"] = phy["enhancer_id"].astype(str)

    contrib = pd.read_csv(CONTRIB_CSV)[["enhancer_id", "Species", "mean_contrib"]]
    contrib["enhancer_id"] = contrib["enhancer_id"].astype(str)
    contrib = contrib[contrib["Species"] == "macaque"].drop(columns="Species")
    contrib = contrib.rename(columns={"mean_contrib": "contrib"})

    # Panel A extras: pick the value for each enhancer's target cell type.
    raw = pd.read_csv(FEATURES_TSV, sep="\t").rename(
        columns={"Enhancer ID": "enhancer_id"})
    raw["enhancer_id"] = raw["enhancer_id"].astype(str)
    targets = raw["Cell type target"].astype(str)

    def _target_col(suffix):
        vals = np.full(len(raw), np.nan, dtype=float)
        for ct, idx in targets.groupby(targets).groups.items():
            col = f"{ct}_{suffix}"
            if col in raw.columns:
                vals[list(idx)] = raw.loc[idx, col].to_numpy(dtype=float)
        return vals

    extra = pd.DataFrame({
        "enhancer_id": raw["enhancer_id"],
        "gc":    raw["GC_content"],
        "skew":  _target_col("skewness"),
        "kurt":  _target_col("kurtosis"),
        "bimod": _target_col("bimodality"),
    })

    ex = pd.read_csv(EXTRA_FEAT_TSV, sep="\t").rename(
        columns={"Enhancer ID": "enhancer_id"})
    ex["enhancer_id"] = ex["enhancer_id"].astype(str)
    ex["tss"] = np.log10(ex["tss_distance_bp"].abs() + 1.0)
    ex = ex[["enhancer_id", "tss"]]

    df = (feats.merge(pred,    on="enhancer_id", how="inner")
                .merge(phy,     on="enhancer_id", how="left")
                .merge(contrib, on="enhancer_id", how="inner")
                .merge(extra,   on="enhancer_id", how="left")
                .merge(ex,      on="enhancer_id", how="left"))
    for c in FEATS_PANEL_A:
        df[c] = df[c].fillna(df[c].median())
    return df


def lr_cv(X, y, k, seed):
    n = X.shape[0]
    oof = np.zeros(n)
    fold_auc = np.zeros(k)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for i, (tr, te) in enumerate(skf.split(X, y)):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(
            penalty="l2", C=1.0, class_weight="balanced",
            solver="lbfgs", max_iter=1000, random_state=seed,
        )
        lr.fit(sc.transform(X[tr]), y[tr])
        prob = lr.predict_proba(sc.transform(X[te]))[:, 1]
        oof[te] = prob
        fold_auc[i] = roc_auc_score(y[te], prob)
    return oof, fold_auc


def univariate_cv_auc(x, y, k, seed):
    """Per-fold AUC for one feature; direction-invariant (max(auc, 1-auc))."""
    fold_auc = np.zeros(k)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for i, (_, te) in enumerate(skf.split(x, y)):
        a = roc_auc_score(y[te], x[te])
        fold_auc[i] = max(a, 1.0 - a)
    return float(fold_auc.mean()), float(fold_auc.std(ddof=1)), fold_auc


def compute_univariate_aucs(df):
    """Returns (order, means, sds) with features sorted by descending mean AUC."""
    y = df["label"].to_numpy()
    means, sds = {}, {}
    for f in FEATS_PANEL_A:
        x = df[f].to_numpy(dtype=float)
        m, s, _ = univariate_cv_auc(x, y, N_FOLDS, RNG_SEED)
        means[f] = m
        sds[f]   = s
    order = sorted(FEATS_PANEL_A, key=lambda k: means[k], reverse=True)
    return order, means, sds


# ----------------------------------------------------------------- Panel A
def feature_direction(df, feat):
    """+1 if feature is higher in on-target, -1 if higher in off-target."""
    on  = df.loc[df["label"] == 1, feat].to_numpy(dtype=float)
    off = df.loc[df["label"] == 0, feat].to_numpy(dtype=float)
    return 1 if np.nanmean(on) >= np.nanmean(off) else -1


def draw_panel_A(ax, df, order, means, sds):
    """Univariate CV AUC per feature; bar = mean, error bar = SD across folds.

    Arrow above each bar indicates direction of effect:
        \u25b2  higher in on-target enhancers
        \u25bc  higher in off-target enhancers
    """
    m_vals = [means[f] for f in order]
    s_vals = [sds[f]   for f in order]
    dirs   = [feature_direction(df, f) for f in order]

    xs = np.arange(len(order))
    colors = [FEAT_COLORS[f] for f in order]

    ax.bar(xs, m_vals, 0.65, yerr=s_vals, color=colors,
           edgecolor="white", linewidth=1.2,
           error_kw=dict(ecolor="#333333", lw=1.0, capsize=3),
           zorder=3)
    ax.axhline(0.5, color=GREY, lw=0.7, ls="--", alpha=0.8, zorder=1,
               label="chance (0.5)")

    y_top = max(m + s for m, s in zip(m_vals, s_vals))
    ax.set_ylim(0.45, y_top * 1.22)
    ax.set_xlim(-0.6, len(xs) - 0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([FEAT_LABELS[f] for f in order],
                       rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("AUC (univariate, CV)", fontsize=10)
    ax.set_title(
        f"Univariate predictive power  "
        f"(n = {len(df)}, {N_FOLDS}-fold CV, mean $\\pm$ SD)",
        fontsize=11)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(axis="y", color=LIGHT_GREY, lw=0.5, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for x, m, s, d in zip(xs, m_vals, s_vals, dirs):
        ax.text(x, m + s + 0.010, f"{m:.3f}", ha="center",
                fontsize=8.5, fontweight="bold", color="#222222")
        arrow  = "\u25b2" if d > 0 else "\u25bc"
        acolor = "#1A7F37" if d > 0 else "#B0242A"
        ax.text(x, m + s + 0.045, arrow, ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=acolor)

    # Legend with chance line + direction key.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=GREY, lw=0.7, ls="--", label="chance (0.5)"),
        Line2D([0], [0], marker="^", color="#1A7F37", lw=0, markersize=7,
               label="higher in on-target"),
        Line2D([0], [0], marker="v", color="#B0242A", lw=0, markersize=7,
               label="higher in off-target"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8.5,
              frameon=False, handlelength=1.8, borderpad=0.3,
              labelspacing=0.3)


# ----------------------------------------------------------------- Panel B
def draw_panel_B(ax, df, top_feats):
    """ROC — ATAC alone vs LR fit on the top-K features from panel A."""
    y = df["label"].to_numpy()
    X = df[top_feats].to_numpy(dtype=float)

    x_atac = df["atac"].to_numpy(dtype=float)
    fpr_a, tpr_a, _ = roc_curve(y, x_atac)
    auc_atac = roc_auc_score(y, x_atac)

    oof, fold_auc = lr_cv(X, y, N_FOLDS, RNG_SEED)
    fpr_l, tpr_l, _ = roc_curve(y, oof)
    auc_lr = roc_auc_score(y, oof)
    auc_mean = float(fold_auc.mean())
    auc_sem  = float(fold_auc.std(ddof=1) / np.sqrt(N_FOLDS))

    k = len(top_feats)
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=0.8, alpha=0.7,
            label="chance  (AUC = 0.5)", zorder=1)
    ax.plot(fpr_a, tpr_a, color=FEAT_COLORS["atac"], lw=2.2,
            label=f"ATAC specificity alone  (AUC = {auc_atac:.3f})", zorder=2)
    ax.plot(fpr_l, tpr_l, color="#1A1A1A", lw=2.8,
            label=f"LR top-{k} ({N_FOLDS}-fold OOF)  (AUC = {auc_lr:.3f})",
            zorder=3)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1.005)
    ax.set_xlabel("False positive rate", fontsize=10)
    ax.set_ylabel("True positive rate",  fontsize=10)
    feat_str = ", ".join(FEAT_LABELS[f] for f in top_feats)
    ax.set_title(
        f"ROC \u2014 ATAC alone vs LR top-{k} features\n"
        f"per-fold AUC = {auc_mean:.3f} $\\pm$ {auc_sem:.3f}",
        fontsize=11)
    ax.text(0.5, -0.22, f"top-{k}: {feat_str}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8.5, color="#555555", style="italic")
    ax.legend(loc="lower right", fontsize=9, frameon=False, handlelength=2.5)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(True, color=LIGHT_GREY, lw=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------- Panel C
def draw_panel_C(fig, gs_slot, df):
    feats  = ["atac", "crested"]
    titles = {"atac":    "ATAC proportion\nspecificity",
              "crested": "CREsted base\nprediction specificity"}
    ylabels = {"atac": "atac specificity", "crested": "pred specificity"}
    n_on  = int(df["label"].sum())
    n_off = int((1 - df["label"]).sum())

    sub_gs = gs_slot.subgridspec(1, len(feats), wspace=0.55)
    OFF_COLOR, ON_COLOR = "#D9D9D9", FEAT_COLORS["atac"]
    OFF_SCATTER, ON_SCATTER = "#888888", "#1f3a60"

    for k, f in enumerate(feats):
        ax = fig.add_subplot(sub_gs[0, k])
        off_vals = df.loc[df["label"] == 0, f].dropna().to_numpy()
        on_vals  = df.loc[df["label"] == 1, f].dropna().to_numpy()

        bp = ax.boxplot(
            [off_vals, on_vals], positions=[0, 1], widths=0.55,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="#222222", lw=1.4),
            whiskerprops=dict(color="#444444", lw=0.9),
            capprops=dict(color="#444444", lw=0.9),
            boxprops=dict(linewidth=0.9),
        )
        bp["boxes"][0].set_facecolor(OFF_COLOR); bp["boxes"][0].set_edgecolor("#666666")
        bp["boxes"][1].set_facecolor(ON_COLOR);  bp["boxes"][1].set_edgecolor("#1f3a60")

        rng = np.random.default_rng(42 + k)
        ax.scatter(rng.normal(0, 0.07, size=len(off_vals)), off_vals,
                   s=8, color=OFF_SCATTER, alpha=0.5, edgecolor="none", zorder=2)
        ax.scatter(1 + rng.normal(0, 0.07, size=len(on_vals)), on_vals,
                   s=8, color=ON_SCATTER, alpha=0.55, edgecolor="none", zorder=2)

        _, pval = mannwhitneyu(on_vals, off_vals, alternative="two-sided")
        stars = ("***" if pval < 1e-3 else "**" if pval < 1e-2
                 else "*" if pval < 5e-2 else "ns")

        y_top = max(off_vals.max(), on_vals.max())
        y_bot = min(off_vals.min(), on_vals.min())
        y_pad = 0.07 * (y_top - y_bot + 1e-9)
        y_br  = y_top + y_pad
        ax.plot([0, 0, 1, 1],
                [y_br, y_br + y_pad*0.5, y_br + y_pad*0.5, y_br],
                color="#333333", lw=0.9)
        ax.text(0.5, y_br + y_pad*0.9, stars,
                ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.text(0.5, y_br + y_pad*2.6, f"P = {pval:.3g}",
                ha="center", va="bottom", fontsize=9, color="#444444")
        ax.set_ylim(top=y_br + y_pad*5.0)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"Off\n(n={n_off})", f"On\n(n={n_on})"], fontsize=10)
        ax.set_ylabel(ylabels[f], fontsize=10)
        ax.set_title(titles[f], fontsize=11)
        ax.grid(axis="y", color=LIGHT_GREY, lw=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)

        if k == 0:
            ax.text(-0.28, 1.06, "C", transform=ax.transAxes,
                    fontsize=18, weight="bold", ha="left", va="bottom",
                    color="#111111")


# ----------------------------------------------------------------- Panel D
def draw_panel_D(ax, g):
    g = g[(g["n_on"] >= 2) & (g["n_off"] >= 2)].copy()
    g["delta"] = g["model_ratio"] - g["atac_ratio"]
    g = g.sort_values("delta", ascending=False).reset_index(drop=True)
    y = np.arange(len(g))
    row_color = np.where(g["delta"].values > 0, BLUE, RED)

    XMIN, XMAX = 0.4, 3.0
    ax.axvline(1.0, color="gray", lw=0.7, alpha=0.7)

    for i in range(len(g)):
        c = row_color[i]
        xm = float(g["model_ratio"].iloc[i])
        xa = float(g["atac_ratio"].iloc[i])
        ax.plot([min(xm, XMAX), min(xa, XMAX)], [y[i], y[i]],
                color=c, lw=2.2, alpha=0.85, zorder=1)
        if xm > XMAX:
            ax.scatter([XMAX], [y[i]], marker=">", s=70,
                       facecolors="white", edgecolors=c, linewidths=1.6,
                       zorder=3, clip_on=False)
            ax.text(XMAX + 0.04, y[i], f"{xm:.1f}",
                    va="center", ha="left", fontsize=8, color=c)
        else:
            ax.scatter([xm], [y[i]], s=45, facecolors="white",
                       edgecolors=c, linewidths=1.6, zorder=3)
        if xa > XMAX:
            ax.scatter([XMAX], [y[i]], marker=">", s=70,
                       facecolors=c, edgecolors=c, linewidths=1.6,
                       zorder=3, clip_on=False)
            ax.text(XMAX + 0.04, y[i] + 0.18, f"{xa:.1f}",
                    va="center", ha="left", fontsize=8, color=c)
        else:
            ax.scatter([xa], [y[i]], s=45, facecolors=c,
                       edgecolors=c, linewidths=1.6, zorder=3)

    labels = [f"{t}  (on={non}, off={noff})"
              for t, non, noff in zip(g["Cell type target"], g["n_on"], g["n_off"])]
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean rated-on / mean none-rated specificity  (same target)", fontsize=10)
    ax.set_xlim(XMIN, XMAX)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=LIGHT_GREY, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title(
        "On/off specificity per cell type  "
        "(filtered: n_on $\\geq$ 2 and n_off $\\geq$ 2)", fontsize=11)

    h_model = plt.Line2D([0], [0], marker="o", linestyle="None",
                         markerfacecolor="white", markeredgecolor="#222222",
                         markeredgewidth=1.4, markersize=8, label="Model (base 99)")
    h_atac  = plt.Line2D([0], [0], marker="o", linestyle="None",
                         markerfacecolor="#222222", markeredgecolor="#222222",
                         markersize=8, label="ATAC")
    h_blue  = plt.Line2D([0], [0], marker="s", linestyle="None",
                         markerfacecolor=BLUE, markeredgecolor="none",
                         markersize=8, label="Model > ATAC")
    h_red   = plt.Line2D([0], [0], marker="s", linestyle="None",
                         markerfacecolor=RED, markeredgecolor="none",
                         markersize=8, label="ATAC > Model")
    ax.legend(handles=[h_model, h_atac, h_blue, h_red],
              loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=4, fontsize=9, frameon=False,
              handletextpad=0.4, columnspacing=1.4, borderpad=0.4)


# ----------------------------------------------------------------- assemble
def main():
    for p in (FEATURES_TSV, EXTRA_FEAT_TSV, PRED_SPEC_CSV, PHYLOP_FRAC_CSV,
              CONTRIB_CSV, ON_OFF_CSV):
        if not os.path.exists(p):
            sys.exit(f"missing input: {p}")

    if not OVERWRITE:
        existing = [p for p in (OUT_PDF, OUT_SVG, OUT_PNG) if os.path.exists(p)]
        if existing:
            sys.exit("refusing to overwrite:\n  " + "\n  ".join(existing) +
                     "\nset OVERWRITE = True to allow.")

    df = load_lr_df()
    g  = pd.read_csv(ON_OFF_CSV)

    order, means, sds = compute_univariate_aucs(df)
    top_feats = order[:TOP_K]

    fig = plt.figure(figsize=(17.0, 12.5), facecolor="white")
    gs = fig.add_gridspec(
        nrows=2, ncols=4,
        height_ratios=[1.05, 1.30],
        hspace=0.70, wspace=0.42,
        left=0.16, right=0.97, top=0.93, bottom=0.11,
    )

    ax_A = fig.add_subplot(gs[0, 0:2])
    ax_B = fig.add_subplot(gs[0, 2:4])

    gs_row2 = gs[1, :].subgridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.65)
    gs_C = gs_row2[0, 0]
    ax_D = fig.add_subplot(gs_row2[0, 1])

    for ax, letter in [(ax_A, "A"), (ax_B, "B"), (ax_D, "D")]:
        ax.text(-0.085, 1.06, letter, transform=ax.transAxes,
                fontsize=18, weight="bold", ha="left", va="bottom",
                color="#111111")

    draw_panel_A(ax_A, df, order, means, sds)
    draw_panel_B(ax_B, df, top_feats)
    draw_panel_C(fig, gs_C, df)
    draw_panel_D(ax_D, g)

    fig.suptitle(
        "Macaque enhancer scoring \u2014 panels A, B, C, D",
        fontsize=15, y=0.985, color="#222222", weight="semibold")

    fig.savefig(OUT_PDF)
    fig.savefig(OUT_SVG)
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print("saved:", OUT_PDF, OUT_SVG, OUT_PNG, sep="\n  ")


if __name__ == "__main__":
    main()
