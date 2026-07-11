"""NUDAP restoration panel — Illustrator-editable, standalone version.

Reproduces the two-panel NUDAP figure (contribution stack + restoration
bar chart) fully as native vector objects, using only the cached CSVs
shipped alongside the group stack. No keras / no model compute — the
per-variant NUDAP predictions are read from
``nudap_restoration_predictions.csv``.

Panel A  (top)
    Three contribution-score rows (macaque / human / mouse) aligned to
    the shared MSA, with:
      * yellow shading + labelled downward arrow for each
        LOST-in-mouse region (L0..L2)
      * purple shading + labelled downward arrow for each
        GAINED-in-mouse region (G0..G2)
      * mouse-row overlay: the actual mouse base is drawn at every
        column inside a region; if it differs from the q+h consensus
        (or is a gap) it is drawn in RED, otherwise in black.

Panel B  (bottom)
    Two stacked bar charts sharing the same x-axis (7 bars: L0..L2,
    G0..G2, ALL):
      * absolute NUDAP prediction with horizontal reference lines for
        mouse-WT / macaque-WT / human-WT
      * Delta NUDAP (restored - mouse WT) with horizontal reference
        lines for macaque - mouse and human - mouse

------------------------------------------------------------------------
Required inputs (--phylop-dir defaults to the workspace-local copy)
------------------------------------------------------------------------
    <phylop-dir>/make_group_stacks.py                       (imported)
    <phylop-dir>/plot_seq_phylop.py                         (imported)
    <phylop-dir>/seq_phylop_plots/groups/strv_d1_nudap/
        fimo_hits.csv
    <phylop-dir>/seq_phylop_plots/nudap_lost_in_mouse/
        nudap_lost_in_mouse_regions.csv
        nudap_gained_in_mouse_regions.csv
        nudap_restoration_predictions.csv
    plus whatever ``make_group_stacks.load_combined_npz`` /
    ``load_coords`` / ``load_standardized_seqs`` expect (the master
    contribution / one-hot bundle for AiE1485q/h/m).

------------------------------------------------------------------------
Usage
------------------------------------------------------------------------
    # Copy this script into your own working directory so your outputs
    # go there, then activate the crested env (only needed for the
    # sibling helpers, no keras is actually loaded):

    source /allen/scratch/aibstemp/miniconda3/etc/profile.d/conda.sh
    conda activate /allen/programs/celltypes/workgroups/hct/sarojaS/\\
        250506_bg_figure/merged_Fig_7and8/PyPeakRankr/pattern_analysis/\\
        2502_CREsted_tutorial/hct_run/hpc_run/latest_crested_env

    # Default: writes nudap_restoration_panel.{pdf,svg,png} next to the
    # script. Refuses to overwrite existing files unless --force.
    python nudap_restoration_panel.py

    # Custom output directory + overwrite:
    python nudap_restoration_panel.py \\
        --out-dir ~/nudap_figs --force

    # Custom source data root:
    python nudap_restoration_panel.py \\
        --phylop-dir /path/to/phylop_analysis

Outputs (written to --out-dir, defaults to the script's own directory):
    nudap_restoration_panel.pdf         Illustrator-editable vector
    nudap_restoration_panel.svg         same, SVG
    nudap_restoration_panel.png         300 dpi raster preview
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec


# ── Defaults (override via CLI flags) ────────────────────────────────────
DEFAULT_PHYLOP_DIR = Path(
    "/allen/programs/celltypes/workgroups/hct/sarojaS/260604_grammar_flex/"
    "phylop_analysis"
)
DEFAULT_CRESTED_UTILS = Path(
    "/home/sarojas/.local/lib/python3.9/site-packages/"
    "crested/pl/patterns/_utils.py"
)

SCRIPT_STEM = "nudap_restoration_panel"

# ── Constants ────────────────────────────────────────────────────────────
IDS = ["AiE1485q", "AiE1485h", "AiE1485m"]
SPECIES = {"AiE1485q": "macaque (rheMac10)",
           "AiE1485h": "human (hg38)",
           "AiE1485m": "mouse (mm10)"}
NUDAP_CLASS_IDX = 51
NUDAP_CLASS_NAME = "STRv_D1_NUDAP_MSN"

LOST_COLOR = "#ffcc00"    # yellow — LOST in mouse
GAIN_COLOR = "#9467bd"    # purple — GAINED in mouse
ALL_COLOR  = "#888888"    # grey   — all restored simultaneously
EDIT_COLOR = "#d62728"    # red    — mouse base differs from q+h consensus
REGION_ALPHA = 0.18
GAP_COLOR = "0.88"
BASE_FONT = "DejaVu Sans Mono"


# ── One-time module setup (imports helpers from the phylop project) ──────
def import_helpers(phylop_dir: Path, crested_utils_path: Path):
    """Return (mgs_module, plot_seq_phylop_helpers, grad_and_plot_fns)."""
    if not phylop_dir.exists():
        raise SystemExit(f"phylop-dir does not exist: {phylop_dir}")
    sys.path.insert(0, str(phylop_dir))
    import make_group_stacks as mgs                              # noqa: F401
    from plot_seq_phylop import (                                # noqa: F401
        _msa_mafft, _nat_idx_from_aligned, _three_way_align,
        _project_to_msa, _project_hits, _seq_from_one_hot,
    )
    if not crested_utils_path.exists():
        raise SystemExit(
            "crested contribution-utils not found: "
            f"{crested_utils_path}\n"
            "Pass --crested-utils to point at the local install "
            "(crested/pl/patterns/_utils.py).")
    spec = importlib.util.spec_from_file_location(
        "_crested_utils_local", crested_utils_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plot_seq_phylop_helpers = {
        "_msa_mafft": _msa_mafft,
        "_nat_idx_from_aligned": _nat_idx_from_aligned,
        "_three_way_align": _three_way_align,
        "_project_to_msa": _project_to_msa,
        "_project_hits": _project_hits,
        "_seq_from_one_hot": _seq_from_one_hot,
    }
    grad_and_plot = {
        "grad_times_input_to_df": module.grad_times_input_to_df,
        "_plot_attribution_map":  module._plot_attribution_map,
    }
    return mgs, plot_seq_phylop_helpers, grad_and_plot


# ── Illustrator-friendly vector defaults ─────────────────────────────────
def apply_rc():
    mpl.rcParams["pdf.fonttype"]      = 42
    mpl.rcParams["ps.fonttype"]       = 42
    mpl.rcParams["svg.fonttype"]      = "none"
    mpl.rcParams["font.family"]       = "DejaVu Sans"
    mpl.rcParams["axes.linewidth"]    = 0.7
    mpl.rcParams["axes.edgecolor"]    = "#333333"
    mpl.rcParams["xtick.major.width"] = 0.7
    mpl.rcParams["ytick.major.width"] = 0.7


# ── Data loaders ─────────────────────────────────────────────────────────
def load_fimo_hits(fimo_csv, native_lengths):
    out = {eid: [] for eid in native_lengths}
    if not fimo_csv.exists():
        return out
    df = pd.read_csv(fimo_csv)
    for _, r in df.iterrows():
        eid = str(r["sequence_name"])
        if eid not in native_lengths:
            continue
        a = max(0, int(r["start"]) - 1)
        b = min(native_lengths[eid], int(r["stop"]))
        if b > a:
            out[eid].append((str(r["motif_id"]), a, b,
                             str(r.get("strand", "+")),
                             float(r.get("score", 0.0))))
    return out


def build_bundles(mgs, phylop_dir, fimo_csv):
    npz      = mgs.load_combined_npz()
    coords   = mgs.load_coords()
    std_seqs = mgs.load_standardized_seqs()

    bundles = {}
    for eid in IDS:
        oh, sc, s_start, s_end, flipped = mgs.trim_and_orient(
            eid, npz, std_seqs, verbose=True)
        seq = mgs.one_hot_to_seq(oh)
        pp  = mgs.aligned_phylop(eid, oh.shape[0], coords,
                                 phylop_dir / "seq_phylop_plots", flipped)
        bundles[eid] = {
            "one_hot": oh, "scores": sc, "seq": seq, "pp": pp,
            "flipped": flipped, "s_start": s_start, "s_end": s_end,
            "L": oh.shape[0],
        }
    native_lengths = {eid: b["L"] for eid, b in bundles.items()}
    for eid, hits in load_fimo_hits(fimo_csv, native_lengths).items():
        bundles[eid]["hits"] = hits
    return bundles


def align_bundles(bundles, helpers):
    seqs = [helpers["_seq_from_one_hot"](bundles[e]["one_hot"]) for e in IDS]
    rows = helpers["_msa_mafft"](seqs, IDS)
    if rows is not None:
        return rows, [helpers["_nat_idx_from_aligned"](r) for r in rows]
    return helpers["_three_way_align"](seqs[0], seqs[1], seqs[2])


def project_to_msa(bundles, rows, nats, helpers):
    n, L_aln = len(IDS), len(rows[0])
    scores_aln  = np.zeros((n, L_aln, 4))
    one_hot_aln = np.zeros((n, L_aln, 4))
    for i, eid in enumerate(IDS):
        sc, oh, _pp = helpers["_project_to_msa"](bundles[eid], nats[i])
        scores_aln[i], one_hot_aln[i] = sc, oh
    return scores_aln, one_hot_aln


def load_regions(lost_csv, gain_csv):
    lost = pd.read_csv(lost_csv)
    gain = pd.read_csv(gain_csv)
    lost["direction"] = "lost_in_mouse"
    gain["direction"] = "gained_in_mouse"
    lost["label"] = [f"L{int(r)}" for r in lost["region"]]
    gain["label"] = [f"G{int(r)}" for r in gain["region"]]
    keep = ["label", "direction", "region",
            "msa_start", "msa_end",
            "macaque_seq", "human_seq", "mouse_seq",
            "n_sub_mouse", "n_del_in_mouse", "n_ins_in_mouse",
            "mouse_total_edits"]
    regs = pd.concat([lost[keep], gain[keep]], ignore_index=True)
    return regs.reset_index(drop=True)


# ── Panel A: contribution stack + shading + arrows + mouse letters ────────
def draw_panel_A(fig, gs_slot, bundles, rows, scores_aln, one_hot_aln,
                 regions, grad_and_plot):
    n = len(IDS)
    L_aln = len(rows[0])

    grad_df = grad_and_plot["grad_times_input_to_df"]
    plot_attr = grad_and_plot["_plot_attribution_map"]

    saliency_dfs = [grad_df(one_hot_aln[i], scores_aln[i]) for i in range(n)]
    y_min = min((scores_aln[i] * one_hot_aln[i]).min() for i in range(n))
    y_max = max((scores_aln[i] * one_hot_aln[i]).max() for i in range(n))
    y_min -= 0.25 * abs(y_min)
    y_max += 0.25 * abs(y_max)

    inner = gs_slot.subgridspec(n, 1, hspace=0.38)
    axs = []
    for i in range(n):
        ax = fig.add_subplot(inner[i, 0])
        if axs:
            ax.sharex(axs[0])
        axs.append(ax)

    for i, ax in enumerate(axs):
        plot_attr(saliency_dfs[i], ax=ax, return_ax=False,
                  figsize=None, spines=False)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(-0.5, L_aln - 0.5)
        ax.set_xticks(np.arange(0, L_aln, 50))
        ax.tick_params(axis="y", labelsize=7)
        ax.set_ylabel("contrib.", fontsize=8)
        eid = IDS[i]
        n_gaps = rows[i].count("-")
        ax.set_title(
            f"{eid}  ({SPECIES[eid]})   L={bundles[eid]['L']} bp   "
            f"MSA gaps: {n_gaps}/{L_aln}",
            fontsize=9.5, loc="left", color="0.10", fontweight="bold")

        for c, ch in enumerate(rows[i]):
            if ch == "-":
                ax.axvspan(c - 0.5, c + 0.5, color=GAP_COLOR,
                           alpha=0.55, lw=0, zorder=0)

    q_ax = axs[0]
    y0, y1 = q_ax.get_ylim()
    q_ax.set_ylim(y0, y1 + 0.32 * (y1 - y0))

    for _, r in regions.iterrows():
        s, e = int(r["msa_start"]), int(r["msa_end"])
        color = LOST_COLOR if r["direction"] == "lost_in_mouse" else GAIN_COLOR
        for ax in axs:
            ax.axvspan(s - 0.5, e - 0.5,
                       facecolor=color, alpha=REGION_ALPHA,
                       edgecolor="none", zorder=0.5)
        ya, yb = q_ax.get_ylim()
        arr_top = yb - 0.05 * (yb - ya)
        arr_bot = yb - 0.20 * (yb - ya)
        mid = (s + e - 1) / 2.0
        q_ax.annotate(
            "",
            xy=(mid, arr_bot), xytext=(mid, arr_top),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                            mutation_scale=15),
            zorder=6,
        )
        q_ax.text(
            mid, arr_top + 0.02 * (yb - ya), r["label"],
            ha="center", va="bottom",
            fontsize=7.5, color="#111111", fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.6, edgecolor="none", pad=1.5),
            zorder=7,
        )

    m_ax = axs[2]
    y0m, y1m = m_ax.get_ylim()
    m_ax.set_ylim(y0m, y1m + 0.38 * (y1m - y0m))
    y0m2, y1m2 = m_ax.get_ylim()
    text_y = y1m2 - 0.09 * (y1m2 - y0m2)
    # Scale with per-column width; wider figure -> larger letters.
    fig_w_in = fig.get_size_inches()[0]
    pt_per_col = fig_w_in * 72.0 / max(1, L_aln)
    fontsize = float(np.clip(pt_per_col * 1.4, 5.5, 9.0))
    for _, r in regions.iterrows():
        s, e = int(r["msa_start"]), int(r["msa_end"])
        for col in range(s, e):
            qch, hch, mch = rows[0][col], rows[1][col], rows[2][col]
            consensus = qch if (qch == hch and qch != "-") else "N"
            is_edit = (
                (mch == "-" and (qch != "-" or hch != "-")) or
                (consensus != "N" and mch != consensus and mch != "-")
            )
            color_t = EDIT_COLOR if is_edit else "black"
            display = mch if mch != "-" else "\u2212"
            m_ax.text(col, text_y, display,
                      ha="center", va="center",
                      fontsize=fontsize, color=color_t,
                      fontweight="bold" if is_edit else "normal",
                      family=BASE_FONT, zorder=5)

    axs[-1].set_xlabel("Alignment position (bp)", fontsize=9)
    return axs


# ── Panel B: two bar charts (absolute NUDAP, Δ NUDAP) ────────────────────
def _bar_order(regions):
    order, labels, kinds = [], [], []
    for direction in ("lost_in_mouse", "gained_in_mouse"):
        sub = regions[regions["direction"] == direction].sort_values("region")
        for _, r in sub.iterrows():
            order.append(f"mouse_restore_{r['label']}")
            labels.append(r["label"])
            kinds.append(direction)
    order.append("mouse_restore_ALL")
    labels.append("ALL")
    kinds.append("combined")
    return order, labels, kinds


def draw_panel_B(fig, gs_slot, regions, pred_df):
    order, labels, kinds = _bar_order(regions)
    df = pred_df.set_index("variant").loc[order].reset_index()

    colors = [{"lost_in_mouse": LOST_COLOR,
               "gained_in_mouse": GAIN_COLOR,
               "combined": ALL_COLOR}[k] for k in kinds]
    n_subs = df["n_subs"].astype(int).to_numpy()
    x = np.arange(len(df))
    xtick_labels = [f"{lbl}\n(n_subs={n})" for lbl, n in zip(labels, n_subs)]

    mouse_wt   = float(pred_df.loc[pred_df["variant"] == "mouse_WT",   "nudap_pred"].iloc[0])
    macaque_wt = float(pred_df.loc[pred_df["variant"] == "macaque_WT", "nudap_pred"].iloc[0])
    human_wt   = float(pred_df.loc[pred_df["variant"] == "human_WT",   "nudap_pred"].iloc[0])

    inner = gs_slot.subgridspec(2, 1, hspace=0.32)
    ax_abs   = fig.add_subplot(inner[0, 0])
    ax_delta = fig.add_subplot(inner[1, 0], sharex=ax_abs)

    ax_abs.bar(x, df["nudap_pred"].to_numpy(), width=0.72,
               color=colors, edgecolor="black", linewidth=0.6, zorder=3)
    ax_abs.axhline(mouse_wt,   color="black",   lw=1.0, ls="--",
                   label=f"mouse WT ({mouse_wt:.3f})",   zorder=2)
    ax_abs.axhline(macaque_wt, color="#1f77b4", lw=1.0, ls=":",
                   label=f"macaque WT ({macaque_wt:.3f})", zorder=2)
    ax_abs.axhline(human_wt,   color="#d62728", lw=1.0, ls=":",
                   label=f"human WT ({human_wt:.3f})",   zorder=2)
    ax_abs.set_ylabel(f"NUDAP prediction\n(class idx {NUDAP_CLASS_IDX})",
                      fontsize=9)
    ax_abs.set_title(
        "NUDAP CREsted prediction \u2014 mouse with each region "
        "restored to q+h consensus",
        fontsize=10, loc="left", color="0.10", fontweight="bold", pad=6)
    ax_abs.legend(loc="upper left", fontsize=7.5, frameon=False,
                  handlelength=2.0, borderaxespad=0.3)
    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels([""] * len(x))
    ax_abs.yaxis.grid(True, color="0.90", lw=0.5, zorder=0)
    ax_abs.set_axisbelow(True)
    for sp in ("top", "right"):
        ax_abs.spines[sp].set_visible(False)

    ymax_abs = ax_abs.get_ylim()[1]
    for xi, v in zip(x, df["nudap_pred"].to_numpy()):
        ax_abs.text(xi, v + 0.015 * ymax_abs, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7.5, color="0.15",
                    zorder=4)

    delta = df["delta_vs_mouse_WT"].to_numpy()
    ax_delta.bar(x, delta, width=0.72,
                 color=colors, edgecolor="black", linewidth=0.6, zorder=3)
    ax_delta.axhline(0, color="black", lw=0.8, zorder=2)
    ax_delta.axhline(macaque_wt - mouse_wt,
                     color="#1f77b4", lw=1.0, ls=":",
                     label=f"macaque \u2212 mouse ({macaque_wt - mouse_wt:+.3f})",
                     zorder=2)
    ax_delta.axhline(human_wt - mouse_wt,
                     color="#d62728", lw=1.0, ls=":",
                     label=f"human \u2212 mouse ({human_wt - mouse_wt:+.3f})",
                     zorder=2)
    ax_delta.set_ylabel("\u0394 NUDAP\n(restored \u2212 mouse WT)",
                        fontsize=9)
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels(xtick_labels, fontsize=8, color="0.10")
    ax_delta.tick_params(axis="x", length=0, pad=3)
    ax_delta.legend(loc="upper left", fontsize=7.5, frameon=False,
                    handlelength=2.0, borderaxespad=0.3)
    ax_delta.yaxis.grid(True, color="0.90", lw=0.5, zorder=0)
    ax_delta.set_axisbelow(True)
    for sp in ("top", "right"):
        ax_delta.spines[sp].set_visible(False)

    lim_abs = float(np.nanmax(np.abs(delta))) if delta.size else 1.0
    ax_delta.set_ylim(-lim_abs * 0.30, lim_abs * 1.30)

    y_off_d = 0.03 * lim_abs
    for xi, v in zip(x, delta):
        va = "bottom" if v >= 0 else "top"
        ax_delta.text(xi, v + (y_off_d if v >= 0 else -y_off_d),
                      f"{v:+.1f}",
                      ha="center", va=va, fontsize=7.5, color="0.15",
                      zorder=4)

    ax_abs.set_xlim(-0.7, len(x) - 0.3)
    return ax_abs, ax_delta


# ── CLI ─────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description=("Standalone NUDAP restoration panel — "
                     "contribution stack + Δ prediction bars, "
                     "Illustrator-editable vector output."))
    p.add_argument(
        "--phylop-dir", type=Path, default=DEFAULT_PHYLOP_DIR,
        help=("Root of the phylop analysis project containing "
              "make_group_stacks.py + seq_phylop_plots/. "
              f"Default: {DEFAULT_PHYLOP_DIR}"))
    p.add_argument(
        "--crested-utils", type=Path, default=DEFAULT_CRESTED_UTILS,
        help=("Path to crested/pl/patterns/_utils.py (used for the "
              "attribution logo plotter). "
              f"Default: {DEFAULT_CRESTED_UTILS}"))
    p.add_argument(
        "--out-dir", type=Path, default=Path(__file__).resolve().parent,
        help=("Directory for the output PDF/SVG/PNG. "
              "Default: directory of this script."))
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output files.")
    return p.parse_args()


def main():
    args = parse_args()

    phylop_dir = args.phylop_dir
    nudap_dir  = phylop_dir / "seq_phylop_plots" / "nudap_lost_in_mouse"
    group_dir  = phylop_dir / "seq_phylop_plots" / "groups" / "strv_d1_nudap"

    fimo_csv          = group_dir / "fimo_hits.csv"
    lost_regions_csv  = nudap_dir / "nudap_lost_in_mouse_regions.csv"
    gain_regions_csv  = nudap_dir / "nudap_gained_in_mouse_regions.csv"
    preds_csv         = nudap_dir / "nudap_restoration_predictions.csv"

    for p in (fimo_csv, lost_regions_csv, gain_regions_csv, preds_csv):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / f"{SCRIPT_STEM}.pdf"
    out_svg = out_dir / f"{SCRIPT_STEM}.svg"
    out_png = out_dir / f"{SCRIPT_STEM}.png"
    existing = [p for p in (out_pdf, out_svg, out_png) if p.exists()]
    if existing and not args.force:
        raise SystemExit(
            "refusing to overwrite existing output files:\n  "
            + "\n  ".join(str(p) for p in existing)
            + "\npass --force to allow overwrite.")

    apply_rc()
    mgs, helpers, grad_and_plot = import_helpers(
        phylop_dir, args.crested_utils)

    bundles = build_bundles(mgs, phylop_dir, fimo_csv)
    rows, nats = align_bundles(bundles, helpers)
    scores_aln, one_hot_aln = project_to_msa(bundles, rows, nats, helpers)
    regions = load_regions(lost_regions_csv, gain_regions_csv)
    pred_df = pd.read_csv(preds_csv)

    L_aln = len(rows[0])
    # Per-column width so the letters inside the contribution logo are not
    # squished. ~0.05 in / column gives comfortably readable bases; capped
    # so absurdly long alignments still render.
    fig_w = max(18.0, min(42.0, L_aln * 0.05 + 4.0))
    fig_h = 16.0
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    gs = gridspec.GridSpec(
        nrows=2, ncols=1, figure=fig,
        height_ratios=[4.2, 2.2],
        left=0.045, right=0.99, top=0.95, bottom=0.055,
        hspace=0.30,
    )

    draw_panel_A(fig, gs[0, 0], bundles, rows, scores_aln, one_hot_aln,
                 regions, grad_and_plot)
    # Center panel B horizontally: only the middle column of a 3-col grid
    # holds the bars, so widening the figure for panel A doesn't stretch
    # the bars into slabs.
    gs_B = gs[1, 0].subgridspec(1, 3, width_ratios=[1.0, 3.2, 1.0],
                                wspace=0.0)
    ax_abs, _ax_delta = draw_panel_B(fig, gs_B[0, 1], regions, pred_df)

    fig.text(0.008, 0.945, "A", fontsize=20, fontweight="bold",
             ha="left", va="top", color="0.10")
    bbox_abs = ax_abs.get_position()
    fig.text(0.008, bbox_abs.y1 + 0.008, "B",
             fontsize=20, fontweight="bold",
             ha="left", va="top", color="0.10")

    fig.suptitle(
        f"NUDAP (AiE1485) \u2014 full alignment ({L_aln} cols); "
        "yellow = LOST in mouse, purple = GAINED in mouse, "
        "red letters = mouse substitution/gap vs q+h consensus",
        fontsize=10, y=0.982, color="0.10")

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[save] {out_pdf}")
    print(f"[save] {out_svg}")
    print(f"[save] {out_png}")


if __name__ == "__main__":
    main()
