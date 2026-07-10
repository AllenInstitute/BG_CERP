"""
Fig 7 add-ons: composes four supplementary panels the user will drop
into the existing Figure 7.ai:

    A  Conserved-base count per peak (macaque On vs Off) — right subplot
       of plot_phylop_conserved_bar.py.  Native / Illustrator-editable.

    B  phyloP + pattern stack for OT_D1_ICj
       (phylop_analysis/.../groups/ot_d1_icj/stack.pdf, rasterized).

    C  phyloP + pattern stack for STRv_D1_NUDAP
       (phylop_analysis/.../groups/strv_d1_nudap/stack.pdf, rasterized).

    D  phyloP + pattern stack for STR_SST-CHODL_GABA
       (phylop_analysis/.../groups/str_sst_chodl/stack.pdf, rasterized).

    E  NUDAP lost-in-mouse in-silico restoration panels A+B
       (phylop_analysis/.../nudap_lost_in_mouse/nudap_panels_AB_oneup.pdf,
        rasterized).

Outputs (illustrator/):
    fig7_addons.pdf   fig7_addons.svg   fig7_addons.png
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import stats

import pypdfium2 as pdfium

# Illustrator-friendly text in vector outputs.
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.family"] = "DejaVu Sans"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

FEATURES_TSV = ROOT / "macaque_209_features.tsv"
PHYLOP_CSV = ROOT / "macaque_209_phylop_conserved.csv"

STACK_ICJ = ROOT / "phylop_analysis" / "seq_phylop_plots" / "groups" / "ot_d1_icj" / "stack.pdf"
STACK_NUDAP = ROOT / "phylop_analysis" / "seq_phylop_plots" / "groups" / "strv_d1_nudap" / "stack.pdf"
STACK_SST_CHODL = ROOT / "phylop_analysis" / "seq_phylop_plots" / "groups" / "str_sst_chodl" / "stack.pdf"
NUDAP_AB = ROOT / "phylop_analysis" / "seq_phylop_plots" / "nudap_lost_in_mouse" / "nudap_panels_AB_oneup.pdf"

OUT_PDF = HERE / "fig7_addons.pdf"
OUT_SVG = HERE / "fig7_addons.svg"
OUT_PNG = HERE / "fig7_addons.png"

ON_COLOR = "#08519c"
OFF_COLOR = "#bdbdbd"


def render_pdf(pdf_path: Path, dpi: int = 300) -> np.ndarray:
    """Rasterize the first page of a PDF into an RGB image array."""
    doc = pdfium.PdfDocument(str(pdf_path))
    page = doc[0]
    scale = dpi / 72.0
    pil = page.render(scale=scale).to_pil().convert("RGB")
    return np.asarray(pil)


def draw_count_bar(ax):
    feats = pd.read_csv(FEATURES_TSV, sep="\t").rename(
        columns={"Enhancer ID": "enhancer_id"})
    feats["enhancer_id"] = feats["enhancer_id"].astype(str)
    feats["On Target"] = feats["On Target"].str.lower()
    phy = pd.read_csv(PHYLOP_CSV)
    phy["enhancer_id"] = phy["enhancer_id"].astype(str)
    df = feats.merge(phy, on="enhancer_id", how="left").dropna(
        subset=["phyloP_conserved_frac"]).copy()
    df["group"] = np.where(df["On Target"] != "none", "On", "Off")
    on = df[df["group"] == "On"]["phyloP_conserved_bp"].to_numpy()
    off = df[df["group"] == "Off"]["phyloP_conserved_bp"].to_numpy()
    n_on, n_off = len(on), len(off)
    m_on, s_on = on.mean(), on.std(ddof=1) / np.sqrt(n_on)
    m_off, s_off = off.mean(), off.std(ddof=1) / np.sqrt(n_off)
    _, p = stats.mannwhitneyu(on, off, alternative="two-sided")

    positions = [1, 2]
    ax.bar(positions, [m_off, m_on], width=0.6,
           color=[OFF_COLOR, ON_COLOR], edgecolor="black", linewidth=1.0)
    ax.errorbar(positions, [m_off, m_on], yerr=[s_off, s_on],
                fmt="none", ecolor="black", capsize=4, lw=1.0)

    rng = np.random.default_rng(42)
    for pos, vals, c in zip(positions, [off, on], [OFF_COLOR, ON_COLOR]):
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals,
                   s=10, color=c, edgecolor="black", lw=0.25,
                   alpha=0.6, zorder=3)

    top = max(on.max(), off.max())
    bar_y = top * 1.05
    h = top * 0.025
    ax.plot([1, 1, 2, 2], [bar_y, bar_y + h, bar_y + h, bar_y],
            lw=1.0, color="black")
    stars = "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 5e-2 else "n.s."))
    ax.text(1.5, bar_y + h + top * 0.01,
            f"{stars}\nP = {p:.3g}",
            ha="center", va="bottom", fontsize=10)
    for pos, m in zip(positions, [m_off, m_on]):
        ax.text(pos, m * 0.5, f"{m:.1f}",
                ha="center", va="center", fontsize=10,
                color="white", fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels([f"Off\n(n={n_off})", f"On\n(n={n_on})"], fontsize=10)
    ax.set_ylabel("Bases with phyloP > 2 (count)", fontsize=10)
    ax.set_title("A  Conserved-base count per peak\n"
                 f"macaque (n={len(df)}), On = high/med/low, Off = none",
                 fontsize=10, loc="left", fontweight="bold")
    ax.set_ylim(0, bar_y + h + top * 0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3)


def draw_pdf_panel(ax, pdf_path: Path, letter: str, caption: str):
    img = render_pdf(pdf_path, dpi=300)
    ax.imshow(img, aspect="auto", interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"{letter}  {caption}",
                 fontsize=10, loc="left", fontweight="bold")


def main():
    # width chosen so wide PDFs remain legible
    W = 16.0

    # aspect ratios of the rasterized PDFs (h/w)
    asp = {}
    for name, pth in [("icj", STACK_ICJ), ("nudap", STACK_NUDAP),
                      ("sst", STACK_SST_CHODL), ("ab", NUDAP_AB)]:
        doc = pdfium.PdfDocument(str(pth))
        w_pt, h_pt = doc[0].get_size()
        asp[name] = h_pt / w_pt

    h_bar = 5.6           # portrait bar chart
    h_icj = W * asp["icj"] + 0.4
    h_nudap = W * asp["nudap"] + 0.4
    h_sst = W * asp["sst"] + 0.4
    h_ab = W * asp["ab"] + 0.4
    top_pad = 0.9         # room for suptitle
    H = h_bar + h_icj + h_nudap + h_sst + h_ab + top_pad + 0.6

    fig = plt.figure(figsize=(W, H))
    gs = GridSpec(
        nrows=5, ncols=4, figure=fig,
        height_ratios=[h_bar, h_icj, h_nudap, h_sst, h_ab],
        hspace=0.30, wspace=0.0,
        top=1 - top_pad / H, bottom=0.3 / H, left=0.05, right=0.97,
    )
    ax_bar = fig.add_subplot(gs[0, 1:3])   # centered narrow bar panel
    ax_icj = fig.add_subplot(gs[1, :])
    ax_nudap = fig.add_subplot(gs[2, :])
    ax_sst = fig.add_subplot(gs[3, :])
    ax_ab = fig.add_subplot(gs[4, :])

    draw_count_bar(ax_bar)
    draw_pdf_panel(ax_icj, STACK_ICJ,
                   "B", "OT_D1_ICj — phyloP + high-contrib patterns "
                   "(macaque / human / mouse stack)")
    draw_pdf_panel(ax_nudap, STACK_NUDAP,
                   "C", "STRv_D1_NUDAP — phyloP + high-contrib patterns "
                   "(macaque / human / mouse stack)")
    draw_pdf_panel(ax_sst, STACK_SST_CHODL,
                   "D", "STR_SST-CHODL_GABA — phyloP + high-contrib patterns "
                   "(macaque / human / mouse stack)")
    draw_pdf_panel(ax_ab, NUDAP_AB,
                   "E", "NUDAP (AiE1485) — full alignment & in-silico "
                   "region-by-region restoration of mouse to q+h consensus")

    fig.suptitle(
        "Figure 7 add-on panels — conservation of ON-target enhancers "
        "and NUDAP lost-in-mouse restoration",
        fontsize=12, fontweight="bold", y=1 - 0.25 / H,
    )

    fig.savefig(OUT_PNG, dpi=200)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_SVG)
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_SVG}")


if __name__ == "__main__":
    main()
