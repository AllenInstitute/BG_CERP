"""CREsted contribution stack + pattern-family KO specificity heatmap
for the STR_SST-CHODL enhancer trio (macaque AiE1255q, human AiE1443h,
mouse AiE1438m).

Single-file pipeline: if the KO summary CSV is missing it re-computes
it using the crested/keras env; otherwise it just re-renders the figure.

Outputs (next to this script):
    chodl_pattern_ko.pdf                Illustrator-ready figure
    chodl_pattern_ko.png                300 dpi raster preview
    chodl_pattern_ko.fasta              oriented sequences used
    chodl_pattern_ko_regions.csv        shaded core coords
    chodl_pattern_ko_summary.csv        per (region, ortholog) Δspec
    chodl_pattern_ko_predictions.csv    per class long-format Δpred

Usage:
    python chodl_pattern_ko.py                # render (auto-compute if missing)
    python chodl_pattern_ko.py --force-compute # re-run KO before rendering

To run WITHOUT overwriting the shared outputs in this directory, copy the
script (and the cached CSVs, if you want to skip the ~2 min keras compute)
into your own working directory first, e.g.

    mkdir -p ~/my_chodl_run && cd ~/my_chodl_run
    cp /allen/programs/celltypes/workgroups/hct/sarojaS/260709/chodl_pattern_ko.py .
    cp /allen/programs/celltypes/workgroups/hct/sarojaS/260709/chodl_pattern_ko_summary.csv .      # optional
    cp /allen/programs/celltypes/workgroups/hct/sarojaS/260709/chodl_pattern_ko_predictions.csv .  # optional
    source /allen/scratch/aibstemp/miniconda3/etc/profile.d/conda.sh
    conda activate /allen/programs/celltypes/workgroups/hct/sarojaS/250506_bg_figure/merged_Fig_7and8/PyPeakRankr/pattern_analysis/2502_CREsted_tutorial/hct_run/hpc_run/latest_crested_env
    python chodl_pattern_ko.py

All outputs are written next to the script via `HERE = Path(__file__).resolve().parent`,
so running from your own copy will only touch files in that directory.
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
from matplotlib import cm, colors as mcolors, gridspec
from matplotlib.patches import ConnectionPatch, Rectangle

# ── Paths (all absolute) ──────────────────────────────────────────────────
ROOT       = Path("/allen/programs/celltypes/workgroups/hct/sarojaS/260604_grammar_flex")
PHYLOP_DIR = ROOT / "phylop_analysis"
SLUG_DIR   = PHYLOP_DIR / "seq_phylop_plots" / "groups" / "str_sst_chodl_loose_1p0em3"
FIMO_HITS_CSV    = SLUG_DIR / "fimo_hits.csv"
PHYLOP_CACHE_DIR = PHYLOP_DIR / "seq_phylop_plots"

HERE = Path(__file__).resolve().parent
OUT_PDF     = HERE / "chodl_pattern_ko.pdf"
OUT_PNG     = HERE / "chodl_pattern_ko.png"
OUT_FASTA   = HERE / "chodl_pattern_ko.fasta"
OUT_REGIONS = HERE / "chodl_pattern_ko_regions.csv"
OUT_SUMMARY = HERE / "chodl_pattern_ko_summary.csv"
OUT_PREDS   = HERE / "chodl_pattern_ko_predictions.csv"

# ── Vendored helpers from the sister project ──────────────────────────────
sys.path.insert(0, str(PHYLOP_DIR))
sys.path.insert(0, str(ROOT / "illustrator"))
import make_group_stacks as mgs                       # noqa: E402
from plot_seq_phylop import (                          # noqa: E402
    _msa_mafft, _nat_idx_from_aligned, _three_way_align,
    _project_to_msa, _project_hits, _seq_from_one_hot,
)

# Pull crested's logo plotter directly (skip the broken package __init__).
_CRESTED_UTILS = Path(
    "/home/sarojas/.local/lib/python3.9/site-packages/crested/pl/patterns/_utils.py"
)
_spec = importlib.util.spec_from_file_location("_crested_utils_local",
                                                _CRESTED_UTILS)
_cru = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cru)
grad_times_input_to_df = _cru.grad_times_input_to_df
_plot_attribution_map  = _cru._plot_attribution_map

# ── Constants ─────────────────────────────────────────────────────────────
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"]  = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.family"]  = "DejaVu Sans"

IDS     = ["AiE1255q", "AiE1443h", "AiE1438m"]
SPECIES = {"AiE1255q": "macaque (rheMac10)",
           "AiE1443h": "human (hg38)",
           "AiE1438m": "mouse (mm10)"}
TARGET_CLASS = "STR_SST-CHODL_GABA"

PHYLOP_THR       = mgs.PHYLOP_THR         # 2.0
IMP_FRAC_OF_MAX  = mgs.IMP_FRAC_OF_MAX    # 0.10
MIN_RUN_BP       = mgs.MIN_HIT_KEEP_BP    # 3
SHADE_MIN_BP     = 5
STRONG_DSPEC     = 100.0                  # emphasis cutoff on |Δspec|

GAP_COLOR            = "0.85"
SHADED_CORE_COLOR    = "#1f77b4"
SHADED_CORE_ALPHA    = 0.22
EXTRA_KO_COLOR       = "#f0a020"
EXTRA_KO_EDGE        = "#a67308"
EXTRA_KO_ALPHA       = 0.30
EXTRA_KO_LABEL_COLOR = "#8b5a00"

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


# ── Small utilities ───────────────────────────────────────────────────────
def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def find_all(hay: str, needle: str) -> list[int]:
    """All (possibly overlapping) starts of `needle` in `hay`."""
    out, start = [], 0
    if not needle:
        return out
    while True:
        i = hay.find(needle, start)
        if i < 0:
            return out
        out.append(i)
        start = i + 1


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if not mask.any():
        return []
    idx = np.flatnonzero(mask)
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends   = np.r_[idx[breaks], idx[-1]] + 1
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def parse_occurrences(s: str) -> list[tuple[int, int, str]]:
    """'a1-b1:strand;a2-b2:strand;...' -> [(a, b, strand), ...]."""
    out: list[tuple[int, int, str]] = []
    for tok in str(s or "").split(";"):
        tok = tok.strip()
        if not tok:
            continue
        rng, _, strand = tok.partition(":")
        try:
            a, b = rng.split("-")
            out.append((int(a), int(b), strand or "fwd"))
        except ValueError:
            continue
    return out


def model_ivl_to_msa(m_a: int, m_b: int, bundle: dict,
                     nats_i: list[int]) -> tuple[int, int] | None:
    """Map a [m_a, m_b) interval in model-input coords to MSA columns."""
    s_start = int(bundle["s_start"])
    L_trim  = int(bundle["L"])
    flipped = bool(bundle["flipped"])
    t_a, t_b = m_a - s_start, m_b - s_start
    if t_a < 0 or t_b > L_trim or t_a >= t_b:
        return None
    d_a, d_b = (L_trim - t_b, L_trim - t_a) if flipped else (t_a, t_b)
    cols = [c for c, n in enumerate(nats_i)
            if n is not None and n >= 0 and d_a <= n < d_b]
    if not cols:
        return None
    return (min(cols), max(cols) + 1)


# ── Load oriented sequences + MSA (no keras needed) ───────────────────────
def load_fimo_hits(fimo_csv: Path,
                   native_lengths: dict[str, int]) -> dict[str, list[tuple]]:
    out: dict[str, list[tuple]] = {eid: [] for eid in native_lengths}
    if not fimo_csv.exists():
        print(f"[warn] no FIMO hits file at {fimo_csv}")
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


def build_bundles() -> dict[str, dict]:
    npz      = mgs.load_combined_npz()
    coords   = mgs.load_coords()
    std_seqs = mgs.load_standardized_seqs()

    bundles: dict[str, dict] = {}
    for eid in IDS:
        oh, sc, s_start, s_end, flipped = mgs.trim_and_orient(
            eid, npz, std_seqs, verbose=True)
        seq = mgs.one_hot_to_seq(oh)
        pp  = mgs.aligned_phylop(eid, oh.shape[0], coords,
                                 PHYLOP_CACHE_DIR, flipped)
        bundles[eid] = {
            "one_hot": oh, "scores": sc, "seq": seq, "pp": pp,
            "flipped": flipped, "s_start": s_start, "s_end": s_end,
            "L": oh.shape[0],
        }
    native_lengths = {eid: b["L"] for eid, b in bundles.items()}
    for eid, hits in load_fimo_hits(FIMO_HITS_CSV, native_lengths).items():
        bundles[eid]["hits"] = hits
    return bundles


def align_bundles(bundles: dict[str, dict]
                  ) -> tuple[list[str], list[list[int]]]:
    seqs = [_seq_from_one_hot(bundles[e]["one_hot"]) for e in IDS]
    rows = _msa_mafft(seqs, IDS)
    if rows is not None:
        return rows, [_nat_idx_from_aligned(r) for r in rows]
    return _three_way_align(seqs[0], seqs[1], seqs[2])


# ── Shaded-core detection ─────────────────────────────────────────────────
def project_to_msa(bundles: dict[str, dict], rows: list[str],
                   nats: list[list[int]]):
    n, L_aln = len(IDS), len(rows[0])
    scores_aln  = np.zeros((n, L_aln, 4))
    one_hot_aln = np.zeros((n, L_aln, 4))
    pp_aln:   list[np.ndarray] = [None] * n
    hits_aln: list[list[tuple]] = [[] for _ in range(n)]
    for i, eid in enumerate(IDS):
        sc, oh, pp = _project_to_msa(bundles[eid], nats[i])
        scores_aln[i], one_hot_aln[i], pp_aln[i] = sc, oh, pp
        hits_aln[i] = _project_hits(bundles[eid]["hits"], nats[i])
    return scores_aln, one_hot_aln, pp_aln, hits_aln


def find_shaded_cores(rows, one_hot_aln, scores_aln, hits_aln):
    n, L_aln = len(IDS), len(rows[0])

    aligned_all = np.ones(L_aln, dtype=bool)
    high_per_sp, pp_per_sp = [], []
    fimo_any_per_sp = [np.zeros(L_aln, dtype=bool) for _ in range(n)]
    for i in range(n):
        aligned_all &= np.array([ch != "-" for ch in rows[i]])
        contrib = np.abs(scores_aln[i]).sum(axis=-1)
        cmax = float(contrib.max()) if contrib.size else 0.0
        high_per_sp.append(contrib > (IMP_FRAC_OF_MAX * cmax))
        for _m, a, b, _st, _sc in hits_aln[i]:
            fimo_any_per_sp[i][a:b] = True

    same_base = one_hot_aln[0].sum(axis=1) > 0
    ref_idx = one_hot_aln[0].argmax(axis=1)
    for i in range(1, n):
        has = one_hot_aln[i].sum(axis=1) > 0
        same_base &= has & (one_hot_aln[i].argmax(axis=1) == ref_idx)

    high_all = np.logical_and.reduce(high_per_sp)
    fimo_all = np.logical_and.reduce(fimo_any_per_sp)

    mask = aligned_all & same_base & high_all & fimo_all
    core_runs = [(a, b) for a, b in contiguous_runs(mask)
                 if b - a >= SHADE_MIN_BP]
    print(f"  [shade] {len(core_runs)} conserved shaded cores")
    return core_runs


# ── Pattern-family KO (uses keras) ────────────────────────────────────────
def compute_pattern_ko(regions_msa: list[tuple[int, int]],
                       out_summary: Path, out_preds: Path) -> None:
    import h5py
    import keras
    import scramble_chodl_regions as scr    # noqa: E402

    with h5py.File(scr.ADATA_PATH, "r") as h:
        raw = h["obs"]["_index"][:]
    class_names = [v.decode() if isinstance(v, bytes) else str(v) for v in raw]
    n_cls = len(class_names)
    tgt = class_names.index(TARGET_CLASS)
    other = np.ones(n_cls, dtype=bool); other[tgt] = False
    print(f"[class] target={TARGET_CLASS} idx={tgt} of {n_cls}")

    oriented = scr.load_oriented()
    rows = scr.load_msa_rows()
    nats = [mgs._nat_idx_from_aligned(r) for r in rows]

    print(f"[model] {scr.MODEL_PATH}")
    model = keras.models.load_model(scr.MODEL_PATH, compile=False)

    seq_str = {e: mgs.one_hot_to_seq(oriented[e]["oh_full_model"])
               for e in IDS}
    wt_batch = np.stack([oriented[e]["oh_full_model"] for e in IDS])
    wt_full = model.predict(wt_batch, batch_size=4, verbose=0)
    if wt_full.shape[1] != n_cls:
        raise RuntimeError("model outputs != class list")
    wt_by = {e: wt_full[i] for i, e in enumerate(IDS)}
    wt_spec = {e: float(wt_by[e][tgt] - wt_by[e][other].mean()) for e in IDS}
    for e in IDS:
        print(f"  [WT] {e}: CHODL={wt_by[e][tgt]:.2f} "
              f"other_mean={wt_by[e][other].mean():.2f} "
              f"spec={wt_spec[e]:+.2f}")

    long_rows: list[dict] = []
    sum_rows:  list[dict] = []
    for r_idx, (msa_a, msa_b) in enumerate(regions_msa):
        for i, eid in enumerate(IDS):
            proj = scr.project_region_to_model((msa_a, msa_b),
                                               nats[i], oriented[eid])
            if proj is None:
                print(f"  [skip] R{r_idx+1} {eid}: no aligned bases")
                continue
            m_a, m_b = proj
            full = seq_str[eid]
            pat  = full[m_a:m_b]
            rc   = revcomp(pat)
            fwd  = find_all(full, pat)
            rev  = [] if rc == pat else find_all(full, rc)
            width = m_b - m_a

            ko_oh = oriented[eid]["oh_full_model"].copy()
            intervals: list[tuple[int, int, str]] = []
            for h in fwd:
                ko_oh[h:h + width] = 0.0
                intervals.append((h, h + width, "fwd"))
            for h in rev:
                ko_oh[h:h + width] = 0.0
                intervals.append((h, h + width, "rc"))
            intervals.sort()

            ko_full = model.predict(ko_oh[None, ...],
                                    batch_size=1, verbose=0)[0]
            ko_sp = float(ko_full[tgt] - ko_full[other].mean())
            dsp   = ko_sp - wt_spec[eid]
            occ_str = ";".join(f"{a}-{b}:{k}" for a, b, k in intervals)
            n_occ = len(fwd) + len(rev)

            print(f"  R{r_idx+1} {eid} pat={pat} rc={rc} "
                  f"occ={n_occ} (fwd={len(fwd)},rc={len(rev)}) "
                  f"spec {wt_spec[eid]:+.1f}->{ko_sp:+.1f} "
                  f"Δspec={dsp:+.1f}")

            for ci, cname in enumerate(class_names):
                long_rows.append({
                    "region_idx": r_idx, "msa_start": msa_a, "msa_end": msa_b,
                    "enhancer_id": eid, "species": SPECIES[eid],
                    "pattern": pat, "revcomp": rc,
                    "n_occurrences": n_occ,
                    "n_fwd": len(fwd), "n_rc": len(rev),
                    "occurrences": occ_str,
                    "class_idx": ci, "class_name": cname,
                    "is_target": cname == TARGET_CLASS,
                    "wt_pred": float(wt_by[eid][ci]),
                    "ko_pred": float(ko_full[ci]),
                    "delta_vs_wt": float(ko_full[ci] - wt_by[eid][ci]),
                })
            sum_rows.append({
                "region_idx": r_idx, "msa_start": msa_a, "msa_end": msa_b,
                "aln_length": msa_b - msa_a,
                "enhancer_id": eid, "species": SPECIES[eid],
                "model_start": m_a, "model_end": m_b,
                "region_bp_model": width,
                "pattern": pat, "revcomp": rc,
                "n_occurrences": n_occ,
                "n_fwd": len(fwd), "n_rc": len(rev),
                "occurrences": occ_str,
                "wt_chodl": float(wt_by[eid][tgt]),
                "ko_chodl": float(ko_full[tgt]),
                "wt_other_mean": float(wt_by[eid][other].mean()),
                "ko_other_mean": float(ko_full[other].mean()),
                "wt_spec": wt_spec[eid],
                "ko_spec": ko_sp,
                "dspec": dsp,
            })

    pd.DataFrame(long_rows).to_csv(out_preds, index=False)
    pd.DataFrame(sum_rows).to_csv(out_summary, index=False)
    print(f"[save] {out_preds}  ({len(long_rows)} rows)")
    print(f"[save] {out_summary} ({len(sum_rows)} rows)")


def load_ko_summary(csv_path: Path, ordered_regions: list[tuple[int, int, str]]
                    ) -> dict[tuple[str, int], dict]:
    """Load per-(ortholog, region) Δspec dict from the summary CSV."""
    df = pd.read_csv(csv_path)
    needed = {"enhancer_id", "msa_start", "msa_end",
              "wt_spec", "ko_spec", "dspec", "n_occurrences", "pattern"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"{csv_path.name} missing columns: {sorted(missing)}")
    key2col = {(int(a), int(b)): j
               for j, (a, b, _m) in enumerate(ordered_regions)}
    out: dict[tuple[str, int], dict] = {}
    for _, row in df.iterrows():
        eid = row["enhancer_id"]
        j = key2col.get((int(row["msa_start"]), int(row["msa_end"])))
        if j is None or eid not in IDS:
            continue
        out[(eid, j)] = {
            "wt_spec":     float(row["wt_spec"]),
            "ko_spec":     float(row["ko_spec"]),
            "dspec":       float(row["dspec"]),
            "n_occ":       int(row["n_occurrences"]),
            "n_fwd":       int(row.get("n_fwd", 0)),
            "n_rc":        int(row.get("n_rc", 0)),
            "pattern":     str(row["pattern"]),
            "occurrences": str(row.get("occurrences", "")),
        }
    return out


# ── Figure ────────────────────────────────────────────────────────────────
def render_figure(bundles, rows, nats, scores_aln, one_hot_aln,
                  shade_regions, spec_by):
    n, L_aln = len(IDS), len(rows[0])
    have_ko = bool(spec_by)
    ordered = sorted(shade_regions, key=lambda t: t[0])
    n_reg   = len(ordered)

    saliency_dfs = [grad_times_input_to_df(one_hot_aln[i], scores_aln[i])
                    for i in range(n)]
    y_min = min((scores_aln[i] * one_hot_aln[i]).min() for i in range(n))
    y_max = max((scores_aln[i] * one_hot_aln[i]).max() for i in range(n))
    y_min -= 0.25 * abs(y_min)
    y_max += 0.25 * abs(y_max)

    plot_w, row_h = max(12.0, L_aln / 10.0), 2.0
    stack_h = row_h * n
    top_pad = 0.55

    if have_ko:
        hm_h, hm_gap, bot_pad = 2.6, 1.15, 1.05
        total_h = stack_h + hm_gap + hm_h + bot_pad + top_pad
        fig = plt.figure(figsize=(plot_w, total_h))
        gs = gridspec.GridSpec(n, 1, figure=fig, hspace=0.25,
                               top=1.0 - (top_pad / total_h),
                               bottom=(hm_h + hm_gap + bot_pad) / total_h,
                               left=0.06, right=0.92)
    else:
        total_h = stack_h + top_pad
        fig = plt.figure(figsize=(plot_w, total_h))
        gs = gridspec.GridSpec(n, 1, figure=fig, hspace=0.25,
                               top=1.0 - (top_pad / total_h))
    axs = [fig.add_subplot(gs[i, 0]) for i in range(n)]
    for ax in axs[1:]:
        ax.sharex(axs[0])

    # ── stack rows ──────────────────────────────────────────────────────
    for i, ax in enumerate(axs):
        _plot_attribution_map(saliency_dfs[i], ax=ax, return_ax=False,
                              figsize=None, spines=False)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(-0.5, L_aln - 0.5)
        ax.set_xticks(np.arange(0, L_aln, 50))
        eid = IDS[i]
        n_gaps = rows[i].count("-")
        ax.set_title(f"{eid}  ({SPECIES[eid]})   L={bundles[eid]['L']} bp   "
                     f"MSA gaps: {n_gaps}/{L_aln}",
                     fontsize=10, loc="left")
        ax.set_ylabel("contrib.", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)

        for c, ch in enumerate(rows[i]):
            if ch == "-":
                ax.axvspan(c - 0.5, c + 0.5, color=GAP_COLOR,
                           alpha=0.5, lw=0, zorder=0)

        for a, b in [(a, b) for a, b, _ in shade_regions]:
            ax.add_patch(Rectangle(
                (a - 0.5, y_min), b - a, y_max - y_min,
                facecolor=SHADED_CORE_COLOR, edgecolor=SHADED_CORE_COLOR,
                alpha=SHADED_CORE_ALPHA, linewidth=0.0, zorder=0.5))

        # Extra pattern copies KO'd along with the primary R#, per-ortholog.
        if have_ko:
            for j, (a, b, _m) in enumerate(ordered):
                entry = spec_by.get((eid, j))
                if entry is None:
                    continue
                for m_a, m_b, strand in parse_occurrences(entry["occurrences"]):
                    span = model_ivl_to_msa(m_a, m_b, bundles[eid], nats[i])
                    if span is None or span == (a, b):
                        continue
                    ma, mb = span
                    ax.add_patch(Rectangle(
                        (ma - 0.5, y_min), mb - ma, y_max - y_min,
                        facecolor=EXTRA_KO_COLOR, edgecolor=EXTRA_KO_EDGE,
                        alpha=EXTRA_KO_ALPHA, hatch="////",
                        linewidth=0.9, zorder=0.6))
                    ax.annotate(
                        f"R{j+1}\u2032 ({strand})",
                        xy=((ma + mb - 1) / 2.0, y_max),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=7, fontweight="bold",
                        color=EXTRA_KO_LABEL_COLOR)

    for j, (a, b, _m) in enumerate(ordered):
        axs[0].annotate(f"R{j+1}",
                        xy=((a + b - 1) / 2.0, y_max),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color="0.20")

    axs[-1].set_xlabel("Alignment position (bp)", fontsize=9)

    # ── heatmap band ────────────────────────────────────────────────────
    if have_ko:
        ref_seq_by_reg = [rows[0][a:b].replace("-", "")
                          for a, b, _m in ordered]
        dspec_mat = np.full((n, n_reg), np.nan)
        for i, eid in enumerate(IDS):
            for j in range(n_reg):
                info = spec_by.get((eid, j))
                if info is not None:
                    dspec_mat[i, j] = info["dspec"]

        finite = dspec_mat[np.isfinite(dspec_mat)]
        vmax_data = float(np.max(np.abs(finite))) if finite.size else 1.0
        vmax = max(25.0, float(np.ceil(vmax_data / 25.0)) * 25.0)
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap = mpl.colormaps.get_cmap("RdBu_r")

        hm_bot = (bot_pad + 0.30) / total_h
        hm_hgt = (hm_h - 0.30) / total_h
        hm_ax = fig.add_axes([0.06, hm_bot, 0.86, hm_hgt])
        hm_ax.set_xlim(-0.5, L_aln - 0.5)
        hm_ax.set_ylim(0, n)
        hm_ax.add_patch(Rectangle((-0.5, 0), L_aln, n,
                                  facecolor="0.965", edgecolor="none", zorder=0))
        for i in range(1, n):
            hm_ax.axhline(i, color="white", linewidth=1.2, zorder=1)

        for i, eid in enumerate(IDS):
            y_bot = n - 1 - i
            for j, (a, b, _m) in enumerate(ordered):
                v = dspec_mat[i, j]
                info = spec_by.get((eid, j))
                if not np.isfinite(v):
                    hm_ax.add_patch(Rectangle(
                        (a - 0.5, y_bot), b - a, 1.0,
                        facecolor="0.85", edgecolor="black",
                        linewidth=0.5, hatch="///", zorder=2))
                    continue
                is_strong = abs(v) >= STRONG_DSPEC
                face = list(cmap(norm(v)))
                if not is_strong:
                    face = [0.60 * c + 0.40 for c in face[:3]] + [1.0]
                hm_ax.add_patch(Rectangle(
                    (a - 0.5, y_bot), b - a, 1.0,
                    facecolor=tuple(face),
                    edgecolor="black" if is_strong else "0.55",
                    linewidth=0.9 if is_strong else 0.45,
                    linestyle="solid" if is_strong else (0, (2, 1)),
                    zorder=2))

                if abs(v) >= 100:
                    txt = f"{v:+.0f}"
                elif abs(v) >= 10:
                    txt = f"{v:+.1f}"
                else:
                    txt = f"{v:+.2f}"
                if is_strong:
                    tc = "white" if abs(v) > 0.55 * vmax else "black"
                    tfw, tsty, tsz = "bold", "normal", 9
                else:
                    tc, tfw, tsty, tsz = "0.45", "normal", "italic", 8
                xc = (a + b - 1) / 2.0
                hm_ax.text(xc, y_bot + 0.62, txt,
                           ha="center", va="center", zorder=3,
                           fontsize=tsz, fontweight=tfw, fontstyle=tsty,
                           color=tc)
                if info:
                    occ = (f"N={info['n_occ']}  "
                           f"({info['n_fwd']}f/{info['n_rc']}rc)"
                           if (info['n_fwd'] or info['n_rc'])
                           else f"N={info['n_occ']}")
                    hm_ax.text(xc, y_bot + 0.22, occ,
                               ha="center", va="center", zorder=3,
                               fontsize=7, fontstyle="italic",
                               color=tc if is_strong else "0.45")

        for j, (a, b, _m) in enumerate(ordered):
            xc = (a + b - 1) / 2.0
            x_frac = (xc + 0.5) / L_aln
            hm_ax.text(x_frac, -0.40,
                       f"R{j+1}\n{ref_seq_by_reg[j]}\n[{a}:{b}]",
                       ha="center", va="top",
                       fontsize=9, color="0.10", fontweight="bold",
                       transform=hm_ax.transAxes)

        hm_ax.set_yticks([n - i - 0.5 for i in range(n)])
        hm_ax.set_yticklabels(
            [f"{e}\n({SPECIES[e].split()[0]})" for e in IDS], fontsize=10)
        hm_ax.tick_params(axis="y", length=0, pad=4)
        hm_ax.set_xticks(np.arange(0, L_aln, 50))
        hm_ax.tick_params(axis="x", labelsize=9)
        hm_ax.set_xlabel("Alignment position (bp)", fontsize=10, labelpad=6)
        for sp in ("top", "right"):
            hm_ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            hm_ax.spines[sp].set_linewidth(0.6)

        hm_ax.text(0.0, 1.15,
                   "\u0394 CHODL specificity   "
                   "(pattern-family KO  \u2212  WT)",
                   ha="left", va="bottom",
                   fontsize=12, fontweight="bold", color="0.10",
                   transform=hm_ax.transAxes)
        hm_ax.text(0.0, 1.02,
                   f"specificity = pred({TARGET_CLASS}) \u2212 "
                   f"mean pred(other 57 classes)      \u2022      "
                   f"KO = N-mask every fwd + revcomp copy of the core's "
                   f"exact sequence across the full model input      "
                   f"\u2022      cell = spec(KO) \u2212 spec(WT)",
                   ha="left", va="bottom",
                   fontsize=9, color="0.30",
                   transform=hm_ax.transAxes)

        for j, (a, b, _m) in enumerate(ordered):
            xc = (a + b - 1) / 2.0
            fig.add_artist(ConnectionPatch(
                xyA=(xc, y_min), coordsA=axs[-1].transData,
                xyB=(xc, n),     coordsB=hm_ax.transData,
                arrowstyle="-|>", mutation_scale=10,
                color="0.30", linewidth=1.0, zorder=5,
                shrinkA=1.0, shrinkB=0.5))

        sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        cbar_ax = fig.add_axes([0.928, hm_ax.get_position().y0,
                                0.010, hm_ax.get_position().height])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(
            "\u0394 CHODL-specificity\n(pattern-family KO \u2212 WT, "
            "raw model-output units)",
            fontsize=9, labelpad=6)
        cbar.ax.tick_params(labelsize=9)
        ticks = np.linspace(-vmax, vmax, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:+.0f}" for t in ticks])

    # ── legend ──────────────────────────────────────────────────────────
    handles = [
        Rectangle((0, 0), 1, 1,
                  facecolor=SHADED_CORE_COLOR, edgecolor=SHADED_CORE_COLOR,
                  alpha=SHADED_CORE_ALPHA,
                  label=(f"Shaded core (R1\u2013R{n_reg}): \u2265 "
                         f"{SHADE_MIN_BP} bp with, in ALL {n} orthologs, "
                         "same base +\n"
                         f"|contribution| > {IMP_FRAC_OF_MAX:.0%} of the "
                         "seq max + inside an atlas FIMO motif hit")),
        Rectangle((0, 0), 1, 1,
                  facecolor=EXTRA_KO_COLOR, edgecolor=EXTRA_KO_EDGE,
                  alpha=EXTRA_KO_ALPHA, hatch="////", linewidth=0.9,
                  label=("Extra pattern copy N-masked with the primary "
                         "R# core (per-ortholog)\n"
                         "labelled R#\u2032 (fwd) or R#\u2032 (rc)")),
        Rectangle((0, 0), 1, 1,
                  facecolor=GAP_COLOR, alpha=0.5, edgecolor="none",
                  label="MSA gap (this ortholog missing that column)"),
    ]
    if have_ko:
        handles += [
            Rectangle((0, 0), 1, 1,
                      facecolor=cmap(norm(-vmax * 0.75)),
                      edgecolor="black", linewidth=0.9,
                      label=(f"STRONG BLUE cell (\u0394spec \u2264 "
                             f"-{STRONG_DSPEC:.0f}): KO drops CHODL "
                             "specificity \u2192 load-bearing driver")),
            Rectangle((0, 0), 1, 1,
                      facecolor=cmap(norm(vmax * 0.75)),
                      edgecolor="black", linewidth=0.9,
                      label=(f"STRONG RED cell (\u0394spec \u2265 "
                             f"+{STRONG_DSPEC:.0f}): KO raises CHODL "
                             "specificity \u2192 competing motif")),
            Rectangle((0, 0), 1, 1,
                      facecolor=(0.60 * cmap(norm(-vmax * 0.75))[0] + 0.40,
                                 0.60 * cmap(norm(-vmax * 0.75))[1] + 0.40,
                                 0.60 * cmap(norm(-vmax * 0.75))[2] + 0.40),
                      edgecolor="0.55", linewidth=0.45, linestyle=(0, (2, 1)),
                      label=(f"FADED cell (|\u0394spec| < "
                             f"{STRONG_DSPEC:.0f}): KO barely moves "
                             "CHODL specificity")),
            Rectangle((0, 0), 1, 1,
                      facecolor="white", edgecolor="0.55", linewidth=0.6,
                      label=("Cell text: top = \u0394spec, "
                             "bottom = N=k (Xf/Yrc) pattern copies")),
        ]
    axs[0].legend(
        handles=handles, loc="upper left",
        bbox_to_anchor=(0.965, 0.985), bbox_transform=fig.transFigure,
        fontsize=10, framealpha=0.90, borderaxespad=0.0,
        labelspacing=1.0, handleheight=1.8, handlelength=2.2,
        title="STR SST-CHODL CREsted stack + pattern-family KO heatmap",
        title_fontsize=11)

    fig.suptitle(
        "STR SST-CHODL enhancer trio  \u2014  CREsted contribution stack "
        "+ per-region CHODL-specificity \u0394 under pattern-family "
        "knock-out (all fwd + revcomp copies N-masked)",
        fontsize=13, fontweight="bold", y=1.0 - (0.10 / total_h))

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[save] {OUT_PDF}")
    print(f"[save] {OUT_PNG}")


# ── Regions + FASTA export ────────────────────────────────────────────────
def write_regions_csv(shade_regions, bundles, rows, nats):
    n_rows: list[dict] = []
    for a, b, motif in shade_regions:
        n_rows.append({"kind": "shade_pattern", "enhancer_id": "ALL",
                       "species": "all 3", "motif": motif,
                       "aln_start_0b": a, "aln_end_excl": b,
                       "aln_length": b - a,
                       "native_start_0b": None, "native_end_excl": None,
                       "native_sequence": ""})
    pd.DataFrame(n_rows).to_csv(OUT_REGIONS, index=False)
    print(f"[save] {OUT_REGIONS}  ({len(shade_regions)} shaded cores)")


def write_fasta(rows):
    with open(OUT_FASTA, "w") as fh:
        for eid, r in zip(IDS, rows):
            fh.write(f">{eid}  {SPECIES[eid]}\n{r}\n")
    print(f"[save] {OUT_FASTA}")


# ── Entry point ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-compute", action="store_true",
                    help="Re-run the pattern-family KO (needs crested/keras).")
    args = ap.parse_args()

    bundles = build_bundles()
    rows, nats = align_bundles(bundles)
    print(f"[msa] mafft L-INS-i  L_aln={len(rows[0])}")

    scores_aln, one_hot_aln, _pp, hits_aln = project_to_msa(
        bundles, rows, nats)
    core_runs = find_shaded_cores(rows, one_hot_aln, scores_aln, hits_aln)
    shade_regions = [(a, b, "shared_atlas_hit") for a, b in core_runs]
    ordered = sorted(shade_regions, key=lambda t: t[0])

    write_regions_csv(shade_regions, bundles, rows, nats)
    write_fasta(rows)

    if args.force_compute or not OUT_SUMMARY.exists():
        print(f"[compute] running pattern-family KO "
              f"({len(ordered)} regions x {len(IDS)} orthologs)")
        compute_pattern_ko([(a, b) for a, b, _m in ordered],
                           OUT_SUMMARY, OUT_PREDS)
    else:
        print(f"[compute] using cached {OUT_SUMMARY.name} "
              "(pass --force-compute to re-run)")

    spec_by = load_ko_summary(OUT_SUMMARY, ordered) if OUT_SUMMARY.exists() else {}
    render_figure(bundles, rows, nats, scores_aln, one_hot_aln,
                  shade_regions, spec_by)


if __name__ == "__main__":
    main()
