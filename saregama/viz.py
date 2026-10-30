"""Contour sheets: the tonic-relative pitch contour with sargam gridlines,
sustained swaras, meends and kampitas, plus sections, sama and (when aligned)
lyric syllables, all on the mix clock.  Mirrors ``pitch_contour_sargam.pdf``.

Page 1: whole-recording overview and the share of stable time per swara.
Then 30 s panels, three per page.  Light mode only (made for print/PDF).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# reference categorical palette, slots 1-3 (validated all-pairs), plus ink
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SUSTAIN = "#2a78d6"      # slot 1, blue
MEEND = "#eb6834"        # slot 2, orange
KAMPITA = "#1baf7a"      # slot 3, aqua

PANEL_S = 30.0
PANELS_PER_PAGE = 3


def _swara_lines(positions: Dict[str, float], lo: float, hi: float):
    """(cents, label) for every allowed swara in [lo, hi] across octaves."""
    out = []
    for octave in range(int(np.floor(lo / 1200)) - 1, int(np.ceil(hi / 1200)) + 1):
        for pc, p in positions.items():
            c = p + 1200 * octave
            if lo <= c <= hi:
                name = ("." * -octave if octave < 0 else "") + pc + ("'" * octave if octave > 0 else "")
                out.append((c, name))
    return sorted(out)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=7, length=2)


def _panel(ax, t, cents, segs, positions, tonic, t0, t1, ylim, sections, sama, syllables):
    m = (t >= t0) & (t < t1)
    for c, name in _swara_lines(positions, *ylim):
        ax.axhline(c, color=GRID, lw=0.6, zorder=0)
    ax.set_yticks([c for c, _ in _swara_lines(positions, *ylim)])
    ax.set_yticklabels([f"{n} {tonic * 2 ** (c / 1200):.0f} Hz" if n.strip(".'") == "S" else n
                        for c, n in _swara_lines(positions, *ylim)], fontsize=5.5)
    for s in segs:
        if s["end"] < t0 or s["start"] > t1:
            continue
        if s["kind"] == "kampita":
            ax.axvspan(max(s["start"], t0), min(s["end"], t1), color=KAMPITA, alpha=0.12, lw=0, zorder=1)
        elif s["kind"] == "glide":
            k = (t >= s["start"]) & (t <= s["end"]) & m
            ax.plot(t[k], cents[k], color=MEEND, lw=2.0, zorder=3, solid_capstyle="round")
        elif s["kind"] == "stable" and s["sustained"]:
            ax.plot([s["start"], s["end"]], [s["cents"], s["cents"]], color=SUSTAIN, lw=3.0, alpha=0.85,
                    zorder=2, solid_capstyle="round")
    ax.plot(t[m], cents[m], color=INK, lw=0.7, zorder=4)
    top = ylim[1]
    for sec in sections:
        if sec["end"] < t0 or sec["start"] > t1:
            continue
        a = max(sec["start"], t0)
        ax.axvline(sec["start"], color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1) if sec["start"] >= t0 else None
        ax.text(a + 0.2, top, sec["label"], fontsize=6.5, color=INK_2, va="top", ha="left")
    ss = [x for x in sama if t0 <= x < t1]
    ax.plot(ss, [ylim[0]] * len(ss), marker="^", ms=4, ls="none", color=INK_2, zorder=5, clip_on=False)
    for syl in syllables:
        if t0 <= syl["start"] < t1:
            ax.text(syl["start"], ylim[0] + 40, syl["text"], fontsize=5.5, color=INK, rotation=0,
                    va="bottom", ha="left", zorder=6)
    ax.set_xlim(t0, t1)
    ax.set_ylim(*ylim)
    ax.set_xlabel(f"{t0:.0f}-{t1:.0f} s", fontsize=6.5, color=INK_2, labelpad=1)
    _style(ax)


def contour_pdf(out: Path, title: str, tonic: float, raga: str, f0_npz: Path, swaras: dict,
                sections: List[dict], sama: List[float], syllables: Optional[List[dict]] = None,
                max_pages: Optional[int] = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    z = np.load(f0_npz)
    cents = z["cents"].astype(float)
    fps = float(z["fps"])
    t = np.arange(len(cents)) / fps
    segs = swaras["segments"]
    positions = swaras["positions_cents"]
    syllables = syllables or []
    v = ~np.isnan(cents)
    lo, hi = np.percentile(cents[v], [0.5, 99.5]) if v.any() else (-600, 1800)
    ylim = (float(np.floor(lo / 100) * 100 - 100), float(np.ceil(hi / 100) * 100 + 100))
    legend = [Line2D([], [], color=INK, lw=0.8, label="pitch contour"),
              Line2D([], [], color=SUSTAIN, lw=3, label="sustained swara (>= 0.25 s)"),
              Line2D([], [], color=MEEND, lw=2, label="meend / glide (>= 150 c)"),
              Patch(color=KAMPITA, alpha=0.25, label="kampita"),
              Line2D([], [], marker="^", ls="none", color=INK_2, label="sama")]
    header = f"{title}  |  {raga}  |  Sa = {tonic:.1f} Hz  (dot below = mandra, dot above = tara)"
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        # page 1: overview + histogram
        fig = plt.figure(figsize=(11.69, 8.27), facecolor=SURFACE)
        gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 1], wspace=0.15, hspace=0.3)
        ax = fig.add_subplot(gs[0, :])
        step = max(1, int(fps / 10))
        ax.plot(t[::step], cents[::step], color=INK, lw=0.4)
        for sec in sections:
            ax.axvline(sec["start"], color=MUTED, lw=0.6, ls=(0, (2, 2)))
            ax.text(sec["start"] + 1, ylim[1], sec["label"], fontsize=6.5, color=INK_2, va="top")
        lines = _swara_lines(positions, *ylim)
        ax.set_yticks([c for c, _ in lines])
        ax.set_yticklabels([n for _, n in lines], fontsize=6)
        ax.set_ylim(*ylim)
        ax.set_xlim(0, t[-1] if len(t) else 1)
        ax.set_xlabel("time (s)", fontsize=7, color=INK_2)
        _style(ax)
        ax.set_title(header, fontsize=9, color=INK, loc="left")
        hx = fig.add_subplot(gs[1, 1])
        share = swaras["summary"]["stable_time_by_swara"]
        tot = sum(share.values()) or 1.0
        order = sorted(share, key=lambda n: positions[n.strip(".'")] + 1200 * (n.count("'") - n.count(".")))
        hx.barh(range(len(order)), [100 * share[n] / tot for n in order], color=SUSTAIN, height=0.7)
        hx.set_yticks(range(len(order)))
        hx.set_yticklabels(order, fontsize=7)
        for i, n in enumerate(order):
            hx.text(100 * share[n] / tot + 0.5, i, f"{100 * share[n] / tot:.0f}%", va="center", fontsize=6,
                    color=INK_2)
        hx.set_xlabel("share of stable time (%)", fontsize=7, color=INK_2)
        _style(hx)
        sx = fig.add_subplot(gs[1, 0])
        sx.axis("off")
        summ = swaras["summary"]
        txt = [f"voiced {summ['voiced_s'] / 60:.1f} min",
               "time share: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in summ["share"].items()),
               f"{summ['n_stable']} stable swaras ({summ['n_sustained']} sustained >= 0.25 s), "
               f"{summ['n_glide']} glides",
               "swara positions (cents above Sa): " + ", ".join(f"{k} {v:.0f}" for k, v in positions.items())]
        sx.text(0, 0.9, "\n".join(txt), fontsize=8, color=INK, va="top", family="monospace")
        sx.legend(handles=legend, loc="lower left", fontsize=7, frameon=False)
        pdf.savefig(fig, facecolor=SURFACE)
        plt.close(fig)
        # 30 s panels
        n_panels = int(np.ceil((t[-1] if len(t) else 0) / PANEL_S))
        pages = int(np.ceil(n_panels / PANELS_PER_PAGE))
        for pg in range(pages if max_pages is None else min(pages, max_pages)):
            fig, axes = plt.subplots(PANELS_PER_PAGE, 1, figsize=(11.69, 8.27), facecolor=SURFACE)
            fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.05, hspace=0.45)
            fig.suptitle(header, fontsize=8.5, color=INK, x=0.08, ha="left")
            fig.legend(handles=legend, loc="upper right", ncol=5, fontsize=6.5, frameon=False)
            for i, ax in enumerate(axes):
                k = pg * PANELS_PER_PAGE + i
                if k >= n_panels:
                    ax.axis("off")
                    continue
                _panel(ax, t, cents, segs, positions, tonic, k * PANEL_S, (k + 1) * PANEL_S, ylim,
                       sections, sama, syllables)
            pdf.savefig(fig, facecolor=SURFACE)
            plt.close(fig)
    return out
