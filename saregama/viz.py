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
    """Shared axis style: recessive spines and ticks on the light surface."""
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=7, length=2)


def _panel(ax, t, cents, segs, positions, tonic, t0, t1, ylim, sections, sama, syllables):
    """Draw one 30 s panel ``[t0, t1)``: swara gridlines (Hz on Sa lines),
    kampita bands, glide and sustained-swara highlights, the contour, section
    labels, sama ticks and lyric syllables (sargam in italics).
    """
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
            ax.text(syl["start"], ylim[0] + 40, syl["text"], fontsize=5.5,
                    color=INK_2 if syl.get("sargam") else INK, style="italic" if syl.get("sargam") else "normal",
                    rotation=0, va="bottom", ha="left", zorder=6)
    ax.set_xlim(t0, t1)
    ax.set_ylim(*ylim)
    ax.set_xlabel(f"{t0:.0f}-{t1:.0f} s", fontsize=6.5, color=INK_2, labelpad=1)
    _style(ax)


def contour_pdf(out: Path, title: str, tonic: float, raga: str, f0_npz: Path, swaras: dict,
                sections: List[dict], sama: List[float], syllables: Optional[List[dict]] = None,
                max_pages: Optional[int] = None) -> Path:
    """Write the contour sheets for one recording to ``out`` (A4 landscape).

    Page 1 is the whole-recording overview with the stable-time histogram per
    swara and a text summary; then ``PANELS_PER_PAGE`` 30 s panels per page.
    ``syllables`` = [{text, start, sargam}] from ``lyrics.json``; ``max_pages``
    limits the panel pages (quick previews).
    """
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



# colours of the reference sheet (files_and_context/pitch_contour_sargam.pdf)
REF_BG = "#faf6ef"        # warm paper background
REF_LINE = "#1f4e79"      # pitch contour
REF_SUS = "#c0392b"       # sustained swara bar
REF_SUS_TXT = "#7b1f1f"   # swara label above a bar
REF_MEEND = "#f5cba7"     # meend / glide band
REF_GRID = "#d9bf9a"      # swara gridline
REF_SA = "#a0703c"        # Sa gridlines (every octave)


def _dot_name(pc: str, octave: int) -> str:
    """Swara with dot octave marks: dot below = mandra, dot above = tara
    (combining U+0323 / U+0307, as in the reference sheet)."""
    if octave < 0:
        return pc + "\u0323" * -octave
    return pc + "\u0307" * octave


def _ref_grid(positions: Dict[str, float], lo_c: float, hi_c: float):
    """(cents, dotted name, is_sa) for every swara position within [lo_c, hi_c]."""
    out = []
    for octave in range(int(np.floor(lo_c / 1200)) - 1, int(np.ceil(hi_c / 1200)) + 1):
        for pc, p in positions.items():
            c = p + 1200 * octave
            if lo_c <= c <= hi_c:
                out.append((c, _dot_name(pc, octave), pc == "S"))
    return sorted(out)


def _seg_octave_name(swara: str) -> str:
    """``swaras.segment`` name (``.D``, ``S'``) -> dotted name (Ḍ, Ṡ)."""
    return _dot_name(swara.strip(".'"), swara.count("'") - swara.count("."))


def _hz_axis(ax, tonic: float, positions: Dict[str, float], lo_c: float, hi_c: float):
    """Log-Hz y axis: swara names on the left (bold), Hz ticks on the right."""
    from matplotlib.ticker import FixedLocator, NullLocator
    lo, hi = tonic * 2 ** (lo_c / 1200), tonic * 2 ** (hi_c / 1200)
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    grid = _ref_grid(positions, lo_c, hi_c)
    for c, _, sa in grid:
        ax.axhline(tonic * 2 ** (c / 1200), color=REF_SA if sa else REF_GRID, lw=1.0 if sa else 0.7, zorder=0)
    ax.yaxis.set_major_locator(FixedLocator([tonic * 2 ** (c / 1200) for c, _, _ in grid]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([n for _, n, _ in grid], fontsize=10, fontweight="bold")
    rx = ax.twinx()
    rx.set_yscale("log")
    rx.set_ylim(lo, hi)
    span = hi / lo
    step = 10 if span < 1.5 else 25 if span < 2.5 else 50
    ticks = [v for v in range(int(np.ceil(lo / step) * step), int(hi) + 1, step)]
    rx.yaxis.set_major_locator(FixedLocator(ticks))
    rx.yaxis.set_minor_locator(NullLocator())
    rx.set_yticklabels([str(v) for v in ticks], fontsize=8)
    rx.set_ylabel("Hz", fontsize=8)
    return rx


def sargam_report_pdf(out: Path, title: str, tonic: float, cents: np.ndarray, fps: float, swaras: dict,
                      words: Optional[List[dict]] = None, note: str = "", panel_s: float = 30.0) -> Path:
    """Pitch-contour sargam report in the style of the reference sheet
    ``pitch_contour_sargam.pdf``.

    Page 1: the whole contour in Hz (log axis) with swara gridlines (dot
    below = mandra, dot above = tara), meend / glide bands (glides moving
    >= 150 cents), and a bar chart of the share of sustained time per swara.
    Then one ``panel_s`` window per page: contour, sustained swaras
    (stable >= 0.25 s) as red bars labelled with their swara, meend bands,
    and (optional) lyric ``words`` = [{text, start, line_start}] along the
    bottom; a word starting a lyric line is prefixed with its line number.
    ``note`` is printed under the page-1 title (e.g. how Sa was found).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    words = words or []
    t = np.arange(len(cents)) / fps
    hz = tonic * 2 ** (cents / 1200)                 # NaN stays NaN (unvoiced gap)
    pos = swaras["positions_cents"]
    segs = swaras["segments"]
    meends = [s for s in segs if s["kind"] == "glide" and abs(s["cents_end"] - s["cents"]) >= 150]
    sus = [s for s in segs if s["kind"] == "stable" and s["sustained"]]
    v = ~np.isnan(cents)
    glo, ghi = np.percentile(cents[v], [1, 99.5]) if v.any() else (-500, 1500)
    glo, ghi = float(glo) - 60, float(ghi) + 60
    dur = float(t[-1]) if len(t) else 0.0
    legend = [Line2D([], [], color=REF_LINE, lw=1.5, label="pitch contour"),
              Line2D([], [], color=REF_SUS, lw=3, label="sustained swara (\u2265 0.25 s)"),
              Patch(color=REF_MEEND, label="meend / glide (\u2265 150 \u00a2)")]
    sa_name = f"Sa = {tonic:.1f} Hz"
    out.parent.mkdir(parents=True, exist_ok=True)

    def style(ax):
        """Reference look: paper background, plain spines."""
        ax.set_facecolor(REF_BG)
        ax.tick_params(labelsize=8)

    with PdfPages(out) as pdf:
        # page 1: whole recording + share of sustained time per swara
        fig = plt.figure(figsize=(16, 7))
        gs = fig.add_gridspec(1, 2, width_ratios=[6, 1], wspace=0.12, left=0.03, right=0.98, top=0.88, bottom=0.09)
        ax = fig.add_subplot(gs[0])
        for s in meends:
            ax.axvspan(s["start"], s["end"], color=REF_MEEND, alpha=0.6, lw=0, zorder=1)
        ax.plot(t, hz, color=REF_LINE, lw=1.0, zorder=3)
        rx = _hz_axis(ax, tonic, pos, glo, ghi)
        ax.set_xlim(0, dur)
        ax.set_xlabel("time (s)", fontsize=9)
        style(ax)
        ax.set_title(f"{title}: pitch contour, full recording, {sa_name}", fontsize=13, loc="left")
        if note:
            fig.text(0.03, 0.935, note, fontsize=9, color="#555")
        ax.legend(handles=legend[:1] + legend[2:], loc="upper right", fontsize=8)
        # share of sustained time, on the same Hz rows as the gridlines
        hx = fig.add_subplot(gs[1], sharey=rx)
        tot: Dict[str, float] = {}
        for s in sus:
            tot[s["swara"]] = tot.get(s["swara"], 0.0) + s["end"] - s["start"]
        allt = sum(tot.values()) or 1.0
        grid_c = [c for c, _, _ in _ref_grid(pos, glo, ghi)]
        for name, sec in tot.items():
            pc = name.strip(".'")
            c = pos.get(pc, 0.0) + 1200 * (name.count("'") - name.count("."))
            if not grid_c or c < glo or c > ghi:
                continue
            y = tonic * 2 ** (c / 1200)
            share = 100 * sec / allt
            hx.barh(y, share, height=y * 0.03, color=REF_SUS)
            hx.text(share + 0.5, y, f"{share:.0f}%", va="center", fontsize=8)
        for c in grid_c:
            hx.axhline(tonic * 2 ** (c / 1200), color=REF_GRID, lw=0.6, zorder=0)
        hx.set_xlim(0, max(10, 1.3 * max([100 * x / allt for x in tot.values()] or [10])))
        hx.tick_params(labelleft=False, labelright=False, labelsize=8)
        hx.set_title("share of sustained time", fontsize=9)
        style(hx)
        pdf.savefig(fig)
        plt.close(fig)

        # one panel per page
        for k in range(int(np.ceil(dur / panel_s))):
            t0, t1 = k * panel_s, min((k + 1) * panel_s, dur)
            m = (t >= t0) & (t < t1)
            vc = cents[m][~np.isnan(cents[m])]
            if len(vc):
                lo_c, hi_c = np.percentile(vc, [1, 99])
                lo_c, hi_c = max(glo, float(lo_c) - 120), min(ghi, float(hi_c) + 120)
                if hi_c - lo_c < 500:
                    mid = (lo_c + hi_c) / 2
                    lo_c, hi_c = mid - 250, mid + 250
            else:
                lo_c, hi_c = -100, 800
            fig, ax = plt.subplots(figsize=(16, 6))
            fig.subplots_adjust(left=0.035, right=0.955, top=0.9, bottom=0.1)
            for s in meends:
                if s["end"] > t0 and s["start"] < t1:
                    ax.axvspan(max(s["start"], t0), min(s["end"], t1), color=REF_MEEND, alpha=0.7, lw=0, zorder=1)
            ax.plot(t[m], hz[m], color=REF_LINE, lw=1.3, zorder=3)
            ylo, yhi = tonic * 2 ** (lo_c / 1200), tonic * 2 ** (hi_c / 1200)
            for s in sus:
                if s["end"] <= t0 or s["start"] >= t1:
                    continue
                y = tonic * 2 ** (s["cents"] / 1200)
                if not ylo <= y <= yhi:
                    continue
                ax.plot([max(s["start"], t0), min(s["end"], t1)], [y, y], color=REF_SUS, lw=3, alpha=0.9,
                         zorder=4, solid_capstyle="round")
                ax.text((max(s["start"], t0) + min(s["end"], t1)) / 2, y * 2 ** (40 / 1200),
                        _seg_octave_name(s["swara"]), color=REF_SUS_TXT, fontsize=10, fontweight="bold",
                        ha="center", va="bottom", zorder=5)
            _hz_axis(ax, tonic, pos, lo_c, hi_c)
            row = 0
            for w in words:
                if t0 <= w["start"] < t1:
                    txt = (f"L{w['line'] + 1}: " if w.get("line_start") else "") + w["text"]
                    ax.text(w["start"], 0.015 + 0.035 * (row % 3), txt, transform=ax.get_xaxis_transform(),
                            fontsize=7.5, color="#333" if not w.get("line_start") else "#000",
                            fontweight="bold" if w.get("line_start") else "normal", ha="left", va="bottom",
                            zorder=6)
                    row += 1
            ax.set_xlim(t0, t0 + panel_s)
            ax.set_xlabel("time (s)", fontsize=9)
            style(ax)
            ax.set_title(f"{title}: {t0:.0f}\u2013{t1:.0f} s   ({sa_name}; dot below = mandra saptak, "
                         "dot above = taar saptak)", fontsize=11, loc="left")
            ax.legend(handles=legend, loc="upper right", ncol=3, fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)
    return out


def song_contour_pdf(song_dir: Path, out: Path, title: str = "", fps: float = 50.0) -> Path:
    """Pitch-contour sargam report for a legacy film-song workspace song
    (``work/songs/<id>``), via :func:`sargam_report_pdf`.

    Reads ``annotation.json`` (tonic, syllables; times in frames at ``fps``)
    and ``f0.npy`` (Hz at ``fps``, 0 = unvoiced). Film songs have no raga, so
    the swaras are the pitch-histogram peaks relative to the estimated Sa
    (``swaras.segment`` with an unknown raga; lower case = komal, ``M`` =
    tivra). Lyric words are printed romanised (their phones joined;
    matplotlib cannot shape Devanagari). The cents track and the swara
    segments are written next to the PDF (``<stem>.f0.npz``, ``<stem>.swaras.json``).
    """
    import json

    from .stages.swaras import segment, to_json

    song_dir, out = Path(song_dir), Path(out)
    ann = json.loads((song_dir / "annotation.json").read_text(encoding="utf-8"))
    hz = np.load(song_dir / "f0.npy").astype(float)
    tonic = float(ann["tonic_hz"])
    cents = np.full(len(hz), np.nan)
    v = hz > 0
    cents[v] = 1200 * np.log2(hz[v] / tonic)
    segs, pos = segment(cents, "", fps=fps)
    sw = to_json(segs, pos, "")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".f0.npz"), cents=cents.astype(np.float32), fps=fps)
    out.with_suffix(".swaras.json").write_text(json.dumps(sw, ensure_ascii=False), encoding="utf-8")
    # words: syllable phones joined, grouped by (line, word), timed at the first syllable
    words: List[dict] = []
    seen_lines = set()
    for s in ann.get("syllables", []):
        key = (s["line"], s["word"])
        rom = "".join(s.get("phones") or []) or s["text"]
        if words and words[-1]["key"] == key:
            words[-1]["text"] += rom
            continue
        words.append({"key": key, "text": rom, "start": s["start"] / fps, "line": s["line"],
                      "line_start": s["line"] not in seen_lines})
        seen_lines.add(s["line"])
    note = (f"Sa estimated automatically from the vocals ({ann.get('tonic_source', 'auto')}, "
            f"confidence {ann.get('tonic_conf', float('nan')):.2f}); film song, no raga: the swara gridlines "
            "are the pitch-histogram peaks (lower case = komal, M = tivra). Lyrics romanised from the alignment.")
    return sargam_report_pdf(out, title or ann.get("song_id", song_dir.name), tonic, cents, fps, sw, words, note)
