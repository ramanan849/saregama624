"""SVC rendering check: does SoulX-Singer's F0 conditioning keep gamakas?

SoulX-Singer has no training code, so before anyone writes a trainer this check
asks whether the released SVC model (``model-svc.pt``) can *render* a Carnatic
pitch contour at all.  Its F0 input is coarse (50 fps, 20-cent bins: see
``SoulXSingerSVC.f0_to_coarse``), and gamakas move fast.

Per probe phrase (``pick_phrases``):
- **target**: an ``dur``-second phrase of a recording's ``vocal_clean``, with our
  100 Hz F0 sampled at 50 fps (every second frame, never interpolated across
  unvoiced frames);
- **prompt**: ``prompt_s`` seconds of the same singer from elsewhere in the
  recording (the timbre reference);
- SoulX renders the target's content with the prompt's voice following our F0
  (``pitch_shift`` 0: same singer, same register);
- the rendered audio's F0 is re-measured with the pipeline's own ensemble and
  compared with the F0 we asked for (:func:`contour_metrics`).

Two floors put the numbers in context:
- **measure**: the same ensemble re-run on the *original* phrase against the
  target F0. This is the error of the measurement itself.
- **quant**: the target F0 against its own 20-cent quantisation. This is the
  best SoulX's input encoding allows.

Metrics follow the Feasibility Report: cents RMSE on frames voiced in both, the
same on gamaka frames (target slope |dc/dt| > ``GAMAKA_CENTS_PER_S``), Pearson r of
the cents curves, and voicing recall.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SOULX_FPS = 50.0               # SoulX's F0 / mel frame rate (24 kHz, hop 480)
F0_BIN_CENTS = 20.0            # SoulX's coarse F0 bin width
GAMAKA_CENTS_PER_S = 300.0     # target slope above which a frame counts as gamaka


@dataclass
class PhraseCfg:
    """How probe phrases are chosen."""
    n: int = 20                    # phrases wanted
    dur: float = 8.0               # target phrase length, s
    prompt_s: float = 20.0         # prompt length, s
    min_voiced: float = 0.6        # voiced share needed in the target window
    min_prompt_voiced: float = 0.5 # voiced share needed in the prompt window
    step_s: float = 1.0            # window search step


def at_50fps(f0_100: np.ndarray) -> np.ndarray:
    """100 Hz F0 to SoulX's 50 fps grid by taking every second frame (frame k
    is time k·20 ms); unvoiced frames stay 0, nothing is interpolated."""
    return np.asarray(f0_100[::2], dtype=np.float32)


def cents_of(f0: np.ndarray, ref_hz: float) -> np.ndarray:
    """Cents re ``ref_hz``; NaN where unvoiced (f0 <= 0)."""
    c = np.full(len(f0), np.nan, dtype=np.float64)
    v = f0 > 0
    c[v] = 1200 * np.log2(f0[v] / ref_hz)
    return c


def quantise_like_soulx(f0: np.ndarray, f0_min: float = 32.7031956625) -> np.ndarray:
    """``f0`` rounded to SoulX's 20-cent coarse bins and back to Hz (0 stays 0)."""
    out = np.zeros_like(f0, dtype=np.float64)
    v = f0 > 0
    bins = np.round(1200 * np.log2(np.maximum(f0[v], f0_min) / f0_min) / F0_BIN_CENTS)
    out[v] = f0_min * 2 ** (bins * F0_BIN_CENTS / 1200)
    return out


def gamaka_mask(cents: np.ndarray, fps: float = SOULX_FPS, thresh: float = GAMAKA_CENTS_PER_S) -> np.ndarray:
    """Frames where the contour moves faster than ``thresh`` cents/s. The slope
    is a central difference, taken only where the frame and both neighbours are
    voiced (no slope across a voicing gap)."""
    m = np.zeros(len(cents), dtype=bool)
    if len(cents) < 3:
        return m
    d = (cents[2:] - cents[:-2]) * fps / 2
    ok = np.isfinite(d)
    m[1:-1] = ok & (np.abs(np.where(ok, d, 0)) > thresh)
    return m


def contour_metrics(ref_hz: np.ndarray, out_hz: np.ndarray, fps: float = SOULX_FPS) -> dict:
    """Compare a rendered/measured F0 ``out_hz`` with the requested ``ref_hz``
    (same grid; lengths are cut to the shorter).

    Returns cents RMSE on frames voiced in both, RMSE on gamaka frames of the
    reference, Pearson r of the cents, voicing recall (share of reference-voiced
    frames voiced in the output), the gamaka share, and the frame counts.
    """
    n = min(len(ref_hz), len(out_hz))
    ref_hz, out_hz = np.asarray(ref_hz[:n], float), np.asarray(out_hz[:n], float)
    ref_c = cents_of(ref_hz, 100.0)
    out_c = cents_of(out_hz, 100.0)
    both = np.isfinite(ref_c) & np.isfinite(out_c)
    gam = gamaka_mask(ref_c, fps) & both
    err = out_c - ref_c

    def rmse(mask):
        """Root-mean-square cents error over ``mask`` (None if empty)."""
        return round(float(np.sqrt(np.mean(err[mask] ** 2))), 1) if mask.any() else None

    r = float(np.corrcoef(ref_c[both], out_c[both])[0, 1]) if both.sum() > 2 else None
    ref_v = np.isfinite(ref_c)
    return {"rmse_cents": rmse(both), "rmse_gamaka_cents": rmse(gam),
            "pearson_r": round(r, 4) if r is not None else None,
            "voicing_recall": round(float(both.sum() / max(ref_v.sum(), 1)), 3),
            "gamaka_share": round(float(gam.sum() / max(both.sum(), 1)), 3),
            "frames_both": int(both.sum()), "frames_gamaka": int(gam.sum())}


def best_window(voiced: np.ndarray, score: np.ndarray, fps: float, dur: float, step_s: float,
                min_voiced: float, exclude: tuple[float, float] | None = None) -> float | None:
    """Start time (s) of the ``dur``-second window with the highest mean ``score``
    among windows whose voiced share is at least ``min_voiced`` and that don't
    overlap ``exclude`` = (start, end); None if there is none."""
    w, step = int(dur * fps), max(1, int(step_s * fps))
    if len(voiced) < w:
        return None
    cv = np.concatenate([[0], np.cumsum(voiced)])
    cs = np.concatenate([[0], np.cumsum(score)])
    best, best_t = -np.inf, None
    for i in range(0, len(voiced) - w + 1, step):
        t0 = i / fps
        if exclude and t0 < exclude[1] and t0 + dur > exclude[0]:
            continue
        if (cv[i + w] - cv[i]) / w < min_voiced:
            continue
        s = (cs[i + w] - cs[i]) / w
        if s > best:
            best, best_t = s, t0
    return best_t


def pick_phrases(recs: list[dict], cfg: PhraseCfg = PhraseCfg()) -> list[dict]:
    """Choose up to ``cfg.n`` (target, prompt) pairs, spreading over ragas first.

    ``recs``: dicts with ``mbid``, ``raga``, ``singer``, ``f0`` (100 Hz, Hz) and
    ``tonic_hz``; ordered by preference. Round-robin over ragas: each raga gives
    its next recording until ``cfg.n`` phrases are found, so at least
    min(n, ragas) ragas appear. In each recording the target is the voiced window
    with the most gamaka (fast-moving) frames, and the prompt is the most voiced
    window that doesn't overlap it.
    """
    by_raga: dict[str, list[dict]] = {}
    for r in recs:
        by_raga.setdefault(r["raga"], []).append(r)
    order, out = [], []
    while any(by_raga.values()):
        for raga in list(by_raga):
            if by_raga[raga]:
                order.append(by_raga[raga].pop(0))
    for r in order:
        if len(out) >= cfg.n:
            break
        f0 = np.asarray(r["f0"], float)
        voiced = (f0 > 0).astype(float)
        gam = gamaka_mask(cents_of(f0, r["tonic_hz"]), 100.0).astype(float)
        t0 = best_window(voiced, gam, 100.0, cfg.dur, cfg.step_s, cfg.min_voiced)
        if t0 is None:
            continue
        p0 = best_window(voiced, voiced, 100.0, cfg.prompt_s, cfg.step_s, cfg.min_prompt_voiced,
                         exclude=(t0, t0 + cfg.dur))
        if p0 is None:
            continue
        out.append({"mbid": r["mbid"], "title": r.get("title", ""), "raga": r["raga"], "singer": r["singer"],
                    "tonic_hz": r["tonic_hz"], "target_start": round(t0, 2), "prompt_start": round(p0, 2),
                    "dur": cfg.dur, "prompt_s": cfg.prompt_s})
    return out


def summarise(rows: list[dict]) -> dict:
    """Medians of each metric over phrases, per comparison (``generated``,
    ``measure`` floor, ``quant`` floor), plus how many ragas were covered."""
    out = {"phrases": len(rows), "ragas": len({r["raga"] for r in rows})}
    for kind in ("generated", "measure", "quant"):
        out[kind] = {}
        for k in ("rmse_cents", "rmse_gamaka_cents", "pearson_r", "voicing_recall"):
            v = [r[kind][k] for r in rows if r.get(kind) and r[kind].get(k) is not None]
            out[kind][k] = round(float(np.median(v)), 3) if v else None
    return out
