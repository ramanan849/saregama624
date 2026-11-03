"""Tala grid: laya, beats and sama for every recording (ICM-IR's matra index).

ICM-IR v0 asks for the laya (tempo) at the global level and a matra index and
sam flag on every syllable / swara.  Saraga annotates sama for 63 of the 168
multitrack recordings, and gives a tempo file for 60.  Its tempo file is
``apm, bpm, cycle_s, aksharas_per_cycle, aksharas_per_beat`` (aksharas per
minute, beats per minute, sama-to-sama seconds; verified against the sama
annotations), e.g. Adi ``343, 86, 5.591, 32, 4``: 8 beats of 4 aksharas.

For every recording this stage produces ``ir['tala']``:

* ``source``: ``manual_sama`` (beats subdivided evenly between annotated samas)
  or ``estimated`` (below), and ``phase_conf`` for estimated samas;
* ``beats_per_cycle``, ``aksharas_per_beat``, ``laya_bpm``;
* ``beats`` (times) and ``beat_index`` (0 = sama), ``samas`` (times).

Estimation, from the clean percussion stems:

1. onset strength of mridangam (+ ghatam) at 100 Hz (spectral flux);
2. beat period: the autocorrelation peak of the onset strength within the
   tala's plausible tempo range (``TEMPO_RANGE``), refined by the cycle-lag
   peak (beats_per_cycle x period);
3. beats: dynamic-programming tracking at that period (``librosa.beat``);
4. sama: of the ``beats_per_cycle`` phases, the one whose beats carry the
   strongest bass-head (``mridangam-left``, the thoppi) strokes on average;
   ``phase_conf`` is the margin of that phase over the runner-up.

Tani and alapana have no tala, but percussion and composition both keep one
cycle, so the grid is tracked over the whole recording and the matra index is
only *used* where sections say a composition or tani is under way.
"""
from __future__ import annotations

import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np

ONSET_FPS = 100
# beats per cycle and aksharas per beat (default nadai) per tala name
TALAS: Dict[str, Tuple[int, int]] = {
    "adi": (8, 4), "rupaka": (3, 4), "misra chapu": (7, 2), "khanda chapu": (5, 2),
    "khanda-jati eka": (5, 2), "tisra-jati eka": (3, 4), "misra-jati jhampe": (10, 2),
    "atta": (14, 2),
}
TEMPO_RANGE = (40.0, 200.0)          # plausible beats per minute
PRIOR_BPM = 95.0                     # Saraga tempo files: 63-171 bpm, most 85-110
PRIOR_OCTAVES = 0.6                  # width of the log-normal tempo prior


def tala_key(name: str) -> str:
    """Normalised tala name (no diacritics, lower case) for ``TALAS`` lookup."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower().strip()
    return s.replace("ṁ", "m")


def tala_shape(name: str, tempo_file: Optional[List[float]] = None) -> Optional[Tuple[int, int]]:
    """(beats per cycle, aksharas per beat).  Saraga's tempo file wins over the
    tala name (some recordings' tala metadata disagrees with their tempo file)."""
    if tempo_file and len(tempo_file) >= 5 and tempo_file[4]:
        return int(round(tempo_file[3] / tempo_file[4])), int(tempo_file[4])
    return TALAS.get(tala_key(name)) if name else None


def load_tempo(path) -> List[float]:
    """Saraga ``tempo-manual.txt``: [apm, bpm, cycle_s, aksharas_per_cycle, aksharas_per_beat]."""
    with open(path, encoding="utf-8") as fh:
        return [float(x) for x in fh.read().replace(",", " ").split()]


# ----------------------------------------------------------------------------- grids

def grid_from_samas(samas: np.ndarray, beats_per_cycle: int) -> Tuple[np.ndarray, np.ndarray]:
    """Beats subdivided evenly between consecutive annotated samas.  Cycles
    more than 1.6x the median cycle (a gap in the annotation) are skipped.
    Returns (beat times, beat index within the cycle; 0 = sama)."""
    samas = np.sort(np.asarray(samas, dtype=float))
    if len(samas) < 2:
        return samas, np.zeros(len(samas), dtype=np.int64)
    med = float(np.median(np.diff(samas)))
    times, idx = [], []
    for a, b in zip(samas[:-1], samas[1:]):
        if b - a > 1.6 * med:
            times.append(a), idx.append(0)
            continue
        for k in range(beats_per_cycle):
            times.append(a + (b - a) * k / beats_per_cycle)
            idx.append(k)
    times.append(samas[-1]), idx.append(0)
    return np.array(times), np.array(idx, dtype=np.int64)


def onset_strength(x: np.ndarray, sr: int, fps: int = ONSET_FPS) -> np.ndarray:
    """Spectral-flux onset strength of percussion audio at ``fps`` (librosa).
    ``sr`` must be a multiple of ``fps``: a rounded hop (22050 / 100 -> 220)
    would stretch time by 0.2 % and make beats drift."""
    import librosa
    if sr % fps:
        raise ValueError(f"sample rate {sr} is not a multiple of {fps} frames/s")
    hop = sr // fps
    return librosa.onset.onset_strength(y=x, sr=sr, hop_length=hop, aggregate=np.median).astype(np.float32)


def beat_period(env: np.ndarray, beats_per_cycle: int, fps: int = ONSET_FPS) -> float:
    """Beat period (s): autocorrelation peak of the onset strength within
    ``TEMPO_RANGE``, then refined with the cycle-lag peak near
    beats_per_cycle x that period."""
    e = env - env.mean()
    n = len(e)
    f = np.fft.rfft(e, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    lo, hi = int(fps * 60 / TEMPO_RANGE[1]), int(fps * 60 / TEMPO_RANGE[0])
    # a click train correlates almost as well at 2 beats as at 1 (tempo octave
    # errors), so weight lags by a log-normal tempo prior, as librosa does
    bpm = 60.0 * fps / np.arange(lo, hi)
    prior = np.exp(-0.5 * (np.log2(bpm / PRIOR_BPM) / PRIOR_OCTAVES) ** 2)
    lag = lo + int(np.argmax(np.maximum(ac[lo:hi], 0) * prior))
    c = lag * beats_per_cycle                         # refine on the (sharper) cycle peak
    if c + c // 20 + 1 < n:
        w = max(2, c // 20)
        c = c - w + int(np.argmax(ac[c - w:c + w + 1]))
        y0, y1, y2 = ac[c - 1], ac[c], ac[c + 1]      # parabolic peak: sub-frame period
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
        return float((c + off) / beats_per_cycle / fps)
    return float(lag / fps)


def track_beats(env: np.ndarray, period_s: float, fps: int = ONSET_FPS) -> np.ndarray:
    """Beat times (s) by dynamic programming at a fixed tempo (``librosa.beat``)."""
    import librosa
    # the tempo is fixed; a moderate tightness lets beats lock onto the strokes
    # instead of drifting with a slightly wrong period
    _, beats = librosa.beat.beat_track(onset_envelope=env, sr=fps, hop_length=1, bpm=60.0 / period_s,
                                       tightness=100, trim=False, units="time")
    return np.asarray(beats, dtype=float)


def sama_phase(beats: np.ndarray, bass_env: np.ndarray, beats_per_cycle: int,
               fps: int = ONSET_FPS) -> Tuple[int, float]:
    """Which beat phase is sama: the phase whose beats carry the most bass-head
    onset strength.  Returns (phase, confidence = relative margin over the
    runner-up phase)."""
    # a stroke's onset peak can sit a frame or two off the tracked beat, so take
    # the strongest bass onset within +-30 ms of each beat
    w = max(1, int(round(0.03 * fps)))
    peak = np.lib.stride_tricks.sliding_window_view(np.pad(bass_env, w, mode="edge"), 2 * w + 1).max(axis=1)
    idx = np.clip(np.round(beats * fps).astype(int), 0, len(bass_env) - 1)
    strength = peak[idx]
    means = np.array([strength[p::beats_per_cycle].mean() if len(strength[p::beats_per_cycle]) else 0.0
                      for p in range(beats_per_cycle)])
    order = np.argsort(means)[::-1]
    best, second = means[order[0]], means[order[1]] if len(order) > 1 else 0.0
    return int(order[0]), float((best - second) / (best + 1e-9))


def estimate_grid(perc: np.ndarray, bass: np.ndarray, sr: int, beats_per_cycle: int
                  ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Beats, beat indices (0 = sama), beat period (s) and sama-phase confidence
    from percussion audio ``perc`` and bass-head audio ``bass`` (mono, ``sr``)."""
    env = onset_strength(perc, sr)
    period = beat_period(env, beats_per_cycle)
    beats = track_beats(env, period)
    if len(beats) < beats_per_cycle:
        return beats, np.zeros(len(beats), dtype=np.int64), period, 0.0
    phase, conf = sama_phase(beats, onset_strength(bass, sr), beats_per_cycle)
    index = (np.arange(len(beats)) - phase) % beats_per_cycle
    return beats, index.astype(np.int64), period, conf


# ----------------------------------------------------------------------------- evaluation

def beat_f_measure(est: np.ndarray, ref: np.ndarray, tol: float = 0.07) -> float:
    """F-measure of estimated vs reference beat times within ±``tol`` s
    (each reference beat matched at most once)."""
    if len(est) == 0 or len(ref) == 0:
        return 0.0
    used = np.zeros(len(ref), bool)
    hits = 0
    for t in est:
        j = int(np.argmin(np.abs(ref - t)))
        if abs(ref[j] - t) <= tol and not used[j]:
            used[j] = True
            hits += 1
    p, r = hits / len(est), hits / len(ref)
    return 2 * p * r / (p + r) if p + r else 0.0


def sama_hit_rate(est_samas: np.ndarray, ref_samas: np.ndarray, tol: float = 0.07) -> float:
    """Share of reference samas with an estimated sama within ±``tol`` s."""
    if len(ref_samas) == 0:
        return float("nan")
    if len(est_samas) == 0:
        return 0.0
    d = np.abs(ref_samas[:, None] - np.asarray(est_samas)[None, :]).min(axis=1)
    return float(np.mean(d <= tol))


def matra_of(t: float, beats: np.ndarray, beat_index: np.ndarray, aksharas_per_beat: int
             ) -> Optional[Tuple[int, int, bool]]:
    """(beat index, akshara index within the cycle, is-sama) of time ``t``, or
    None outside the beat grid."""
    if len(beats) < 2 or t < beats[0] or t > beats[-1]:
        return None
    k = int(np.searchsorted(beats, t, side="right") - 1)
    k = min(k, len(beats) - 2)
    frac = (t - beats[k]) / max(beats[k + 1] - beats[k], 1e-6)
    sub = min(aksharas_per_beat - 1, int(frac * aksharas_per_beat))
    b = int(beat_index[k])
    return b, b * aksharas_per_beat + sub, bool(b == 0 and sub == 0)
