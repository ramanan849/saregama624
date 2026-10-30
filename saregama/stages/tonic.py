"""Stage 3 - tonic (Sa) estimation.

Pitch-class histogram method: fold voiced F0 into one octave at 10-cent
resolution, smooth, and score each candidate pitch class by the mass at the
candidate (Sa) plus weighted mass a fifth (Pa) and a fourth (Ma) above.  Sa and Pa
are the most-sung and most-stable pitches in Hindustani, Carnatic, bhajan and
most film music, so the fifth term separates Sa from a strong Ga or Pa.

The octave is chosen so that Sa sits at or just below the singer's median pitch
(madhya saptak).  Known tonics (from liner notes or the reviewer) override this;
pass them in songs.csv.  For film songs "tonic" simply means the key centre.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

BIN = 10                      # cents per histogram bin
NB = 1200 // BIN
REF = 55.0                    # A1, arbitrary reference for pitch classes


def pitch_class_histogram(f0: np.ndarray, weights: Optional[np.ndarray] = None,
                          stable_only: bool = True) -> np.ndarray:
    v = f0 > 0
    c = 1200 * np.log2(f0[v] / REF)
    w = np.ones_like(c) if weights is None else np.asarray(weights)[v]
    if stable_only and len(c) > 3:
        # weight steady frames more: sustained Sa/Pa dominate, glides do not
        slope = np.abs(np.gradient(c))
        w = w * (slope < 20)
    h = np.bincount(np.mod(np.round(c / BIN).astype(int), NB), weights=w, minlength=NB)
    return gaussian_filter1d(h.astype(float), sigma=1.5, mode="wrap")


def estimate_tonic(f0: np.ndarray, pa_weight: float = 0.6, ma_weight: float = 0.2,
                   weights: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Return (tonic_hz, confidence 0..1)."""
    if (f0 > 0).sum() < 50:
        return 0.0, 0.0
    h = pitch_class_histogram(f0, weights)
    h = h / (h.sum() + 1e-12)
    idx = np.arange(NB)
    score = h + pa_weight * h[(idx + 70) % NB] + ma_weight * h[(idx + 50) % NB]
    best = int(np.argmax(score))
    order = np.sort(score)[::-1]
    # confidence: margin of the best over the best candidate at least 50 cents away
    far = score.copy()
    for d in range(-5, 6):
        far[(best + d) % NB] = -1
    conf = float(np.clip((score[best] - far.max()) / (score[best] + 1e-12) * 3, 0, 1))
    # refine the peak to sub-bin precision
    a, b, c = score[(best - 1) % NB], score[best], score[(best + 1) % NB]
    den = a - 2 * b + c
    off = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
    pc_cents = (best + off) * BIN
    # octave placement: highest Sa at or below the median sung pitch
    med = 1200 * np.log2(np.median(f0[f0 > 0]) / REF)
    k = np.floor((med - pc_cents) / 1200)
    tonic = REF * 2 ** ((pc_cents + 1200 * k) / 1200)
    return float(tonic), conf


def snap_to_tonic_octave(tonic_hz: float, f0: np.ndarray) -> float:
    """Move a given tonic (any octave) to the octave just below the median pitch."""
    if tonic_hz <= 0 or (f0 > 0).sum() == 0:
        return tonic_hz
    med = np.median(f0[f0 > 0])
    k = np.floor(np.log2(med / tonic_hz))
    return float(tonic_hz * 2 ** k)
