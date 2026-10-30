"""Stage 6 - voice-quality curves on the 50 fps grid.

energy_db  RMS level (40 ms window).                       -> loudness knob
hnr_db     harmonics-to-noise ratio from the normalised autocorrelation at the
           F0 period; low HNR = breathy or noisy.             -> breathiness knob
h1h2_db    level of the first harmonic minus the second; high = breathy/lax,
           low or negative = pressed/tense phonation.         -> tension knob

HNR and H1-H2 are only meaningful where F0 > 0; elsewhere they are NaN here and
0 in the exported clips (the F0 track tells the model which frames count).
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..audio import SR, frame_signal, rms_db


def quality_curves(x: np.ndarray, f0: np.ndarray) -> Dict[str, np.ndarray]:
    n = len(f0)
    energy = rms_db(x, win=960)[:n]
    energy = np.pad(energy, (0, n - len(energy)), constant_values=energy[-1] if len(energy) else -100)

    win = 2048
    fr = frame_signal(x.astype(np.float64), win, n=n) * np.hanning(win)[None, :]
    spec = np.abs(np.fft.rfft(fr, axis=1)) + 1e-9
    df = SR / win
    hnr = np.full(n, np.nan)
    h1h2 = np.full(n, np.nan)
    voiced = np.where(f0 > 0)[0]
    if len(voiced):
        # autocorrelation via FFT of the windowed frames
        ac = np.fft.irfft(spec[voiced] ** 2, axis=1)
        ac0 = ac[:, :1]
        acn = ac / np.maximum(ac0, 1e-12)
        lag = np.clip(np.round(SR / f0[voiced]).astype(int), 2, win // 2 - 2)
        rows = np.arange(len(voiced))
        r = np.max(np.stack([acn[rows, lag - 1], acn[rows, lag], acn[rows, lag + 1]]), axis=0)
        r = np.clip(r, 1e-4, 0.9999)
        hnr[voiced] = np.clip(10 * np.log10(r / (1 - r)), -10, 40)

        def harmonic_db(mult):
            centre = f0[voiced] * mult / df
            lo = np.clip(np.floor(centre * 0.9).astype(int), 1, spec.shape[1] - 2)
            hi = np.clip(np.ceil(centre * 1.1).astype(int) + 1, 2, spec.shape[1] - 1)
            out = np.empty(len(voiced))
            for i in range(len(voiced)):
                out[i] = spec[voiced[i], lo[i]:hi[i]].max()
            return 20 * np.log10(out)

        h1h2[voiced] = harmonic_db(1) - harmonic_db(2)
    return {"energy_db": energy.astype(np.float32), "hnr_db": hnr.astype(np.float32),
            "h1h2_db": h1h2.astype(np.float32)}
