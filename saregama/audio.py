"""Audio I/O and SoulX-Singer's frame grid.

SoulX-Singer works on 24 kHz audio with hop 480 samples, so one frame = 20 ms
and frame k covers time k * 0.02 s.  The legacy film-song workspace and the
SoulX exporter store syllable / note / clip boundaries as integer frames on
this grid, which makes ``sum(note durations) * 50 == len(f0)`` hold exactly.

The Saraga pipeline keeps its master curves at 100 Hz on the mix clock
(``ir.F0_HZ``) and only derives the 50 fps grid at export time.
"""
from __future__ import annotations

import shutil
import subprocess
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 24000
HOP = 480
FPS = SR // HOP            # 50 frames per second
FRAME_S = HOP / SR         # 0.02 s


def load_audio(path: str | Path, sr: int = SR, mono: bool = True) -> np.ndarray:
    """Load any audio file as float32 at ``sr``.  Falls back to ffmpeg for
    formats libsndfile cannot read (m4a, some mp3)."""
    path = str(path)
    try:
        x, file_sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        if not shutil.which("ffmpeg"):
            raise
        cmd = ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1" if mono else "2",
               "-ar", str(sr), "-"]
        raw = subprocess.run(cmd, check=True, capture_output=True).stdout
        x = np.frombuffer(raw, dtype=np.float32)
        return x.copy() if mono else x.reshape(-1, 2).T.copy()
    x = x.T                                   # (channels, samples)
    if mono:
        x = x.mean(axis=0)
    if file_sr != sr:
        g = gcd(int(file_sr), int(sr))
        x = resample_poly(x, sr // g, int(file_sr) // g, axis=-1).astype(np.float32)
    return x.astype(np.float32)


def save_audio(path: str | Path, x: np.ndarray, sr: int = SR) -> None:
    """Write ``x`` (mono, or channels-first) as 16-bit PCM; creates parent folders."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(x, dtype=np.float32).T, sr, subtype="PCM_16")


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase-resample ``x`` from ``sr_in`` to ``sr_out`` (no-op if equal)."""
    if sr_in == sr_out:
        return x
    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g, axis=-1).astype(np.float32)


def n_frames(n_samples: int) -> int:
    """Number of 20 ms frames covering ``n_samples`` (SoulX: ceil)."""
    return int(np.ceil(n_samples / HOP))


def frame_to_s(f) -> float:
    """Start time in seconds of 20 ms frame ``f``."""
    return float(f) * FRAME_S


def s_to_frame(t: float) -> int:
    """Nearest 20 ms frame index of time ``t`` (seconds)."""
    return int(round(t * FPS))


def frame_signal(x: np.ndarray, win: int, hop: int = HOP, n: int | None = None) -> np.ndarray:
    """Centered frames (frame k centred on sample k*hop) -> (n, win)."""
    if n is None:
        n = n_frames(len(x))
    pad = win // 2
    xp = np.pad(x, (pad, pad + win + hop))
    idx = np.arange(win)[None, :] + (np.arange(n) * hop)[:, None]
    return xp[idx]


def rms_db(x: np.ndarray, win: int = 960) -> np.ndarray:
    """Per-frame RMS level in dB on the 20 ms grid, ``win``-sample centred windows."""
    fr = frame_signal(x, win)
    return (10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-10)).astype(np.float32)


def loudness_normalize(x: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    """Scale ``x`` to ``target_dbfs`` RMS, then pull the peak below 0.99
    if that clipped.  Used for model inputs only, never for stems that must
    stay sample-aligned with the mix.
    """
    r = np.sqrt(np.mean(x ** 2) + 1e-12)
    g = 10 ** (target_dbfs / 20) / r
    y = x * g
    peak = np.max(np.abs(y)) + 1e-9
    if peak > 0.99:
        y = y * (0.99 / peak)
    return y.astype(np.float32)
