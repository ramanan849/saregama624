"""Stem-to-mix sync: a known shift is recovered and undone."""
import numpy as np
import soundfile as sf

from saregama.stages.sync import SYNC_SR, estimate_offsets, shift, stem_lag


def _signal(sr, dur, seed):
    rng = np.random.default_rng(seed)
    n = int(sr * dur)
    t = np.arange(n) / sr
    # voice-like: harmonic tone with wandering pitch, plus percussive clicks
    f0 = 220 * 2 ** (0.3 * np.sin(2 * np.pi * 0.2 * t))
    ph = 2 * np.pi * np.cumsum(f0) / sr
    voice = sum(np.sin(k * ph) / k for k in range(1, 8)) * 0.1
    clicks = np.zeros(n)
    clicks[rng.integers(0, n, int(dur * 4))] = 1.0
    perc = np.convolve(clicks, np.exp(-np.arange(400) / 60.0) * rng.standard_normal(400), "same") * 0.3
    return voice.astype(np.float32), perc.astype(np.float32)


def test_stem_lag_recovers_shift():
    sr = SYNC_SR
    voice, perc = _signal(sr, 120.0, 0)
    lag_s = 0.00731
    mix = shift(voice, lag_s, sr, len(voice)) + shift(perc, lag_s, sr, len(perc))
    res = stem_lag(voice, mix)
    assert abs(res.lag_s - lag_s) < 1.0 / sr
    assert res.iqr_ms < 0.1 and res.n_windows >= 3


def test_estimate_offsets_and_alignment(tmp_path):
    sr = 44100
    voice, perc = _signal(sr, 120.0, 1)
    lag_s = -0.0042
    mix = shift(voice, lag_s, sr, len(voice)) + 0.7 * shift(perc, lag_s, sr, len(perc))
    paths = {"vocal": tmp_path / "v.wav", "mridangam-right": tmp_path / "m.wav"}
    sf.write(paths["vocal"], voice, sr)
    sf.write(paths["mridangam-right"], perc, sr)
    sf.write(tmp_path / "mix.wav", np.stack([mix, mix], 1), sr)
    res = estimate_offsets(tmp_path / "mix.wav", paths)
    assert abs(res.offset_s - lag_s) < 2.0 / SYNC_SR
    assert not res.flags
    aligned = shift(voice, res.offset_s, sr, len(voice))
    residual = stem_lag(aligned, mix, sr=sr)
    assert abs(residual.lag_s) < 0.5 / SYNC_SR
