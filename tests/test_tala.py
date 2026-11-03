"""Tala grid: shapes, grids from samas, matra lookup, metrics, and estimation on
a synthetic percussion track with a known tempo and sama."""
import numpy as np

from saregama.stages.tala import (beat_f_measure, estimate_grid, grid_from_samas, matra_of, sama_hit_rate,
                                  tala_shape)


def test_tala_shape_prefers_tempo_file():
    """The tempo file's aksharas decide; the name is the fallback."""
    assert tala_shape("Ādi") == (8, 4)
    assert tala_shape("Ādi", [340, 170, 2.471, 14, 2]) == (7, 2)       # metadata says Adi, file says chapu
    assert tala_shape("") is None


def test_grid_from_samas_and_matra():
    """Even subdivision between samas; a gap in the annotation is not filled."""
    beats, idx = grid_from_samas(np.array([0.0, 8.0, 16.0, 40.0, 48.0]), 8)
    assert idx[:9].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 0] and beats[1] == 1.0
    assert not ((beats > 16.0) & (beats < 40.0)).any()
    assert matra_of(1.6, beats, idx, 4) == (1, 6, False)               # beat 1, akshara 4 + 2
    assert matra_of(8.0, beats, idx, 4) == (0, 0, True)
    assert matra_of(-1.0, beats, idx, 4) is None


def test_metrics():
    """Beat F-measure and sama hit rate within 70 ms."""
    ref = np.arange(0, 10, 0.5)
    assert beat_f_measure(ref + 0.03, ref) == 1.0
    assert beat_f_measure(ref + 0.25, ref) == 0.0
    assert sama_hit_rate(np.array([0.0, 4.05]), np.array([0.0, 4.0, 8.0])) == 2 / 3


def _clicks(times, gain, sr, dur, rng):
    """Short decaying noise bursts at ``times`` with per-click ``gain``."""
    x = np.zeros(int(sr * dur), np.float32)
    burst = (rng.standard_normal(int(0.03 * sr)) * np.exp(-np.arange(int(0.03 * sr)) / (0.004 * sr)))
    for t, g in zip(times, gain):
        i = int(t * sr)
        x[i:i + len(burst)] += g * burst[:len(x) - i]
    return x


def test_estimate_grid_on_synthetic_percussion():
    """92 bpm, 8 beats per cycle, sama (strong bass) on beat 3 of the track."""
    sr, dur, rng = 16000, 90.0, np.random.default_rng(0)
    period = 60 / 92
    beats = np.arange(0.3, dur - 0.5, period)
    sama = (np.arange(len(beats)) % 8) == 3
    perc = _clicks(beats, np.where(sama, 1.0, 0.7), sr, dur, rng) + 0.01 * rng.standard_normal(int(sr * dur))
    bass = _clicks(beats[sama], np.ones(sama.sum()), sr, dur, rng) + _clicks(beats[~sama], 0.2 * np.ones((~sama).sum()), sr, dur, rng)
    est, index, est_period, conf = estimate_grid(perc.astype(np.float32), bass.astype(np.float32), sr, 8)
    assert abs(est_period / period - 1) < 0.02
    assert beat_f_measure(est, beats) > 0.9
    assert sama_hit_rate(est[index == 0], beats[sama]) > 0.9 and conf > 0.1     # clear margin
