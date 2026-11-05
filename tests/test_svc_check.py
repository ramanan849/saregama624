"""SVC check maths: 50 fps sampling, SoulX quantisation, gamaka frames, contour
metrics and probe-phrase selection (no model needed)."""
import numpy as np

from saregama.svc_check import (PhraseCfg, at_50fps, best_window, contour_metrics, gamaka_mask, pick_phrases,
                                quantise_like_soulx, summarise)


def _gamaka(n, fps, hz=200.0, depth=150.0, rate=5.0):
    """An oscillating contour (kampita-like) around ``hz``: depth in cents, rate in Hz."""
    t = np.arange(n) / fps
    return hz * 2 ** (depth * np.sin(2 * np.pi * rate * t) / 1200)


def test_at_50fps_takes_every_second_frame():
    """Frame k of the output is input frame 2k; zeros are kept, nothing interpolated."""
    f = np.array([100, 0, 0, 110, 120, 130], dtype=np.float32)
    assert at_50fps(f).tolist() == [100, 0, 120]


def test_quantisation_error_is_bounded():
    """SoulX's 20-cent bins: at most 10 cents off; unvoiced stays 0."""
    f = np.array([0.0, 150.0, 151.3, 233.7])
    q = quantise_like_soulx(f)
    assert q[0] == 0 and np.all(np.abs(1200 * np.log2(q[1:] / f[1:])) <= 10.0 + 1e-6)


def test_gamaka_mask_and_metrics():
    """A fast oscillation is mostly gamaka; a copy scores zero error and r = 1; a
    flattened copy loses the gamaka (large gamaka RMSE)."""
    ref = _gamaka(400, 50.0)
    c = 1200 * np.log2(ref / 100.0)
    assert gamaka_mask(c).mean() > 0.5
    m = contour_metrics(ref, ref.copy())
    assert m["rmse_cents"] == 0 and m["pearson_r"] > 0.999 and m["voicing_recall"] == 1.0
    flat = np.full_like(ref, 200.0)
    assert contour_metrics(ref, flat)["rmse_gamaka_cents"] > 100
    half = ref.copy()
    half[::2] = 0
    assert abs(contour_metrics(ref, half)["voicing_recall"] - 0.5) < 0.01


def test_best_window_and_pick_phrases():
    """Windows must be voiced enough and must not overlap the excluded span; the
    phrase picker spreads over ragas first."""
    voiced = np.zeros(3000)
    voiced[500:2500] = 1
    score = np.zeros(3000)
    score[1500:1800] = 1
    t = best_window(voiced, score, 100.0, 8.0, 1.0, 0.6)
    assert t is not None and 7.0 <= t <= 15.0
    assert best_window(voiced, voiced, 100.0, 8.0, 1.0, 0.6, exclude=(5.0, 25.0)) is None
    f0 = np.zeros(6000)
    f0[200:5800] = _gamaka(5600, 100.0)
    recs = [{"mbid": f"m{i}", "raga": r, "singer": "s", "tonic_hz": 150.0, "f0": f0}
            for i, r in enumerate(["a", "a", "a", "b", "c"])]
    ph = pick_phrases(recs, PhraseCfg(n=3))
    assert [p["raga"] for p in ph] == ["a", "b", "c"]
    for p in ph:
        assert p["prompt_start"] + 20 <= p["target_start"] or p["target_start"] + 8 <= p["prompt_start"]
    rows = [{"raga": "a", "generated": {"rmse_cents": 30.0, "rmse_gamaka_cents": 60.0, "pearson_r": 0.9,
                                        "voicing_recall": 0.9}}]
    s = summarise(rows)
    assert s["generated"]["rmse_cents"] == 30.0 and s["measure"]["rmse_cents"] is None
