"""Stage 4 - the 100 Hz F0 master track of an IR bundle.

For each cleaned vocal stem (``vocal_clean``, and ``vocal-s_clean`` for duos)
this writes ``f0_100hz.npz`` (or ``f0_100hz_vocal-s.npz``) with, per 10 ms frame
k centred on t = k / 100 s of the mix clock:

* ``f0_hz``      ensemble F0, 0 = unvoiced
* ``cents``      1200 log2(f0 / tonic), NaN when unvoiced
* ``agree``      ensemble agreement 0..1 (loss weight / QC)
* ``conf``       RMVPE peak salience 0..1
* ``energy_db``  RMS level (20 ms window)

The tonic is chosen among Saraga's ``ctonic``, the pitch-class histogram
estimate and their fifth/fourth relatives (Saraga's ctonic is sometimes Pa: it
jumps by a fifth between recordings of one concert) by how well each explains
the stable pitches of the raga; its octave is placed just below the singer's
median pitch.
QC compares against Saraga's own pitch track; that track is mix-based and its
timestamps lead the mix clock by ~13 ms, so the lag is searched, not assumed.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf

from ..ir import F0_HZ, Bundle
from .f0 import F0Estimator
from .tonic import estimate_tonic, snap_to_tonic_octave

TONIC_MARGIN = 0.05           # top-2 candidate scores closer than this -> review


def energy_db(x: np.ndarray, sr: int, n: int, fps: float = F0_HZ, win_s: float = 0.02) -> np.ndarray:
    hop = sr / fps
    w = int(win_s * sr)
    xp = np.pad(x.astype(np.float64), (w // 2, w + int(hop) + 1))
    starts = np.round(np.arange(n) * hop).astype(int)
    c = np.concatenate([[0.0], np.cumsum(xp ** 2)])
    e = (c[starts + w] - c[starts]) / w
    return (10 * np.log10(e + 1e-10)).astype(np.float32)


def compare_with_saraga(f0: np.ndarray, ref_t: np.ndarray, ref_f: np.ndarray,
                        fps: float = F0_HZ, max_lag_ms: int = 30) -> dict:
    """Raw pitch accuracy (50 cents) of ``f0`` against Saraga's track at the
    best lag; lag_ms > 0 means Saraga's timestamps must be delayed."""
    t = np.arange(len(f0)) / fps
    best = None
    for lag in range(-max_lag_ms, max_lag_ms + 1):
        ref = np.interp(t, ref_t + lag / 1000, ref_f)
        rv = np.interp(t, ref_t + lag / 1000, (ref_f > 0).astype(float)) > 0.99
        m = (f0 > 0) & rv
        if m.sum() < 100:
            continue
        c = 1200 * np.log2(f0[m] / ref[m])
        rpa = float(np.mean(np.abs(c) < 50))
        if best is None or rpa > best["rpa50"]:
            best = {"lag_ms": lag, "rpa50": round(rpa, 4),
                    "octave_err": round(float(np.mean(np.abs(np.abs(c) - 1200) < 100)), 4),
                    "both_voiced": round(float(m.mean()), 4)}
    return best or {}


def tonic_score(f0: np.ndarray, tonic: float, raga: str, fps: float = F0_HZ) -> dict:
    """How well ``tonic`` explains the stable pitches: share of stable time on
    the raga's swaras (if the raga is known) plus the share on Sa and half the
    share on Pa (Sa and Pa dominate Carnatic melody)."""
    from .swaras import PC_CENTS, SwaraCfg, raga_swaras, stable_histogram
    c = np.full(len(f0), np.nan)
    v = f0 > 0
    c[v] = 1200 * np.log2(f0[v] / tonic)
    bins, h = stable_histogram(c, SwaraCfg(), fps)
    near = lambda pos: h[np.abs((bins - pos + 600) % 1200 - 600) <= 35].sum()
    allowed = raga_swaras(raga)
    fit = float(sum(near(PC_CENTS[p]) for p in allowed)) if allowed else 0.0
    sa, pa = float(near(0)), float(near(700))
    use_pa = not allowed or "P" in allowed
    return {"tonic_hz": round(float(tonic), 3), "raga_fit": round(fit, 4), "sa": round(sa, 4),
            "pa": round(pa, 4), "score": round(fit + sa + (0.5 * pa if use_pa else 0.0), 4)}


def choose_tonic(f0: np.ndarray, saraga_tonic: Optional[float], raga: str = ""
                 ) -> Tuple[float, str, List[dict]]:
    """Pick Sa among Saraga's ctonic, the histogram estimate, and their fifth /
    fourth relatives (Saraga's ctonic is sometimes Pa or Ma), then place it
    just below the singer's median pitch.  Returns (tonic, source, candidates)."""
    est, _ = estimate_tonic(f0)
    cands = []
    for src, base in (("saraga_ctonic", saraga_tonic), ("histogram", est)):
        if not base:
            continue
        for rel, ratio in (("", 1.0), ("*3/2", 1.5), ("*4/3", 4 / 3)):
            t = base * ratio
            if any(abs((1200 * np.log2(t / c["tonic_hz"]) + 600) % 1200 - 600) < 30 for c in cands):
                continue
            cands.append({**tonic_score(f0, t, raga), "source": src + rel})
    if not cands:
        return 0.0, "none", []
    best = max(cands, key=lambda c: c["score"])
    return snap_to_tonic_octave(best["tonic_hz"], f0), best["source"], cands


def run_f0(b: Bundle, est: F0Estimator, saraga_pitch=None) -> dict:
    ir = b.load()
    report = {}
    for stem, suffix in (("vocal_clean", ""), ("vocal-s_clean", "_vocal-s")):
        if stem not in ir["stems"]:
            continue
        x, sr = sf.read(str(b.dir / ir["stems"][stem]["path"]), dtype="float32")
        out = est(x, sr)
        f0 = out["f0"]
        n = len(f0)
        conf = out["conf"].get("rmvpe", (f0 > 0).astype(np.float32))
        if stem == "vocal_clean":
            saraga_tonic = ir.get("tonic_saraga_hz", ir.get("tonic_hz"))
            tonic, source, cands = choose_tonic(f0, saraga_tonic, ir.get("raaga", ""))
            ir["tonic_saraga_hz"] = saraga_tonic
            ir["tonic_hz"], ir["tonic_source"] = round(tonic, 3), source
            ir["tonic_candidates"] = cands
            scores = sorted((c["score"] for c in cands), reverse=True)
            if len(scores) > 1 and scores[0] - scores[1] < TONIC_MARGIN:
                ir["flags"] = sorted(set(ir.get("flags", [])) | {"tonic_ambiguous"})
        tonic = ir["tonic_hz"]
        cents = np.full(n, np.nan, dtype=np.float32)
        v = f0 > 0
        cents[v] = 1200 * np.log2(f0[v] / tonic)
        np.savez_compressed(b.path(f"f0_100hz{suffix}.npz"), fps=np.float32(F0_HZ), f0_hz=f0,
                            cents=cents, agree=out["f0_agree"], conf=conf[:n],
                            energy_db=energy_db(x, sr, n),
                            **{f"track_{k}": t for k, t in out["tracks"].items()})
        rep = {"frames": n, "voiced": round(float(v.mean()), 4),
               "agree_voiced": round(float(out["f0_agree"][v].mean()) if v.any() else 0.0, 4),
               "octave_fixed_frames": int(out["octave_fixed_frames"])}
        if saraga_pitch is not None:
            rep["vs_saraga"] = compare_with_saraga(f0, *saraga_pitch)
        report[stem] = rep
    ir.setdefault("qc", {})["f0"] = report
    b.save(ir)
    return report


def concert_consensus(tonics: List[Tuple[str, float, float]], tol_cents: float = 40.0
                      ) -> Optional[float]:
    """Sa pitch class shared by a concert: the duration-weighted pitch class
    (as Hz in some octave) that most recordings agree with within
    ``tol_cents``.  ``tonics`` = [(mbid, tonic_hz, duration_s)]."""
    if not tonics:
        return None
    pcs = np.array([np.mod(1200 * np.log2(t / 440.0), 1200) for _, t, _ in tonics])
    w = np.array([d for _, _, d in tonics])
    dist = lambda a, b: np.abs((a - b + 600) % 1200 - 600)
    support = [w[dist(pcs, p) <= tol_cents].sum() for p in pcs]
    best = int(np.argmax(support))
    members = dist(pcs, pcs[best]) <= tol_cents
    ref = pcs[best]
    mean_pc = ref + np.average(((pcs[members] - ref + 600) % 1200) - 600, weights=w[members])
    return float(440.0 * 2 ** (mean_pc / 1200))


def apply_tonic(b: Bundle, tonic: float, source: str) -> None:
    """Set a bundle's tonic and recompute the cents curves it stores."""
    ir = b.load()
    for name in ("f0_100hz.npz", "f0_100hz_vocal-s.npz"):
        path = b.path(name)
        if not path.exists():
            continue
        z = dict(np.load(path))
        f0 = z["f0_hz"]
        cents = np.full(len(f0), np.nan, dtype=np.float32)
        v = f0 > 0
        cents[v] = 1200 * np.log2(f0[v] / tonic)
        z["cents"] = cents
        np.savez_compressed(path, **z)
    ir["tonic_hz"], ir["tonic_source"] = round(float(tonic), 3), source
    b.save(ir)
