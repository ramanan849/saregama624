"""Stage 2 - F0 on the 50 fps grid, from an ensemble of trackers.

Backends
--------
rmvpe     SoulX-Singer's RMVPE wrapper (needs the SoulX repo + Preprocess weights).
fcpe      torchfcpe (pip install torchfcpe; weights bundled).
crepe     torchcrepe (pip install torchcrepe; weights bundled).
annotator The unified annotator model trained in ``annotator_model`` (later rounds).
yin       Pure-numpy YIN.  No GPU, no weights; less robust to breath and bleed but
          a useful extra vote and the fallback for tests.

The ensemble octave-aligns every member to the first backend, takes the per-frame
median in cents, and stores the agreement fraction as a confidence curve.  No
smoothing is ever applied to voiced frames: gamak, kampita and meend must survive.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..audio import FPS, HOP, SR, frame_signal, n_frames, resample
from ..config import F0Cfg

Track = Tuple[np.ndarray, np.ndarray]     # (f0 Hz with 0 = unvoiced, confidence 0..1)


# ---------------------------------------------------------------------------
# YIN (numpy)
# ---------------------------------------------------------------------------
def yin(x: np.ndarray, sr: int = SR, fmin: float = 60.0, fmax: float = 1400.0,
        threshold: float = 0.15, n: Optional[int] = None,
        silence_db: float = -55.0) -> Track:
    hop = int(round(sr / FPS))
    tau_min = max(2, int(sr / fmax))
    tau_max = int(np.ceil(sr / fmin))
    w = max(tau_max, int(0.025 * sr))           # integration window
    win = w + tau_max + 2
    if n is None:
        n = int(np.ceil(len(x) / hop))
    # frame k: analysis window (first w samples) centred on sample k*hop
    xp = np.pad(x.astype(np.float64), (w // 2, win + hop))
    fr = xp[np.arange(win)[None, :] + (np.arange(n) * hop)[:, None]]
    nfft = 1 << int(np.ceil(np.log2(win + w)))
    X1 = np.fft.rfft(fr[:, :w], nfft)
    X2 = np.fft.rfft(fr, nfft)
    r = np.fft.irfft(np.conj(X1) * X2, nfft)[:, :tau_max + 2]
    sq = np.concatenate([np.zeros((n, 1)), np.cumsum(fr ** 2, axis=1)], axis=1)
    taus = np.arange(tau_max + 2)
    e0 = sq[:, w][:, None]
    et = sq[:, taus + w] - sq[:, taus]
    d = np.maximum(e0 + et - 2 * r, 0.0)
    d[:, 0] = 0
    cums = np.cumsum(d[:, 1:], axis=1)
    cmnd = np.ones_like(d)
    cmnd[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(cums, 1e-12)

    f0 = np.zeros(n)
    conf = np.zeros(n)
    energy_db = 10 * np.log10(np.mean(fr[:, :w] ** 2, axis=1) + 1e-12)
    sub = cmnd[:, tau_min:tau_max + 1]
    below = sub < threshold
    has = below.any(axis=1)
    first = np.where(has, below.argmax(axis=1), sub.argmin(axis=1))
    for k in range(n):
        if energy_db[k] < silence_db:
            continue
        t = first[k]
        if has[k]:                                    # walk to the local minimum
            while t + 1 < sub.shape[1] and sub[k, t + 1] < sub[k, t]:
                t += 1
        val = sub[k, t]
        if val > 0.35:
            continue
        tau = t + tau_min
        if 1 <= t < sub.shape[1] - 1:                 # parabolic interpolation
            a, b, c = sub[k, t - 1], sub[k, t], sub[k, t + 1]
            den = a - 2 * b + c
            if abs(den) > 1e-12:
                tau = tau + 0.5 * (a - c) / den
        f0[k] = sr / tau
        conf[k] = 1.0 - val
    return f0.astype(np.float32), conf.astype(np.float32)


# ---------------------------------------------------------------------------
# neural backends (imported lazily)
# ---------------------------------------------------------------------------
class _Rmvpe:
    def __init__(self, cfg: F0Cfg, soulx_repo: str):
        repo = Path(soulx_repo).resolve()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from preprocess.tools.f0_extraction import F0Extractor  # type: ignore
        self.m = F0Extractor(model_path=cfg.rmvpe_model, device=cfg.device,
                             target_sr=SR, hop_size=HOP, max_duration=3600, verbose=False)

    def __call__(self, x: np.ndarray, n: int) -> Track:
        x16 = resample(x, SR, 16000)
        f16 = self.m.model.infer_from_audio(x16, thred=self.m.thred)
        f0 = self.m.interpolate_f0(f16, original_length=len(x16), original_sr=16000,
                                   target_sr=SR, hop_size=HOP, max_duration=3600)
        f0 = _fit(np.asarray(f0, dtype=np.float32), n)
        # RMVPE's linear interpolation smears onsets into unvoiced frames; mark
        # frames where the 16 kHz track is unvoiced on either side as unvoiced.
        t16 = np.arange(len(f16)) * 0.01
        idx = np.clip(np.searchsorted(t16, np.arange(n) / FPS), 1, len(f16) - 1)
        uv = (f16[idx] <= 0) | (f16[idx - 1] <= 0)
        f0[uv] = 0
        return f0, (f0 > 0).astype(np.float32)


class _Fcpe:
    def __init__(self, cfg: F0Cfg):
        import torch
        from torchfcpe import spawn_bundled_infer_model  # type: ignore
        self.torch = torch
        self.cfg = cfg
        self.m = spawn_bundled_infer_model(device=cfg.device)

    def __call__(self, x: np.ndarray, n: int) -> Track:
        t = self.torch.from_numpy(resample(x, SR, 16000)).float()[None, :, None].to(self.cfg.device)
        f0 = self.m.infer(t, sr=16000, decoder_mode="local_argmax", threshold=0.006,
                          f0_min=self.cfg.fmin, f0_max=self.cfg.fmax, interp_uv=False,
                          output_interp_target_length=n)
        f0 = f0.squeeze().cpu().numpy().astype(np.float32)
        return _fit(f0, n), (f0 > 0).astype(np.float32)


class _Crepe:
    def __init__(self, cfg: F0Cfg):
        import torch
        import torchcrepe  # type: ignore
        self.torch, self.tc, self.cfg = torch, torchcrepe, cfg

    def __call__(self, x: np.ndarray, n: int) -> Track:
        x16 = self.torch.from_numpy(resample(x, SR, 16000)).float()[None]
        pitch, per = self.tc.predict(x16, 16000, hop_length=320, fmin=self.cfg.fmin,
                                     fmax=self.cfg.fmax, model="full", batch_size=512,
                                     device=self.cfg.device, return_periodicity=True)
        pitch = _fit(pitch.squeeze(0).cpu().numpy(), n)
        per = _fit(per.squeeze(0).cpu().numpy(), n)
        pitch[per < 0.5] = 0
        return pitch.astype(np.float32), per.astype(np.float32)


def _fit(a: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(a).reshape(-1)
    return a[:n] if len(a) >= n else np.pad(a, (0, n - len(a)))


# ---------------------------------------------------------------------------
# ensemble and repair
# ---------------------------------------------------------------------------
def to_cents(f0: np.ndarray, ref_hz: float = 440.0) -> np.ndarray:
    out = np.full(len(f0), np.nan)
    v = f0 > 0
    out[v] = 1200 * np.log2(f0[v] / ref_hz)
    return out


def from_cents(c: np.ndarray, ref_hz: float = 440.0) -> np.ndarray:
    out = np.zeros(len(c), dtype=np.float32)
    v = ~np.isnan(c)
    out[v] = ref_hz * 2 ** (c[v] / 1200)
    return out


def ensemble(tracks: Dict[str, Track], order: List[str], agree_cents: float = 50.0
             ) -> Tuple[np.ndarray, np.ndarray]:
    names = [k for k in order if k in tracks]
    if not names:
        raise ValueError("no F0 tracks to combine")
    C = np.stack([to_cents(tracks[k][0]) for k in names])          # (m, n)
    ref = C[0]
    for i in range(1, len(names)):                                  # octave-align to ref
        diff = C[i] - ref
        for o in (-2400, -1200, 1200, 2400):
            m = np.abs(diff - o) < 150
            C[i, m] -= o
    voiced = ~np.isnan(C)
    nv = voiced.sum(axis=0)
    m = len(names)
    keep = nv * 2 >= m if m > 1 else nv > 0
    if m > 1:                                                       # ties: trust the reference
        keep |= (nv * 2 == m) & voiced[0]
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.where(voiced, C, np.nan), axis=0)
    med[~keep] = np.nan
    close = (np.abs(C - med[None]) <= agree_cents) & voiced
    if m == 1:
        agree = np.where(keep, tracks[names[0]][1], 0.0)
    else:
        agree = np.where(keep, close.sum(axis=0) / np.maximum(nv, 1), 0.0)
    return from_cents(med), agree.astype(np.float32)


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    out, k, n = [], 0, len(mask)
    while k < n:
        if mask[k]:
            j = k
            while j < n and mask[j]:
                j += 1
            out.append((k, j))
            k = j
        else:
            k += 1
    return out


def fix_octave_islands(f0: np.ndarray, max_island_frames: int) -> Tuple[np.ndarray, int]:
    """Fold back short stretches that sit ~one octave away from both neighbours."""
    c = to_cents(f0)
    fixed = 0
    for a, b in _runs(f0 > 0):
        seg = c[a:b]
        if len(seg) < 3:
            continue
        jumps = np.diff(seg)
        idx = np.where(np.abs(np.abs(jumps) - 1200) < 200)[0]
        k = 0
        while k < len(idx):
            s = idx[k] + 1
            sign = np.sign(jumps[idx[k]])
            # find the matching jump back
            e = None
            for j in idx[k + 1:]:
                if np.sign(jumps[j]) == -sign:
                    e = j + 1
                    break
            if e is not None and e - s <= max_island_frames:
                seg[s:e] -= sign * 1200
                fixed += e - s
                k = int(np.searchsorted(idx, e - 1)) + 1
            else:
                k += 1
        c[a:b] = seg
    return from_cents(c), fixed


def fill_short_gaps(f0: np.ndarray, max_gap: int, max_jump_cents: float = 300.0) -> np.ndarray:
    if max_gap <= 0:
        return f0
    c = to_cents(f0)
    for a, b in _runs(f0 <= 0):
        if a == 0 or b >= len(f0) or b - a > max_gap:
            continue
        if abs(c[b] - c[a - 1]) > max_jump_cents:
            continue
        c[a:b] = np.linspace(c[a - 1], c[b], b - a + 2)[1:-1]
    return from_cents(c)


class F0Estimator:
    def __init__(self, cfg: F0Cfg, soulx_repo: str = "../SoulX-Singer", annotator=None):
        self.cfg = cfg
        self.backends = {}
        for name in cfg.backends:
            if name == "yin":
                self.backends[name] = lambda x, n: yin(x, SR, cfg.fmin, cfg.fmax, n=n)
            elif name == "rmvpe":
                self.backends[name] = _Rmvpe(cfg, soulx_repo)
            elif name == "fcpe":
                self.backends[name] = _Fcpe(cfg)
            elif name == "crepe":
                self.backends[name] = _Crepe(cfg)
            elif name == "annotator":
                if annotator is None:
                    raise ValueError("f0 backend 'annotator' needs a loaded annotator model")
                self.backends[name] = lambda x, n: annotator.f0(x, n)
            else:
                raise ValueError(f"unknown f0 backend {name!r}")

    def __call__(self, vocals: np.ndarray) -> Dict[str, np.ndarray]:
        n = n_frames(len(vocals))
        tracks = {k: fn(vocals, n) for k, fn in self.backends.items()}
        f0, agree = ensemble(tracks, self.cfg.backends, self.cfg.agree_cents)
        octave_fixed = 0
        if self.cfg.fix_octave_jumps:
            f0, octave_fixed = fix_octave_islands(f0, int(self.cfg.max_octave_island_s * FPS))
        f0 = fill_short_gaps(f0, self.cfg.fill_gap_frames)
        f0[(f0 < self.cfg.fmin * 0.9) | (f0 > self.cfg.fmax * 1.1)] = 0
        return {"f0": f0.astype(np.float32), "f0_agree": agree,
                "octave_fixed_frames": octave_fixed,
                "tracks": {k: v[0] for k, v in tracks.items()}}
