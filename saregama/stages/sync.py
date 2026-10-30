"""Put Saraga stems on the mix's clock.

The stems and the stereo mix are separate mp3 files whose start times differ
by a few milliseconds (measured 0-9 ms, constant over a recording, no drift).
For each stem we estimate its lag against the mix with GCC-PHAT on 10 s
windows every 20 s.  The melodic close-mic stems (vocal, violin) agree to
the sample; the mridangam and ghatam stems often scatter or lock onto another
path in the mix.  The recording's offset is the median lag of the melodic
stems whose windows agree (IQR below ``tight_ms``), and every stem is shifted
by that one offset, so all stems stay mutually aligned.

Sign convention: ``lag_s > 0`` means an event at stem time t is heard in the
mix at t + lag_s, so the stem is delayed by ``lag_s`` to put it on the mix clock.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

from ..audio import load_audio

SYNC_SR = 16000          # analysis rate; lag resolution 62.5 us
# The stems come from one multitrack recorder and are mutually synchronous, so
# one offset fits all of them.  Melodic close-mic stems give it reliably; the
# mridangam stems often lock onto a different path in the mix (seen: -11 to
# -36 ms against the vocal) and are only used when no melodic stem is tight.
REFERENCE_STEMS = ("vocal", "vocal-s", "violin")
WRITE_SR = 44100         # stems are written at Saraga's native rate


@dataclass
class StemLag:
    lag_s: float             # median lag over good windows
    iqr_ms: float            # spread of the window lags
    n_windows: int
    drift_ms_per_min: float
    tight: bool              # used for the consensus


@dataclass
class SyncResult:
    offset_s: float                              # applied to every stem
    stems: Dict[str, StemLag] = field(default_factory=dict)
    consensus_stems: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"offset_s": self.offset_s, "consensus_stems": self.consensus_stems,
                "flags": self.flags, "stems": {k: asdict(v) for k, v in self.stems.items()}}


def gcc_phat(a: np.ndarray, b: np.ndarray, max_lag: int):
    """Lag (samples) of ``a`` inside ``b`` and the PHAT peak-to-median ratio."""
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    x = np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n))
    x /= np.abs(x) + 1e-9
    c = np.fft.irfft(x, n)
    c = np.concatenate([c[-max_lag:], c[:max_lag + 1]])
    k = int(np.argmax(c))
    return k - max_lag, float(c[k] / (np.median(np.abs(c)) + 1e-9))


def stem_lag(stem: np.ndarray, mix: np.ndarray, sr: int = SYNC_SR, win_s: float = 10.0,
             step_s: float = 20.0, max_lag_s: float = 0.1, min_rms: float = 1e-3,
             min_quality: float = 8.0) -> StemLag:
    n = min(len(stem), len(mix))
    win, step, max_lag = int(win_s * sr), int(step_s * sr), int(max_lag_s * sr)
    rows = []
    for s0 in range(step, n - win - step + 1, step):
        a, b = stem[s0:s0 + win], mix[s0:s0 + win]
        if np.sqrt(np.mean(a ** 2)) < min_rms:
            continue
        lag, q = gcc_phat(a, b, max_lag)
        if q >= min_quality:
            rows.append((s0 / sr, lag / sr))
    if not rows:
        return StemLag(0.0, float("inf"), 0, 0.0, False)
    t, lags = np.array(rows).T
    iqr = float(np.subtract(*np.percentile(lags, [75, 25])) * 1000)
    drift = float(np.polyfit(t, lags, 1)[0] * 60 * 1000) if len(rows) > 3 else 0.0
    return StemLag(float(np.median(lags)), iqr, len(rows), drift, False)


def estimate_offsets(mix_path: Path, stem_paths: Dict[str, Path], tight_ms: float = 0.5,
                     agree_ms: float = 0.5) -> SyncResult:
    mix = load_audio(mix_path, SYNC_SR)
    lags = {k: stem_lag(load_audio(p, SYNC_SR), mix) for k, p in stem_paths.items()}
    for v in lags.values():
        v.tight = v.n_windows >= 3 and v.iqr_ms <= tight_ms
    tight = [k for k, v in lags.items() if v.tight and k in REFERENCE_STEMS]
    tight = tight or [k for k, v in lags.items() if v.tight]
    flags = []
    if not tight:
        flags.append("no_tight_stem")
        pool = [k for k in ("vocal", "violin") if k in lags and lags[k].n_windows] or list(lags)
    else:
        pool = tight
    offset = float(np.median([lags[k].lag_s for k in pool]))
    spread = max(abs(lags[k].lag_s - offset) for k in pool) * 1000
    if spread > agree_ms:
        flags.append(f"tight_stems_disagree_{spread:.2f}ms")
    if "vocal" in lags and not lags["vocal"].tight:
        flags.append("vocal_not_tight")
    return SyncResult(offset, lags, sorted(pool), flags)


def shift(x: np.ndarray, lag_s: float, sr: int, n_out: int) -> np.ndarray:
    """Delay ``x`` by ``lag_s`` (negative = advance) and fit it to ``n_out`` samples."""
    k = int(round(lag_s * sr))
    y = np.concatenate([np.zeros(k, x.dtype), x]) if k >= 0 else x[-k:]
    if len(y) >= n_out:
        return y[:n_out]
    return np.concatenate([y, np.zeros(n_out - len(y), x.dtype)])


def write_aligned_stems(mix_path: Path, stem_paths: Dict[str, Path], offset_s: float,
                        out_dir: Path, sr: int = WRITE_SR) -> Dict[str, Path]:
    """Offset-corrected mono FLAC stems with exactly the mix's length; the two
    mridangam heads are also summed into ``mridangam``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_out = len(load_audio(mix_path, sr))
    out, cache = {}, {}
    for k, p in stem_paths.items():
        y = shift(load_audio(p, sr), offset_s, sr, n_out)
        cache[k] = y
        out[k] = out_dir / f"{k}.flac"
        sf.write(out[k], y, sr, subtype="PCM_16")
    if "mridangam-left" in cache and "mridangam-right" in cache:
        y = cache["mridangam-left"] + cache["mridangam-right"]
        peak = float(np.max(np.abs(y))) if len(y) else 0.0
        out["mridangam"] = out_dir / "mridangam.flac"
        sf.write(out["mridangam"], y / peak * 0.99 if peak > 0.99 else y, sr, subtype="PCM_16")
    return out


def bleed_scores(stems: Dict[str, Path], sr: int = 4000, hop_s: float = 0.05) -> Dict[str, Dict[str, float]]:
    """Correlation of each stem's log-energy envelope with every other stem's.
    High values between a stem and another source's stem indicate bleed."""
    env = {}
    hop = int(hop_s * sr)
    for k, p in stems.items():
        if k == "mridangam":
            continue
        x = load_audio(p, sr)
        n = len(x) // hop
        e = np.log(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-8)
        env[k] = (e - e.mean()) / (e.std() + 1e-9)
    n = min(len(e) for e in env.values())
    return {a: {b: round(float(np.mean(env[a][:n] * env[b][:n])), 3) for b in env if b != a}
            for a in env}
