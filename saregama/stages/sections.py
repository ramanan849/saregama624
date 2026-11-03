"""Automatic concert sections for recordings without Saraga's manual labels.

Only 46 of the 168 multitrack recordings carry manual sections, but the
aligner needs to know where lyrics can be (``align_kriti.region_lines``).  This
stage labels every second of a recording as one of

=================  ============================================  ==================
class              what is happening                             region label used
=================  ============================================  ==================
``vocal_alapana``  voice, no percussion (alapana, sloka, viruttam)  ``Vocal ālāp``
``violin_alapana`` violin, no voice, no percussion               ``Violin ālāp``
``kriti``          voice + percussion: the composition, neraval  ``Kriti``
``svara``          voice + percussion singing sargam (kalpana    ``Kalpanā svara``
                   svara, citta svara)
``tani``           percussion solo                               ``Tani āvartana``
=================  ============================================  ==================

The region labels are ones ``region_lines`` already understands ("Kriti"
means lyrics with every line allowed).

Features (one row per second, all from files earlier stages wrote):

* level of the lead vocal, violin and percussion (mridangam + ghatam) stems,
  relative to each stem's loud level in this recording, and the percussion
  onset rate;
* the voiced share of the lead (and second) singer (``f0_100hz*.npz``);
* MMS letter-spike rate, consonant share, and the share of spikes on the
  sargam consonants s r g m p d n (``mms_50hz.npz``);
* swara movement: stable-swara onsets per second, glide and kampita shares
  (``swaras.json``);

each also averaged over ±2 s and ±10 s of context.  A gradient-boosted
classifier trained on the labelled recordings gives per-second class
probabilities; a Viterbi pass with a switching cost turns them into sections
(no flicker), and regions shorter than ``min_region_s`` merge into neighbours.
``rules`` is a model-free baseline used for comparison.

Manual labels always win: ``sections`` writes ``ir['sections_auto']`` and the
aligner uses ``ir['sections']`` when it is non-empty.
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

from ..ir import Bundle

CLASSES = ("vocal_alapana", "violin_alapana", "kriti", "svara", "tani")
REGION_LABEL = {"vocal_alapana": "Vocal ālāp", "violin_alapana": "Violin ālāp", "kriti": "Kriti",
                "svara": "Kalpanā svara", "tani": "Tani āvartana"}
SARGAM_CONSONANTS = set("srgmpdn")
VOWELS = set("aeiou")
HOP_S = 1.0                     # one feature row per second
ENV_FPS = 100                   # energy envelope rate for levels and onsets
CONTEXT_S = (2, 10)             # half-widths of the context averages


def _plain(label: str) -> str:
    """Lower-case label without diacritics."""
    s = unicodedata.normalize("NFKD", label)
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower().strip()


def class_of_label(label: str) -> Optional[str]:
    """Map a Saraga section label to a class, or None for unusable labels ('-')."""
    lab = _plain(label)
    if "violin" in lab and "alap" in lab:
        return "violin_alapana"
    if "alap" in lab or "sloka" in lab or "viruttam" in lab:
        return "vocal_alapana"
    if "tani" in lab:
        return "tani"
    if "svara" in lab and "sahitya" not in lab:
        return "svara"
    if any(k in lab for k in ("pallavi", "caranam", "charanam", "nereval", "neraval", "mel kalam",
                              "tisram", "kriti")):
        return "kriti"
    return None


# ----------------------------------------------------------------------------- features

def envelope_db(path: Path, fps: int = ENV_FPS, n: Optional[int] = None) -> np.ndarray:
    """RMS level (dB) of an audio file on a ``fps`` grid, read block by block so
    hour-long stems never sit in memory at once."""
    info = sf.info(str(path))
    hop = int(round(info.samplerate / fps))
    out = []
    for blk in sf.blocks(str(path), blocksize=hop * 1000, dtype="float32", always_2d=True):
        x = blk.mean(axis=1)
        m = len(x) // hop
        if m:
            out.append(10 * np.log10(np.mean(x[:m * hop].reshape(m, hop) ** 2, axis=1) + 1e-10))
    e = np.concatenate(out) if out else np.zeros(0)
    if n is not None:
        e = e[:n] if len(e) >= n else np.pad(e, (0, n - len(e)), constant_values=-100.0)
    return e.astype(np.float32)


def _per_second(x: np.ndarray, fps: float, n_sec: int, fn=np.mean) -> np.ndarray:
    """Aggregate a frame-rate curve into ``n_sec`` one-second values with ``fn``."""
    k = int(round(fps * HOP_S))
    x = x[:n_sec * k]
    if len(x) < n_sec * k:
        x = np.pad(x, (0, n_sec * k - len(x)), constant_values=x[-1] if len(x) else 0)
    return fn(x.reshape(n_sec, k), axis=1)


def _relative(db: np.ndarray) -> np.ndarray:
    """Level relative to the stem's loud level in this recording (95th
    percentile), so gain differences between recordings do not matter."""
    return db - np.percentile(db, 95)


def onset_rate(env_db: np.ndarray, n_sec: int, fps: int = ENV_FPS, rise_db: float = 6.0) -> np.ndarray:
    """Percussion strokes per second: local maxima of the 20 ms level rise above ``rise_db``."""
    rise = np.zeros_like(env_db)
    rise[2:] = env_db[2:] - env_db[:-2]
    peak = (rise > rise_db) & (rise >= np.roll(rise, 1)) & (rise > np.roll(rise, -1))
    return _per_second(peak.astype(np.float32), fps, n_sec, np.sum)


def spike_features(logp: np.ndarray, n_sec: int, fps: int = 50) -> np.ndarray:
    """Per second: MMS letter spikes, consonant share, sargam-consonant share.
    ``logp`` is the (T, C) MMS_FA log-probability matrix (blank = column 0)."""
    import torchaudio
    labels = torchaudio.pipelines.MMS_FA.get_labels(star=None)
    am = logp.argmax(axis=1)
    start = np.concatenate([[True], am[1:] != am[:-1]]) & (am != 0)      # first frame of each run
    sec = (np.nonzero(start)[0] / fps).astype(int)
    letters = [labels[c] for c in am[start]]
    cons = np.array([l not in VOWELS for l in letters], dtype=float)
    sarg = np.array([l in SARGAM_CONSONANTS for l in letters], dtype=float)
    keep = sec < n_sec
    rate = np.bincount(sec[keep], minlength=n_sec).astype(float)
    c = np.bincount(sec[keep], weights=cons[keep], minlength=n_sec)
    g = np.bincount(sec[keep], weights=sarg[keep], minlength=n_sec)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.stack([rate, np.nan_to_num(c / rate), np.nan_to_num(g / rate)], axis=1)


def swara_features(swaras: dict, n_sec: int) -> np.ndarray:
    """Per second: stable-swara onsets, glide share, kampita share."""
    onsets = np.zeros(n_sec)
    glide = np.zeros(n_sec)
    kamp = np.zeros(n_sec)
    for s in swaras.get("segments", []):
        a, b = s["start"], s["end"]
        if s["kind"] == "stable" and int(a) < n_sec:
            onsets[int(a)] += 1
        target = glide if s["kind"] == "glide" else (kamp if s["kind"] == "kampita" else None)
        if target is not None:
            for k in range(int(a), min(n_sec, int(np.ceil(b)))):
                target[k] += max(0.0, min(b, k + 1) - max(a, k))
    return np.stack([onsets, glide, kamp], axis=1)


FEATURE_NAMES = ("vocal_db", "violin_db", "perc_db", "perc_onsets", "voiced", "voiced_second",
                 "spikes", "spike_cons", "spike_sargam", "stable_onsets", "glide", "kampita")


def features(b: Bundle) -> np.ndarray:
    """(seconds, features) matrix of one bundle: ``FEATURE_NAMES`` plus their
    ±2 s and ±10 s context averages."""
    ir = b.load()
    z = np.load(b.path("f0_100hz.npz"))
    fps = float(z["fps"])
    n_sec = int(len(z["f0_hz"]) / fps / HOP_S)
    st = ir["stems"]
    n_env = n_sec * ENV_FPS
    level = lambda name: (envelope_db(b.dir / st[name]["path"], n=n_env) if name in st
                          else np.full(n_env, -100.0, np.float32))
    vocal, violin = level("vocal_clean"), level("violin")
    perc_lin = 10 ** (level("mridangam") / 10) + 10 ** (level("ghatam") / 10)
    perc = 10 * np.log10(perc_lin + 1e-10)
    sec = lambda env: _relative(_per_second(env, ENV_FPS, n_sec))
    voiced = _per_second((z["f0_hz"] > 0).astype(np.float32), fps, n_sec)
    second = (_per_second((np.load(b.path("f0_100hz_vocal-s.npz"))["f0_hz"] > 0).astype(np.float32), fps, n_sec)
              if b.path("f0_100hz_vocal-s.npz").exists() else np.zeros(n_sec))
    spk = (spike_features(np.load(b.path("mms_50hz.npz"))["logp"].astype(np.float32), n_sec)
           if b.path("mms_50hz.npz").exists() else np.zeros((n_sec, 3)))
    sw = (swara_features(json.loads(b.path("swaras.json").read_text(encoding="utf-8")), n_sec)
          if b.path("swaras.json").exists() else np.zeros((n_sec, 3)))
    base = np.column_stack([sec(vocal), sec(violin), sec(perc), onset_rate(perc, n_sec), voiced, second,
                            spk, sw]).astype(np.float32)
    ctx = [_smooth(base, w) for w in CONTEXT_S]
    return np.concatenate([base] + ctx, axis=1)


def _smooth(x: np.ndarray, half: int) -> np.ndarray:
    """Centred moving average over ±``half`` rows (edges use the rows available)."""
    c = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    idx = np.arange(len(x))
    lo, hi = np.clip(idx - half, 0, len(x)), np.clip(idx + half + 1, 0, len(x))
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


def frame_labels(sections: Sequence[dict], n_sec: int) -> np.ndarray:
    """Class index per second from manual sections; -1 where unlabelled or unusable."""
    y = np.full(n_sec, -1, dtype=np.int64)
    for s in sections:
        c = class_of_label(s["label"])
        if c is None:
            continue
        a, e = int(round(s["start"] / HOP_S)), int(round(s["end"] / HOP_S))
        y[max(0, a):min(n_sec, e)] = CLASSES.index(c)
    return y


# ----------------------------------------------------------------------------- decisions

def rules(x: np.ndarray, perc_on_db: float = -25.0, voice_on: float = 0.3) -> np.ndarray:
    """Model-free baseline on the ±10 s context features: percussion + voice is
    kriti, percussion alone is tani, voice alone is vocal alapana, neither is
    violin alapana.  (Cannot see svara.)"""
    n = len(FEATURE_NAMES)
    ctx = x[:, 2 * n:3 * n]                           # the ±10 s averages
    perc = ctx[:, FEATURE_NAMES.index("perc_db")] > perc_on_db
    voice = np.maximum(ctx[:, FEATURE_NAMES.index("voiced")], ctx[:, FEATURE_NAMES.index("voiced_second")]) > voice_on
    out = np.where(perc & voice, CLASSES.index("kriti"),
                   np.where(perc, CLASSES.index("tani"),
                            np.where(voice, CLASSES.index("vocal_alapana"), CLASSES.index("violin_alapana"))))
    return out.astype(np.int64)


def train(X: List[np.ndarray], Y: List[np.ndarray], seed: int = 0):
    """Gradient-boosted classifier on the labelled seconds of the given recordings.
    Classes are re-weighted to equal total weight (svara and tani are rare)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    x = np.concatenate(X)
    y = np.concatenate(Y)
    keep = y >= 0
    x, y = x[keep], y[keep]
    counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
    w = (len(y) / (len(CLASSES) * np.maximum(counts, 1)))[y]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=seed)
    clf.fit(x, y, sample_weight=w)
    return clf


def proba(clf, x: np.ndarray) -> np.ndarray:
    """(seconds, len(CLASSES)) class probabilities; classes absent from training get 0."""
    p = np.zeros((len(x), len(CLASSES)))
    p[:, clf.classes_] = clf.predict_proba(x)
    return p


def viterbi_smooth(p: np.ndarray, switch_cost: float = 8.0) -> np.ndarray:
    """Most likely class sequence given per-second probabilities, paying
    ``switch_cost`` (log domain) for every change of class."""
    lp = np.log(np.maximum(p, 1e-6))
    T, K = lp.shape
    score = lp[0].copy()
    back = np.zeros((T, K), dtype=np.int64)
    for t in range(1, T):
        stay = score
        best_other = np.argmax(score)
        switch = score[best_other] - switch_cost
        back[t] = np.where(stay >= switch, np.arange(K), best_other)
        score = np.maximum(stay, switch) + lp[t]
    out = np.zeros(T, dtype=np.int64)
    out[-1] = int(np.argmax(score))
    for t in range(T - 1, 0, -1):
        out[t - 1] = back[t, out[t]]
    return out


def to_regions(cls: np.ndarray, min_region_s: float = 15.0, length_s: Optional[float] = None) -> List[dict]:
    """Merge per-second classes into regions [{start, end, label, class}];
    regions shorter than ``min_region_s`` are absorbed by the longer neighbour."""
    runs = []
    for k, c in enumerate(cls):
        if runs and runs[-1][2] == c:
            runs[-1][1] = k + 1
        else:
            runs.append([k, k + 1, int(c)])
    changed = True
    while changed and len(runs) > 1:
        changed = False
        i = min(range(len(runs)), key=lambda j: runs[j][1] - runs[j][0])
        if (runs[i][1] - runs[i][0]) * HOP_S >= min_region_s:
            break
        left = runs[i - 1] if i > 0 else None
        right = runs[i + 1] if i + 1 < len(runs) else None
        tgt = left if right is None or (left is not None and left[1] - left[0] >= right[1] - right[0]) else right
        tgt[0], tgt[1] = min(tgt[0], runs[i][0]), max(tgt[1], runs[i][1])
        del runs[i]
        merged = []
        for r in runs:                               # neighbours of one class join up
            if merged and merged[-1][2] == r[2]:
                merged[-1][1] = r[1]
            else:
                merged.append(r)
        runs, changed = merged, True
    out = []
    for a, e, c in runs:
        end = e * HOP_S if length_s is None or e < len(cls) else max(e * HOP_S, length_s)
        out.append({"start": round(a * HOP_S, 3), "end": round(end, 3), "label": REGION_LABEL[CLASSES[c]],
                    "class": CLASSES[c]})
    return out


def frame_accuracy(pred: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Accuracy over labelled seconds, and recall per class."""
    m = y >= 0
    acc = float(np.mean(pred[m] == y[m])) if m.any() else float("nan")
    rec = {c: round(float(np.mean(pred[m & (y == k)] == k)), 3) for k, c in enumerate(CLASSES) if (m & (y == k)).any()}
    return acc, rec
