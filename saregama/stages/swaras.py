"""Stage 5 - raga-aware swara segmentation of the tonic-relative contour.

The 100 Hz cents track (0 = Sa) is cut into segments of three kinds:

* ``stable``   the voice holds a svarasthana (pitch range < ``stable_range``
               cents over a ``win_s`` window, within ``snap_cents`` of an
               allowed position) for at least ``min_stable_s``;
               ``sustained`` = held for >= ``sustain_s`` (0.25 s, as in the
               reference contour plots)
* ``glide``    a monotonic piece (between turning points) moving
               >= ``meend_cents`` (meend / jaaru)
* ``kampita``  oscillation: >= 3 consecutive quick pieces (each half-cycle
               <= ``kampita_half_s``) between reversals of >= ``kampita_cents``,
               staying within ``kampita_max_span``
* ``transit``  any other voiced movement

Allowed positions come from the raga (``RAGAS``; letters are the 12 pitch
classes S r R g G m M P d D n N) and are then *refined per recording*: each
position moves to the nearest peak of the stable-frame histogram within 50
cents, so shruti offsets (e.g. a low Ga) are kept.  Unknown ragas use the
histogram peaks alone.  Every stable segment stores its exact median cents.

The flattened contour (stable segments at their swara, straight lines across
glides and transits) is the "flat" input for a flat -> expressive gamaka model.
"""
from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d

PCS = ("S", "r", "R", "g", "G", "m", "M", "P", "d", "D", "n", "N")
PC_CENTS = {p: 100 * i for i, p in enumerate(PCS)}
# Carnatic svarasthana names per pitch class (both spellings where they coincide)
CARNATIC = {"S": "S", "r": "R1", "R": "R2/G1", "g": "G2/R3", "G": "G3", "m": "M1", "M": "M2",
            "P": "P", "d": "D1", "D": "D2/N1", "n": "N2/D3", "N": "N3"}

# Union of arohana and avarohana swaras.  Keys are normalised names (see raga_key).
RAGAS: Dict[str, str] = {
    "mohanam": "S R G P D", "mohana": "S R G P D",
    "hamsadhvani": "S R G P N", "hindolam": "S g m d n", "madhyamavati": "S R m P n",
    "abhogi": "S R g m D", "suddhasaveri": "S R m P D", "suddha saveri": "S R m P D",
    "sankarabharanam": "S R G m P D N", "kalyani": "S R G M P D N",
    "kharaharapriya": "S R g m P D n", "todi": "S r g m P d n", "hanumatodi": "S r g m P d n",
    "mayamalavagaula": "S r G m P d N", "bhairavi": "S R g m P d D n",
    "kambhoji": "S R G m P D n N", "harikambhoji": "S R G m P D n",
    "natabhairavi": "S R g m P d n", "simhendramadhyamam": "S R g M P d N",
    "sanmukhapriya": "S R g M P d n", "pantuvarali": "S r G M P d N",
    "kamavardani": "S r G M P d N", "kamavardani/pantuvarali": "S r G M P d N",
    "charukesi": "S R G m P d n", "vacaspati": "S R G M P D n",
    "ritigaula": "S R g m P D n", "sriranjani": "S R g m D n", "srirañjani": "S R g m D n",
    "saveri": "S r m P d N", "vasanta": "S r G m D N", "saramati": "S R g m P d n",
    "harikambodhi": "S R G m P D n", "nata": "S g G m P n N", "arabhi": "S R G m P D N",
    "begada": "S R G m P D n N", "sahana": "S R G m P D n", "surati": "S R G m P D n",
    "kedaragaula": "S R G m P D n", "dhanyasi": "S r g m P d n", "atana": "S R G m P D n N",
    "anandabhairavi": "S R g m P D n", "kanada": "S R g m P D n", "bilahari": "S R G m P D n N",
    "kapi": "S R g G m P D n N", "behag": "S R G m M P D n N", "sindhubhairavi": "S r R g G m P d D n N",
    "suddhadhanyasi": "S g m P n", "kumudakriya": "S r G M P d N", "bahudari": "S G m P D n",
    "cittaranjani": "S R G m P D n", "jaganmohini": "S G m P N", "sindhumandari": "S r G m P d N",
    "padi": "S r m P d N", "saurastram": "S r G m P D n N", "karnataka devagandhari": "S R g m P D n",
    "hamir kalyani": "S R G m M P D N", "hamirkalyani": "S R G m M P D N",
    "manji": "S R g m P d D n", "suddha dhanyasi": "S g m P n", "sucaritra": "S g G M P d D",
    "mandu": "S R G m P D n N", "ragamalika": "",
}


def raga_key(name: str) -> str:
    s = unicodedata.normalize("NFKD", name)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = s.replace("ṁ", "m").strip()
    for a, b in (("sh", "s"), ("aa", "a"), ("ee", "i"), ("oo", "u"), ("th", "t"), ("dh", "d"),
                 ("bh", "bh")):
        s = s.replace(a, b)
    return s


def raga_swaras(name: str) -> Optional[List[str]]:
    table = {raga_key(k).replace(" ", ""): v for k, v in RAGAS.items()}
    for part in [name] + [p.strip() for p in name.replace(";", "/").split("/")]:
        k = raga_key(part).replace(" ", "")
        if k in table:
            return table[k].split() or None
    return None


@dataclass
class SwaraCfg:
    win_s: float = 0.1
    stable_range: float = 40.0
    snap_cents: float = 35.0
    min_stable_s: float = 0.08
    sustain_s: float = 0.25
    meend_cents: float = 150.0
    kampita_cents: float = 50.0       # minimum reversal that counts as a turn
    kampita_half_s: float = 0.25      # a kampita half-cycle is at most this long
    kampita_max_span: float = 500.0   # an oscillation stays within this range
    min_glide_s: float = 0.06         # shorter moves are transits (attacks, flicks)
    refine_cents: float = 50.0
    hist_min_share: float = 0.015      # histogram peak share to count as a used swara


@dataclass
class Segment:
    start: float
    end: float
    kind: str                  # stable | glide | kampita | transit
    swara: str = ""            # stable: e.g. "G", ".D", "S'"
    svarasthana: str = ""      # stable: Carnatic name, e.g. "G3"
    cents: float = 0.0         # stable: median cents above Sa; others: start cents
    cents_end: float = 0.0     # glide / transit / kampita: end cents
    span: float = 0.0          # max - min cents inside the segment
    sustained: bool = False


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def stable_histogram(cents: np.ndarray, cfg: SwaraCfg, fps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Folded (0..1200) 5-cent histogram of stable frames, normalised."""
    st = stable_mask(cents, cfg, fps)
    c = np.mod(cents[st], 1200)
    h = np.bincount((c // 5).astype(int) % 240, minlength=240).astype(float)
    h = gaussian_filter1d(h, 2.0, mode="wrap")
    return np.arange(240) * 5 + 2.5, h / (h.sum() + 1e-12)


def stable_mask(cents: np.ndarray, cfg: SwaraCfg, fps: float) -> np.ndarray:
    v = ~np.isnan(cents)
    w = max(3, int(round(cfg.win_s * fps)))
    hi = maximum_filter1d(np.where(v, cents, -1e9), w, mode="nearest")
    lo = minimum_filter1d(np.where(v, cents, 1e9), w, mode="nearest")
    return v & (hi - lo < cfg.stable_range)


def swara_positions(cents: np.ndarray, raga: str, cfg: SwaraCfg, fps: float) -> Dict[str, float]:
    """Pitch class -> refined position in cents (0..1200) for this recording."""
    bins, h = stable_histogram(cents, cfg, fps)
    allowed = raga_swaras(raga)
    peaks = [i for i in range(240) if h[i] >= h[i - 1] and h[i] >= h[(i + 1) % 240] and h[i] > 0]
    pos = {}
    if allowed:
        for pc in allowed:
            nom = PC_CENTS[pc]
            near = [i for i in peaks if abs((bins[i] - nom + 600) % 1200 - 600) <= cfg.refine_cents]
            if near:
                i = max(near, key=lambda i: h[i])
                pos[pc] = float(nom + ((bins[i] - nom + 600) % 1200 - 600))
            else:
                pos[pc] = float(nom)
    else:
        win = int(cfg.refine_cents // 5)
        for i in peaks:
            lo = [h[(i + d) % 240] for d in range(-win, win + 1)]
            if h[i] < max(lo) or h[i] * win * 2 < cfg.hist_min_share:
                continue
            pc = PCS[int(round(bins[i] / 100)) % 12]
            if pc not in pos or h[i] > h[int(pos[pc] // 5) % 240]:
                pos[pc] = float(bins[i] if bins[i] < 1150 else bins[i] - 1200)
        pos.setdefault("S", 0.0)
    pos["S"] = 0.0
    return dict(sorted(pos.items(), key=lambda kv: kv[1]))


def swara_name(pc: str, octave: int) -> str:
    return ("." * -octave if octave < 0 else "") + pc + ("'" * octave if octave > 0 else "")


def _nearest(c: float, positions: Dict[str, float]) -> Tuple[str, int, float]:
    """(pitch class, octave, distance) of the allowed position nearest to c."""
    best = None
    for pc, p in positions.items():
        octave = int(np.round((c - p) / 1200))
        d = abs(c - (p + 1200 * octave))
        if best is None or d < best[2]:
            best = (pc, octave, d)
    return best


def turning_points(c: np.ndarray, min_amp: float) -> List[int]:
    """Indices of the extrema between reversals of at least ``min_amp`` cents."""
    pts, direction, ext_i = [], 0, 0
    hi_i = lo_i = 0
    for k in range(1, len(c)):
        x = c[k]
        if direction == 0:
            if x > c[hi_i]:
                hi_i = k
            if x < c[lo_i]:
                lo_i = k
            if x - c[lo_i] >= min_amp:
                direction, ext_i = 1, k
                if lo_i > 0:
                    pts.append(lo_i)
            elif c[hi_i] - x >= min_amp:
                direction, ext_i = -1, k
                if hi_i > 0:
                    pts.append(hi_i)
        elif direction == 1:
            if x >= c[ext_i]:
                ext_i = k
            elif c[ext_i] - x >= min_amp:
                pts.append(ext_i)
                direction, ext_i = -1, k
        else:
            if x <= c[ext_i]:
                ext_i = k
            elif x - c[ext_i] >= min_amp:
                pts.append(ext_i)
                direction, ext_i = 1, k
    return pts


def _movement(seg: np.ndarray, base: int, fps: float, cfg: "SwaraCfg") -> List[Tuple[int, int, str]]:
    """Split a non-stable stretch at its turning points and label the pieces:
    runs of >= 3 quick alternating pieces are kampita, other pieces moving
    >= meend_cents are glides, the rest transits.  Returns (start, end, kind)
    in absolute frames, merged so neighbours never share a kind."""
    cuts = [0] + turning_points(seg, cfg.kampita_cents) + [len(seg)]
    pieces = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] > cuts[i]]
    quick = [(e - s) / fps <= cfg.kampita_half_s for s, e in pieces]
    kinds = [""] * len(pieces)
    i = 0
    while i < len(pieces):
        j = i
        lo, hi = np.inf, -np.inf
        while j < len(pieces) and quick[j]:
            s_, e_ = pieces[j]
            lo2, hi2 = min(lo, float(seg[s_:e_].min())), max(hi, float(seg[s_:e_].max()))
            if hi2 - lo2 > cfg.kampita_max_span:
                break
            lo, hi, j = lo2, hi2, j + 1
        if j - i >= 3:
            for k in range(i, j):
                kinds[k] = "kampita"
        i = max(j, i + 1)
    for k, (s, e) in enumerate(pieces):
        if not kinds[k]:
            net = abs(float(seg[min(e, len(seg) - 1)] - seg[s]))
            long_enough = (e - s) / fps >= cfg.min_glide_s
            kinds[k] = "glide" if net >= cfg.meend_cents and long_enough else "transit"
    out: List[Tuple[int, int, str]] = []
    for (s, e), kind in zip(pieces, kinds):
        if out and out[-1][2] == kind and kind != "glide":
            out[-1] = (out[-1][0], base + e, kind)
        else:
            out.append((base + s, base + e, kind))
    return out


def segment(cents: np.ndarray, raga: str, fps: float = 100.0, cfg: SwaraCfg = SwaraCfg()
            ) -> Tuple[List[Segment], Dict[str, float]]:
    positions = swara_positions(cents, raga, cfg, fps)
    v = ~np.isnan(cents)
    st = stable_mask(cents, cfg, fps)
    # label stable frames with their nearest allowed swara (within snap_cents)
    lab = np.full(len(cents), "", dtype=object)
    for k in np.where(st)[0]:
        pc, octv, d = _nearest(float(cents[k]), positions)
        if d <= cfg.snap_cents:
            lab[k] = swara_name(pc, octv)
    min_len = int(round(cfg.min_stable_s * fps))
    segs: List[Segment] = []
    for a, b in _runs(v):
        k = a
        stables = []
        while k < b:                                   # stable runs of one label
            if lab[k]:
                j = k
                while j < b and lab[j] == lab[k]:
                    j += 1
                if j - k >= min_len:
                    stables.append((k, j, lab[k]))
                k = j
            else:
                k += 1
        cursor = a
        for s, e, name in stables + [(b, b, None)]:
            if s > cursor:                             # movement between stables
                for ms, me, kind in _movement(cents[cursor:s], cursor, fps, cfg):
                    seg = cents[ms:me]
                    segs.append(Segment(float(ms / fps), float(me / fps), kind,
                                        cents=round(float(seg[0]), 1),
                                        cents_end=round(float(cents[min(me, b - 1)]), 1),
                                        span=round(float(seg.max() - seg.min()), 1)))
            if name is None:
                break
            seg = cents[s:e]
            pc = name.strip(".'")
            segs.append(Segment(float(s / fps), float(e / fps), "stable", swara=name, svarasthana=CARNATIC[pc],
                                cents=round(float(np.median(seg)), 1),
                                span=round(float(seg.max() - seg.min()), 1),
                                sustained=bool((e - s) / fps >= cfg.sustain_s)))
            cursor = e
    return segs, positions


def flatten(segs: List[Segment], n: int, positions: Dict[str, float], fps: float = 100.0) -> np.ndarray:
    """Flat contour in cents: stable segments at their (refined) swara
    position, straight lines across everything else; NaN where unvoiced."""
    flat = np.full(n, np.nan, dtype=np.float32)
    for s in segs:
        a, b = int(round(s.start * fps)), int(round(s.end * fps))
        if s.kind == "stable":
            pc = s.swara.strip(".'")
            octv = s.swara.count("'") - s.swara.count(".")
            flat[a:b] = positions[pc] + 1200 * octv
    for s in segs:
        if s.kind == "stable":
            continue
        a, b = int(round(s.start * fps)), int(round(s.end * fps))
        left = flat[a - 1] if a > 0 and not np.isnan(flat[a - 1]) else s.cents
        right = flat[b] if b < n and not np.isnan(flat[b]) else s.cents_end
        flat[a:b] = np.linspace(left, right, b - a + 2)[1:-1]
    return flat


def summary(segs: List[Segment]) -> Dict[str, object]:
    dur = lambda kind: sum(s.end - s.start for s in segs if s.kind == kind)
    total = sum(s.end - s.start for s in segs) or 1.0
    by_swara: Dict[str, float] = {}
    for s in segs:
        if s.kind == "stable":
            by_swara[s.swara] = by_swara.get(s.swara, 0.0) + s.end - s.start
    return {"voiced_s": round(total, 2),
            "share": {k: round(dur(k) / total, 4) for k in ("stable", "glide", "kampita", "transit")},
            "n_stable": int(sum(s.kind == "stable" for s in segs)),
            "n_sustained": int(sum(s.sustained for s in segs)),
            "n_glide": int(sum(s.kind == "glide" for s in segs)),
            "stable_time_by_swara": {k: round(v, 2) for k, v in sorted(by_swara.items(), key=lambda kv: -kv[1])}}


def to_json(segs: List[Segment], positions: Dict[str, float], raga: str) -> dict:
    return {"raga": raga, "raga_swaras": raga_swaras(raga), "positions_cents": positions,
            "summary": summary(segs), "segments": [asdict(s) for s in segs]}
