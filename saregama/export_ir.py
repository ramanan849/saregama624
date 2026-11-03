"""ICM-IR v0 dataset export (Phase 1 Model Landscape p2) from the IR bundles.

The bundles stay the canonical, editable record; this writes a flat dataset
for model training into ``<out>/``:

* ``recordings.parquet``  global level, one row per recording: tonic, raga,
  tala, laya, form, singer ids, lyric status, section source, split, flags
* ``syllables.parquet``   syllable level: text (as aligned), ISO-15919, phones,
  onset / duration, line / word / section, singer, matra (beat, akshara, sam),
  confidence flags, duet overlap
* ``swaras.parquet``      swara level: kind, IR name (``Sa Re_k Re Ga_k Ga Ma
  Ma_t Pa Dha_k Dha Ni_k Ni``) with octave, Carnatic svarasthana resolved for
  the raga, cents, matra
* ``frames/<mbid>.npz``    100 Hz frame level: ``cents_q`` (cents re Sa in
  10-cent steps, int16, ``UNVOICED`` when silent), ``voiced``, ``loudness_db``,
  ``phone_id`` (-1 = none; ids in ``phones.json``), ``flat_cents_q`` (the
  gamaka-free contour: S2's flat -> expressive pair), ``conf``; duets add the
  second singer's ``*_second`` arrays
* ``phones.json``         phone id table; ``splits.json`` the train / val /
  test assignment by concert; ``DATASHEET.md``

Frame k of every array is at t = k / 100 s on the mix clock.  Splits are by
concert, so no singer appears in two splits.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ir import Bundle
from .phoneset import PHONES
from .stages.sections import class_of_label
from .stages.swaras import raga_swaras
from .stages.tala import matra_of
from .translit import fit_syllables, is_notation, notation_syllables, syllabify_phones, to_phones, word_tokens

IR_SPEC = "ICM-IR v0"
UNVOICED = -32768                 # cents_q value of unvoiced frames
CENTS_STEP = 10                   # cents_q resolution (10-cent bins, as GaMaDHaNi)
IR_NAMES = {"S": "Sa", "r": "Re_k", "R": "Re", "g": "Ga_k", "G": "Ga", "m": "Ma", "M": "Ma_t",
            "P": "Pa", "d": "Dha_k", "D": "Dha", "n": "Ni_k", "N": "Ni"}
PHONE_ID = {p.name: i for i, p in enumerate(PHONES)}
TALA_SECTIONS = ("kriti", "svara", "tani")   # sections where a tala cycle runs


# ----------------------------------------------------------------------------- names

def swara_parts(name: str) -> Tuple[str, int]:
    """(pitch class letter, octave) of a swara name such as ``.D``, ``G`` or ``S'``."""
    return name.strip(".'"), name.count("'") - name.count(".")


def ir_swara_name(name: str) -> Tuple[str, int]:
    """ICM-IR name and octave (-1 mandra, 0 madhya, 1 tara) of a swara name."""
    pc, octave = swara_parts(name)
    return IR_NAMES[pc], octave


def svarasthana(pc: str, raga_pcs: Optional[Sequence[str]]) -> str:
    """Carnatic svarasthana of pitch class ``pc`` in a raga with pitch classes
    ``raga_pcs``.  Four positions have two names; the raga decides: 200 cents
    is G1 (not R2) when the raga has R1 and no other Ga; 300 is R3 (not G2)
    when the raga has G3 and no other Ri; 900 is N1 (not D2) when it has D1 and
    no other Ni; 1000 is D3 (not N2) when it has N3 and no other Dha."""
    have = set(raga_pcs or ())
    if pc == "R":
        return "G1" if "r" in have and not have & {"g", "G"} else "R2"
    if pc == "g":
        return "R3" if "G" in have and not have & {"r", "R"} else "G2"
    if pc == "D":
        return "N1" if "d" in have and not have & {"n", "N"} else "D2"
    if pc == "n":
        return "D3" if "N" in have and not have & {"d", "D"} else "N2"
    return {"S": "S", "r": "R1", "G": "G3", "m": "M1", "M": "M2", "P": "P", "d": "D1", "N": "N3"}[pc]


def singer_id(name: str) -> str:
    """Stable ASCII id of a singer (``Cherthala Ranganatha Sharma`` -> ``cherthala_ranganatha_sharma``)."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


# ----------------------------------------------------------------------------- splits

def assign_splits(concerts: Sequence[str], n_test: int = 2, n_val: int = 2,
                  test: Sequence[str] = (), val: Sequence[str] = ()) -> Dict[str, str]:
    """Concert -> split.  Concerts named in ``test`` / ``val`` go there; the rest
    of each split is filled deterministically (by a hash of the concert name),
    so the assignment is stable across runs and machines."""
    out = {c: "test" for c in test} | {c: "val" for c in val}
    rest = sorted((c for c in concerts if c not in out), key=lambda c: hashlib.sha1(c.encode()).hexdigest())
    for c in rest:
        if sum(v == "test" for v in out.values()) < n_test:
            out[c] = "test"
        elif sum(v == "val" for v in out.values()) < n_val:
            out[c] = "val"
        else:
            out[c] = "train"
    return out


# ----------------------------------------------------------------------------- per bundle

def section_class_at(t: float, sections: Sequence[dict]) -> Optional[str]:
    """Section class (``stages.sections.CLASSES``) at time ``t``, or None."""
    for s in sections:
        if s["start"] <= t < s["end"]:
            return s.get("class") or class_of_label(s["label"])
    return None


def _matra(t: float, tala: dict, sections: Sequence[dict]):
    """(beat, akshara, sam) at ``t`` when a reliable tala grid (annotated samas)
    covers a section where a cycle runs, else Nones."""
    if not tala or not tala.get("reliable") or section_class_at(t, sections) not in TALA_SECTIONS:
        return None, None, None
    m = matra_of(t, np.asarray(tala["beats"]), np.asarray(tala["beat_index"]), tala["aksharas_per_beat"])
    return m if m is not None else (None, None, None)


def instance_phones(inst: dict, lines: Dict[int, dict]) -> List[List[List[str]]]:
    """Phones per aligned syllable of one instance (words -> syllables ->
    phones), fitted to the aligned syllable count of every word."""
    if inst["section"] == "sargam":
        return [[to_phones(sy["text"]) for sy in w["syllables"]] for w in inst["words"]]
    line = lines.get(inst["line_id"], {})
    text = line.get("text", "")
    if is_notation(text):                          # notation lines are sung as sargam
        sarg = iter(notation_syllables(text))
        return [[to_phones(next(sarg, "sa")) for _ in w["syllables"]] for w in inst["words"]]
    tokens = word_tokens(text)
    roman = line.get("words", [])
    out, k = [], 0
    for w in inst["words"]:                        # instance words are an in-order subset of the line
        while k < len(roman) and roman[k] != w["text"]:
            k += 1
        token = tokens[k] if k < len(tokens) else w["text"]
        k += 1
        out.append(fit_syllables(syllabify_phones(to_phones(token)), len(w["syllables"])))
    return out


def syllable_rows(mbid: str, lyr: dict, singer: str, tala: dict, sections: Sequence[dict],
                  verified: bool) -> List[dict]:
    """One row per aligned syllable of one singer's ``lyrics.json``."""
    lines = {l["id"]: l for l in lyr.get("lines", [])}
    rows = []
    for ii, inst in enumerate(lyr.get("instances", [])):
        try:
            phones = instance_phones(inst, lines)
        except ValueError:
            phones = [[[] for _ in w["syllables"]] for w in inst["words"]]
        for wi, w in enumerate(inst["words"]):
            for si, sy in enumerate(w["syllables"]):
                ph = phones[wi][si] if wi < len(phones) and si < len(phones[wi]) else []
                beat, akshara, sam = _matra(sy["start"], tala, sections)
                rows.append({
                    "mbid": mbid, "singer": singer, "instance": ii, "line_id": inst["line_id"],
                    "section": inst["section"], "region": inst.get("region", ""), "word": w["text"],
                    "word_index": wi, "syllable": sy["text"], "syllable_index": si,
                    "iso": "".join(_iso_of_phones(ph)), "phones": "-".join(ph),
                    "start": round(sy["start"], 3), "end": round(sy["end"], 3),
                    "duration": round(sy["end"] - sy["start"], 3), "sargam": inst["section"] == "sargam",
                    "beat": beat, "akshara": akshara, "sam": sam,
                    "low_conf": "low_conf" in inst.get("flags", []), "overlap": inst.get("overlap", 0.0),
                    "lyrics_verified": verified})
    return rows


_PHONE_ISO = {"aa": "ā", "ii": "ī", "uu": "ū", "ee": "ē", "oo": "ō", "T": "ṭ", "Th": "ṭh", "D": "ḍ", "Dh": "ḍh",
              "N": "ṇ", "L": "ḷ", "sh": "ś", "Sh": "ṣ", "zh": "ḻ", "rr": "ṟ", "ny": "ñ", "ng": "ṅ"}


def _iso_of_phones(phones: Sequence[str]) -> List[str]:
    """ISO-15919 letters of a phone sequence (for syllable-level ISO text)."""
    return [_PHONE_ISO.get(p, p) for p in phones]


def swara_rows(mbid: str, sw: dict, raga: str, tala: dict, sections: Sequence[dict]) -> List[dict]:
    """One row per swara segment (stable segments get names; movement keeps cents)."""
    pcs = raga_swaras(raga)
    rows = []
    for s in sw.get("segments", []):
        name, octave, sthana = "", None, ""
        if s["kind"] == "stable" and s["swara"]:
            name, octave = ir_swara_name(s["swara"])
            sthana = svarasthana(swara_parts(s["swara"])[0], pcs)
        beat, akshara, sam = _matra(s["start"], tala, sections)
        rows.append({"mbid": mbid, "start": round(s["start"], 3), "end": round(s["end"], 3),
                     "duration": round(s["end"] - s["start"], 3), "kind": s["kind"], "swara": name,
                     "octave": octave, "svarasthana": sthana, "cents": s["cents"], "cents_end": s["cents_end"],
                     "span": s["span"], "sustained": bool(s.get("sustained")),
                     "beat": beat, "akshara": akshara, "sam": sam})
    return rows


def quantise_cents(cents: np.ndarray) -> np.ndarray:
    """Cents re Sa in ``CENTS_STEP`` steps as int16 (``UNVOICED`` where NaN)."""
    q = np.full(len(cents), UNVOICED, dtype=np.int16)
    v = ~np.isnan(cents)
    q[v] = np.round(cents[v] / CENTS_STEP).astype(np.int16)
    return q


def phone_track(n: int, rows: Sequence[dict], fps: int = 100, max_consonant_frames: int = 4) -> np.ndarray:
    """Per-frame phone id from syllable rows: consonants before the vowel take up
    to ``max_consonant_frames`` each at the start, consonants after it the same
    at the end, and the vowel fills the rest (an approximation: within a
    syllable only its span is aligned, not its phones)."""
    track = np.full(n, -1, dtype=np.int16)
    for r in rows:
        ph = [p for p in r["phones"].split("-") if p]
        a, b = int(round(r["start"] * fps)), min(n, int(round(r["end"] * fps)))
        if not ph or b <= a:
            continue
        ids = [PHONE_ID[p] for p in ph]
        vowels = [i for i, p in enumerate(ph) if PHONES[PHONE_ID[p]].kind == "vowel"]
        nuc = vowels[0] if vowels else len(ph) - 1
        pre, post = ph[:nuc], ph[nuc + 1:]
        c = min(max_consonant_frames, max(0, (b - a - 1) // max(1, len(pre) + len(post))))
        pos = a
        for i in range(len(pre)):
            track[pos:pos + c] = ids[i]
            pos += c
        end = b - c * len(post)
        track[pos:max(pos, end)] = ids[nuc]
        for i in range(len(post)):
            track[end + i * c:end + (i + 1) * c] = ids[nuc + 1 + i]
    return track


def frames(b: Bundle, rows_by_singer: Dict[str, List[dict]]) -> Dict[str, np.ndarray]:
    """The 100 Hz frame arrays of one bundle (see the module docstring)."""
    out = {}
    for suffix, tag in (("", ""), ("_vocal-s", "_second")):
        path = b.path(f"f0_100hz{suffix}.npz")
        if not path.exists():
            continue
        z = np.load(path)
        n = len(z["f0_hz"])
        out[f"cents_q{tag}"] = quantise_cents(z["cents"].astype(np.float64))
        out[f"voiced{tag}"] = z["f0_hz"] > 0
        out[f"loudness_db{tag}"] = z["energy_db"].astype(np.float16)
        out[f"conf{tag}"] = np.round(np.asarray(z["agree"], dtype=np.float32) * 255).astype(np.uint8)
        out[f"phone_id{tag}"] = phone_track(n, rows_by_singer.get("second" if tag else "lead", []))
    if b.path("flat_100hz.npz").exists():
        out["flat_cents_q"] = quantise_cents(np.load(b.path("flat_100hz.npz"))["flat_cents"].astype(np.float64))
    out["fps"] = np.int16(100)
    return out
