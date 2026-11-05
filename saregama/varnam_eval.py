"""Swara segmentation and tonic, scored on the Carnatic-Varnam dataset.

Carnatic-Varnam 1.1 (Koduri et al. 2014) has 28 solo vocal varnams (7 ragas, 5
singers, all Adi tala, drone only) with three things we can use as ground truth:
- the **notation** of each varnam (``notations/<raga>.yaml``): per section, a list
  of cycles, each 8 divisions of 4 svaras; ``,`` continues the previous svara,
  ``^`` / ``_`` mark the upper / lower octave;
- the **structure** per singer (``notations/<raga>/<singer>.yaml``): the order in
  which sections are sung and at how many svaras per tala tick (2 = first speed,
  4 = second speed). Keys repeat, so it is parsed line by line;
- the **tala ticks** (``annotations/taalas/<raga>/<singer>.svl``): 8 per cycle,
  labelled ``m.n``. The ``m.1`` ticks are manual, the rest are even divisions.
- **tonics** per singer (``annotations/tonics.yaml``).

:func:`expected_notes` lays the notation onto the ticks: each tick holds 2 or 4
svaras, evenly spaced, and ``,`` extends the previous note. Times are therefore
approximate inside a tick, since singers don't sing perfectly evenly; that is why
onset scores are reported at two tolerances.

Scores (:func:`score`):
- **label accuracy**: for each expected note, our stable-segment labels in the
  middle half of it; the note is right when their majority equals the notated
  svara (octave included, variant ignored: the raga fixes it). Notes with no
  stable frame there count as "unlabelled", not wrong (gamaka-only renditions).
  ``label_accuracy_any_octave`` ignores the octave too: the notation often leaves
  the tara S of an ascending "N S" unmarked.
- **change onsets**: at each notated change of svara, the nearest of our stable
  segments with that svara starting within ±``search_s``; hit rates within 100 ms
  and 200 ms, and the median absolute error of the matches.
- **tonic**: our raga-aware estimate against the dataset's, in cents.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class Note:
    """One notated svara laid onto the tala ticks (seconds)."""
    start: float
    end: float
    svara: str          # notation style: "S", "R", ... with "^" (tara) / "_" (mandra)


def parse_structure(path: Path) -> list[tuple[str, int]]:
    """``[(section, svaras_per_tick), ...]`` in sung order. Read line by line
    because the YAML repeats keys (a YAML loader would keep only the last)."""
    out, inside = [], False
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("structure:"):
            inside = True
            continue
        m = re.match(r"^\s+(\w+):\s*(\d+)\s*$", line)
        if inside and m:
            out.append((m.group(1), int(m.group(2))))
        elif inside and line and not line.startswith(" "):
            break
    return out


def parse_ticks(svl: Path) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Tick times (s) and ``(cycle, division)`` labels from a Sonic Visualiser
    time-instants layer."""
    root = ET.parse(svl).getroot()
    sr = float(root.find(".//model").get("sampleRate"))
    pts = [(int(p.get("frame")) / sr, p.get("label")) for p in root.iter("point")]
    pts.sort()
    labels = [tuple(int(x) for x in lab.split(".")) for _, lab in pts]
    return np.array([t for t, _ in pts]), labels


def notation_sections(path: Path) -> dict[str, list[str]]:
    """Section name -> flat list of notated svaras (``,`` kept)."""
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {k: [str(s) for cyc in v for div in cyc for s in div]
            for k, v in d.items() if isinstance(v, list)}


def tick_spans(times: np.ndarray) -> list[tuple[float, float]]:
    """``(start, end)`` of each tick. A tick ends at the next tick unless the gap
    is longer than 1.5 median ticks (a pause between sections); then it gets
    one median tick."""
    if len(times) == 0:
        return []
    d = np.diff(times)
    med = float(np.median(d)) if len(d) else 1.0
    spans = []
    for i, t in enumerate(times):
        nxt = times[i + 1] if i + 1 < len(times) else None
        spans.append((float(t), float(nxt) if nxt is not None and nxt - t <= 1.5 * med else float(t + med)))
    return spans


def expected_notes(sections: dict[str, list[str]], structure: list[tuple[str, int]],
                   spans: list[tuple[float, float]]) -> tuple[list[Note], dict]:
    """Lay the notation onto the ticks in sung order.

    Each structure entry ``(section, n)`` consumes ``len(section) / n`` ticks, n
    svaras per tick, evenly spaced; ``,`` extends the previous note. Stops when
    the ticks run out. Returns the notes and a report of ticks used / available
    and sections that didn't fit.
    """
    notes: list[Note] = []
    k = 0
    skipped = []
    for name, n in structure:
        svs = sections.get(name)
        if not svs:
            skipped.append(name)
            continue
        need = int(np.ceil(len(svs) / n))
        if k + need > len(spans):
            skipped.append(name)
            break
        for i, s in enumerate(svs):
            t0, t1 = spans[k + i // n]
            a = t0 + (t1 - t0) * (i % n) / n
            b = t0 + (t1 - t0) * (i % n + 1) / n
            if s == "," and notes:
                notes[-1].end = b
            elif s != ",":
                notes.append(Note(a, b, s))
        k += need
    return notes, {"ticks_used": k, "ticks": len(spans), "skipped": skipped}


def to_notation(swara: str) -> str:
    """Our swara name (``.D``, ``g``, ``S'``) in notation style (``D_``, ``G``,
    ``S^``): the variant letter case is dropped, octave marks converted."""
    if not swara:
        return ""
    low = swara.count(".")
    high = swara.count("'")
    base = swara.strip(".'").upper()
    return base + ("^" * high) + ("_" * low)


def label_track(segments: list[dict], n: int, fps: float = 100.0) -> np.ndarray:
    """Per-frame notation-style label of our stable segments ("" elsewhere)."""
    lab = np.full(n, "", dtype=object)
    for s in segments:
        if s.get("kind") == "stable" and s.get("swara"):
            lab[int(round(s["start"] * fps)):int(round(s["end"] * fps))] = to_notation(s["swara"])
    return lab


def score(notes: list[Note], segments: list[dict], n_frames: int, fps: float = 100.0,
          search_s: float = 0.3) -> dict:
    """Label accuracy and change-onset scores of our segments against the notes
    (see the module docstring)."""
    lab = label_track(segments, n_frames, fps)
    right = wrong = unlabelled = right_pc = 0
    for nt in notes:
        q = (nt.end - nt.start) / 4
        seg = [x for x in lab[int((nt.start + q) * fps):int((nt.end - q) * fps)] if x]
        if not seg:
            unlabelled += 1
            continue
        vals, counts = np.unique(seg, return_counts=True)
        top = vals[np.argmax(counts)]
        right_pc += top[0] == nt.svara[0]            # same svara, any octave
        if top == nt.svara:
            right += 1
        else:
            wrong += 1
    starts = [(s["start"], to_notation(s["swara"])) for s in segments if s.get("kind") == "stable" and s.get("swara")]
    st_t = np.array([t for t, _ in starts])
    errs, changes = [], 0
    for prev, nt in zip(notes, notes[1:]):
        if nt.svara == prev.svara:
            continue
        changes += 1
        if not len(st_t):
            continue
        near = np.where(np.abs(st_t - nt.start) <= search_s)[0]
        cand = [st_t[i] - nt.start for i in near if starts[i][1] == nt.svara]
        if cand:
            errs.append(min(cand, key=abs))
    errs = np.abs(np.array(errs))
    labelled = right + wrong
    return {"notes": len(notes), "labelled_share": round(labelled / max(len(notes), 1), 3),
            "label_accuracy": round(right / max(labelled, 1), 3),
            "label_accuracy_any_octave": round(right_pc / max(labelled, 1), 3), "unlabelled": unlabelled,
            "changes": changes,
            "onset_hit_100ms": round(float((errs <= 0.1).sum() / max(changes, 1)), 3),
            "onset_hit_200ms": round(float((errs <= 0.2).sum() / max(changes, 1)), 3),
            "onset_median_err_ms": round(1000 * float(np.median(errs)), 1) if len(errs) else None}


def recordings(root: Path) -> list[dict]:
    """Every (raga, singer) pair with audio, notation, structure and ticks."""
    na = Path(root) / "Notations_Annotations"
    tonics = yaml.safe_load((na / "annotations" / "tonics.yaml").read_text())
    out = []
    for svl in sorted((na / "annotations" / "taalas").glob("*/*.svl")):
        raga, singer = svl.parent.name, svl.stem
        audio = list((Path(root) / "Audio").glob(f"*by-{singer}-in-{raga}-raaga.mp3"))
        struct = na / "notations" / raga / f"{singer}.yaml"
        if audio and struct.exists():
            out.append({"raga": raga, "singer": singer, "audio": audio[0], "ticks": svl, "structure": struct,
                        "notation": na / "notations" / f"{raga}.yaml", "tonic_hz": float(tonics[singer])})
    return out
