"""SoulX-Singer training clips from the IR bundles (melody mode: lyrics + F0).

SoulX-Singer has no training code in its repository; this writes data in the
metadata format its preprocessing produces and its ``DataProcessor`` reads
(24 kHz, hop 480, 50 fps), so a fine-tuning loop written later can consume it.

Per singer track of a recording (lead; the second singer of a duet too):

1. **Notes.** Every aligned syllable is split into notes at the boundaries of
   the swara segments it spans (``swaras.json``; computed on the fly for the
   second singer).  The first note is type 2 and carries the syllable's phones;
   the rest are type 3 (slurs) carrying only its vowel, because SoulX cycles a
   note's phones over all its frames (a long ``ma`` would otherwise sound
   ``m-a-m-a``).  A note needs ``phones + 3`` frames (``<BOW>``, phones,
   ``<SEP>``, ``<EOW>``); shorter ones merge into their neighbour.
2. **Akaram.** Voiced stretches in alapana sections with no syllables become
   vowel notes (``in_a`` then slurs), so alapana is usable training data.
3. **Rests and unlabelled singing.** Gaps are ``<SP>`` notes.  Voiced audio
   that nothing labels (singing the aligner missed) must not be called a rest:
   such a stretch ends a phrase like a long rest does, so it never lies inside
   a clip (at most ``max_unlabelled_s`` of it) while the labelled singing
   around it is kept.
4. **Clips** of ``min_s``–``max_s`` seconds are cut inside rests of at least
   ``split_rest_s``, padded by ``pad_s`` into the rest; nothing over ``hard_max_s``
   (SoulX's own limit is 30 s per prompt) is written.

``note_pitch`` is absolute MIDI (score mode); ``f0`` is the 100 Hz track taken
at every second frame (k x 20 ms), never interpolated across unvoiced frames.
Extras beyond SoulX's fields: tonic, per-note cents and swara names, singer id,
recording id, split, and whether the lyrics are verified.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .phoneset import BY_NAME, note_phoneme_string

FPS = 50
NOTE_REST, NOTE_LYRIC, NOTE_SLUR = 1, 2, 3
LANGUAGE_NAMES = {"tel": "Telugu", "san": "Sanskrit", "tam": "Tamil", "kan": "Kannada", "mal": "Malayalam",
                  "hin": "Hindi", "mar": "Marathi"}


@dataclass
class ClipCfgS:
    """Clip planning thresholds (seconds)."""
    min_s: float = 2.0
    max_s: float = 16.0
    hard_max_s: float = 30.0
    split_rest_s: float = 0.3
    pad_s: float = 0.1
    max_unlabelled_s: float = 0.1     # unlabelled voiced audio a clip may contain
    akaram_min_s: float = 0.3         # shortest voiced alapana stretch made into akaram notes
    max_overlap: float = 0.3          # duets: instances sung along more than this are skipped


@dataclass
class Note:
    """One note on the 50 fps grid: [a, b) frames."""
    a: int
    b: int
    kind: int                         # NOTE_REST | NOTE_LYRIC | NOTE_SLUR
    text: str = "<SP>"
    phones: List[str] = field(default_factory=list)
    midi: int = 0
    cents: Optional[float] = None
    swara: str = ""


def _vowel(phones: Sequence[str]) -> List[str]:
    """The syllable's vowel (its first vowel phone) for slur notes."""
    v = [p for p in phones if BY_NAME[p].kind == "vowel"]
    return v[:1] or list(phones[-1:])


def midi_of(f0: np.ndarray) -> int:
    """MIDI note of the median voiced F0 of a span (0 if unvoiced)."""
    v = f0[f0 > 0]
    return int(np.clip(round(69 + 12 * np.log2(np.median(v) / 440.0)), 1, 127)) if len(v) else 0


def syllable_notes(a: int, b: int, text: str, phones: List[str], cuts: Sequence[int], f0: np.ndarray,
                   cents: np.ndarray, swara_at) -> List[Note]:
    """Notes of one syllable on frames [a, b): split at ``cuts`` (segment
    boundaries inside the syllable), then too-short notes merged so that each
    note has room for its phones."""
    bounds = [a] + sorted(c for c in set(cuts) if a < c < b) + [b]
    notes = []
    for i, (x, y) in enumerate(zip(bounds[:-1], bounds[1:])):
        ph = phones if i == 0 else _vowel(phones)
        notes.append(Note(x, y, NOTE_LYRIC if i == 0 else NOTE_SLUR, text, list(ph)))
    k = 0                                             # merge short notes into the previous one
    while k < len(notes):
        n = notes[k]
        if n.b - n.a < len(n.phones) + 3 and len(notes) > 1:
            if k > 0:
                notes[k - 1].b = n.b
                del notes[k]
                continue
            notes[1].a, notes[1].kind, notes[1].phones = n.a, NOTE_LYRIC, n.phones   # first note: keep its phones
            del notes[0]
            continue
        k += 1
    for n in notes:
        seg = cents[n.a * 2:n.b * 2]                  # 100 Hz arrays: 2 frames per 50 fps frame
        seg = seg[~np.isnan(seg)]
        n.midi = midi_of(f0[n.a * 2:n.b * 2])
        n.cents = round(float(np.median(seg)), 1) if len(seg) else None
        n.swara = swara_at((n.a + n.b) / 2 / FPS)
    return notes


def track_notes(syllables: Sequence[dict], extra: Sequence[Tuple[float, float]], segments: Sequence[dict],
                f0: np.ndarray, cents: np.ndarray, n_frames: int) -> List[Note]:
    """All notes of one singer track: aligned syllables (rows from
    ``export_ir.syllable_rows`` with ``phones``) plus akaram spans ``extra``,
    with rests in between.  ``segments`` are swara segments (for note cuts and
    names); f0 / cents are the 100 Hz curves."""
    cuts = sorted({int(round(s["start"] * FPS)) for s in segments} | {int(round(s["end"] * FPS)) for s in segments})
    cuts = np.array(cuts, dtype=np.int64)
    starts = np.array([s["start"] for s in segments])

    def swara_at(t: float) -> str:
        """Name of the stable swara segment containing time t, else ''."""
        i = int(np.searchsorted(starts, t, side="right")) - 1
        if 0 <= i < len(segments) and segments[i]["end"] > t and segments[i]["kind"] == "stable":
            return segments[i]["swara"]
        return ""

    events = [(int(round(s["start"] * FPS)), int(round(s["end"] * FPS)), s["syllable"],
               [p for p in s["phones"].split("-") if p]) for s in syllables if s["phones"]]
    events += [(int(round(x * FPS)), int(round(y * FPS)), "a", ["a"]) for x, y in extra]
    events.sort()
    notes: List[Note] = []
    pos = 0
    for a, b, text, phones in events:
        a = max(a, pos)
        if b - a < 2:
            continue
        if a > pos:
            notes.append(Note(pos, a, NOTE_REST))
        inside = cuts[(cuts > a) & (cuts < b)]
        notes += syllable_notes(a, b, text, phones, inside.tolist(), f0, cents, swara_at)
        pos = b
    if pos < n_frames:
        notes.append(Note(pos, n_frames, NOTE_REST))
    return notes


def akaram_spans(voiced: np.ndarray, sections: Sequence[dict], syllables: Sequence[dict],
                 min_s: float, fps: int = 100) -> List[Tuple[float, float]]:
    """Voiced stretches (seconds) inside vocal-alapana sections that no syllable covers."""
    from .export_ir import section_class_at
    covered = np.zeros(len(voiced), bool)
    for s in syllables:
        covered[int(s["start"] * fps):int(np.ceil(s["end"] * fps))] = True
    out, k, n = [], 0, len(voiced)
    while k < n:
        if voiced[k] and not covered[k]:
            j = k
            while j < n and voiced[j] and not covered[j]:
                j += 1
            if (j - k) / fps >= min_s and section_class_at(k / fps, sections) == "vocal_alapana":
                out.append((k / fps, j / fps))
            k = j
        else:
            k += 1
    return out


def plan_clips(notes: Sequence[Note], voiced50: np.ndarray, cfg: ClipCfgS) -> List[Tuple[int, int]]:
    """[a, b) frame spans of clips.

    Phrases are the note runs between *cut* rests: rests of at least
    ``split_rest_s``, and rests holding more than ``max_unlabelled_s`` of
    voiced audio (singing nothing labels).  Unlabelled singing therefore ends
    a phrase instead of spoiling it, and never lies inside a clip.  Phrases are
    joined while the clip stays under ``max_s``; edges are padded by up to
    ``pad_s`` into silent rest frames only.
    """
    long_rest = int(cfg.split_rest_s * FPS)
    max_bad = int(round(cfg.max_unlabelled_s * FPS))
    bad = np.zeros(len(voiced50), bool)              # voiced frames inside rests
    for n in notes:
        if n.kind == NOTE_REST:
            bad[n.a:n.b] = voiced50[n.a:n.b]
    groups, cur = [], []                              # phrases separated by cut rests
    for n in notes:
        cut = n.kind == NOTE_REST and (n.b - n.a >= long_rest or bad[n.a:n.b].sum() > max_bad)
        if cut:
            if cur:
                groups.append((cur[0].a, cur[-1].b))
            cur = []
        else:
            cur.append(n)
    if cur:
        groups.append((cur[0].a, cur[-1].b))
    pad = int(cfg.pad_s * FPS)

    def padded(a: int, b: int) -> Tuple[int, int]:
        """Extend [a, b) into neighbouring silent frames, at most ``pad`` each side."""
        lo = a
        while lo > 0 and a - lo < pad and not voiced50[lo - 1]:
            lo -= 1
        hi = b
        while hi < len(voiced50) and hi - b < pad and not voiced50[hi]:
            hi += 1
        return lo, hi

    clips, i = [], 0
    while i < len(groups):
        a, b = groups[i]
        j = i
        while j + 1 < len(groups) and groups[j + 1][1] - a <= cfg.max_s * FPS \
                and not bad[groups[j][1]:groups[j + 1][0]].any():
            j += 1                                   # join phrases only across silent rests
            b = groups[j][1]
        i = j + 1
        a2, b2 = padded(a, b)
        if not cfg.min_s * FPS <= b2 - a2 <= cfg.hard_max_s * FPS:
            continue
        if bad[a2:b2].sum() > max_bad:
            continue
        clips.append((a2, b2))
    return clips


def clip_notes(notes: Sequence[Note], a: int, b: int) -> List[Note]:
    """Notes of [a, b): cut at the clip edges (edge pieces become rests if too short)."""
    out = []
    for n in notes:
        if n.b <= a or n.a >= b:
            continue
        m = Note(max(n.a, a), min(n.b, b), n.kind, n.text, list(n.phones), n.midi, n.cents, n.swara)
        if m.kind != NOTE_REST and (m.b - m.a < len(m.phones) + 3 or m.a != n.a and m.kind == NOTE_LYRIC):
            m = Note(m.a, m.b, NOTE_REST)              # a syllable cut by the clip edge is not sung here
        if out and out[-1].kind == NOTE_REST and m.kind == NOTE_REST:
            out[-1].b = m.b
        else:
            out.append(m)
    return out


def clip_item(notes: Sequence[Note], a: int, b: int, f0_100: np.ndarray, meta: dict) -> dict:
    """SoulX metadata item of one clip (fields as SoulX's preprocessing writes
    them, plus extras); durations sum exactly to the clip's frames."""
    durs = [(n.b - n.a) / FPS for n in notes]
    f0 = f0_100[a * 2:b * 2:2]                         # every second 100 Hz frame = k x 20 ms
    f0 = np.where(f0 > 0, f0, 0.0)
    return {
        "index": meta["clip_id"],
        "language": meta["language"],
        "time": [int(a * 1000 / FPS), int(b * 1000 / FPS)],
        # SoulX takes int(sum(durations) * 50) frames: a 0.1 ms nudge on the last
        # note keeps float rounding from losing a frame
        "duration": " ".join([f"{d:.2f}" for d in durs[:-1]] + [f"{durs[-1] + 1e-4:.4f}"]),
        "text": " ".join(n.text if n.kind != NOTE_REST else "<SP>" for n in notes),
        "phoneme": " ".join(note_phoneme_string(n.phones) if n.kind != NOTE_REST else "<SP>" for n in notes),
        "note_pitch": " ".join(str(n.midi if n.kind != NOTE_REST else 0) for n in notes),
        "note_type": " ".join(str(n.kind) for n in notes),
        "f0": " ".join(f"{x:.1f}" for x in f0),
        "n_frames": b - a,
        "tonic_hz": meta["tonic_hz"],
        "note_cents": " ".join("-" if n.cents is None or n.kind == NOTE_REST else f"{n.cents:.1f}" for n in notes),
        "swara": " ".join(n.swara or "-" for n in notes),
        "singer_id": meta["singer_id"], "mbid": meta["mbid"], "split": meta["split"],
        "lyrics_verified": meta["lyrics_verified"], "wav": meta["wav"],
    }
