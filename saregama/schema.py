"""Data model of the annotator.

Two levels:

* **Song annotation** (``work/songs/<song_id>/annotation.json`` + ``*.npy``): the
  master record for a whole song.  Humans review and correct this.  Frame-level
  curves are full-song arrays on the 50 fps grid; syllables and notes carry
  integer frame boundaries.
* **Training clip** (``dataset/<split>.jsonl`` + ``dataset/wavs/*.wav``): what the
  SoulX-Singer fine-tune reads.  Built from song annotations by
  :mod:`saregama.export`.  The first nine fields are exactly SoulX-Singer's
  metadata format; the rest are extensions the model changes can use.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .audio import FPS

NOTE_REST, NOTE_LYRIC, NOTE_SLUR = 1, 2, 3
STATUS = ("auto", "flagged", "reviewed", "gold", "rejected")
LANG_NAMES = {"hi": "Hindi", "mr": "Marathi", "ta": "Tamil"}
STYLES = ("film", "bhajan", "hindustani", "carnatic", "bhavgeet", "natyasangeet",
          "abhang", "ghazal", "pop", "devotional", "folk", "other")


@dataclass
class SyllableAnn:
    text: str                  # native script, e.g. "मन" or "ணக்"
    phones: List[str]          # phone names, e.g. ["m", "a", "n"]
    start: int                 # frame (inclusive)
    end: int                   # frame (exclusive)
    line: int                  # lyric line index
    word: int                  # word index within the line
    conf: float = 0.0          # alignment confidence (mean token log-posterior)
    flags: List[str] = field(default_factory=list)   # e.g. ["vocalise_after"]


@dataclass
class NoteAnn:
    start: int
    end: int
    type: int                  # 1 rest, 2 lyric (syllable onset), 3 slur
    syllable: int              # index into syllables, -1 for rests
    midi: int = 0              # integer MIDI pitch for SoulX score mode, 0 for rests
    cents: Optional[float] = None   # median pitch in cents above the tonic (exact)
    swara: str = ""            # e.g. "S", "r", "G", "M", "P'", ".D"


@dataclass
class SongAnnotation:
    song_id: str
    language: str              # hi | mr | ta
    style: str = "other"
    singer_id: str = ""
    gender: str = ""
    source: str = ""           # where the recording came from
    license: str = ""          # licence / rights note
    audio_path: str = ""       # original file
    vocals_path: str = ""      # separated dry lead vocal, 24 kHz mono
    n_frames: int = 0
    tonic_hz: float = 0.0
    tonic_source: str = "auto"         # auto | given | reviewed
    tonic_conf: float = 0.0
    lines: List[str] = field(default_factory=list)
    syllables: List[SyllableAnn] = field(default_factory=list)
    notes: List[NoteAnn] = field(default_factory=list)
    status: str = "auto"
    qc: Dict = field(default_factory=dict)
    history: List[str] = field(default_factory=list)
    # frame-level arrays, saved as .npy next to annotation.json
    f0: Optional[np.ndarray] = None            # Hz, 0 = unvoiced
    f0_agree: Optional[np.ndarray] = None      # 0..1 ensemble agreement
    energy_db: Optional[np.ndarray] = None
    hnr_db: Optional[np.ndarray] = None
    h1h2_db: Optional[np.ndarray] = None

    ARRAYS = ("f0", "f0_agree", "energy_db", "hnr_db", "h1h2_db")

    # ------------------------------------------------------------------
    @property
    def duration_s(self) -> float:
        return self.n_frames / FPS

    def cents(self) -> np.ndarray:
        """F0 in cents above the tonic (NaN where unvoiced)."""
        f0 = np.asarray(self.f0, dtype=np.float64)
        out = np.full_like(f0, np.nan)
        v = f0 > 0
        out[v] = 1200 * np.log2(f0[v] / self.tonic_hz)
        return out

    def save(self, song_dir: str | Path) -> None:
        d = Path(song_dir)
        d.mkdir(parents=True, exist_ok=True)
        meta = asdict(self)
        for k in self.ARRAYS:
            meta.pop(k, None)
            arr = getattr(self, k)
            if arr is not None:
                np.save(d / f"{k}.npy", np.asarray(arr, dtype=np.float32))
        (d / "annotation.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                           encoding="utf-8")

    @classmethod
    def load(cls, song_dir: str | Path) -> "SongAnnotation":
        d = Path(song_dir)
        meta = json.loads((d / "annotation.json").read_text(encoding="utf-8"))
        meta["syllables"] = [SyllableAnn(**s) for s in meta.get("syllables", [])]
        meta["notes"] = [NoteAnn(**n) for n in meta.get("notes", [])]
        ann = cls(**meta)
        for k in cls.ARRAYS:
            p = d / f"{k}.npy"
            if p.exists():
                setattr(ann, k, np.load(p))
        return ann

    def validate(self) -> List[str]:
        """Structural checks; returns a list of problems (empty = valid)."""
        errs = []
        if self.language not in LANG_NAMES:
            errs.append(f"language {self.language!r} not in {list(LANG_NAMES)}")
        if self.f0 is not None and len(self.f0) != self.n_frames:
            errs.append(f"f0 has {len(self.f0)} frames, expected {self.n_frames}")
        prev_end = 0
        for i, s in enumerate(self.syllables):
            if not (0 <= s.start < s.end <= self.n_frames):
                errs.append(f"syllable {i} has bad span {s.start}-{s.end}")
            if s.start < prev_end:
                errs.append(f"syllable {i} overlaps the previous one")
            prev_end = s.end
        prev_end = 0
        for i, n in enumerate(self.notes):
            if not (0 <= n.start < n.end <= self.n_frames):
                errs.append(f"note {i} has bad span {n.start}-{n.end}")
            if n.start != prev_end:
                errs.append(f"note {i} starts at {n.start}, previous ended at {prev_end} (notes must tile)")
            if n.type not in (NOTE_REST, NOTE_LYRIC, NOTE_SLUR):
                errs.append(f"note {i} has type {n.type}")
            if n.type == NOTE_REST and n.syllable != -1:
                errs.append(f"rest note {i} points at a syllable")
            if n.type != NOTE_REST and not (0 <= n.syllable < len(self.syllables)):
                errs.append(f"note {i} points at missing syllable {n.syllable}")
            prev_end = n.end
        if self.notes and prev_end != self.n_frames:
            errs.append(f"notes end at {prev_end}, song has {self.n_frames} frames")
        if self.tonic_hz and not (40 < self.tonic_hz < 600):
            errs.append(f"tonic {self.tonic_hz} Hz is implausible")
        return errs
