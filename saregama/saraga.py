"""Saraga 1.5 (Carnatic) loader: recordings, stems and annotations.

Each recording lives in ``<root>/<concert>/<recording>/`` and every file there
shares one prefix (the metadata JSON's basename) followed by a suffix:

* ``.mp3.mp3`` or ``.mp3``          the stereo mix (t = 0 of the master clock)
* ``.multitrack-<stem>.mp3``        mono stems (vocal, vocal-s, violin,
                                    mridangam-left, mridangam-right, ghatam)
* ``.pitch.txt``, ``.pitch-vocal.txt``   ``time<TAB>f0_hz`` (0 = unvoiced)
* ``.ctonic.txt``                   tonic in Hz
* ``.sections-manual-p.txt``        ``start<TAB>?<TAB>duration<TAB>label``
* ``.sama-manual.txt``              one time per line
* ``.mphrases-manual.txt``          ``start<TAB>?<TAB>duration<TAB>svaras``

Prefixes are not uniform (some concerts put the artist first), so files are
matched by suffix only.  Every annotation time is on the mix's clock.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

DEFAULT_ROOT = Path("/home/kartiksingh/Gokul/624/Datasets/Saraga/files/saraga1.5_carnatic")
STEMS = ("vocal", "vocal-s", "violin", "mridangam-left", "mridangam-right", "ghatam")
ANNOTATIONS = ("pitch", "pitch-vocal", "ctonic", "sections-manual-p", "sections-manual",
               "sama-manual", "bpm-manual", "tempo-manual", "mphrases-manual")

MANIFEST_HEADER = ["mbid", "title", "concert", "dir", "mix", "length_s", "raaga", "taala", "form",
                   "work_mbid", "work_title", "lead", "stems", "has_stems", "has_tonic",
                   "has_pitch", "has_sections", "has_sama"]


@dataclass
class Recording:
    mbid: str
    title: str
    concert: str
    dir: Path
    mix: Optional[Path]
    length_s: float
    stems: Dict[str, Path] = field(default_factory=dict)
    ann: Dict[str, Path] = field(default_factory=dict)
    raaga: str = ""
    taala: str = ""
    form: str = ""
    work_mbid: str = ""
    work_title: str = ""
    lead: List[str] = field(default_factory=list)      # lead vocalists

    @property
    def has_stems(self) -> bool:
        return "vocal" in self.stems

    def tonic(self) -> Optional[float]:
        return load_tonic(self.ann["ctonic"]) if "ctonic" in self.ann else None

    def pitch(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Saraga's own predominant pitch (vocal version preferred)."""
        for key in ("pitch-vocal", "pitch"):
            if key in self.ann:
                return load_pitch(self.ann[key])
        return None

    def sections(self) -> List[dict]:
        for key in ("sections-manual-p", "sections-manual"):
            if key in self.ann:
                return load_sections(self.ann[key])
        return []

    def sama(self) -> np.ndarray:
        return load_times(self.ann["sama-manual"]) if "sama-manual" in self.ann else np.zeros(0)


# ----------------------------------------------------------------------------- files

def _classify(name: str, prefix: str) -> Tuple[str, str]:
    """('mix'|'stem'|'ann'|'meta'|'', key) for a file in a recording directory."""
    if not name.startswith(prefix):
        return "", ""
    rest = name[len(prefix):]
    if rest == ".json":
        return "meta", ""
    if rest.startswith(".multitrack-") and rest.endswith(".mp3"):
        return "stem", rest[len(".multitrack-"):-len(".mp3")]
    if rest in (".mp3.mp3", ".mp3"):
        return "mix", ""
    if rest.endswith(".txt"):
        return "ann", rest[1:-len(".txt")]
    return "", ""


def _names(items, key="name") -> str:
    return "; ".join(x.get(key, "") for x in items or [])


def load_recording(rec_dir: Path) -> Optional[Recording]:
    metas = [p for p in rec_dir.glob("*.json")]
    if len(metas) != 1:
        return None
    meta_path = metas[0]
    prefix = meta_path.name[:-len(".json")]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rec = Recording(
        mbid=meta["mbid"], title=meta.get("title", prefix), concert=rec_dir.parent.name,
        dir=rec_dir, mix=None, length_s=meta.get("length", 0) / 1000.0,
        raaga=_names(meta.get("raaga")), taala=_names(meta.get("taala")),
        form=_names(meta.get("form")),
        work_mbid=_names(meta.get("work"), "mbid"), work_title=_names(meta.get("work"), "title"),
        lead=[a["artist"]["name"] for a in meta.get("artists", [])
              if a.get("lead") and a.get("instrument", {}).get("name") == "Voice"],
    )
    for p in sorted(rec_dir.iterdir()):
        kind, key = _classify(p.name, prefix)
        if kind == "mix":
            rec.mix = p
        elif kind == "stem":
            rec.stems[key] = p
        elif kind == "ann":
            rec.ann[key] = p
    return rec


def scan(root: str | Path = DEFAULT_ROOT) -> List[Recording]:
    root = Path(root)
    recs = []
    for concert in sorted(p for p in root.iterdir() if p.is_dir()):
        for rec_dir in sorted(p for p in concert.iterdir() if p.is_dir()):
            rec = load_recording(rec_dir)
            if rec is not None and rec.mix is not None:
                recs.append(rec)
    return recs


def write_manifest(recs: List[Recording], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(MANIFEST_HEADER)
        for r in recs:
            w.writerow([r.mbid, r.title, r.concert, str(r.dir), str(r.mix), f"{r.length_s:.3f}",
                        r.raaga, r.taala, r.form, r.work_mbid, r.work_title, "; ".join(r.lead),
                        " ".join(sorted(r.stems)), int(r.has_stems), int("ctonic" in r.ann),
                        int("pitch" in r.ann or "pitch-vocal" in r.ann),
                        int("sections-manual-p" in r.ann or "sections-manual" in r.ann),
                        int("sama-manual" in r.ann)])


def read_manifest(path: str | Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def by_mbid(recs: List[Recording]) -> Dict[str, Recording]:
    return {r.mbid: r for r in recs}


# ----------------------------------------------------------------------------- annotations

def load_tonic(path: Path) -> float:
    return float(Path(path).read_text().split()[0])


def load_pitch(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """(times_s, f0_hz) with 0 for unvoiced."""
    a = np.loadtxt(path, dtype=np.float64, ndmin=2)
    return a[:, 0], a[:, 1]


def load_times(path: Path) -> np.ndarray:
    vals = [float(line.split()[0]) for line in Path(path).read_text().splitlines() if line.strip()]
    return np.asarray(vals, dtype=np.float64)


def load_sections(path: Path) -> List[dict]:
    """[{start, end, label}] from a ``start ? duration label`` file."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4:
            continue
        start, dur = float(parts[0]), float(parts[2])
        out.append({"start": start, "end": start + dur, "label": parts[3].strip()})
    return out
