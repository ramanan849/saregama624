"""The per-recording ICM-IR bundle.

``<annotations>/carnatic/<mbid>/``:

* ``ir.json``        global metadata, stems (+ sync), tonic, sections, sama,
                     per-stage status and QC
* ``f0_100hz.npz``   t, f0_hz, cents, voiced, agree, energy_db on the 100 Hz grid
* ``swaras.json``    swara segments
* ``lyrics.json``    lines -> words -> syllables
* ``stems/*.flac``   offset-corrected stems
* ``review/``        Sonic Visualiser layers, sync preview, contour PDF

Every time in the bundle is in seconds on the mix's clock (t = 0 is the first
sample of the Saraga mix file).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .saraga import Recording

DEFAULT_ROOT = Path("/home/kartiksingh/Gokul/624/annotations/carnatic")
IR_VERSION = "icm-ir-0.1"
F0_HZ = 100                  # master frame rate of every frame-level curve


class Bundle:
    def __init__(self, root: str | Path, mbid: str):
        self.dir = Path(root) / mbid
        self.mbid = mbid

    @property
    def ir_path(self) -> Path:
        return self.dir / "ir.json"

    @property
    def stems_dir(self) -> Path:
        return self.dir / "stems"

    @property
    def review_dir(self) -> Path:
        return self.dir / "review"

    def path(self, name: str) -> Path:
        return self.dir / name

    def exists(self) -> bool:
        return self.ir_path.exists()

    def load(self) -> dict:
        return json.loads(self.ir_path.read_text(encoding="utf-8"))

    def save(self, ir: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.ir_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(ir, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.ir_path)

    def update(self, **fields) -> dict:
        ir = self.load()
        ir.update(fields)
        self.save(ir)
        return ir

    def stage_done(self, stage: str) -> bool:
        return self.exists() and stage in self.load().get("stages", {})

    def mark(self, stage: str, **info) -> dict:
        ir = self.load()
        ir.setdefault("stages", {})[stage] = info
        self.save(ir)
        return ir


def init_bundle(rec: Recording, root: str | Path = DEFAULT_ROOT) -> Bundle:
    """Create ``ir.json`` from Saraga metadata (keeps an existing one)."""
    b = Bundle(root, rec.mbid)
    if b.exists():
        return b
    tonic: Optional[float] = rec.tonic()
    b.save({
        "version": IR_VERSION,
        "mbid": rec.mbid, "title": rec.title, "concert": rec.concert,
        "source_dir": str(rec.dir), "mix": str(rec.mix), "length_s": rec.length_s,
        "lead": rec.lead, "raaga": rec.raaga, "taala": rec.taala, "form": rec.form,
        "work_mbid": rec.work_mbid, "work_title": rec.work_title,
        "tonic_hz": tonic, "tonic_source": "saraga_ctonic" if tonic else None,
        "sections": rec.sections(), "sama": rec.sama().round(4).tolist(),
        "stems": {}, "sync": None, "status": "auto", "stages": {}, "qc": {}, "flags": [],
    })
    return b
