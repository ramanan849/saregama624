"""The per-recording ICM-IR bundle.

``<annotations>/carnatic/<mbid>/``:

* ``ir.json``          global metadata, stems (+ sync), tonic, sections, sama,
                       per-stage status (``stages``), QC, flags, review status
* ``f0_100hz.npz``     fps, f0_hz, cents, agree, conf, energy_db, track_<backend>
                       on the 100 Hz grid (frame k = time k / 100 s); see stages/f0_ir.py
* ``flat_100hz.npz``   fps, flat_cents: the gamaka-free swara contour
* ``swaras.json``      raga, refined swara positions, summary, segments
* ``lyrics.json``      source, lines, instances -> words -> syllables, regions
* ``mms_50hz.npz``     cached MMS_FA log-probabilities (alignment input)
* ``stems/*.flac``     offset-corrected stems, exactly the mix's length
* ``review/``          contour PDF, Sonic Visualiser layers, sync preview

Every time in the bundle is in seconds on the mix's clock (t = 0 is the first
sample of the Saraga mix file).  ``ir.json`` keeps the absolute Saraga paths
it was built from (``mix``, ``source_dir``, stem ``source``) for provenance
only; the pipeline itself re-locates recordings by scanning ``--saraga-root``,
and bundle-internal paths (stems) are relative to the bundle folder.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .paths import ANN_CARNATIC
from .saraga import Recording

DEFAULT_ROOT = ANN_CARNATIC  # see paths.py; override with SAREGAMA_ANNOTATIONS or --ann-root
IR_VERSION = "icm-ir-0.1"
F0_HZ = 100                  # master frame rate of every frame-level curve


class Bundle:
    """Handle on one recording's bundle folder ``<root>/<mbid>/``.

    ``ir.json`` is the index: metadata, stems, tonic, sections, sama, QC, flags
    and ``stages`` (which pipeline steps have run, with their summaries).  Stages
    are resumable: a CLI command skips a recording whose stage is already marked
    unless ``--force`` is given.
    """
    def __init__(self, root: str | Path, mbid: str):
        """``root`` is the folder holding all bundles; ``mbid`` the recording."""
        self.dir = Path(root) / mbid
        self.mbid = mbid

    @property
    def ir_path(self) -> Path:
        """Path of ``ir.json``."""
        return self.dir / "ir.json"

    @property
    def stems_dir(self) -> Path:
        """Folder of the offset-corrected FLAC stems."""
        return self.dir / "stems"

    @property
    def review_dir(self) -> Path:
        """Folder of review material (contour PDF, Sonic Visualiser layers)."""
        return self.dir / "review"

    def path(self, name: str) -> Path:
        """Path of a file ``name`` inside the bundle."""
        return self.dir / name

    def exists(self) -> bool:
        """True once ``ir.json`` has been written (``init_bundle``)."""
        return self.ir_path.exists()

    def load(self) -> dict:
        """Read ``ir.json``."""
        return json.loads(self.ir_path.read_text(encoding="utf-8"))

    def save(self, ir: dict) -> None:
        """Write ``ir.json`` atomically (temp file + rename), so an interrupted
        run never leaves a half-written index.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.ir_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(ir, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.ir_path)

    def update(self, **fields) -> dict:
        """Merge top-level ``fields`` into ``ir.json`` and return the result."""
        ir = self.load()
        ir.update(fields)
        self.save(ir)
        return ir

    def stage_done(self, stage: str) -> bool:
        """True if ``stage`` is recorded in ``ir.json['stages']``."""
        return self.exists() and stage in self.load().get("stages", {})

    def mark(self, stage: str, **info) -> dict:
        """Record ``stage`` as done, with ``info`` as its summary."""
        ir = self.load()
        ir.setdefault("stages", {})[stage] = info
        self.save(ir)
        return ir


def init_bundle(rec: Recording, root: str | Path = DEFAULT_ROOT) -> Bundle:
    """Create ``ir.json`` from Saraga metadata (keeps an existing one, so
    re-running ``sync`` never discards later stages' results)."""
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
