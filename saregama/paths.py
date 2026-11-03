"""Where the project's data lives, in one place.

The repository is expected to sit inside a project folder next to its data::

    <project>/
        saregama624/        this repository
        SoulX-Singer/       clone of Soul-AILab/SoulX-Singer (+ pretrained_models/)
        Datasets/Saraga/files/saraga1.5_carnatic/
        annotations/        written by this pipeline (carnatic/, lyrics/, logs/)

Every default below is derived from that layout, so moving the whole project
folder to another machine needs no code change.  Each location can also be
overridden with an environment variable (useful when the data sits elsewhere):

==========================  ==============================================
``SAREGAMA_PROJECT``        the project folder (default: repo's parent)
``SAREGAMA_SOULX``          SoulX-Singer clone
``SAREGAMA_SARAGA``         Saraga Carnatic root (``saraga1.5_carnatic``)
``SAREGAMA_ANNOTATIONS``    annotation output root (holds ``carnatic/``, ``lyrics/``)
==========================  ==============================================

The CLI's ``--saraga-root``, ``--ann-root`` and ``--lyrics-dir`` override these
again per run.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env(name: str, default: Path) -> Path:
    """``$name`` as a Path if set and non-empty, else ``default``."""
    v = os.environ.get(name, "").strip()
    return Path(v).expanduser() if v else default


REPO = Path(__file__).resolve().parents[1]
PROJECT = _env("SAREGAMA_PROJECT", REPO.parent)
SOULX_REPO = _env("SAREGAMA_SOULX", PROJECT / "SoulX-Singer")
SOULX_MODELS = SOULX_REPO / "pretrained_models" / "SoulX-Singer-Preprocess"
SARAGA_CARNATIC = _env("SAREGAMA_SARAGA", PROJECT / "Datasets" / "Saraga" / "files" / "saraga1.5_carnatic")
ANNOTATIONS = _env("SAREGAMA_ANNOTATIONS", PROJECT / "annotations")
ANN_CARNATIC = ANNOTATIONS / "carnatic"          # one IR bundle per recording
LYRICS_DIR = ANNOTATIONS / "lyrics"              # one YAML per composition (work)
