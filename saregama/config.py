"""Model and stage settings as dataclasses.

Defaults live here (model locations come from :mod:`saregama.paths`).  The
legacy workspace commands read overrides from a TOML file (see
``configs/default.toml``) passed as ``--config``; the Saraga commands build
the sections they need directly from their CLI options.
"""
from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import List, Optional

from .paths import SOULX_MODELS, SOULX_REPO


@dataclass
class SeparationCfg:
    """Vocal separation / de-reverb (stage ``stems``; legacy ``run``).

    ``backend`` soulx uses SoulX-Singer's Mel-RoFormer karaoke + de-reverb models
    from ``soulx_models``; audio_separator runs the same checkpoint through the
    ``audio-separator`` package; none means the input is already a dry vocal.
    """
    backend: str = "soulx"            # soulx | audio_separator | none
    soulx_repo: str = str(SOULX_REPO)
    soulx_models: str = str(SOULX_MODELS)
    dereverb: bool = True
    audio_separator_model: str = "mel_band_roformer_karaoke_becruily.ckpt"
    device: str = "cuda"


@dataclass
class F0Cfg:
    """F0 ensemble.  The first backend is the reference the others are
    octave-aligned to; the Saraga pipeline runs ``rmvpe,fcpe,crepe``.
    """
    backends: List[str] = field(default_factory=lambda: ["rmvpe", "yin"])
    rmvpe_model: str = str(SOULX_MODELS / "rmvpe" / "rmvpe.pt")
    fmin: float = 60.0
    fmax: float = 1400.0
    agree_cents: float = 50.0          # ensemble members "agree" within this
    fix_octave_jumps: bool = True
    max_octave_island_s: float = 0.15  # isolated octave jumps shorter than this are folded back
    fill_gap_frames: int = 2           # interpolate unvoiced gaps this short inside voiced runs
    device: str = "cuda"


@dataclass
class TonicCfg:
    """Legacy tonic estimation (histogram); the Saraga pipeline uses
    ``stages.f0_ir.choose_tonic`` and the concert consensus instead.
    """
    method: str = "histogram"          # histogram | given
    pa_weight: float = 0.6             # weight of the fifth (Pa) in candidate scoring
    ma_weight: float = 0.2


@dataclass
class AlignCfg:
    """Legacy frame-level CTC alignment (``stages.align``) used by the
    film-song workspace.  Saraga kritis use ``stages.align_spikes``.
    """
    backend: str = "mms_fa"            # mms_fa | hf_ctc | annotator | heuristic
    hf_model: str = ""                 # HF id or path, for hf_ctc
    annotator_ckpt: str = ""           # for the trained unified annotator
    use_star: bool = True              # '*' absorbs non-lyric vocalising (aalap, humming)
    chunk_s: float = 30.0
    chunk_overlap_s: float = 2.0
    min_gap_rest_s: float = 0.12       # silence this long between syllables becomes a rest
    vocalise_min_s: float = 0.6        # star-absorbed voiced spans longer than this are flagged
    vocalise_policy: str = "review"    # review | drop | akaar
    device: str = "cuda"


@dataclass
class NotesCfg:
    """Legacy note segmentation for SoulX clips (``stages.notes``)."""
    backend: str = "grid"              # grid | rosvot | some
    min_note_s: float = 0.08
    smooth_frames: int = 5          # stability window (frames) for note segmentation
    grid: str = "12tet"                # 12tet relative to tonic, or a raga name file later


@dataclass
class ClipCfg:
    """How songs are cut into SoulX-Singer training clips."""
    min_s: float = 2.0
    max_s: float = 16.0
    split_rest_s: float = 0.3          # cut only at rests at least this long
    pad_s: float = 0.1                 # keep this much rest at both ends


@dataclass
class QcCfg:
    """Thresholds that route legacy songs to auto / flagged review."""
    min_align_conf: float = -2.5       # mean log-posterior per token (MMS scale)
    min_f0_agree: float = 0.85
    max_octave_jump_rate: float = 0.02
    min_voiced_ratio: float = 0.35
    max_syll_rate: float = 12.0        # syllables per second of voiced time
    min_syll_s: float = 0.04
    review_fraction_cap: float = 0.25  # if more than this would go to review, keep the worst only


@dataclass
class G2PCfg:
    """Rule-based G2P options (Hindi, Marathi, Tamil) plus lexicon files."""
    lexicon: List[str] = field(default_factory=lambda: ["lexicons/common.tsv"])
    keep_final_schwa: bool = False
    delete_medial_schwa: bool = True
    marathi_affricates: bool = True
    tamil_initial_c_as_s: bool = True
    tamil_intervocalic_k: str = "g"


@dataclass
class Config:
    """All sections of the legacy workspace configuration (``--config`` TOML)."""
    work_dir: str = "work"
    out_dir: str = "dataset"
    separation: SeparationCfg = field(default_factory=SeparationCfg)
    f0: F0Cfg = field(default_factory=F0Cfg)
    tonic: TonicCfg = field(default_factory=TonicCfg)
    align: AlignCfg = field(default_factory=AlignCfg)
    notes: NotesCfg = field(default_factory=NotesCfg)
    clips: ClipCfg = field(default_factory=ClipCfg)
    qc: QcCfg = field(default_factory=QcCfg)
    g2p: G2PCfg = field(default_factory=G2PCfg)

    def to_dict(self) -> dict:
        """Plain nested dict of every setting (for logging / provenance)."""
        return asdict(self)


def _update(obj, data: dict):
    """Recursively copy ``data`` (from TOML) onto dataclass ``obj``;
    unknown keys raise ``KeyError`` so typos never pass silently.
    """
    names = {f.name: f for f in fields(obj)}
    for k, v in data.items():
        if k not in names:
            raise KeyError(f"unknown config key {k!r} in section {type(obj).__name__}")
        cur = getattr(obj, k)
        if is_dataclass(cur) and isinstance(v, dict):
            _update(cur, v)
        else:
            setattr(obj, k, v)


def load_config(path: Optional[str | Path] = None, **overrides) -> Config:
    """Defaults, then the TOML at ``path`` (if any), then ``overrides``."""
    cfg = Config()
    if path:
        with open(path, "rb") as f:
            _update(cfg, tomllib.load(f))
    if overrides:
        _update(cfg, overrides)
    return cfg
