"""Pipeline configuration.  Defaults live here; override them with a TOML file
(see ``configs/default.toml``) passed as ``--config``."""
from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class SeparationCfg:
    backend: str = "soulx"            # soulx | audio_separator | none
    soulx_repo: str = "../SoulX-Singer"
    soulx_models: str = "../SoulX-Singer/pretrained_models/SoulX-Singer-Preprocess"
    dereverb: bool = True
    audio_separator_model: str = "mel_band_roformer_karaoke_becruily.ckpt"
    device: str = "cuda"


@dataclass
class F0Cfg:
    backends: List[str] = field(default_factory=lambda: ["rmvpe", "yin"])
    rmvpe_model: str = "../SoulX-Singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt"
    fmin: float = 60.0
    fmax: float = 1400.0
    agree_cents: float = 50.0          # ensemble members "agree" within this
    fix_octave_jumps: bool = True
    max_octave_island_s: float = 0.15  # isolated octave jumps shorter than this are folded back
    fill_gap_frames: int = 2           # interpolate unvoiced gaps this short inside voiced runs
    device: str = "cuda"


@dataclass
class TonicCfg:
    method: str = "histogram"          # histogram | given
    pa_weight: float = 0.6             # weight of the fifth (Pa) in candidate scoring
    ma_weight: float = 0.2


@dataclass
class AlignCfg:
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
    backend: str = "grid"              # grid | rosvot | some
    min_note_s: float = 0.08
    smooth_frames: int = 5          # stability window (frames) for note segmentation
    grid: str = "12tet"                # 12tet relative to tonic, or a raga name file later


@dataclass
class ClipCfg:
    min_s: float = 2.0
    max_s: float = 16.0
    split_rest_s: float = 0.3          # cut only at rests at least this long
    pad_s: float = 0.1                 # keep this much rest at both ends


@dataclass
class QcCfg:
    min_align_conf: float = -2.5       # mean log-posterior per token (MMS scale)
    min_f0_agree: float = 0.85
    max_octave_jump_rate: float = 0.02
    min_voiced_ratio: float = 0.35
    max_syll_rate: float = 12.0        # syllables per second of voiced time
    min_syll_s: float = 0.04
    review_fraction_cap: float = 0.25  # if more than this would go to review, keep the worst only


@dataclass
class G2PCfg:
    lexicon: List[str] = field(default_factory=lambda: ["lexicons/common.tsv"])
    keep_final_schwa: bool = False
    delete_medial_schwa: bool = True
    marathi_affricates: bool = True
    tamil_initial_c_as_s: bool = True
    tamil_intervocalic_k: str = "g"


@dataclass
class Config:
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
        return asdict(self)


def _update(obj, data: dict):
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
    cfg = Config()
    if path:
        with open(path, "rb") as f:
            _update(cfg, tomllib.load(f))
    if overrides:
        _update(cfg, overrides)
    return cfg
