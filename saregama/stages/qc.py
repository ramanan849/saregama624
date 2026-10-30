"""Stage 7 - automatic quality control and review routing.

Every song gets metrics and flags.  Status after QC:
  auto      passed every check; goes to training (and to random audits)
  flagged   at least one check failed; queued for human review, worst first
  rejected  unusable (no singing found, alignment collapsed); excluded
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..audio import FPS
from ..config import QcCfg
from ..schema import NOTE_REST, SongAnnotation


def song_qc(ann: SongAnnotation, cfg: QcCfg, align_info: Dict, f0_info: Dict,
            leakage_db: float | None = None) -> Tuple[Dict, List[str], str]:
    f0 = ann.f0
    voiced = f0 > 0
    n = ann.n_frames
    lyr = [s for s in ann.syllables if "vocalise" not in s.flags]
    voc = [s for s in ann.syllables if "vocalise" in s.flags]
    m: Dict = {}
    m["voiced_ratio"] = float(voiced.mean()) if n else 0.0
    m["f0_agree"] = float(np.mean(ann.f0_agree[voiced])) if voiced.any() else 0.0
    m["octave_fixed_rate"] = float(f0_info.get("octave_fixed_frames", 0)) / max(int(voiced.sum()), 1)
    confs = np.array([s.conf for s in lyr], dtype=float)
    finite = confs[np.isfinite(confs)]
    m["align_conf_mean"] = float(finite.mean()) if len(finite) else None
    m["align_low_conf_frac"] = float((finite < cfg.min_align_conf).mean()) if len(finite) else None
    sung_s = sum(s.end - s.start for s in lyr) / FPS
    m["syllables"] = len(lyr)
    m["syll_rate"] = len(lyr) / max(voiced.sum() / FPS, 1e-6)
    m["short_syllables"] = int(sum(1 for s in lyr if (s.end - s.start) / FPS < cfg.min_syll_s))
    m["vocalise_spans"] = len(voc)
    m["vocalise_s"] = sum(s.end - s.start for s in voc) / FPS
    m["sung_s"] = sung_s
    m["tonic_hz"] = ann.tonic_hz
    m["tonic_conf"] = ann.tonic_conf
    m["star_frames"] = align_info.get("star_frames")
    if leakage_db is not None:
        m["leakage_db"] = leakage_db

    flags: List[str] = []
    if m["voiced_ratio"] < 0.05 or not lyr:
        return m, ["no_singing"], "rejected"
    if m["voiced_ratio"] < cfg.min_voiced_ratio:
        flags.append("low_voiced_ratio")
    if m["f0_agree"] < cfg.min_f0_agree:
        flags.append("f0_disagreement")
    if m["octave_fixed_rate"] > cfg.max_octave_jump_rate:
        flags.append("octave_jumps")
    if m["align_conf_mean"] is not None and m["align_conf_mean"] < cfg.min_align_conf:
        flags.append("low_align_conf")
    if m["syll_rate"] > cfg.max_syll_rate:
        flags.append("syllables_too_dense")
    if m["short_syllables"] > max(3, 0.05 * len(lyr)):
        flags.append("many_short_syllables")
    if m["vocalise_spans"]:
        flags.append("unlabelled_vocalise")
    if ann.tonic_source == "auto" and ann.tonic_conf < 0.3:
        flags.append("tonic_uncertain")
    if leakage_db is not None and leakage_db > -15:
        flags.append("separation_leakage")
    # alignment collapse: many syllables piled into a tiny span
    if lyr:
        spans = np.array([s.end - s.start for s in lyr])
        if np.median(spans) <= 1:
            return m, flags + ["alignment_collapsed"], "rejected"
    status = "flagged" if flags else "auto"
    return m, flags, status


def review_priority(qc: Dict) -> float:
    """Higher = review sooner."""
    p = 0.0
    m = qc.get("metrics", {})
    flags = qc.get("flags", [])
    p += 3.0 * ("low_align_conf" in flags) + 2.0 * ("octave_jumps" in flags)
    p += 2.0 * ("tonic_uncertain" in flags) + 1.5 * ("unlabelled_vocalise" in flags)
    p += 1.0 * ("f0_disagreement" in flags) + 1.0 * ("separation_leakage" in flags)
    p += 1.0 * ("many_short_syllables" in flags) + 1.0 * ("syllables_too_dense" in flags)
    if m.get("align_low_conf_frac") is not None:
        p += 2.0 * m["align_low_conf_frac"]
    return p


def clip_qc(notes, f0_agree: np.ndarray, cfg: QcCfg) -> Dict:
    sung = [n for n in notes if n.type != NOTE_REST]
    return {"n_notes": len(notes), "n_sung": len(sung),
            "f0_agree": float(np.nanmean(f0_agree)) if len(f0_agree) else 0.0}
