"""Stage 1 - lead-vocal separation and de-reverberation.

Backends
--------
soulx            SoulX-Singer's own wrapper (Mel-Band RoFormer karaoke + de-reverb),
                 imported from a local clone of the SoulX-Singer repo with the
                 SoulX-Singer-Preprocess weights downloaded.  Recommended.
audio_separator  The ``audio-separator`` package (pip install audio-separator[gpu]),
                 which can run the same RoFormer checkpoints by file name.
none             Input is already a dry vocal (your vocal-only files).

Output: dry lead vocal, mono, 24 kHz, loudness-normalised, plus the accompaniment
estimate when the backend gives one (used for a leakage score in QC).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..audio import SR, load_audio, loudness_normalize, resample, save_audio
from ..config import SeparationCfg


class Separator:
    def __init__(self, cfg: SeparationCfg):
        self.cfg = cfg
        self._impl = None

    def _load_soulx(self):
        repo = Path(self.cfg.soulx_repo).resolve()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from preprocess.tools.vocal_separation.model import VocalSeparator  # type: ignore
        m = Path(self.cfg.soulx_models)
        return VocalSeparator(
            sep_model_path=str(m / "mel-band-roformer-karaoke/mel_band_roformer_karaoke_becruily.ckpt"),
            sep_config_path=str(m / "mel-band-roformer-karaoke/config_karaoke_becruily.yaml"),
            der_model_path=str(m / "dereverb_mel_band_roformer/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"),
            der_config_path=str(m / "dereverb_mel_band_roformer/dereverb_mel_band_roformer_anvuew.yaml"),
            use_der=self.cfg.dereverb,
            device=self.cfg.device,
            verbose=False,
        )

    def _load_audio_separator(self):
        from audio_separator.separator import Separator as AS  # type: ignore
        sep = AS(output_dir=tempfile.mkdtemp(prefix="sep_"), output_format="WAV")
        sep.load_model(model_filename=self.cfg.audio_separator_model)
        return sep

    def process(self, audio_path: str, out_vocals: str,
                out_accomp: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        backend = self.cfg.backend
        accomp = None
        if backend == "none":
            vocals = load_audio(audio_path, SR)
        elif backend == "soulx":
            if self._impl is None:
                self._impl = self._load_soulx()
            res = self._impl.process(audio_path)
            v = res.vocals_dereverbed if res.vocals_dereverbed is not None else res.vocals
            v = np.asarray(v, dtype=np.float32)
            v = v.mean(axis=0) if v.ndim == 2 and v.shape[0] <= 2 else (
                v.mean(axis=1) if v.ndim == 2 else v)
            vocals = resample(v, int(res.sample_rate), SR)
            if res.accompaniment is not None:
                a = np.asarray(res.accompaniment, dtype=np.float32)
                a = a.mean(axis=0) if a.ndim == 2 and a.shape[0] <= 2 else (
                    a.mean(axis=1) if a.ndim == 2 else a)
                accomp = resample(a, int(res.sample_rate), SR)
        elif backend == "audio_separator":
            if self._impl is None:
                self._impl = self._load_audio_separator()
            files = self._impl.separate(audio_path)
            out_dir = Path(self._impl.output_dir)
            files = [out_dir / f if not os.path.isabs(f) else Path(f) for f in files]
            voc = [f for f in files if "vocal" in f.name.lower() and "instrument" not in f.name.lower()]
            ins = [f for f in files if f not in voc]
            if not voc:
                raise RuntimeError(f"audio-separator returned no vocal stem: {files}")
            vocals = load_audio(voc[0], SR)
            accomp = load_audio(ins[0], SR) if ins else None
        else:
            raise ValueError(f"unknown separation backend {backend!r}")

        vocals = loudness_normalize(vocals)
        save_audio(out_vocals, vocals)
        if accomp is not None and out_accomp:
            save_audio(out_accomp, accomp)
        return vocals, accomp


def leakage_score(vocals: np.ndarray, accomp: Optional[np.ndarray], f0: np.ndarray,
                  energy_db: np.ndarray) -> float:
    """Rough separation-quality proxy: energy in the vocal stem where the singer is
    *not* voiced, relative to voiced energy (dB).  Drone/violin bleed raises it.
    Lower is better; typical clean studio vocals sit below -25 dB."""
    v = f0 > 0
    if v.sum() < 10 or (~v).sum() < 10:
        return 0.0
    lin = 10 ** (energy_db / 10)
    return float(10 * np.log10(np.median(lin[~v]) + 1e-12) - 10 * np.log10(np.median(lin[v]) + 1e-12))
