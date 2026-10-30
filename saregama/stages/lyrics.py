"""Lyrics: read the lyric sheet, or draft one with ASR for a human to correct.

Lyric sheet format (UTF-8 .txt):
  * one sung line per text line, in singing order - write repeated choruses out
    every time they are sung (the aligner follows the text literally);
  * blank lines and lines starting with '#' are ignored;
  * a line in square brackets, e.g. ``[aalap]`` or ``[sargam]``, marks expected
    non-lyric singing; it is not aligned, the '*' token absorbs it instead.

When a song has no lyric sheet, ``draft_lyrics`` transcribes the separated vocal
phrase by phrase with an Indic ASR model and writes ``lyrics.draft.txt``.  A human
must correct the draft and save it as the lyric sheet before alignment: ASR on
sung, ornamented Indian vocals is not reliable enough to train on.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import List

import numpy as np

from ..audio import FPS, SR, save_audio


def read_lyrics(path: str | Path) -> List[str]:
    lines = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = unicodedata.normalize("NFC", raw.strip())
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            continue
        lines.append(line)
    return lines


class IndicConformerASR:
    """AI4Bharat IndicConformer through NeMo (AI4Bharat's NeMo fork, branch nemo-v2).

    ``model_path`` is a downloaded .nemo file (multilingual or per language).
    The call signature follows the IndicConformer README; check it against the
    version you install.
    """

    def __init__(self, model_path: str, device: str = "cuda", decoder: str = "ctc"):
        import nemo.collections.asr as nemo_asr  # type: ignore
        self.model = nemo_asr.models.ASRModel.restore_from(model_path, map_location=device)
        self.model.eval()
        try:
            self.model.cur_decoder = decoder
        except Exception:
            pass

    def __call__(self, wav_paths: List[str], lang: str) -> List[str]:
        out = self.model.transcribe(wav_paths, batch_size=8, language_id=lang)
        if isinstance(out, tuple):
            out = out[0]
        return [o if isinstance(o, str) else getattr(o, "text", str(o)) for o in out]


def draft_lyrics(vocals: np.ndarray, f0: np.ndarray, lang: str, asr, out_txt: str | Path,
                 tmp_dir: str | Path) -> List[str]:
    """Split the vocal at pauses >= 0.4 s into phrases, transcribe each, and write
    one draft line per phrase."""
    from .align import _phrases
    phrases = _phrases(f0 > 0, min_gap=int(0.4 * FPS), min_len=int(0.5 * FPS))
    tmp = Path(tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (a, b) in enumerate(phrases):
        p = tmp / f"phrase_{i:04d}.wav"
        seg = vocals[max(0, a * 480 - 2400): b * 480 + 2400]
        from ..audio import resample
        save_audio(p, resample(seg, SR, 16000), 16000)
        paths.append(str(p))
    texts = asr(paths, lang) if paths else []
    lines = [t.strip() for t in texts if t and t.strip()]
    header = ("# DRAFT from ASR - correct every line, write repeats out in full,\n"
              "# then save as the song's lyric sheet.\n")
    Path(out_txt).write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return lines
