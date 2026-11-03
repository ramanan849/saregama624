"""Use a trained annotator inside the pipeline (F0 backend "annotator" and
align backend "annotator")."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from saregama.audio import SR, resample

from .model import PHONE_VOCAB, SR_IN, Annotator


class AnnotatorRunner:
    """Run a trained annotator checkpoint on long audio, chunk by chunk; the
    same interface the pipeline's F0 and alignment backends expect.
    """
    def __init__(self, ckpt: str, device: str = "cuda", chunk_s: float = 30.0, overlap_s: float = 2.0):
        """Load ``ckpt`` on ``device``; chunks are ``chunk_s`` long with
        ``overlap_s`` of context on each side.
        """
        if not ckpt:
            raise ValueError("align.annotator_ckpt is empty")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        m = state["meta"]
        self.model = Annotator(m["encoder"], m["d"], m["layers"])
        self.model.load_state_dict(state["model"])
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.model.to(self.device).eval()
        self.chunk, self.ov = int(chunk_s * 50), int(overlap_s * 50)

    @torch.no_grad()
    def _run(self, x24k: np.ndarray, n: int) -> Dict[str, np.ndarray]:
        """All head outputs on ``n`` 50 fps frames for 24 kHz audio."""
        x16 = resample(x24k, SR, SR_IN)
        hop = SR_IN // 50
        outs = {"pitch": [], "phones": [], "onset": [], "ntype": []}
        pos = 0
        while pos < n:
            a = max(0, pos - self.ov)
            b = min(n, pos + self.chunk + self.ov)
            seg = torch.from_numpy(x16[a * hop: b * hop]).float()[None].to(self.device)
            o = self.model(seg, b - a, None)
            fa, fb = pos - a, pos - a + min(self.chunk, n - pos)
            for k in outs:
                outs[k].append(o[k][0, fa:fb].float().cpu())
            pos += self.chunk
        return {k: torch.cat(v, 0) for k, v in outs.items()}

    def f0(self, x24k: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """(f0 Hz, voicing confidence) on ``n`` frames."""
        o = self._run(x24k, n)
        f0, pv = Annotator.decode_pitch(o["pitch"][None])
        return f0[0].numpy().astype(np.float32), pv[0].numpy().astype(np.float32)

    def phone_vocab(self) -> Dict[str, int]:
        """Phone name -> CTC column (blank = 0)."""
        return dict(PHONE_VOCAB)

    def phone_logprobs(self, x24k: np.ndarray, n: int) -> np.ndarray:
        """(n, phones) CTC log-probabilities for forced alignment."""
        o = self._run(x24k, n)
        return torch.log_softmax(o["phones"], -1).numpy()
