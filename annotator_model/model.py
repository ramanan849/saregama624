"""Unified annotator: one audio encoder, four frame-level heads at 50 fps.

  pitch    721 classes = unvoiced + 720 bins of 10 cents from C1 (32.70 Hz) up
           six octaves; trained with Gaussian-blurred targets (CREPE-style) and
           decoded with a local weighted average, so output is continuous.
  phones   CTC over blank + the Indic phone set + '*' (non-lyric vocalising);
           drives forced alignment directly in the phone inventory.
  onset    probability that a syllable starts at this frame.
  ntype    rest / lyric / slur per frame.

Encoders
  "conformer"   log-mel frontend + a small Conformer-style stack trained from
                scratch.  Good enough once you have tens of hours; cheap.
  "hf:<id>"     any Hugging Face wav2vec2-family model (e.g. facebook/mms-300m,
                ai4bharat IndicWav2Vec checkpoints).  16 kHz input, 20 ms frames,
                so its frames already sit on the 50 fps grid.  Recommended.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from saregama.phoneset import PHONES

F0_MIN = 32.703195662574764       # C1
N_PITCH_BINS = 720                # 10-cent bins, 6 octaves
CENTS_PER_BIN = 10
SR_IN = 16000
HOP_IN = 320                      # 20 ms at 16 kHz

PHONE_VOCAB: Dict[str, int] = {"<blank>": 0}
for _p in PHONES:
    PHONE_VOCAB[_p.name] = len(PHONE_VOCAB)
PHONE_VOCAB["*"] = len(PHONE_VOCAB)


def hz_to_bin(f0: torch.Tensor) -> torch.Tensor:
    """Continuous bin index (1..720) for voiced frames, 0 for unvoiced."""
    c = 1200 * torch.log2(torch.clamp(f0, min=F0_MIN) / F0_MIN) / CENTS_PER_BIN + 1
    return torch.where(f0 > 0, torch.clamp(c, 1, N_PITCH_BINS), torch.zeros_like(f0))


def bin_to_hz(b: torch.Tensor) -> torch.Tensor:
    return torch.where(b > 0, F0_MIN * 2 ** ((b - 1) * CENTS_PER_BIN / 1200), torch.zeros_like(b))


def _mel_fb(n_fft: int, n_mels: int, sr: int, fmin: float = 30.0, fmax: float = 8000.0) -> np.ndarray:
    def hz2mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595) - 1)
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    hz = mel2hz(mels)
    bins = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_mels, len(bins)))
    for i in range(n_mels):
        l, c, r = hz[i], hz[i + 1], hz[i + 2]
        fb[i] = np.maximum(0, np.minimum((bins - l) / (c - l), (r - bins) / (r - c)))
    return fb.astype(np.float32)


class LogMel(nn.Module):
    def __init__(self, n_mels: int = 128, n_fft: int = 1024):
        super().__init__()
        self.n_fft = n_fft
        self.register_buffer("fb", torch.from_numpy(_mel_fb(n_fft, n_mels, SR_IN)), persistent=False)
        self.register_buffer("win", torch.hann_window(n_fft), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:         # x: (B, samples)
        S = torch.stft(x, self.n_fft, HOP_IN, window=self.win, center=True, return_complex=True)
        m = torch.matmul(self.fb, S.abs() ** 2)                    # (B, mels, T)
        return torch.log(m + 1e-6).transpose(1, 2)                 # (B, T, mels)


class ConvModule(nn.Module):
    def __init__(self, d: int, k: int = 15, p: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, 2 * d)
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Linear(d, d)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        y = F.glu(self.pw1(self.ln(x)), dim=-1).transpose(1, 2)
        y = F.silu(self.bn(self.dw(y))).transpose(1, 2)
        return x + self.drop(self.pw2(y))


class ConformerBlock(nn.Module):
    def __init__(self, d: int, heads: int, p: float = 0.1):
        super().__init__()
        self.ff1 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.SiLU(), nn.Dropout(p), nn.Linear(4 * d, d))
        self.ln_att = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.conv = ConvModule(d, p=p)
        self.ff2 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.SiLU(), nn.Dropout(p), nn.Linear(4 * d, d))
        self.ln = nn.LayerNorm(d)

    def forward(self, x, pad_mask: Optional[torch.Tensor] = None):
        x = x + 0.5 * self.ff1(x)
        h = self.ln_att(x)
        x = x + self.att(h, h, h, key_padding_mask=pad_mask, need_weights=False)[0]
        x = self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.ln(x)


class ConformerEncoder(nn.Module):
    def __init__(self, d: int = 256, layers: int = 8, heads: int = 4, n_mels: int = 128):
        super().__init__()
        self.mel = LogMel(n_mels)
        self.inp = nn.Sequential(nn.Linear(n_mels, d), nn.LayerNorm(d))
        self.blocks = nn.ModuleList([ConformerBlock(d, heads) for _ in range(layers)])
        self.dim = d

    def forward(self, wav16: torch.Tensor, n_frames: int, pad_mask=None) -> torch.Tensor:
        x = self.inp(self.mel(wav16))
        x = _fit_time(x, n_frames)
        for b in self.blocks:
            x = b(x, pad_mask)
        return x


class HFEncoder(nn.Module):
    def __init__(self, model_id: str, freeze_feature_encoder: bool = True):
        super().__init__()
        from transformers import AutoModel  # type: ignore
        self.m = AutoModel.from_pretrained(model_id)
        if freeze_feature_encoder and hasattr(self.m, "freeze_feature_encoder"):
            self.m.freeze_feature_encoder()
        self.dim = self.m.config.hidden_size

    def forward(self, wav16: torch.Tensor, n_frames: int, pad_mask=None) -> torch.Tensor:
        x = (wav16 - wav16.mean(dim=1, keepdim=True)) / (wav16.std(dim=1, keepdim=True) + 1e-7)
        h = self.m(x).last_hidden_state
        return _fit_time(h, n_frames)


def _fit_time(x: torch.Tensor, n: int) -> torch.Tensor:
    T = x.shape[1]
    if T == n:
        return x
    if T > n:
        return x[:, :n]
    return torch.cat([x, x[:, -1:].expand(-1, n - T, -1)], dim=1)


class Annotator(nn.Module):
    def __init__(self, encoder: str = "conformer", d: int = 256, layers: int = 8, heads: int = 4):
        super().__init__()
        self.encoder_name = encoder
        if encoder == "conformer":
            self.enc = ConformerEncoder(d, layers, heads)
        elif encoder.startswith("hf:"):
            self.enc = HFEncoder(encoder[3:])
        else:
            raise ValueError(encoder)
        h = self.enc.dim
        self.pitch = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, N_PITCH_BINS + 1))
        self.phones = nn.Linear(h, len(PHONE_VOCAB))
        self.onset = nn.Linear(h, 1)
        self.ntype = nn.Linear(h, 3)

    def forward(self, wav16: torch.Tensor, n_frames: int, pad_mask=None) -> Dict[str, torch.Tensor]:
        z = self.enc(wav16, n_frames, pad_mask)
        return {"pitch": self.pitch(z), "phones": self.phones(z),
                "onset": self.onset(z).squeeze(-1), "ntype": self.ntype(z)}

    @staticmethod
    def decode_pitch(logits: torch.Tensor, voicing_threshold: float = 0.5):
        """(B, T, 721) -> f0 Hz (B, T), voicing prob (B, T)."""
        p = logits.softmax(-1)
        pv = 1 - p[..., 0]
        pb = p[..., 1:]
        centre = pb.argmax(-1, keepdim=True)                       # 0..719
        idx = centre + torch.arange(-4, 5, device=logits.device)
        idx = idx.clamp(0, N_PITCH_BINS - 1)
        w = torch.gather(pb, -1, idx)
        b = (w * idx.float()).sum(-1) / w.sum(-1).clamp_min(1e-9) + 1   # back to 1..720
        f0 = bin_to_hz(b)
        f0 = torch.where(pv >= voicing_threshold, f0, torch.zeros_like(f0))
        return f0, pv


def pitch_targets(f0: torch.Tensor, sigma_bins: float = 2.5) -> torch.Tensor:
    """Gaussian-blurred soft targets over 721 classes."""
    b = hz_to_bin(f0)                                              # (B, T)
    grid = torch.arange(1, N_PITCH_BINS + 1, device=f0.device).float()
    g = torch.exp(-0.5 * ((grid[None, None] - b[..., None]) / sigma_bins) ** 2)
    g = g / g.sum(-1, keepdim=True).clamp_min(1e-9)
    out = torch.zeros(*f0.shape, N_PITCH_BINS + 1, device=f0.device)
    voiced = (f0 > 0)
    out[..., 1:] = torch.where(voiced[..., None], g, torch.zeros_like(g))
    out[..., 0] = (~voiced).float()
    return out


def annotator_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
                   w: Dict[str, float] | None = None) -> Dict[str, torch.Tensor]:
    w = w or {"pitch": 1.0, "ctc": 0.5, "onset": 1.0, "ntype": 0.5}
    mask = batch["frame_mask"]                                     # (B, T) bool
    sw = batch["weight"]                                           # (B,)
    fw = (mask.float() * sw[:, None])
    denom = fw.sum().clamp_min(1)
    tgt = pitch_targets(batch["f0"])
    lp = F.log_softmax(out["pitch"], -1)
    l_pitch = (-(tgt * lp).sum(-1) * fw).sum() / denom
    l_on = (F.binary_cross_entropy_with_logits(out["onset"], batch["onset"], reduction="none",
                                               pos_weight=torch.tensor(8.0, device=mask.device)) * fw).sum() / denom
    l_nt = (F.cross_entropy(out["ntype"].transpose(1, 2), batch["ntype"], reduction="none") * fw).sum() / denom
    logp = F.log_softmax(out["phones"], -1).transpose(0, 1)        # (T, B, C)
    l_ctc_each = F.ctc_loss(logp, batch["phone_ids"], batch["n_frames"], batch["phone_len"],
                            blank=0, reduction="none", zero_infinity=True)
    l_ctc = (l_ctc_each / batch["phone_len"].clamp_min(1) * sw).sum() / sw.sum().clamp_min(1e-6)
    total = w["pitch"] * l_pitch + w["ctc"] * l_ctc + w["onset"] * l_on + w["ntype"] * l_nt
    return {"loss": total, "pitch": l_pitch, "ctc": l_ctc, "onset": l_on, "ntype": l_nt}
