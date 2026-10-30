"""Training data for the unified annotator, read from an exported dataset
(``svs-annotate export``): every clip already has wav + f0 + notes + phonemes.

Sample weights follow how much a human has checked the labels:
gold 1.0, reviewed 1.0, auto 0.5, flagged 0.25.  This is what makes the
self-training rounds safe: auto labels help, but never outvote corrected ones.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from saregama.audio import SR, load_audio, resample
from saregama.phoneset import parse_note_phoneme_string

from .model import PHONE_VOCAB, SR_IN

STATUS_WEIGHT = {"gold": 1.0, "reviewed": 1.0, "auto": 0.5, "flagged": 0.25}


class ClipDataset(Dataset):
    def __init__(self, dataset_dir: str, splits=("train",), statuses=("gold", "reviewed", "auto")):
        self.root = Path(dataset_dir)
        self.items: List[Dict] = []
        for sp in splits:
            p = self.root / f"{sp}.jsonl"
            if not p.exists():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                it = json.loads(line)
                if it.get("status", "auto") in statuses:
                    self.items.append(it)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        wav = load_audio(self.root / it["wav"], SR)
        wav16 = resample(wav, SR, SR_IN)
        n = int(it["n_frames"])
        f0 = np.array([float(x) for x in it["f0"].split()], dtype=np.float32)[:n]
        durs = [float(x) for x in it["duration"].split()]
        types = [int(x) for x in it["note_type"].split()]
        phons = it["phoneme"].split()
        onset = np.zeros(n, dtype=np.float32)
        ntype = np.zeros(n, dtype=np.int64)
        phone_ids: List[int] = []
        pos = 0
        for d, t, ph in zip(durs, types, phons):
            k = int(round(d * 50))
            a, b = pos, min(n, pos + k)
            ntype[a:b] = {1: 0, 2: 1, 3: 2}[t]
            if t == 2:
                if a < n:
                    onset[a] = 1.0
                    if a > 0:
                        onset[a - 1] = max(onset[a - 1], 0.5)
                    if a + 1 < n:
                        onset[a + 1] = max(onset[a + 1], 0.5)
                phone_ids += [PHONE_VOCAB[p] for p in parse_note_phoneme_string(ph)]
            pos += k
        return {"wav16": torch.from_numpy(wav16), "f0": torch.from_numpy(f0),
                "onset": torch.from_numpy(onset), "ntype": torch.from_numpy(ntype),
                "phone_ids": torch.tensor(phone_ids, dtype=torch.long), "n_frames": n,
                "weight": STATUS_WEIGHT.get(it.get("status", "auto"), 0.5)}


def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    B = len(batch)
    T = max(b["n_frames"] for b in batch)
    S = max(len(b["wav16"]) for b in batch)
    wav = torch.zeros(B, S)
    f0 = torch.zeros(B, T)
    onset = torch.zeros(B, T)
    ntype = torch.zeros(B, T, dtype=torch.long)
    mask = torch.zeros(B, T, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["n_frames"]
        wav[i, :len(b["wav16"])] = b["wav16"]
        f0[i, :n] = b["f0"][:n]
        onset[i, :n] = b["onset"][:n]
        ntype[i, :n] = b["ntype"][:n]
        mask[i, :n] = True
    return {"wav16": wav, "f0": f0, "onset": onset, "ntype": ntype, "frame_mask": mask,
            "phone_ids": torch.cat([b["phone_ids"] for b in batch]),
            "phone_len": torch.tensor([len(b["phone_ids"]) for b in batch]),
            "n_frames": torch.tensor([b["n_frames"] for b in batch]),
            "weight": torch.tensor([b["weight"] for b in batch], dtype=torch.float32), "T": T}
