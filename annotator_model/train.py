"""Train the unified annotator on an exported dataset.

  python -m annotator_model.train --data dataset --out ckpt/annotator_r1 \\
      --encoder hf:facebook/mms-300m --epochs 30 --batch 8 --lr 5e-5

Round 1: train on the gold + reviewed clips (plus auto clips at half weight).
Round 2+: re-annotate the corpus with this model (align.backend = "annotator",
f0 backends include "annotator"), review what QC flags, export, retrain.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ClipDataset, collate
from .model import Annotator, annotator_loss


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    tot, n = {}, 0
    gross, voiced, on_hit, on_pred, on_gold = 0, 0, 0, 0, 0
    for batch in loader:
        T = batch.pop("T")
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["wav16"], T, None)
        ls = annotator_loss(out, batch)
        for k, v in ls.items():
            tot[k] = tot.get(k, 0.0) + float(v)
        n += 1
        f0p, _ = Annotator.decode_pitch(out["pitch"])
        m = batch["frame_mask"] & (batch["f0"] > 0) & (f0p > 0)
        if m.any():
            d = 1200 * torch.log2(f0p[m] / batch["f0"][m])
            gross += int((d.abs() > 50).sum())
            voiced += int(m.sum())
        pr = (torch.sigmoid(out["onset"]) > 0.5) & batch["frame_mask"]
        gd = (batch["onset"] >= 1.0) & batch["frame_mask"]
        on_pred += int(pr.sum())
        on_gold += int(gd.sum())
        # onset hit within +-2 frames
        near = torch.nn.functional.max_pool1d(pr.float()[:, None], 5, 1, 2)[:, 0] > 0
        on_hit += int((near & gd).sum())
    res = {k: v / max(n, 1) for k, v in tot.items()}
    res["gross_pitch_error"] = gross / max(voiced, 1)
    rec = on_hit / max(on_gold, 1)
    prec = min(1.0, on_hit / max(on_pred, 1))
    res["onset_f1"] = 2 * prec * rec / max(prec + rec, 1e-9)
    model.train()
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="conformer")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-split", default="val")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train = ClipDataset(a.data, ("train",), ("gold", "reviewed", "auto"))
    val = ClipDataset(a.data, (a.val_split,), ("gold", "reviewed", "auto"))
    if len(val) == 0:
        val = ClipDataset(a.data, ("test",), ("gold", "reviewed"))
    print(f"train clips {len(train)}, val clips {len(val)}")
    tl = DataLoader(train, a.batch, shuffle=True, collate_fn=collate, num_workers=a.workers,
                    drop_last=len(train) > a.batch)
    vl = DataLoader(val, a.batch, shuffle=False, collate_fn=collate, num_workers=a.workers)
    model = Annotator(a.encoder, a.d, a.layers).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = a.max_steps or a.epochs * max(1, len(tl))
    warm = min(1000, total // 10 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    use_amp = a.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    meta = {"encoder": a.encoder, "d": a.d, "layers": a.layers}
    best, step = float("inf"), 0
    t0 = time.time()
    for ep in range(a.epochs):
        for batch in tl:
            T = batch.pop("T")
            batch = {k: v.to(a.device) for k, v in batch.items()}
            with torch.autocast("cuda", enabled=use_amp):
                outp = model(batch["wav16"], T, None)
            ls = annotator_loss({k: v.float() for k, v in outp.items()}, batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(ls["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 50 == 0 or step == 1:
                print(f"step {step} ep {ep} " + " ".join(f"{k}={float(v):.3f}" for k, v in ls.items())
                      + f" {time.time() - t0:.0f}s", flush=True)
            if a.max_steps and step >= a.max_steps:
                break
        res = evaluate(model, vl, a.device) if len(val) else {"loss": float(ls["loss"])}
        print(f"epoch {ep} val " + json.dumps({k: round(v, 4) for k, v in res.items()}), flush=True)
        torch.save({"model": model.state_dict(), "meta": meta, "val": res, "step": step}, out / "last.pt")
        if res["loss"] < best:
            best = res["loss"]
            torch.save({"model": model.state_dict(), "meta": meta, "val": res, "step": step}, out / "best.pt")
        if a.max_steps and step >= a.max_steps:
            break
    print(f"done; best val loss {best:.4f}; checkpoints in {out}")


if __name__ == "__main__":
    main()
