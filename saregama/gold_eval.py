"""Score the pipeline's predictions against hand-corrected (gold) recordings.

A recording becomes gold when its Sonic Visualiser layers are corrected and read
back with ``review-import``. That import first keeps the predictions in
``<bundle>/pred/`` (``lyrics.json``, ``ir_pred.json``); the bundle itself then
holds the corrections. ``gold-eval`` compares the two:

- **syllable onsets** (:func:`syllable_scores`): each gold syllable is matched to
  the nearest predicted syllable with the same text within ``max_s``; reported
  as recall, median absolute onset error, and shares within 50 / 100 ms;
- **line instances** (:func:`instance_scores`): a predicted instance is right when
  a gold instance of the same line overlaps it with IoU >= 0.5; precision,
  recall, F1;
- **sections** (:func:`section_scores`): frame accuracy per second over the
  section classes of :mod:`saregama.stages.sections`, comparing the prediction
  the aligner used (manual sections, else ``sections_auto``) with the gold;
- **sama** (:func:`sama_scores`): share of gold samas with a predicted sama
  within 70 ms, using :func:`saregama.stages.tala.sama_hit_rate`.

F0 is not scored here; there is no gold F0 layer, and F0 is already checked
against Saraga's pitch tracks at the ``f0`` stage.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


def _norm(text: str) -> str:
    """Syllable text for matching: lower case letters only."""
    return re.sub(r"[^a-z]", "", (text or "").lower())


def syllables(lyrics: dict) -> list[tuple[float, float, str]]:
    """``(start, end, text)`` of every syllable of every instance, by start time."""
    out = [(s["start"], s["end"], s["text"]) for inst in lyrics.get("instances", [])
           for w in inst.get("words", []) for s in w.get("syllables", [])]
    return sorted(out)


def syllable_scores(pred: list[tuple[float, float, str]], gold: list[tuple[float, float, str]],
                    max_s: float = 1.0) -> dict:
    """Onset agreement of predicted with gold syllables (see the module docstring).

    Each predicted syllable is used at most once (greedy, gold in time order)."""
    used = np.zeros(len(pred), dtype=bool)
    p_on = np.array([p[0] for p in pred]) if pred else np.zeros(0)
    p_tx = [_norm(p[2]) for p in pred]
    errs = []
    for g0, _, gt in gold:
        if not len(p_on):
            break
        cand = np.where((np.abs(p_on - g0) <= max_s) & ~used)[0]
        cand = [i for i in cand if p_tx[i] == _norm(gt)]
        if cand:
            i = min(cand, key=lambda k: abs(p_on[k] - g0))
            used[i] = True
            errs.append(p_on[i] - g0)
    e = np.abs(np.array(errs))
    n = max(len(gold), 1)
    return {"gold_syllables": len(gold), "pred_syllables": len(pred), "recall": round(len(e) / n, 3),
            "median_abs_err_ms": round(1000 * float(np.median(e)), 1) if len(e) else None,
            "within_50ms": round(float((e <= 0.05).sum() / n), 3),
            "within_100ms": round(float((e <= 0.1).sum() / n), 3)}


def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Intersection over union of two time spans."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def instance_scores(pred: dict, gold: dict, min_iou: float = 0.5) -> dict:
    """Line-instance precision / recall / F1: same ``line_id`` and IoU >= ``min_iou``."""
    P = [(i["line_id"], i["start"], i["end"]) for i in pred.get("instances", [])]
    G = [(i["line_id"], i["start"], i["end"]) for i in gold.get("instances", [])]
    hit_p = sum(any(lp == lg and _iou((a, b), (c, d)) >= min_iou for lg, c, d in G) for lp, a, b in P)
    hit_g = sum(any(lp == lg and _iou((a, b), (c, d)) >= min_iou for lp, a, b in P) for lg, c, d in G)
    prec, rec = hit_p / max(len(P), 1), hit_g / max(len(G), 1)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"pred_instances": len(P), "gold_instances": len(G), "precision": round(prec, 3),
            "recall": round(rec, 3), "f1": round(f1, 3)}


def section_scores(pred_sections: list[dict], gold_sections: list[dict], length_s: float) -> dict:
    """Per-second section-class accuracy over seconds the gold labels."""
    from .stages.sections import frame_accuracy, frame_labels
    n = int(np.ceil(length_s))
    y, p = frame_labels(gold_sections, n), frame_labels(pred_sections, n)
    acc, rec = frame_accuracy(p, y)
    return {"accuracy": round(acc, 3) if acc == acc else None, "recall": rec, "gold_s": int((y >= 0).sum())}


def sama_scores(pred_samas: list[float], gold_samas: list[float]) -> dict:
    """Share of gold samas with a predicted sama within 70 ms."""
    from .stages.tala import sama_hit_rate
    if not gold_samas:
        return {"hit_70ms": None, "gold": 0}
    return {"hit_70ms": round(sama_hit_rate(np.array(pred_samas or []), np.array(gold_samas)), 3),
            "gold": len(gold_samas)}


def evaluate_bundle(bundle_dir: Path) -> dict | None:
    """All scores for one reviewed bundle, or None if it has no ``pred/`` snapshot."""
    d = Path(bundle_dir)
    pred_ir = d / "pred" / "ir_pred.json"
    if not pred_ir.exists():
        return None
    ir = json.loads((d / "ir.json").read_text(encoding="utf-8"))
    pr = json.loads(pred_ir.read_text(encoding="utf-8"))
    out = {"mbid": ir.get("mbid"), "title": ir.get("title"), "reviewer": (ir.get("review") or {}).get("reviewer")}
    layers = set((ir.get("review") or {}).get("layers", []))
    if {"lines", "words", "syllables"} <= layers and (d / "pred" / "lyrics.json").exists():
        gold_l = json.loads((d / "lyrics.json").read_text(encoding="utf-8"))
        pred_l = json.loads((d / "pred" / "lyrics.json").read_text(encoding="utf-8"))
        out["syllables"] = syllable_scores(syllables(pred_l), syllables(gold_l))
        out["instances"] = instance_scores(pred_l, gold_l)
    if "sections" in layers:
        out["sections"] = section_scores(pr.get("sections") or pr.get("sections_auto") or [],
                                         ir.get("sections") or [], ir.get("length_s") or 0)
    if "sama" in layers:
        out["sama"] = sama_scores(pr.get("sama") or [], ir.get("sama") or [])
    return out
