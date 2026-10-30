"""Measure the automatic pipeline (or the trained annotator) against gold songs.

For every song that exists both as gold (hand-corrected) and as a prediction,
compute:

  syllables   phone error rate (PER) over the syllable sequence, onset error of
              matched syllables (median ms, % within 40 / 80 ms)
  f0          gross pitch error (> 50 cents) on frames voiced in both, mean
              absolute cents error on the rest, octave error rate, voicing
              recall and false-alarm rate
  tonic       error in cents (pitch class, octave-free) and correct (< 25 cents)
  notes       note-type frame accuracy, note-onset F1 at 50 ms

and check them against the gates proposed in the plan:
  PER < 5 %, median onset error < 40 ms, gross pitch error < 2 %, tonic accuracy > 95 %.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .audio import FPS
from .schema import NOTE_REST, SongAnnotation

GATES = {"per": ("<", 0.05), "onset_median_ms": ("<", 40.0),
         "gross_pitch_error": ("<", 0.02), "tonic_accuracy": (">", 0.95)}


def _edit_align(a: Sequence, b: Sequence) -> Tuple[int, List[Tuple[int, int]]]:
    """Levenshtein distance and the matched index pairs (substitutions included)."""
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        if D[i, j] == D[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif D[i, j] == D[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return int(D[n, m]), pairs[::-1]


def compare(pred: SongAnnotation, gold: SongAnnotation) -> Dict:
    r: Dict = {"song_id": gold.song_id}
    # syllables / phones
    gs = [s for s in gold.syllables if "vocalise" not in s.flags or "akaar" in s.flags]
    ps = [s for s in pred.syllables if "vocalise" not in s.flags or "akaar" in s.flags]
    gph = [p for s in gs for p in s.phones]
    pph = [p for s in ps for p in s.phones]
    dist, _ = _edit_align(pph, gph)
    r["per"] = dist / max(len(gph), 1)
    sd, pairs = _edit_align(["-".join(s.phones) for s in ps], ["-".join(s.phones) for s in gs])
    errs = np.array([abs(ps[i].start - gs[j].start) * 1000 / FPS for i, j in pairs])
    r["syllables_gold"], r["syllables_pred"], r["syllables_matched"] = len(gs), len(ps), len(pairs)
    r["onset_median_ms"] = float(np.median(errs)) if len(errs) else None
    r["onset_within_40ms"] = float((errs <= 40).mean()) if len(errs) else None
    r["onset_within_80ms"] = float((errs <= 80).mean()) if len(errs) else None
    # f0
    n = min(len(pred.f0), len(gold.f0))
    pf, gf = pred.f0[:n], gold.f0[:n]
    both = (pf > 0) & (gf > 0)
    if both.any():
        d = 1200 * np.log2(pf[both] / gf[both])
        gross = np.abs(d) > 50
        octv = np.abs(np.abs(d) - 1200 * np.round(np.abs(d) / 1200)) < 100
        octv &= np.round(np.abs(d) / 1200) >= 1
        r["gross_pitch_error"] = float(gross.mean())
        r["fine_cents_mae"] = float(np.abs(d[~gross]).mean()) if (~gross).any() else None
        r["octave_error"] = float(octv.mean())
    else:
        r["gross_pitch_error"] = r["fine_cents_mae"] = r["octave_error"] = None
    r["voicing_recall"] = float(((pf > 0) & (gf > 0)).sum() / max((gf > 0).sum(), 1))
    r["voicing_false_alarm"] = float(((pf > 0) & (gf <= 0)).sum() / max((gf <= 0).sum(), 1))
    # tonic (octave-free)
    if pred.tonic_hz > 0 and gold.tonic_hz > 0:
        c = 1200 * np.log2(pred.tonic_hz / gold.tonic_hz)
        c = (c + 600) % 1200 - 600
        r["tonic_error_cents"] = float(c)
        r["tonic_correct"] = bool(abs(c) < 25)
    # notes
    def frame_types(a: SongAnnotation, n: int) -> np.ndarray:
        t = np.zeros(n, dtype=int)
        for x in a.notes:
            t[x.start:min(x.end, n)] = x.type
        return t
    nt = min(pred.n_frames, gold.n_frames)
    r["note_type_acc"] = float((frame_types(pred, nt) == frame_types(gold, nt)).mean())
    po = np.array([x.start for x in pred.notes if x.type != NOTE_REST])
    go = np.array([x.start for x in gold.notes if x.type != NOTE_REST])
    tol = int(round(0.05 * FPS))
    if len(po) and len(go):
        used = np.zeros(len(po), dtype=bool)
        hit = 0
        for g in go:
            k = np.where(~used & (np.abs(po - g) <= tol))[0]
            if len(k):
                used[k[np.argmin(np.abs(po[k] - g))]] = True
                hit += 1
        prec, rec = hit / len(po), hit / len(go)
        r["note_onset_f1"] = 2 * prec * rec / max(prec + rec, 1e-9)
    return r


def evaluate(pred_dir: str | Path, gold_dir: str | Path, out_json: Optional[str] = None) -> Dict:
    pred_dir, gold_dir = Path(pred_dir), Path(gold_dir)
    rows = []
    for gd in sorted(gold_dir.glob("*")):
        if not (gd / "annotation.json").exists() or not (pred_dir / gd.name / "annotation.json").exists():
            continue
        gold = SongAnnotation.load(gd)
        if gold.status != "gold":
            continue
        pred = SongAnnotation.load(pred_dir / gd.name)
        rows.append(compare(pred, gold))
    summary: Dict = {"songs": len(rows)}
    if rows:
        def agg(key, fn=np.mean):
            v = [r[key] for r in rows if r.get(key) is not None]
            return float(fn(v)) if v else None
        summary.update({
            "per": agg("per"), "onset_median_ms": agg("onset_median_ms", np.median),
            "onset_within_40ms": agg("onset_within_40ms"), "onset_within_80ms": agg("onset_within_80ms"),
            "gross_pitch_error": agg("gross_pitch_error"), "fine_cents_mae": agg("fine_cents_mae"),
            "octave_error": agg("octave_error"), "voicing_recall": agg("voicing_recall"),
            "voicing_false_alarm": agg("voicing_false_alarm"),
            "tonic_accuracy": agg("tonic_correct"), "note_type_acc": agg("note_type_acc"),
            "note_onset_f1": agg("note_onset_f1"),
        })
        gates = {}
        for k, (op, thr) in GATES.items():
            v = summary.get(k)
            gates[k] = None if v is None else (v < thr if op == "<" else v > thr)
        summary["gates"] = gates
        summary["all_gates_pass"] = all(v for v in gates.values() if v is not None)
    result = {"summary": summary, "per_song": rows}
    if out_json:
        Path(out_json).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return result
