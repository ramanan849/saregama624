"""Build the SoulX-Singer fine-tuning dataset from song annotations.

Output layout::

    dataset/
      wavs/<clip_id>.wav          24 kHz mono 16-bit, length = n_frames * 480 samples
      train.jsonl  val.jsonl  test.jsonl
      train.json   val.json   test.json     same items as a JSON list (SoulX loads lists)
      excluded.jsonl                       clips left out, with the reason
      stats.json  stats.md                 hours by language / style / singer / status
      soulx/phone_set_indic.json           SoulX phone set + the Indic tokens
      soulx/embedding_init_map.json        new token -> SoulX tokens to average
      soulx/data_processor.patch           one-branch change so in_ notes expand

One clip item (fields 1-9 are SoulX-Singer's own metadata format, unchanged)::

    index        "mr_bhavgeet_0007_c012"
    language     "Marathi"
    time         [start_ms, end_ms] within the song
    duration     note durations in s, space-separated ("0.24 0.40 ...")
    text         per note: syllable text, "<SP>" for rests (slurs repeat the text)
    phoneme      per note: "in_m-a", "<SP>" for rests (slurs repeat the phonemes)
    note_pitch   per note: integer MIDI, 0 for rests
    note_type    per note: 1 rest, 2 lyric, 3 slur
    f0           per 20 ms frame: Hz, 0 = unvoiced
    -- extensions --
    lang, style, singer_id, gender, song_id, wav, n_frames
    tonic_hz     Sa in Hz
    note_cents   per note: exact median pitch in cents above Sa ("-" for rests)
    swara        per note: S r R g G m M P d D n N with . (mandra) / ' (taar)
    energy_db, hnr_db, h1h2_db   per frame voice-quality curves
    f0_agree     per frame F0 confidence (0..1)
    status       auto | flagged | reviewed | gold
    prev, next   neighbouring clip ids in the same song (for long-form stage 2)
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .audio import FPS, HOP, SR, load_audio, save_audio
from .config import Config
from .phoneset import embedding_init_map, extend_soulx_phoneset, note_phoneme_string
from .schema import LANG_NAMES, NOTE_REST, NoteAnn, SongAnnotation

DATA_PROCESSOR_PATCH = '''--- a/soulxsinger/utils/data_processor.py
+++ b/soulxsinger/utils/data_processor.py
@@ def preprocess(
-            if phonemes[ph_idx][:3] == "en_":
-                en_phs = ['en_' + x for x in phonemes[ph_idx][3:].split('-')] + ['<SEP>']     # <sep> between en words in one note
+            if phonemes[ph_idx][:3] in ("en_", "in_"):
+                prefix = phonemes[ph_idx][:3]
+                en_phs = [prefix + x for x in phonemes[ph_idx][3:].split('-')] + ['<SEP>']     # <sep> between words in one note
'''


@dataclass
class Clip:
    song: SongAnnotation
    a: int                     # first frame
    b: int                     # end frame (exclusive)
    notes: List[NoteAnn]
    long: bool = False


def _fmt(arr: Iterable[float], nd: int = 1) -> str:
    return " ".join(f"{(0.0 if not np.isfinite(x) else x):.{nd}f}" for x in arr)


def plan_clips(ann: SongAnnotation, cfg: Config) -> List[Clip]:
    """Cut a song into clips at rests, 2-16 s, never inside a syllable."""
    c = cfg.clips
    notes = ann.notes
    split = int(round(c.split_rest_s * FPS))
    pad = int(round(c.pad_s * FPS))
    min_f, max_f = int(c.min_s * FPS), int(c.max_s * FPS)
    # phrases: maximal runs of notes separated by rests >= split
    phrases: List[Tuple[int, int]] = []          # note index ranges [i, j)
    i = 0
    while i < len(notes):
        while i < len(notes) and notes[i].type == NOTE_REST and notes[i].end - notes[i].start >= split:
            i += 1
        if i >= len(notes):
            break
        j = i
        while j < len(notes) and not (notes[j].type == NOTE_REST and notes[j].end - notes[j].start >= split):
            j += 1
        # trim rests at phrase edges (short ones)
        a_, b_ = i, j
        while a_ < b_ and notes[a_].type == NOTE_REST:
            a_ += 1
        while b_ > a_ and notes[b_ - 1].type == NOTE_REST:
            b_ -= 1
        if b_ > a_:
            phrases.append((a_, b_))
        i = j

    def span(p):
        return notes[p[1] - 1].end - notes[p[0]].start

    # greedily merge consecutive phrases while the clip stays <= max
    groups: List[List[Tuple[int, int]]] = []
    for p in phrases:
        if groups:
            g = groups[-1]
            total = notes[p[1] - 1].end - notes[g[0][0]].start
            if span((g[0][0], g[-1][1])) < min_f and total <= max_f:
                g.append(p)
                continue
        groups.append([p])
    # a short last group joins the previous one if it fits
    if len(groups) > 1 and span((groups[-1][0][0], groups[-1][-1][1])) < min_f:
        total = notes[groups[-1][-1][1] - 1].end - notes[groups[-2][0][0]].start
        if total <= max_f:
            groups[-2].extend(groups.pop())

    clips: List[Clip] = []
    for g in groups:
        ni, nj = g[0][0], g[-1][1]
        a = notes[ni].start
        b = notes[nj - 1].end
        # pad into the neighbouring rests
        if ni > 0 and notes[ni - 1].type == NOTE_REST:
            a = max(notes[ni - 1].start, a - pad)
        if nj < len(notes) and notes[nj].type == NOTE_REST:
            b = min(notes[nj].end, b + pad)
        sub: List[NoteAnn] = []
        for n in notes:
            if n.end <= a or n.start >= b:
                continue
            m = NoteAnn(**{**n.__dict__})
            m.start, m.end = max(n.start, a), min(n.end, b)
            sub.append(m)
        clips.append(Clip(ann, a, b, sub, long=(b - a) > max_f))
    return clips


def clip_item(clip: Clip, clip_id: str, wav_rel: str) -> Dict:
    ann = clip.song
    a, b = clip.a, clip.b
    sy = ann.syllables
    texts, phons, pitches, types, cents, swaras, durs = [], [], [], [], [], [], []
    for n in clip.notes:
        durs.append((n.end - n.start) / FPS)
        types.append(n.type)
        if n.type == NOTE_REST:
            texts.append("<SP>")
            phons.append("<SP>")
            pitches.append(0)
            cents.append("-")
            swaras.append("-")
        else:
            s = sy[n.syllable]
            texts.append(s.text.replace(" ", "") or "~")
            phons.append(note_phoneme_string(s.phones))
            pitches.append(int(n.midi))
            cents.append("-" if n.cents is None else f"{n.cents:.1f}")
            swaras.append(n.swara or "-")
    sl = slice(a, b)
    return {
        "index": clip_id,
        "language": LANG_NAMES[ann.language],
        "time": [int(round(a * 1000 / FPS)), int(round(b * 1000 / FPS))],
        # SoulX computes frames as int(sum(durations) * 50); a float sum of exact
        # 0.02-multiples can land at 229.9999 -> 229.  A 0.1 ms nudge on the last
        # note keeps the truncation on the right side (frames stay exact).
        "duration": " ".join([f"{d:.2f}" for d in durs[:-1]] + [f"{durs[-1] + 1e-4:.4f}"]),
        "text": " ".join(texts),
        "phoneme": " ".join(phons),
        "note_pitch": " ".join(map(str, pitches)),
        "note_type": " ".join(map(str, types)),
        "f0": _fmt(ann.f0[sl], 1),
        "lang": ann.language,
        "style": ann.style,
        "singer_id": ann.singer_id,
        "gender": ann.gender,
        "song_id": ann.song_id,
        "wav": wav_rel,
        "n_frames": b - a,
        "tonic_hz": round(float(ann.tonic_hz), 3),
        "note_cents": " ".join(cents),
        "swara": " ".join(swaras),
        "energy_db": _fmt(ann.energy_db[sl], 1),
        "hnr_db": _fmt(ann.hnr_db[sl], 1),
        "h1h2_db": _fmt(ann.h1h2_db[sl], 1),
        "f0_agree": _fmt(ann.f0_agree[sl], 2),
        "status": ann.status,
    }


def _exclusion_reason(clip: Clip, policy: str) -> Optional[str]:
    for n in clip.notes:
        if n.type == NOTE_REST:
            continue
        s = clip.song.syllables[n.syllable]
        if "vocalise" in s.flags and "akaar" not in s.flags:
            return "unlabelled_vocalise"
    if not any(n.type != NOTE_REST for n in clip.notes):
        return "no_singing"
    return None


def assign_splits(songs: List[SongAnnotation], val_frac: float = 0.03, seed: int = 0,
                  holdout_singers: Iterable[str] = ()) -> Dict[str, str]:
    """Song-level splits.  gold songs -> test (they are the evaluation set);
    songs of held-out singers -> test; a random val_frac of the rest -> val."""
    rng = random.Random(seed)
    hold = set(holdout_singers)
    out = {}
    rest = []
    for s in songs:
        if s.status == "gold" or (s.singer_id and s.singer_id in hold):
            out[s.song_id] = "test"
        else:
            rest.append(s.song_id)
    rng.shuffle(rest)
    k = max(1, int(round(val_frac * len(rest)))) if len(rest) > 10 else 0
    for i, sid in enumerate(rest):
        out[sid] = "val" if i < k else "train"
    return out


def export_dataset(cfg: Config, include_status=("auto", "reviewed", "gold"),
                   soulx_phone_set: Optional[str] = None, val_frac: float = 0.03,
                   holdout_singers: Iterable[str] = ()) -> Dict:
    work = Path(cfg.work_dir)
    out = Path(cfg.out_dir)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    songs = []
    for d in sorted((work / "songs").glob("*")):
        if (d / "annotation.json").exists():
            songs.append(SongAnnotation.load(d))
    splits = assign_splits(songs, val_frac, holdout_singers=holdout_singers)
    items: Dict[str, List[Dict]] = defaultdict(list)
    excluded: List[Dict] = []
    stats = defaultdict(float)
    for ann in songs:
        if ann.status not in include_status:
            excluded.append({"song_id": ann.song_id, "reason": f"status={ann.status}"})
            continue
        errs = ann.validate()
        if errs:
            excluded.append({"song_id": ann.song_id, "reason": "invalid", "errors": errs[:5]})
            continue
        vocals = load_audio(ann.vocals_path, SR)
        clips = plan_clips(ann, cfg)
        ids = [f"{ann.song_id}_c{k:03d}" for k in range(len(clips))]
        kept: List[Tuple[str, Dict]] = []
        for cid, clip in zip(ids, clips):
            why = _exclusion_reason(clip, cfg.align.vocalise_policy)
            if why:
                excluded.append({"index": cid, "reason": why})
                continue
            wav_rel = f"wavs/{cid}.wav"
            seg = vocals[clip.a * HOP: clip.b * HOP]
            seg = np.pad(seg, (0, (clip.b - clip.a) * HOP - len(seg)))
            save_audio(out / wav_rel, seg)
            item = clip_item(clip, cid, wav_rel)
            if clip.long:
                item["long"] = True
            kept.append((cid, item))
        for k, (cid, item) in enumerate(kept):
            item["prev"] = kept[k - 1][0] if k > 0 and kept[k - 1][1]["time"][1] <= item["time"][0] else ""
            item["next"] = kept[k + 1][0] if k + 1 < len(kept) else ""
            split = splits[ann.song_id]
            items[split].append(item)
            sec = item["n_frames"] / FPS
            stats[f"total"] += sec
            stats[f"split:{split}"] += sec
            stats[f"lang:{ann.language}"] += sec
            stats[f"style:{ann.language}:{ann.style}"] += sec
            stats[f"status:{ann.status}"] += sec
            stats[f"singer:{ann.language}:{ann.singer_id or 'unknown'}"] += sec
    for split, lst in items.items():
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for it in lst:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        (out / f"{split}.json").write_text(json.dumps(lst, ensure_ascii=False), encoding="utf-8")
    with open(out / "excluded.jsonl", "w", encoding="utf-8") as f:
        for e in excluded:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    hours = {k: round(v / 3600, 3) for k, v in sorted(stats.items())}
    (out / "stats.json").write_text(json.dumps(hours, indent=1, ensure_ascii=False), encoding="utf-8")
    _write_stats_md(out / "stats.md", hours, {k: len(v) for k, v in items.items()}, len(excluded))
    sx = out / "soulx"
    sx.mkdir(exist_ok=True)
    (sx / "embedding_init_map.json").write_text(json.dumps(embedding_init_map(), indent=1), encoding="utf-8")
    (sx / "data_processor.patch").write_text(DATA_PROCESSOR_PATCH, encoding="utf-8")
    if soulx_phone_set:
        extend_soulx_phoneset(soulx_phone_set, sx / "phone_set_indic.json")
    return {"hours": hours, "clips": {k: len(v) for k, v in items.items()}, "excluded": len(excluded)}


def _write_stats_md(path: Path, hours: Dict[str, float], clips: Dict[str, int], n_excl: int) -> None:
    lines = ["# Dataset statistics", "", f"Clips: {clips}; excluded items: {n_excl}", "",
             "| Group | Hours |", "| --- | --- |"]
    for k, v in hours.items():
        if not k.startswith("singer:"):
            lines.append(f"| {k} | {v:.2f} |")
    singers = defaultdict(int)
    for k in hours:
        if k.startswith("singer:"):
            singers[k.split(":")[1]] += 1
    lines += ["", "| Language | Distinct singers |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in sorted(singers.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
