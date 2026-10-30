"""Human review round-trip through Praat TextGrids.

``export_review`` writes, per song, into ``review/<song_id>/``:

    vocals.wav             the separated vocal (open it together with the TextGrid)
    <song_id>.TextGrid     tiers below
    <song_id>.PitchTier    the F0 track, for viewing in Praat
    README.txt             what to fix and how

TextGrid tiers
--------------
syllables   one interval per syllable, label ``text|phones`` e.g. ``मन|m-a-n``.
            Empty intervals are rests; ``<VOC>`` is unlabelled vocalising (aalap,
            humming).  Move boundaries, fix text; to fix pronunciation edit the
            part after '|' (phones separated by '-'), or delete it to re-run G2P.
            Relabel a ``<VOC>`` interval as e.g. ``आ|aa`` to keep it as akaar.
lines       lyric line spans (information only).
notes       swara of each note ('+' marks a slur), recomputed on import unless
            ``--keep-notes`` is given.
f0_fix      add intervals labelled ``x2`` / ``/2`` (octave up/down), ``uv``
            (force unvoiced) or ``+50`` / ``-30`` (shift in cents) to correct F0.
tonic       one interval spanning the song whose label is the tonic in Hz; type
            the correct value (any octave) if the auto value is wrong.
flags       QC hints: ``?`` marks low-confidence syllables (information only).

``import_review`` reads the edited TextGrid back, rebuilds syllables and notes,
applies F0 fixes and the tonic, re-runs QC-free validation, and marks the song
``reviewed`` (or ``gold`` with ``--gold``).
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .audio import FPS
from .config import Config
from .g2p import Lexicon, word_to_syllables
from .phoneset import BY_NAME
from .schema import NOTE_REST, NOTE_SLUR, SongAnnotation, SyllableAnn
from .stages.notes import build_notes
from .stages.tonic import snap_to_tonic_octave

Interval = Tuple[float, float, str]


# ---------------------------------------------------------------------------
# minimal TextGrid / PitchTier I/O (long text format)
# ---------------------------------------------------------------------------
def _q(s: str) -> str:
    return '"' + s.replace('"', '""') + '"'


def write_textgrid(path: Path, xmax: float, tiers: List[Tuple[str, List[Interval]]]) -> None:
    out = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "",
           "xmin = 0", f"xmax = {xmax:.6f}", "tiers? <exists>", f"size = {len(tiers)}", "item []:"]
    for ti, (name, ivs) in enumerate(tiers, 1):
        ivs = _fill_gaps(ivs, xmax)
        out += [f"    item [{ti}]:", '        class = "IntervalTier"', f"        name = {_q(name)}",
                "        xmin = 0", f"        xmax = {xmax:.6f}",
                f"        intervals: size = {len(ivs)}"]
        for k, (a, b, lab) in enumerate(ivs, 1):
            out += [f"        intervals [{k}]:", f"            xmin = {a:.6f}",
                    f"            xmax = {b:.6f}", f"            text = {_q(lab)}"]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _fill_gaps(ivs: List[Interval], xmax: float) -> List[Interval]:
    out: List[Interval] = []
    t = 0.0
    for a, b, lab in sorted(ivs):
        if a > t + 1e-9:
            out.append((t, a, ""))
        a = max(a, t)
        if b > a:
            out.append((a, b, lab))
            t = b
    if t < xmax - 1e-9:
        out.append((t, xmax, ""))
    return out


def read_textgrid(path: Path) -> Dict[str, List[Interval]]:
    txt = path.read_text(encoding="utf-8")
    tiers: Dict[str, List[Interval]] = {}
    for block in re.split(r"\n\s*item \[\d+\]:\s*\n", txt)[1:]:
        m = re.search(r'name = "((?:[^"]|"")*)"', block)
        if not m:
            continue
        name = m.group(1).replace('""', '"')
        ivs = []
        for im in re.finditer(r'xmin = ([\d.eE+-]+)\s*\n\s*xmax = ([\d.eE+-]+)\s*\n\s*text = "((?:[^"]|"")*)"',
                              block, re.S):
            ivs.append((float(im.group(1)), float(im.group(2)), im.group(3).replace('""', '"')))
        tiers[name] = ivs
    return tiers


def write_pitchtier(path: Path, f0: np.ndarray, xmax: float) -> None:
    pts = [(k / FPS, float(f)) for k, f in enumerate(f0) if f > 0]
    out = ['File type = "ooTextFile"', 'Object class = "PitchTier"', "", "xmin = 0",
           f"xmax = {xmax:.6f}", f"points: size = {len(pts)}"]
    for k, (t, f) in enumerate(pts, 1):
        out += [f"points [{k}]:", f"    number = {t:.4f}", f"    value = {f:.2f}"]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
REVIEW_README = """Review checklist for {song_id}  ({language}, {style})
QC flags: {flags}

Open in Praat: Read vocals.wav and {song_id}.TextGrid, select both, View & Edit.
1. tonic tier: listen to a sustained Sa; correct the Hz value if needed (any octave).
2. syllables tier: fix wrong lyrics text and boundaries (onset of each syllable),
   especially intervals marked '?' in the flags tier.  Empty = rest.
3. <VOC> intervals: relabel as akaar (e.g. आ|aa) if you want to keep them, or leave.
4. f0_fix tier: add intervals x2, /2, uv, or +/-cents where the F0 track (PitchTier,
   or Praat's own pitch display) shows octave errors or bleed.
5. Save the TextGrid in place, then run:
   svs-annotate import-review --song {song_id} [--gold]
"""


def export_review(cfg: Config, song_ids: Optional[List[str]] = None, only_status=("flagged",),
                  out_dir: Optional[str] = None) -> List[str]:
    from .stages.qc import review_priority
    work = Path(cfg.work_dir)
    rdir = Path(out_dir or work / "review")
    anns = []
    for d in sorted((work / "songs").glob("*")):
        if not (d / "annotation.json").exists():
            continue
        if song_ids and d.name not in song_ids:
            continue
        ann = SongAnnotation.load(d)
        if not song_ids and ann.status not in only_status:
            continue
        anns.append(ann)
    anns.sort(key=lambda a: -review_priority(a.qc))
    done = []
    queue = []
    for ann in anns:
        d = rdir / ann.song_id
        d.mkdir(parents=True, exist_ok=True)
        xmax = ann.n_frames / FPS
        syl_iv, flag_iv = [], []
        for s in ann.syllables:
            a, b = s.start / FPS, s.end / FPS
            if "vocalise" in s.flags and "akaar" not in s.flags:
                syl_iv.append((a, b, "<VOC>"))
            else:
                syl_iv.append((a, b, f"{s.text}|{'-'.join(s.phones)}"))
            if np.isfinite(s.conf) and s.conf < cfg.qc.min_align_conf:
                flag_iv.append((a, b, "?"))
        note_iv = [(n.start / FPS, n.end / FPS, ("+" if n.type == NOTE_SLUR else "") + n.swara)
                   for n in ann.notes if n.type != NOTE_REST]
        line_iv = []
        for li, line in enumerate(ann.lines):
            ss = [s for s in ann.syllables if s.line == li and "vocalise" not in s.flags]
            if ss:
                line_iv.append((ss[0].start / FPS, ss[-1].end / FPS, line))
        tiers = [("syllables", syl_iv), ("lines", line_iv), ("notes", note_iv), ("f0_fix", []),
                 ("tonic", [(0.0, xmax, f"{ann.tonic_hz:.2f}")]), ("flags", flag_iv)]
        write_textgrid(d / f"{ann.song_id}.TextGrid", xmax, tiers)
        write_pitchtier(d / f"{ann.song_id}.PitchTier", ann.f0, xmax)
        if Path(ann.vocals_path).exists():
            shutil.copyfile(ann.vocals_path, d / "vocals.wav")
        (d / "README.txt").write_text(REVIEW_README.format(
            song_id=ann.song_id, language=ann.language, style=ann.style,
            flags=", ".join(ann.qc.get("flags", [])) or "none"), encoding="utf-8")
        done.append(ann.song_id)
        queue.append(f"{ann.song_id}\t{review_priority(ann.qc):.2f}\t{','.join(ann.qc.get('flags', []))}")
    if queue:
        (rdir / "queue.tsv").write_text("song_id\tpriority\tflags\n" + "\n".join(queue) + "\n",
                                        encoding="utf-8")
    return done


def _apply_f0_fixes(f0: np.ndarray, fixes: List[Interval]) -> Tuple[np.ndarray, int]:
    f0 = f0.copy()
    nfix = 0
    for a, b, lab in fixes:
        lab = lab.strip()
        if not lab:
            continue
        i, j = int(round(a * FPS)), int(round(b * FPS))
        seg = f0[i:j]
        v = seg > 0
        if lab == "x2":
            seg[v] *= 2
        elif lab == "/2":
            seg[v] /= 2
        elif lab == "uv":
            seg[:] = 0
        elif re.fullmatch(r"[+-]\d+(\.\d+)?", lab):
            seg[v] *= 2 ** (float(lab) / 1200)
        else:
            raise ValueError(f"f0_fix label {lab!r} at {a:.2f}s not understood (x2, /2, uv, +/-cents)")
        f0[i:j] = seg
        nfix += 1
    return f0, nfix


def import_review(cfg: Config, song_id: str, textgrid: Optional[str] = None, gold: bool = False,
                  keep_notes: bool = False, reviewer: str = "") -> SongAnnotation:
    from .pipeline import g2p_opts
    work = Path(cfg.work_dir)
    d = work / "songs" / song_id
    ann = SongAnnotation.load(d)
    tg_path = Path(textgrid) if textgrid else work / "review" / song_id / f"{song_id}.TextGrid"
    tiers = read_textgrid(tg_path)
    if "syllables" not in tiers:
        raise ValueError(f"{tg_path} has no 'syllables' tier")
    lex = Lexicon.load(*cfg.g2p.lexicon)
    opts = g2p_opts(cfg)
    old = {(s.start, s.end): s for s in ann.syllables}

    # tonic
    if "tonic" in tiers:
        labs = [l for _, _, l in tiers["tonic"] if l.strip()]
        if labs:
            t = round(snap_to_tonic_octave(float(labs[0]), ann.f0), 3)
            if abs(t - ann.tonic_hz) > 0.05:
                ann.tonic_source = "reviewed"
            ann.tonic_hz = t

    # F0 fixes
    if "f0_fix" in tiers:
        ann.f0, nfix = _apply_f0_fixes(ann.f0, tiers["f0_fix"])
    else:
        nfix = 0

    # syllables
    sylls: List[SyllableAnn] = []
    line_of = _line_lookup(tiers.get("lines", []))
    for a, b, lab in tiers["syllables"]:
        lab = lab.strip()
        if not lab:
            continue
        i, j = int(round(a * FPS)), int(round(b * FPS))
        if j <= i:
            continue
        prev = old.get((i, j))
        line = line_of((a + b) / 2, prev.line if prev else 0)
        if lab == "<VOC>":
            sylls.append(SyllableAnn("<VOC>", ["aa"], i, j, line, -1, 0.0, ["vocalise"]))
            continue
        text, _, ph = lab.partition("|")
        text = text.strip()
        if ph.strip():
            phones = [p for p in ph.strip().split("-") if p]
            bad = [p for p in phones if p not in BY_NAME]
            if bad:
                raise ValueError(f"{song_id} {a:.2f}s: unknown phones {bad} in {lab!r}")
        else:
            res = word_to_syllables(text, ann.language, lex, **opts)
            phones = [p for _, ps in res for p in ps]
        flags = ["vocalise", "akaar"] if (prev and "vocalise" in prev.flags) else []
        conf = prev.conf if prev and prev.text == text else 0.0
        sylls.append(SyllableAnn(text, phones, i, j, line, prev.word if prev else 0, conf, flags))
    ann.syllables = sylls

    # notes
    if keep_notes and "notes" in tiers:
        ann.notes = _notes_from_tier(ann, tiers["notes"])
    else:
        ann.notes = build_notes(ann.syllables, ann.f0, ann.tonic_hz, ann.n_frames, cfg.notes)
    errs = ann.validate()
    if errs:
        raise ValueError(f"{song_id}: reviewed annotation is invalid: {errs[:5]}")
    ann.status = "gold" if gold else "reviewed"
    ann.qc.setdefault("review", []).append({"reviewer": reviewer, "f0_fixes": nfix,
                                            "gold": gold})
    ann.history.append(f"review:{reviewer or 'unknown'}:{'gold' if gold else 'reviewed'}")
    ann.save(d)
    return ann


def _line_lookup(line_ivs: List[Interval]):
    spans = [(a, b, k) for k, (a, b, l) in enumerate([iv for iv in line_ivs if iv[2].strip()])]

    def f(t: float, default: int) -> int:
        for a, b, k in spans:
            if a <= t < b:
                return k
        return default
    return f


def _notes_from_tier(ann: SongAnnotation, ivs: List[Interval]):
    """Keep reviewer-edited note boundaries; pitches are re-measured from F0."""
    from .schema import NOTE_LYRIC, NoteAnn
    from .stages.notes import _midi, swara_name
    notes = []
    cents = ann.cents()
    sy = ann.syllables
    pos = 0
    for a, b, lab in ivs:
        if not lab.strip():
            continue
        i, j = int(round(a * FPS)), int(round(b * FPS))
        owner = next((k for k, s in enumerate(sy) if s.start <= i < s.end), None)
        if owner is None:
            continue
        if i > pos:
            notes.append(NoteAnn(pos, i, NOTE_REST, -1))
        c = cents[i:j]
        c = c[~np.isnan(c)]
        med = float(np.median(c)) if len(c) else 0.0
        typ = NOTE_SLUR if lab.startswith("+") or (notes and notes[-1].syllable == owner) else NOTE_LYRIC
        notes.append(NoteAnn(i, j, typ, owner, _midi(ann.tonic_hz, med), round(med, 1),
                             swara_name(int(round(med / 100)))))
        pos = j
    if pos < ann.n_frames:
        notes.append(NoteAnn(pos, ann.n_frames, NOTE_REST, -1))
    return notes
