"""Sonic Visualiser review round-trip and the sync-preview audio.

``export_review(b)`` writes into ``<bundle>/review/``:

* ``f0.svl``           time-value layer, lead-vocal F0 (Hz, 100 Hz)
* ``flat.svl``         time-value layer, the flat (gamaka-free) swara contour (Hz)
* ``swaras.svl``       region layer: stable / kampita segments at their pitch, label = swara
* ``lines.svl``        region layer, label ``<line_id>|<section>|<text>``
* ``words.svl``        region layer, label = word
* ``syllables.svl``    region layer at the syllable's median F0, label = syllable
* ``sections.svl``     region layer, label = section name
* ``sama.svl``         time-instant layer
* ``f0_vocal-s.svl``, ``lines_vocal-s.svl``, ``words_vocal-s.svl``,
  ``syllables_vocal-s.svl``   duets only: the second singer's own F0 and alignment
  (for viewing; ``review-import`` reads the lead singer's layers)
* ``sync_preview.flac`` stereo: left = cleaned vocal; right = a sine following the F0,
  a short tick at every syllable onset and a lower knock on every sama
* ``README.txt``       how to open everything

Every time is on the mix clock; frames in the .svl files are at 44.1 kHz.

To correct: in Sonic Visualiser edit the lines / words / syllables / sections / sama
layers, export each with *File > Export Annotation Layer* under the same name into
``review/corrected/``, then run ``saregama review-import``.  Lyrics are rebuilt by
containment (a syllable belongs to the word region holding its midpoint, a word to
the line region holding its midpoint), so regions may be added, deleted or moved.
Imported recordings get ``status = reviewed`` and later ``align`` runs skip them.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import quoteattr

import numpy as np
import soundfile as sf

from .ir import Bundle

SV_SR = 44100
EDITABLE = ("lines", "words", "syllables", "sections", "sama")


# ----------------------------------------------------------------- .svl writing

def _svl(kind: str, name: str, points: Sequence[dict], end_s: float, colour: str,
         units: str = "", resolution: int = 1, plot_style: int = 0) -> str:
    """One Sonic Visualiser layer file.  kind: timevalues | regions | timeinstants."""
    dims = {"timeinstants": 1, "timevalues": 2, "regions": 3}[kind]
    vals = [p["value"] for p in points if "value" in p]
    lo, hi = (min(vals), max(vals)) if vals else (0.0, 0.0)
    out = ['<?xml version="1.0" encoding="UTF-8"?>', "<!DOCTYPE sonic-visualiser>", "<sv>", "  <data>",
           f'    <model id="1" name={quoteattr(name)} sampleRate="{SV_SR}" start="0" '
           f'end="{int(round(end_s * SV_SR))}" type="sparse" dimensions="{dims}" resolution="{resolution}" '
           f'notifyOnAdd="true" dataset="0" minimum="{lo:.3f}" maximum="{hi:.3f}" units={quoteattr(units)} />',
           f'    <dataset id="0" dimensions="{dims}">']
    for p in points:
        a = [f'frame="{int(round(p["start"] * SV_SR))}"']
        if dims >= 2:
            a.append(f'value="{p["value"]:.3f}"')
        if dims == 3:
            a.append(f'duration="{max(1, int(round((p["end"] - p["start"]) * SV_SR)))}"')
        a.append(f'label={quoteattr(p.get("label", ""))}')
        out.append(f'      <point {" ".join(a)} />')
    out += ["    </dataset>", "  </data>", "  <display>",
            f'    <layer id="2" type="{kind}" name={quoteattr(name)} model="1" colourName="" '
            f'colour="{colour}" darkBackground="false" plotStyle="{plot_style}" />',
            "  </display>", "</sv>", ""]
    return "\n".join(out)


def read_svl(path: Path) -> List[dict]:
    """Points of a .svl layer as {start, end, value, label} in seconds."""
    root = ET.parse(path).getroot()
    model = root.find("./data/model")
    sr = float(model.get("sampleRate", SV_SR)) if model is not None else SV_SR
    pts = []
    for p in root.iter("point"):
        t = int(p.get("frame", 0)) / sr
        d = int(p.get("duration", 0)) / sr
        pts.append({"start": t, "end": t + d, "value": float(p.get("value", "nan") or "nan"),
                    "label": p.get("label", "")})
    return sorted(pts, key=lambda q: q["start"])


# --------------------------------------------------------------------- export

def _hz(cents: float, tonic: float) -> float:
    """Hz of ``cents`` above ``tonic``."""
    return float(tonic * 2 ** (cents / 1200))


def _curve(t: np.ndarray, hz: np.ndarray) -> List[dict]:
    """Time-value points for the voiced frames of an Hz curve."""
    v = np.isfinite(hz) & (hz > 0)
    return [{"start": float(a), "value": float(b)} for a, b in zip(t[v], hz[v])]


def _median_hz(f0: np.ndarray, fps: float, a: float, b: float, default: float) -> float:
    """Median voiced F0 over [a, b) seconds, or ``default`` if unvoiced there
    (syllable regions are drawn at that height so they sit on the contour).
    """
    seg = f0[int(a * fps):max(int(a * fps) + 1, int(b * fps))]
    seg = seg[seg > 0]
    return float(np.median(seg)) if len(seg) else default


def export_review(b: Bundle, preview: bool = True) -> Dict[str, str]:
    """Write every review layer of a bundle into ``review/`` (see the module
    docstring) and return {layer name: path}.  ``preview=False`` skips the
    sync-preview audio, the slow part on long recordings.
    """
    ir = b.load()
    out_dir = b.review_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tonic = float(ir["tonic_hz"])
    z = np.load(b.path("f0_100hz.npz"))
    fps = float(z["fps"])
    f0 = z["f0_hz"].astype(float)
    t = np.arange(len(f0)) / fps
    end_s = max(float(ir.get("length_s") or 0), len(f0) / fps)
    layers: Dict[str, str] = {}

    def put(name, text):
        """Write one layer file and remember its path."""
        (out_dir / f"{name}.svl").write_text(text, encoding="utf-8")
        layers[name] = str(out_dir / f"{name}.svl")

    res = int(round(SV_SR / fps))
    # discrete-curve style (6) leaves gaps where the voice is silent
    put("f0", _svl("timevalues", "F0 (vocal_clean)", _curve(t, f0), end_s, "#0b0b0b", "Hz", res, 6))
    if b.path("flat_100hz.npz").exists():
        fc = np.load(b.path("flat_100hz.npz"))["flat_cents"].astype(float)
        put("flat", _svl("timevalues", "flat swaras", _curve(t[:len(fc)], tonic * 2 ** (fc / 1200)),
                         end_s, "#2a78d6", "Hz", res, 6))
    if b.path("swaras.json").exists():
        sw = json.loads(b.path("swaras.json").read_text(encoding="utf-8"))
        pts = [{"start": s["start"], "end": s["end"], "value": _hz(s["cents"], tonic),
                "label": s["swara"] if s["kind"] == "stable" else f'{s["swara"] or "~"} kampita'}
               for s in sw["segments"] if s["kind"] in ("stable", "kampita")]
        put("swaras", _svl("regions", "swaras", pts, end_s, "#1baf7a", "Hz"))

    lyr = _read_json(b.path("lyrics.json"))
    if lyr:
        for name, kind, pts, colour, units in _lyric_layers(lyr, f0, fps, tonic):
            put(name, _svl(kind, name, pts, end_s, colour, units))
    # duets: the second singer's own F0 and alignment, as "<layer>_vocal-s"
    if b.path("f0_100hz_vocal-s.npz").exists():
        f0s = np.load(b.path("f0_100hz_vocal-s.npz"))["f0_hz"].astype(float)
        put("f0_vocal-s", _svl("timevalues", "F0 (vocal-s_clean)", _curve(t[:len(f0s)], f0s), end_s,
                               "#8a8985", "Hz", res, 6))
        lyr2 = _read_json(b.path("lyrics_vocal-s.json"))
        if lyr2:
            for name, kind, pts, colour, units in _lyric_layers(lyr2, f0s, fps, tonic):
                put(f"{name}_vocal-s", _svl(kind, f"{name} (second singer)", pts, end_s, colour, units))

    put("sections", _svl("regions", "sections",
                         [{"start": s["start"], "end": s["end"], "value": 0.0, "label": s["label"]}
                          for s in (ir.get("sections") or ir.get("sections_auto") or [])], end_s, "#52514e"))
    put("sama", _svl("timeinstants", "sama", [{"start": float(x), "label": "sama"}
                                               for x in ir.get("sama", [])], end_s, "#0b0b0b"))
    if preview:
        onsets = [s["start"] for inst in (lyr or {}).get("instances", [])
                  for w in inst["words"] for s in w["syllables"]]
        layers["sync_preview"] = str(sync_preview(b.dir / ir["stems"]["vocal_clean"]["path"], f0, fps,
                                                  onsets, ir.get("sama", []), out_dir / "sync_preview.flac"))
    (out_dir / "README.txt").write_text(_readme(ir, b), encoding="utf-8")
    return layers


def _read_json(path: Path) -> Optional[dict]:
    """Parsed JSON file, or None if it does not exist."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _lyric_layers(lyr: dict, f0: np.ndarray, fps: float, tonic: float):
    """(name, kind, points, colour, units) of the lines / words / syllables layers
    of one ``lyrics.json``; syllables sit at their median F0."""
    texts = {l["id"]: l["text"] for l in lyr.get("lines", [])}
    lines, words, syls = [], [], []
    for inst in lyr["instances"]:
        text = texts.get(inst["line_id"], " ".join(w["text"] for w in inst["words"]))
        lines.append({"start": inst["start"], "end": inst["end"], "value": 0.0,
                      "label": f'{inst["line_id"]}|{inst["section"]}|{text}'})
        for w in inst["words"]:
            words.append({"start": w["start"], "end": w["end"], "value": 0.0, "label": w["text"]})
            for s in w["syllables"]:
                syls.append({"start": s["start"], "end": s["end"], "label": s["text"],
                             "value": _median_hz(f0, fps, s["start"], s["end"], tonic)})
    return [("lines", "regions", lines, "#52514e", ""), ("words", "regions", words, "#8a8985", ""),
            ("syllables", "regions", syls, "#eb6834", "Hz")]


def _readme(ir: dict, b: Bundle) -> str:
    """Text of ``review/README.txt``: what is in the folder and the exact
    commands to open and to re-import it.
    """
    vocal = b.dir / ir["stems"]["vocal_clean"]["path"]
    svl = " ".join(sorted(p.name for p in b.review_dir.glob("*.svl")))
    return f"""{ir["title"]} ({ir.get("raaga", "")}, {ir.get("taala", "")}); Sa = {ir["tonic_hz"]} Hz
All times are on the mix clock: {ir["mix"]}

Listen: sync_preview.flac (left: cleaned vocal; right: sine at the F0, a tick per
syllable onset, a knock per sama).  If the sine or ticks drift from the voice,
the sync or the alignment is off there.

Open in Sonic Visualiser:
  cd {b.review_dir}
  sonic-visualiser {vocal} {svl}
or open the audio (vocal_clean.flac or the mix) and use File > Import Annotation
Layer for each .svl.  f0 / flat / swaras / syllables are in Hz: put them on the same
pane as a spectrogram (log frequency) to see them on top of each other.

Correct: edit lines / words / syllables / sections / sama, export each with
File > Export Annotation Layer as corrected/<same name>.svl, then
  saregama review-import --mbids {ir["mbid"]} --reviewer <you>
Line labels are "<line_id>|<section>|<text>": keep the line_id when moving or
copying a line region (it says which lyric line was sung).
"""


# --------------------------------------------------------------- sync preview

def _click(sr: int, freq: float, dur: float, decay: float) -> np.ndarray:
    """A decaying sine burst: the tick / knock of the sync preview."""
    tt = np.arange(int(dur * sr)) / sr
    return np.sin(2 * np.pi * freq * tt) * np.exp(-tt / decay)


def sync_preview(vocal_path: Path, f0: np.ndarray, fps: float, onsets: Sequence[float],
                 sama: Sequence[float], out_path: Path, sine_gain: float = 0.25) -> Path:
    """Write the stereo sync-check file (see the module docstring).

    The right channel's sine follows the F0 with continuous phase (pitch held
    across unvoiced gaps so restarts do not click) and is gated by voicing with
    a 10 ms ramp.  Syllable onsets get a short 2.5 kHz tick, sama a 0.7 kHz knock.
    """
    x, sr = sf.read(str(vocal_path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    peak = float(np.max(np.abs(x))) or 1.0
    left = 0.8 * x / peak
    n = len(left)
    # sine following the F0, gated by voicing with a 10 ms ramp
    tf = np.arange(len(f0)) / fps
    ts = np.arange(n) / sr
    hz = np.interp(ts, tf, np.where(f0 > 0, f0, 0.0))
    voiced = np.interp(ts, tf, (f0 > 0).astype(float))
    k = max(1, int(0.01 * sr))
    c = np.concatenate([[0.0], np.cumsum(voiced)])
    lo = np.clip(np.arange(n) - k // 2, 0, n)
    gate = (c[np.clip(lo + k, 0, n)] - c[lo]) / k          # moving average, O(n)
    hz_fill = np.interp(ts, tf[f0 > 0], f0[f0 > 0]) if (f0 > 0).any() else hz
    phase = 2 * np.pi * np.cumsum(hz_fill) / sr
    right = sine_gain * gate * np.sin(phase)
    for times, click, g in ((onsets, _click(sr, 2500, 0.015, 0.003), 0.35),
                            (sama, _click(sr, 700, 0.06, 0.012), 0.6)):
        for t0 in times:
            i = int(round(t0 * sr))
            if 0 <= i < n:
                j = min(n, i + len(click))
                right[i:j] += g * click[:j - i]
    y = np.stack([left, np.clip(right, -1, 1)], axis=1).astype(np.float32)
    sf.write(str(out_path), y, sr, subtype="PCM_16")
    return out_path


# --------------------------------------------------------------------- import

def _inside(regions: List[dict], t: float) -> Optional[int]:
    """Index of the region containing time ``t``, or None."""
    for i, r in enumerate(regions):
        if r["start"] <= t < r["end"]:
            return i
    return None


def lyrics_from_layers(lines: List[dict], words: List[dict], syls: List[dict]) -> Tuple[List[dict], List[str]]:
    """Rebuild lyrics.json instances from corrected region layers."""
    problems = []
    insts = []
    for ln in lines:
        lid, section, _ = (ln["label"].split("|", 2) + ["", ""])[:3]
        try:
            lid = int(lid)
        except ValueError:
            problems.append(f"line at {ln['start']:.2f}s has no line_id in its label: {ln['label']!r}")
            lid = -1
        insts.append({"line_id": lid, "section": section or "", "start": ln["start"], "end": ln["end"],
                      "complete": True, "region": "", "flags": ["reviewed"], "words": []})
    for w in words:
        i = _inside(lines, (w["start"] + w["end"]) / 2)
        if i is None:
            problems.append(f"word {w['label']!r} at {w['start']:.2f}s is outside every line")
            continue
        insts[i]["words"].append({"text": w["label"], "start": w["start"], "end": w["end"], "syllables": []})
    flat_words = [(w, inst) for inst in insts for w in inst["words"]]
    wregions = [w for w, _ in flat_words]
    for s in syls:
        i = _inside(wregions, (s["start"] + s["end"]) / 2)
        if i is None:
            problems.append(f"syllable {s['label']!r} at {s['start']:.2f}s is outside every word")
            continue
        wregions[i]["syllables"].append({"text": s["label"], "start": s["start"], "end": s["end"]})
    for inst in insts:
        inst["words"].sort(key=lambda w: w["start"])
        for w in inst["words"]:
            w["syllables"].sort(key=lambda s: s["start"])
            if not w["syllables"]:
                problems.append(f"word {w['text']!r} at {w['start']:.2f}s has no syllables")
    return insts, problems


def snapshot_predictions(b: Bundle, ir: dict) -> bool:
    """Copy the pipeline's predictions into ``pred/`` unless already there
    (the first review import keeps them; later imports leave them alone).
    Returns True if a snapshot was written."""
    pred = b.dir / "pred"
    if (pred / "ir_pred.json").exists():
        return False
    pred.mkdir(exist_ok=True)
    for name in ("lyrics.json", "lyrics_vocal-s.json"):
        if b.path(name).exists():
            (pred / name).write_bytes(b.path(name).read_bytes())
    keep = {k: ir.get(k) for k in ("sections", "sections_auto", "sama", "tonic_hz", "tala")}
    (pred / "ir_pred.json").write_text(json.dumps(keep, ensure_ascii=False), encoding="utf-8")
    return True


def import_review(b: Bundle, reviewer: str = "", corrected: Optional[Path] = None) -> dict:
    """Read corrected layers from ``corrected`` (default
    ``review/corrected/``) back into the bundle.

    Lyrics are rebuilt only when lines, words and syllables are all present;
    sections and sama replace the Saraga ones.  The recording is marked
    ``status = reviewed`` (``align`` then skips it).  Returns a report with the
    layers read, the number of instances and any problems found.

    Before anything is overwritten, the pipeline's predictions are kept in
    ``pred/`` (``lyrics.json`` and ``ir_pred.json`` with sections, auto sections,
    sama and tonic) on the first import only, so ``gold-eval`` can score them
    against the corrections later.
    """
    src = corrected or (b.review_dir / "corrected")
    have = {n: src / f"{n}.svl" for n in EDITABLE if (src / f"{n}.svl").exists()}
    if not have:
        raise FileNotFoundError(f"no corrected layers in {src}")
    ir = b.load()
    snapshot_predictions(b, ir)
    report = {"layers": sorted(have)}
    if {"lines", "words", "syllables"} <= set(have):
        lyr = json.loads(b.path("lyrics.json").read_text(encoding="utf-8")) if b.path("lyrics.json").exists() else {}
        insts, problems = lyrics_from_layers(*(read_svl(have[n]) for n in ("lines", "words", "syllables")))
        lyr["instances"] = insts
        lyr.setdefault("source", {})["reviewed_by"] = reviewer
        lyr["reviewed"] = True
        b.path("lyrics.json").write_text(json.dumps(lyr, ensure_ascii=False), encoding="utf-8")
        report.update(instances=len(insts), problems=problems)
    elif have.keys() & {"lines", "words", "syllables"}:
        report["problems"] = ["lyrics need all three of lines, words, syllables; lyrics not imported"]
    if "sections" in have:
        ir["sections"] = [{"start": round(p["start"], 4), "end": round(p["end"], 4), "label": p["label"]}
                          for p in read_svl(have["sections"])]
    if "sama" in have:
        ir["sama"] = [round(p["start"], 4) for p in read_svl(have["sama"])]
    ir["status"] = "reviewed"
    ir["review"] = {"reviewer": reviewer, "layers": sorted(have)}
    b.save(ir)
    return report
