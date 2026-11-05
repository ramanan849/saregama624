"""Command line: ``saregama <command> ...`` (or ``python -m saregama``).

Saraga Carnatic pipeline, in order (each command is resumable per recording;
``--mbids`` restricts it to a list or a file of recording ids, ``--force``
redoes recordings already done):

    manifest          index Saraga recordings -> <ann-root>/manifest.csv
    sync              stem offsets, mix-aligned FLAC stems, ir.json
    stems             de-bleed / de-reverb the vocal stem(s)            (GPU)
    f0                100 Hz F0 ensemble, tonic, Saraga QC              (GPU)
    tonic-consensus   one Sa per concert
    swaras            swara segments and flat contour
    sections-train    evaluate + train the automatic section classifier
    sections          automatic sections where Saraga has none
    tala              beat grid, laya and sama (estimated where not annotated)
    lyrics-fetch      candidate lyrics per composition (web)
    lyrics-status     rebuild lyrics/status.tsv from the YAMLs (offline)
    lyrics-verify     show candidates / record the right lyrics
    lyrics-review-pack  works + audio excerpts for the shared review page
    lyrics-apply      apply decisions exported from the review page
    emissions         cache MMS_FA emissions per vocal stem             (GPU)
    align             time-aligned lyrics (MMS_FA spikes; uses the cache)
    viz               contour sheets (PDF)
    review-export     Sonic Visualiser layers + sync preview
    review-import     read corrected layers back, mark reviewed
    export-ir         ICM-IR v0 dataset: parquet tables + 100 Hz frames
    export-soulx      SoulX-Singer training clips (melody mode)
    report            dataset QC report: report.csv + report.html
    svc-check         SoulX SVC rendering check: does its F0 input keep gamakas? (GPU)

Global options ``--saraga-root``, ``--ann-root`` and ``--lyrics-dir`` default to
the locations in :mod:`saregama.paths`.

Legacy film-song workspace (songs.csv; ported from indic-svs-annotator):
``init``, ``run``, ``draft-lyrics``, ``export-review``, ``audit``,
``import-review``, ``make-eval-csv``, ``evaluate``, ``export``, ``stats``,
``phoneset``, ``g2p``.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .config import load_config
from .ir import DEFAULT_ROOT as ANN_ROOT
from .paths import LYRICS_DIR
from .saraga import DEFAULT_ROOT as SARAGA_ROOT

SONGS_HEADER = ["song_id", "audio", "language", "lyrics", "style", "singer_id", "gender",
                "vocal_only", "tonic_hz", "source", "license"]


def cmd_init(a):
    """``init``: create a legacy film-song workspace (songs.csv, lyrics/, lexicons/, config)."""
    root = Path(a.dir)
    (root / "audio").mkdir(parents=True, exist_ok=True)
    (root / "lyrics").mkdir(exist_ok=True)
    (root / "lexicons").mkdir(exist_ok=True)
    if not (root / "songs.csv").exists():
        with open(root / "songs.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(SONGS_HEADER)
            w.writerow(["hi_bhajan_0001", "audio/hi_bhajan_0001.mp3", "hi", "lyrics/hi_bhajan_0001.txt",
                        "bhajan", "singer_017", "f", "0", "", "own recording", "owned"])
    lex = root / "lexicons" / "common.tsv"
    lex_src = Path(__file__).resolve().parent.parent / "lexicons" / "common.tsv"
    if not lex.exists():
        lex.write_text(lex_src.read_text(encoding="utf-8") if lex_src.exists() else
                       "# word<TAB>phones, '.' between syllables.\n", encoding="utf-8")
    src = Path(__file__).resolve().parent.parent / "configs" / "default.toml"
    if src.exists() and not (root / "config.toml").exists():
        (root / "config.toml").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"workspace ready in {root}: fill songs.csv, put lyric sheets in lyrics/, edit config.toml")


def _cfg(a):
    """Load ``--config`` with ``--work-dir`` / ``--out-dir`` overrides (legacy commands)."""
    over = {}
    if getattr(a, "work_dir", None):
        over["work_dir"] = a.work_dir
    if getattr(a, "out_dir", None):
        over["out_dir"] = a.out_dir
    return load_config(a.config, **over)


def cmd_run(a):
    """``run``: all legacy stages on songs.csv; prints a status count and errors."""
    from .pipeline import Pipeline, read_songs
    cfg = _cfg(a)
    songs = read_songs(a.songs)
    if a.only:
        keep = set(a.only.split(","))
        songs = [s for s in songs if s["song_id"] in keep]
    annot = _load_annotator(cfg)
    res = Pipeline(cfg, annotator=annot).run(songs, force=a.force)
    print(json.dumps(Counter(v if not v.startswith("error") else "error" for v in res.values()), indent=1))
    errs = {k: v for k, v in res.items() if v.startswith("error")}
    if errs:
        print("errors (see work/songs/<id>/error.txt):", json.dumps(errs, indent=1, ensure_ascii=False))


def _load_annotator(cfg):
    """Load the trained annotator model if the config uses it, else None."""
    need = cfg.align.backend == "annotator" or "annotator" in cfg.f0.backends
    if not need:
        return None
    from annotator_model.infer import AnnotatorRunner
    return AnnotatorRunner(cfg.align.annotator_ckpt, device=cfg.align.device)


def cmd_draft_lyrics(a):
    """``draft-lyrics``: ASR draft lyric sheets for songs without lyrics (legacy)."""
    from .audio import SR, load_audio, loudness_normalize, save_audio
    from .pipeline import Pipeline, read_songs
    from .stages.lyrics import IndicConformerASR, draft_lyrics
    cfg = _cfg(a)
    pipe = Pipeline(cfg)
    asr = IndicConformerASR(a.asr_model, device=cfg.align.device)
    for row in read_songs(a.songs):
        if row.get("lyrics"):
            continue
        d = pipe.song_dir(row["song_id"])
        d.mkdir(parents=True, exist_ok=True)
        vp = d / "vocals.wav"
        if vp.exists():
            vocals = load_audio(vp, SR)
        elif row.get("vocal_only") in ("1", "true", "yes"):
            vocals = loudness_normalize(load_audio(row["audio"], SR))
            save_audio(vp, vocals)
        else:
            vocals, _ = pipe.sep.process(row["audio"], str(vp), str(d / "accomp.wav"))
        f0 = pipe.f0est(vocals)["f0"]
        out = Path(a.out) / f"{row['song_id']}.draft.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = draft_lyrics(vocals, f0, row["language"], asr, out, d / "asr_phrases")
        print(f"{row['song_id']}: {len(lines)} draft lines -> {out}")


def cmd_export_review(a):
    """``export-review``: Praat TextGrids for flagged (or given) legacy songs."""
    from .review import export_review
    cfg = _cfg(a)
    ids = a.songs.split(",") if a.songs else None
    done = export_review(cfg, ids, tuple(a.status.split(",")))
    print(f"exported {len(done)} songs for review into {Path(cfg.work_dir) / 'review'}")


def cmd_audit(a):
    """Random sample of auto-accepted songs for spot checks."""
    from .review import export_review
    from .schema import SongAnnotation
    cfg = _cfg(a)
    auto = [d.name for d in (Path(cfg.work_dir) / "songs").glob("*")
            if (d / "annotation.json").exists() and SongAnnotation.load(d).status == "auto"]
    rng = random.Random(a.seed)
    k = max(1, int(round(a.fraction * len(auto)))) if auto else 0
    pick = rng.sample(auto, k) if k else []
    export_review(cfg, pick, ("auto",))
    print(f"audit sample: {len(pick)} of {len(auto)} auto songs exported for review")


def cmd_import_review(a):
    """``import-review``: read a corrected TextGrid back into a legacy song."""
    from .review import import_review
    cfg = _cfg(a)
    ann = import_review(cfg, a.song, a.textgrid, gold=a.gold, keep_notes=a.keep_notes,
                        reviewer=a.reviewer)
    print(f"{a.song}: {ann.status}, {len(ann.syllables)} syllables, {len(ann.notes)} notes, "
          f"tonic {ann.tonic_hz} Hz")


def cmd_make_eval_csv(a):
    """Gold songs -> a songs.csv that re-runs the automatic pipeline on them
    (reusing the separated vocals and the gold lyric lines) into another work dir."""
    from .schema import SongAnnotation
    cfg = _cfg(a)
    out = Path(a.out)
    lyr_dir = out.parent / "eval_lyrics"
    lyr_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SONGS_HEADER)
        for d in sorted((Path(cfg.work_dir) / "songs").glob("*")):
            if not (d / "annotation.json").exists():
                continue
            ann = SongAnnotation.load(d)
            if ann.status != "gold":
                continue
            lp = lyr_dir / f"{ann.song_id}.txt"
            lp.write_text("\n".join(ann.lines) + "\n", encoding="utf-8")
            w.writerow([ann.song_id, str(Path(ann.vocals_path).resolve()), ann.language,
                        str(lp.resolve()), ann.style, ann.singer_id, ann.gender, "1", "", "", ""])
            n += 1
    print(f"{n} gold songs -> {out}; run them with --work-dir <eval_work> then `evaluate`")


def cmd_evaluate(a):
    """``evaluate``: compare predicted with gold legacy annotations."""
    from .evaluate import evaluate
    res = evaluate(a.pred, a.gold, a.out)
    print(json.dumps(res["summary"], indent=1))


def cmd_export(a):
    """``export``: build the SoulX-Singer training set from legacy song annotations."""
    from .export import export_dataset
    cfg = _cfg(a)
    inc = ("auto", "reviewed", "gold") + (("flagged",) if a.include_flagged else ())
    hold = a.holdout_singers.split(",") if a.holdout_singers else ()
    res = export_dataset(cfg, inc, a.soulx_phone_set, a.val_frac, hold)
    print(json.dumps(res, indent=1, ensure_ascii=False))


def cmd_stats(a):
    """``stats``: sung hours by language and status, and gold progress (legacy)."""
    from .audio import FPS
    from .schema import SongAnnotation
    cfg = _cfg(a)
    by = defaultdict(float)
    cnt = Counter()
    for d in (Path(cfg.work_dir) / "songs").glob("*"):
        if not (d / "annotation.json").exists():
            continue
        ann = SongAnnotation.load(d)
        sung = sum(s.end - s.start for s in ann.syllables) / FPS / 3600
        by[(ann.language, ann.status)] += sung
        cnt[ann.status] += 1
    print("songs by status:", dict(cnt))
    print("sung hours by language/status:")
    for (lang, st), h in sorted(by.items()):
        print(f"  {lang:3s} {st:9s} {h:7.2f} h")
    gold = defaultdict(float)
    for (lang, st), h in by.items():
        if st == "gold":
            gold[lang] += h
    for lang in ("hi", "mr", "ta"):
        print(f"gold progress {lang}: {gold[lang]:.2f} h of {a.gold_target} h target")


def cmd_phoneset(a):
    """``phoneset``: list the Indic phones, or extend SoulX's phone_set.json."""
    from .phoneset import PHONES, embedding_init_map, extend_soulx_phoneset
    if a.soulx_phone_set:
        added = extend_soulx_phoneset(a.soulx_phone_set, a.out)
        print(f"added {len(added)} tokens, indices {min(added.values())}-{max(added.values())} -> {a.out}")
    else:
        for p in PHONES:
            print(f"in_{p.name:4s} {p.ipa:5s} {p.kind:9s} init={'+'.join(p.init)}")


def cmd_g2p(a):
    """``g2p``: print syllables and phones of one lyric line."""
    from .g2p import Lexicon, line_to_syllables
    cfg = _cfg(a)
    lex = Lexicon.load(*cfg.g2p.lexicon)
    for s in line_to_syllables(a.text, a.lang, lex):
        print(f"{s.text}\t{'-'.join(s.phones)}")


# ----------------------------------------------------------------------------- Saraga

def _saraga_recs(a):
    """Saraga recordings selected by the common options: only those with stems
    when the command sets ``stems_only``, filtered by ``--mbids`` (comma-separated
    ids, or a file with one id per line).
    """
    from .saraga import scan
    recs = [r for r in scan(a.saraga_root) if r.has_stems or not a.stems_only]
    if a.mbids:
        keep = set(Path(a.mbids).read_text().split()) if Path(a.mbids).exists() else set(a.mbids.split(","))
        recs = [r for r in recs if r.mbid in keep]
    return recs


def cmd_manifest(a):
    """``manifest``: index every Saraga Carnatic recording into manifest.csv."""
    from .saraga import scan, write_manifest
    recs = scan(a.saraga_root)
    out = Path(a.ann_root) / "manifest.csv"
    write_manifest(recs, out)
    print(f"{len(recs)} recordings ({sum(r.has_stems for r in recs)} with stems) -> {out}")


def cmd_sync(a):
    """``sync``: measure the stem offset, write mix-aligned FLAC stems, compute
    bleed scores and create / update the bundle's ``ir.json``.
    """
    from .ir import init_bundle
    from .stages.sync import sync_recording
    recs = _saraga_recs(a)
    for i, r in enumerate(recs, 1):
        b = init_bundle(r, a.ann_root)
        if b.stage_done("sync") and not a.force:
            continue
        res, paths, bleed = sync_recording(r.mix, r.stems, b.stems_dir)
        ir = b.load()
        ir["sync"] = res.to_json()
        ir["stems"] = {k: {"path": str(p.relative_to(b.dir)), "source": str(r.stems.get(k, "")),
                           "bleed": bleed.get(k, {})} for k, p in paths.items()}
        ir["flags"] = sorted(set(ir.get("flags", [])) | {f"sync:{f}" for f in res.flags})
        b.save(ir)
        b.mark("sync", offset_ms=round(res.offset_s * 1000, 3), flags=res.flags)
        print(f"[{i}/{len(recs)}] {r.title}: offset {res.offset_s * 1000:+.2f} ms "
              f"from {','.join(res.consensus_stems)} {' '.join(res.flags)}", flush=True)


def cmd_stems(a):
    """``stems``: de-bleed and de-reverb the vocal stem(s) into ``vocal_clean``
    (and ``vocal-s_clean`` for duos).
    """
    from .config import SeparationCfg
    from .ir import Bundle
    from .stages.separate import Separator, clean_stem
    sep = Separator(SeparationCfg(device=a.device))
    recs = _saraga_recs(a)
    for i, r in enumerate(recs, 1):
        b = Bundle(a.ann_root, r.mbid)
        if not b.stage_done("sync"):
            print(f"[{i}/{len(recs)}] {r.title}: run sync first", flush=True)
            continue
        if b.stage_done("stems") and not a.force:
            continue
        ir = b.load()
        stats = {}
        for k in ("vocal", "vocal-s"):
            if k not in ir["stems"]:
                continue
            name = "vocal_clean" if k == "vocal" else "vocal-s_clean"
            out = b.stems_dir / f"{name}.flac"
            stats[name] = clean_stem(sep, b.dir / ir["stems"][k]["path"], out)
            ir["stems"][name] = {"path": str(out.relative_to(b.dir)), "from": k, **stats[name]}
        b.save(ir)
        b.mark("stems", **stats)
        print(f"[{i}/{len(recs)}] {r.title}: {stats}", flush=True)


def cmd_f0(a):
    """``f0``: 100 Hz F0 ensemble, tonic choice and Saraga comparison; with
    ``--qc-only`` just re-score the stored F0 against Saraga's pitch tracks.
    """
    from .config import F0Cfg, SeparationCfg
    from .ir import F0_HZ, Bundle
    from .stages.f0 import F0Estimator
    from .stages.f0_ir import run_f0, saraga_qc
    recs = _saraga_recs(a)
    if a.qc_only:
        import numpy as np
        for i, r in enumerate(recs, 1):
            b = Bundle(a.ann_root, r.mbid)
            if not b.stage_done("f0"):
                continue
            ir = b.load()
            q = saraga_qc(np.load(b.path("f0_100hz.npz"))["f0_hz"], r.pitch_refs())
            ir.setdefault("qc", {}).setdefault("f0", {}).setdefault("vocal_clean", {})["vs_saraga"] = q
            b.save(ir)
            print(f"[{i}/{len(recs)}] {r.title}: {json.dumps(q)}", flush=True)
        return
    cfg = F0Cfg(backends=a.backends.split(","), device=a.device)
    est = F0Estimator(cfg, SeparationCfg().soulx_repo, fps=F0_HZ)
    for i, r in enumerate(recs, 1):
        b = Bundle(a.ann_root, r.mbid)
        if not b.stage_done("stems"):
            print(f"[{i}/{len(recs)}] {r.title}: run stems first", flush=True)
            continue
        if b.stage_done("f0") and not a.force:
            continue
        rep = run_f0(b, est, r.pitch_refs())
        ir = b.mark("f0", backends=cfg.backends)
        print(f"[{i}/{len(recs)}] {r.title}: tonic {ir['tonic_hz']} ({ir['tonic_source']}, "
              f"histogram {ir.get('tonic_histogram_hz')}) {json.dumps(rep)}", flush=True)


def cmd_swaras(a):
    """``swaras``: raga-aware swara segments (``swaras.json``) and the flat
    contour (``flat_100hz.npz``); needs ``f0``.
    """
    import numpy as np
    from .ir import Bundle
    from .stages.swaras import flatten, segment, to_json
    recs = _saraga_recs(a)
    for i, r in enumerate(recs, 1):
        b = Bundle(a.ann_root, r.mbid)
        if not b.stage_done("f0"):
            print(f"[{i}/{len(recs)}] {r.title}: run f0 first", flush=True)
            continue
        if b.stage_done("swaras") and not a.force:
            continue
        ir = b.load()
        z = np.load(b.path("f0_100hz.npz"))
        segs, pos = segment(z["cents"].astype(float), ir.get("raaga", ""), float(z["fps"]))
        sw = to_json(segs, pos, ir.get("raaga", ""))
        b.path("swaras.json").write_text(json.dumps(sw, ensure_ascii=False), encoding="utf-8")
        np.savez_compressed(b.path("flat_100hz.npz"), fps=z["fps"],
                            flat_cents=flatten(segs, len(z["cents"]), pos, float(z["fps"])))
        ir = b.load()
        ir.setdefault("qc", {})["swaras"] = sw["summary"]
        b.save(ir)
        b.mark("swaras", n_segments=len(segs))
        print(f"[{i}/{len(recs)}] {r.title}: {json.dumps(sw['summary']['share'])} "
              f"positions {json.dumps(pos)}", flush=True)


def cmd_viz(a):
    """``viz``: contour sheets ``review/contour.pdf`` (with lyric syllables when
    ``lyrics.json`` exists); needs ``swaras``.
    """
    from .ir import Bundle
    from .viz import contour_pdf
    recs = _saraga_recs(a)
    for i, r in enumerate(recs, 1):
        b = Bundle(a.ann_root, r.mbid)
        if not b.stage_done("swaras"):
            continue
        ir = b.load()
        sw = json.loads(b.path("swaras.json").read_text(encoding="utf-8"))
        syl = []
        if b.path("lyrics.json").exists():
            lyr = json.loads(b.path("lyrics.json").read_text(encoding="utf-8"))
            syl = [{"text": sy["text"], "start": sy["start"], "sargam": inst["section"] == "sargam"}
                   for inst in lyr.get("instances", []) for w in inst["words"] for sy in w["syllables"]]
        out = contour_pdf(b.review_dir / "contour.pdf", ir["title"], ir["tonic_hz"], ir.get("raaga", ""),
                          b.path("f0_100hz.npz"), sw, ir.get("sections", []), ir.get("sama", []), syl,
                          max_pages=a.max_pages or None)
        print(f"[{i}/{len(recs)}] {r.title}: {out}", flush=True)


def cmd_lyrics_fetch(a):
    """``lyrics-fetch``: candidate lyrics per composition (MusicBrainz +
    karnatik.com) into ``<lyrics-dir>/<work>.yaml`` and ``status.tsv``.
    """
    from .lyrics_fetch import Fetcher, fetch_work, karnatik_index, shivkumar_index, work_status, write_status
    out_dir = Path(a.lyrics_dir)
    f = Fetcher(out_dir / "_cache")
    index = karnatik_index(f)
    second = shivkumar_index(f) if a.second_source else None
    works = {}
    for r in _saraga_recs(a):
        if r.work_mbid and r.work_mbid not in works:
            works[r.work_mbid] = r
    errors = 0
    for i, (wm, r) in enumerate(works.items(), 1):
        try:
            doc = fetch_work(f, index, wm, r.work_title or r.title, r.raaga, out_dir, second_index=second)
        except Exception as e:
            errors += 1
            print(f"[{i}/{len(works)}] error    {r.work_title or r.title}: {e!r}", flush=True)
            continue
        best = doc["candidates"][0] if doc["candidates"] else {}
        print(f"[{i}/{len(works)}] {work_status(doc):8s} {r.work_title or r.title} ({r.raaga}) -> "
              f"{best.get('title', '-')} / {best.get('raga', '-')} {best.get('score', '')}", flush=True)
    counts = write_status(out_dir)                   # the table always covers every work YAML
    print(json.dumps({**counts, "fetch_errors": errors}))


def cmd_lyrics_status(a):
    """``lyrics-status``: rebuild ``status.tsv`` from the work YAMLs (offline)."""
    from .lyrics_fetch import write_status
    print(json.dumps(write_status(Path(a.lyrics_dir))))


def cmd_lyrics_verify(a):
    """Show the candidates of one work and record the reviewer's choice."""
    import yaml
    path = Path(a.lyrics_dir) / f"{a.work}.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if a.pick is None and not a.text:
        print(f"{doc['title']}  |  {doc['raga']}  |  composer {doc['musicbrainz'].get('composer')}")
        for i, c in enumerate(doc["candidates"]):
            print(f"\n[{i}] {c['title']} / {c['raga']} / {c['composer']}  score {c['score']}  {c['url']}")
            for sec in c.get("parsed", {}).get("sections", []):
                print(f"   {sec['name']}: " + " | ".join(sec["lines"]))
        print(f"\nselected: {doc.get('selected')}  verified: {doc.get('verified')}")
        return
    from .lyrics_fetch import apply_decision
    if a.text:
        apply_decision(doc, "text", a.reviewer, text=Path(a.text).read_text(encoding="utf-8"),
                       source=a.source or str(a.text), notes=a.notes)
    else:
        apply_decision(doc, "candidate", a.reviewer, candidate_url=doc["candidates"][a.pick]["url"], notes=a.notes)
    path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
    from .lyrics_fetch import write_status
    write_status(path.parent)                         # keep status.tsv in step with the YAMLs
    print(f"verified {doc['title']} <- {doc['selected']}")


# (cleaned vocal stem, file suffix, singer) -- duets have a second singer's stem
VOICES = (("vocal_clean", "", "lead"), ("vocal-s_clean", "_vocal-s", "second"))


def _mms_emissions(b, ir: dict, stem: str, suffix: str, em=None):
    """MMS_FA log-probabilities (50 fps) of one cleaned vocal stem, from the
    ``mms_50hz<suffix>.npz`` cache, computed with emitter ``em`` and cached if
    absent."""
    import numpy as np
    import soundfile as sf
    from .audio import resample
    path = b.path(f"mms_50hz{suffix}.npz")
    if path.exists():
        return np.load(path)["logp"].astype(np.float32)
    x, sr = sf.read(str(b.dir / ir["stems"][stem]["path"]), dtype="float32")
    logp = em(resample(x, sr, 16000))
    np.savez_compressed(path, logp=logp.astype(np.float16))
    return logp


def cmd_emissions(a):
    """``emissions``: cache MMS_FA log-probabilities of every cleaned vocal stem
    (``mms_50hz[_vocal-s].npz``).  ``align`` and ``sections`` read the cache, so
    this GPU step can run ahead of lyrics and sections."""
    from .ir import Bundle
    from .stages.align_kriti import AlignCfgK, MmsEmitter
    em = None
    recs = _saraga_recs(a)
    for i, r in enumerate(recs, 1):
        b = Bundle(a.ann_root, r.mbid)
        if not b.stage_done("stems"):
            continue
        ir = b.load()
        done = []
        for stem, suffix, _ in VOICES:
            if stem not in ir["stems"] or b.path(f"mms_50hz{suffix}.npz").exists():
                continue
            em = em or MmsEmitter(AlignCfgK(device=a.device))
            _mms_emissions(b, ir, stem, suffix, em)
            done.append(stem)
        if done:
            print(f"[{i}/{len(recs)}] {r.title}: {', '.join(done)}", flush=True)


def cmd_lyrics_review_pack(a):
    """``lyrics-review-pack``: what the lyrics review page shows.  Writes
    ``<out>/works.json`` (per composition: status, candidates with their parsed
    sections, the recordings that use it, the excerpt file) and
    ``<out>/excerpts/<work_mbid>.mp3`` (and ``.mp4``, AAC, when ffmpeg is
    available: the format the review page's asset store accepts): 20 s of a
    recording's mix, starting at its first aligned lyric line, else at its
    pallavi, else a quarter in."""
    import shutil
    import subprocess
    import numpy as np
    import soundfile as sf
    import yaml
    from .audio import load_audio
    from .ir import Bundle
    from .lyrics_fetch import work_status
    out = Path(a.out)
    (out / "excerpts").mkdir(parents=True, exist_ok=True)
    recs_by_work = {}
    for r in _saraga_recs(a):
        if r.work_mbid:
            recs_by_work.setdefault(r.work_mbid, []).append(r)
    works = []
    for i, path in enumerate(sorted(Path(a.lyrics_dir).glob("*.yaml")), 1):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        wm = doc["work_mbid"]
        recs = recs_by_work.get(wm, [])
        cands = [{"site": c.get("site", "karnatik"), "url": c["url"], "title": c.get("title", ""),
                  "raga": c.get("raga", ""), "composer": c.get("composer", ""), "score": c.get("score"),
                  "sections": (c.get("parsed") or {}).get("sections", [])}
                 for c in doc.get("candidates", [])[:4] if (c.get("parsed") or {}).get("sections")]
        excerpt, start = None, None
        for r in recs:                              # prefer a recording with an aligned lyric line
            b = Bundle(a.ann_root, r.mbid)
            if b.path("lyrics.json").exists():
                inst = [x for x in json.loads(b.path("lyrics.json").read_text(encoding="utf-8"))["instances"]
                        if x["section"] != "sargam"]
                if inst:
                    excerpt, start = r, max(0.0, min(x["start"] for x in inst) - 1.0)
                    break
        if excerpt is None and recs:
            r = recs[0]
            pal = [s_ for s_ in r.sections() if s_["label"].lower().startswith("pallavi")]
            excerpt, start = r, (pal[0]["start"] if pal else r.length_s / 4)
        clip = None
        if excerpt is not None:
            sr = 22050
            x = load_audio(excerpt.mix, sr)
            seg = x[int(start * sr):int((start + a.seconds) * sr)]
            if len(seg):
                clip = f"{wm}.mp3"
                fade = np.minimum(1.0, np.minimum(np.arange(len(seg)), np.arange(len(seg))[::-1]) / (0.5 * sr))
                sf.write(str(out / "excerpts" / clip), (0.9 * seg * fade / (np.abs(seg).max() + 1e-9)).astype(np.float32),
                         sr, format="MP3", subtype="MPEG_LAYER_III")
                if shutil.which("ffmpeg"):          # web pages / artifact assets take audio-only MP4 (AAC)
                    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(out / "excerpts" / clip), "-c:a", "aac",
                                    "-b:a", "64k", "-ac", "1", "-movflags", "+faststart",
                                    str(out / "excerpts" / f"{wm}.mp4")], check=True)
        mb = doc.get("musicbrainz") or {}
        works.append({"work_mbid": wm, "title": doc.get("title", ""), "raga": doc.get("raga", ""),
                      "composer": "; ".join(mb.get("composer") or []), "languages": mb.get("languages") or [],
                      "status": work_status(doc), "selected": doc.get("selected"), "site": doc.get("site"),
                      "candidates": cands, "excerpt": clip,
                      "excerpt_from": {"title": excerpt.title, "start_s": round(start, 1)} if clip else None,
                      "recordings": [{"mbid": r.mbid, "title": r.title, "concert": r.concert,
                                      "minutes": round(r.length_s / 60, 1)} for r in recs]})
        print(f"[{i}] {doc.get('title')}: {len(cands)} candidates, excerpt {'yes' if clip else 'no'}", flush=True)
    (out / "works.json").write_text(json.dumps(works, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(works)} works -> {out}", flush=True)


def cmd_lyrics_apply(a):
    """``lyrics-apply``: apply review decisions exported from the lyrics review
    page (a JSON list of ``{work_mbid, decision, candidate_url, text, source,
    notes, reviewer}``) to the work YAMLs, then rebuild ``status.tsv``.
    Re-run ``align --force`` for the recordings of the changed works."""
    import yaml
    from .lyrics_fetch import apply_decision, write_status
    rows = json.loads(Path(a.reviews).read_text(encoding="utf-8"))
    changed = []
    for row in rows:
        path = Path(a.lyrics_dir) / f"{row['work_mbid']}.yaml"
        if not path.exists():
            print(f"skip {row['work_mbid']}: no lyrics YAML", flush=True)
            continue
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        try:
            apply_decision(doc, row["decision"], row.get("reviewer", ""), row.get("candidate_url", ""),
                           row.get("text", ""), row.get("source", ""), row.get("notes", ""))
        except ValueError as e:
            print(f"skip {doc.get('title')}: {e}", flush=True)
            continue
        path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
        changed.append(row["work_mbid"])
        print(f"{doc.get('title')}: {row['decision']}", flush=True)
    print(json.dumps({**write_status(Path(a.lyrics_dir)), "applied": len(changed)}), flush=True)


def cmd_align(a):
    """``align``: time-align the selected lyrics of each recording.

    Every cleaned vocal stem is aligned on its own (``VOICES``: the lead, and the
    second singer of a duet).  Per voice: MMS_FA log-probabilities are cached
    in ``mms_50hz[_vocal-s].npz``; the spike (default) or trellis engine aligns
    each Saraga section; syllables are extended through that voice's voicing;
    in duets each instance gets the share of time the other singer sings along
    (``overlap``).  Writes ``lyrics[_vocal-s].json`` and
    ``ir.json['qc']['align' | 'align_vocal-s']``.  Reviewed recordings are never
    re-aligned.
    """
    import numpy as np
    import torchaudio
    import yaml
    from .ir import Bundle
    from .stages.align_kriti import (AlignCfgK, MmsEmitter, align_regions, extend_sustain, lyric_lines,
                                     mark_overlap, to_json)
    cfg = AlignCfgK(device=a.device, filler_cost=a.filler_cost)
    vocab = dict(torchaudio.pipelines.MMS_FA.get_dict(star=None))
    em = None
    recs = _saraga_recs(a)
    for i, r in enumerate(recs, 1):
        b = Bundle(a.ann_root, r.mbid)
        if not b.stage_done("stems"):
            print(f"[{i}/{len(recs)}] {r.title}: run stems first", flush=True)
            continue
        if b.stage_done("align") and not a.force:
            continue
        if b.load().get("status") == "reviewed":
            print(f"[{i}/{len(recs)}] {r.title}: reviewed, not re-aligned", flush=True)
            continue
        lpath = Path(a.lyrics_dir) / f"{r.work_mbid}.yaml"
        doc = yaml.safe_load(lpath.read_text(encoding="utf-8")) if lpath.exists() else None
        if not doc or not doc.get("lyrics"):
            print(f"[{i}/{len(recs)}] {r.title}: no lyrics selected ({lpath.name})", flush=True)
            continue
        lines = lyric_lines(doc["lyrics"])
        src = {"work_mbid": r.work_mbid, "selected": doc.get("selected"), "source": doc.get("source"),
               "verified": bool(doc.get("verified"))}
        ir = b.load()
        voices = [v for v in VOICES if v[0] in ir["stems"]]
        voiced = {v[1]: np.load(b.path(f"f0_100hz{v[1]}.npz"))["f0_hz"] > 0
                  for v in voices if b.path(f"f0_100hz{v[1]}.npz").exists()}
        summaries = {}
        for stem, suffix, singer in voices:
            em = em or (None if b.path(f"mms_50hz{suffix}.npz").exists() else MmsEmitter(cfg))
            logp = _mms_emissions(b, ir, stem, suffix, em)
            regions = (ir.get("sections") or ir.get("sections_auto")
                       or [{"start": 0.0, "end": ir["length_s"], "label": "unknown"}])
            if a.engine == "spikes":
                from .stages.align_spikes import SpikeCfg, align_regions_spikes
                inst, rep = align_regions_spikes(logp, lines, regions, vocab, vocab["-"], SpikeCfg())
            else:
                inst, rep = align_regions(logp, lines, regions, vocab, vocab["-"], cfg)
            if suffix in voiced:
                extend_sustain(inst, voiced[suffix])
            for x in inst:
                x.singer = singer
            others = [v for k, v in voiced.items() if k != suffix]
            if others:                                  # duet: how much the other voice sings along
                mark_overlap(inst, others[0])
            b.path(f"lyrics{suffix}.json").write_text(
                json.dumps(to_json(lines, inst, rep, src), ensure_ascii=False), encoding="utf-8")
            summaries[singer] = {
                "instances": len(inst), "lyric_s": round(sum(x.end - x.start for x in inst), 1),
                "low_conf": sum("low_conf" in x.flags for x in inst),
                "sargam": sum(x.section == "sargam" for x in inst),
                "mean_llr": round(float(np.mean([x.llr for x in inst])), 3) if inst else None,
                "lines_seen": sorted({x.line_id for x in inst}), "n_lines": len(lines),
                "lyrics_verified": src["verified"]}
            if others:
                summaries[singer]["solo_s"] = round(sum((x.end - x.start) * (1 - x.overlap) for x in inst), 1)
        ir = b.load()
        ir.setdefault("qc", {})["align"] = summaries["lead"]
        if "second" in summaries:
            ir["qc"]["align_vocal-s"] = summaries["second"]
        if not src["verified"]:
            ir["flags"] = sorted(set(ir.get("flags", [])) | {"lyrics_unverified"})
        b.save(ir)
        summ = summaries["lead"]
        b.mark("align", engine=a.engine, **summ, **({"second": summaries["second"]} if "second" in summaries else {}))
        print(f"[{i}/{len(recs)}] {r.title}: {json.dumps(summaries)}", flush=True)


def _section_features(b):
    """Per-second section features of a bundle, cached in ``sections_feat.npz``
    (recomputed if missing or built from an older feature set)."""
    import numpy as np
    from .stages.sections import FEATURE_NAMES, features
    path = b.path("sections_feat.npz")
    if path.exists():
        z = np.load(path)
        if list(z["names"]) == list(FEATURE_NAMES):
            return z["x"]
    x = features(b)
    np.savez_compressed(path, x=x, names=np.array(FEATURE_NAMES))
    return x


def _section_ready(b) -> bool:
    """True when a bundle has every input the section features need."""
    return b.stage_done("f0") and b.stage_done("swaras") and b.path("mms_50hz.npz").exists()


def cmd_sections_train(a):
    """``sections-train``: leave-one-concert-out evaluation of the section
    classifier (rules baseline, model, model + smoothing) on recordings with
    manual sections, then a final model trained on all of them, saved to
    ``--model``."""
    import pickle
    from collections import defaultdict
    import numpy as np
    from .ir import Bundle
    from .stages.sections import (CLASSES, frame_accuracy, frame_labels, proba, rules, train,
                                  viterbi_smooth)
    data = []
    for r in _saraga_recs(a):
        b = Bundle(a.ann_root, r.mbid)
        if not r.sections() or not _section_ready(b):
            continue
        x = _section_features(b)
        data.append((r.concert, r.title, x, frame_labels(r.sections(), len(x))))
    concerts = sorted({d[0] for d in data})
    print(f"{len(data)} labelled recordings from {len(concerts)} concerts", flush=True)
    tot = defaultdict(lambda: [0, 0])
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for c in concerts:
        clf = train([d[2] for d in data if d[0] != c], [d[3] for d in data if d[0] != c])
        for _, title, x, y in (d for d in data if d[0] == c):
            m = y >= 0
            preds = {"rules": rules(x), "model": proba(clf, x).argmax(1),
                     "model+smooth": viterbi_smooth(proba(clf, x), a.switch_cost)}
            for k, pr in preds.items():
                tot[k][0] += int((pr[m] == y[m]).sum())
                tot[k][1] += int(m.sum())
            np.add.at(confusion, (y[m], preds["model+smooth"][m]), 1)
            acc, rec = frame_accuracy(preds["model+smooth"], y)
            print(f"  [{c[:24]}] {title[:28]:28s} acc {acc:.2f} {rec}", flush=True)
    for k, (h, n) in tot.items():
        print(f"leave-one-concert-out frame accuracy, {k}: {h / max(n, 1):.3f} ({n} labelled s)", flush=True)
    print("confusion (rows = truth, cols = predicted, model+smooth):", list(CLASSES), flush=True)
    for k, row in enumerate(confusion):
        print(f"  {CLASSES[k]:15s} {row.tolist()}", flush=True)
    final = train([d[2] for d in data], [d[3] for d in data])
    Path(a.model).parent.mkdir(parents=True, exist_ok=True)
    with open(a.model, "wb") as fh:
        pickle.dump({"model": final, "switch_cost": a.switch_cost, "trained_on": [d[1] for d in data]}, fh)
    print(f"model -> {a.model}", flush=True)


def cmd_sections(a):
    """``sections``: automatic sections (``ir['sections_auto']``) for every
    recording with the needed inputs, using the model from ``sections-train``.
    Manual Saraga sections are kept as they are and still take precedence."""
    import pickle
    from .ir import Bundle
    from .stages.sections import proba, to_regions, viterbi_smooth
    with open(a.model, "rb") as fh:
        m = pickle.load(fh)
    recs = _saraga_recs(a)
    for i, r in enumerate(recs, 1):
        b = Bundle(a.ann_root, r.mbid)
        if not _section_ready(b) or (b.stage_done("sections") and not a.force):
            continue
        x = _section_features(b)
        ir = b.load()
        regions = to_regions(viterbi_smooth(proba(m["model"], x), m["switch_cost"]), a.min_region_s,
                             ir.get("length_s"))
        ir["sections_auto"] = regions
        b.save(ir)
        b.mark("sections", n=len(regions), manual=bool(ir.get("sections")))
        print(f"[{i}/{len(recs)}] {r.title}: " + " | ".join(f"{x['label']} {x['start']:.0f}-{x['end']:.0f}"
                                                            for x in regions), flush=True)


def _percussion(b, ir, sr: int = 16000):
    """(all percussion, bass head) mono audio of a bundle at ``sr``: mridangam
    (+ ghatam when present), and the mridangam's left (bass, thoppi) head."""
    from .audio import load_audio
    st = ir["stems"]
    perc = load_audio(b.dir / st["mridangam"]["path"], sr)
    if "ghatam" in st:
        g = load_audio(b.dir / st["ghatam"]["path"], sr)
        perc = perc[:len(g)] + g[:len(perc)]
    bass = load_audio(b.dir / st["mridangam-left"]["path"], sr) if "mridangam-left" in st else perc
    return perc, bass, sr


def cmd_tala(a):
    """``tala``: beat grid, laya and sama per recording (``ir['tala']``).

    Annotated samas give the grid directly (beats subdivided evenly).  Without
    them only the tala shape and Saraga's laya are stored, unless ``--estimate``
    asks for the percussion-based estimate (marked ``reliable: false``; on the
    63 annotated recordings it reaches a median beat F of only 0.29).  With ``--evaluate``, the
    estimator also runs on recordings *with* annotated samas and is scored
    against them (beat F-measure and sama hit rate within 70 ms, beat period
    error); nothing is written in that mode."""
    import numpy as np
    from .ir import Bundle
    from .stages.tala import (beat_f_measure, estimate_grid, grid_from_samas, load_tempo, sama_hit_rate,
                              tala_shape)
    rows = []
    recs = _saraga_recs(a)
    for i, r in enumerate(recs, 1):
        b = Bundle(a.ann_root, r.mbid)
        if not b.stage_done("sync"):
            continue
        tempo = load_tempo(r.ann["tempo-manual"]) if "tempo-manual" in r.ann else None
        shape = tala_shape(r.taala, tempo)
        samas = r.sama()
        if a.evaluate:
            if shape is None or len(samas) < 4:
                continue
            ref_beats, _ = grid_from_samas(samas, shape[0])
            ir = b.load()
            perc, bass, sr = _percussion(b, ir)
            beats, index, period, conf = estimate_grid(perc, bass, sr, shape[0])
            span = (beats >= samas[0] - 1) & (beats <= samas[-1] + 1)     # where the reference exists
            ref_period = float(np.median(np.diff(samas))) / shape[0]
            row = {"title": r.title[:24], "beats_per_cycle": shape[0],
                   "period_err": round(period / ref_period - 1, 3),
                   "beat_f": round(beat_f_measure(beats[span], ref_beats), 3),
                   "sama_hit": round(sama_hit_rate(beats[span][index[span] == 0], samas), 3),
                   "phase_conf": round(conf, 3)}
            rows.append(row)
            print(f"[{i}/{len(recs)}] {json.dumps(row, ensure_ascii=False)}", flush=True)
            continue
        if b.stage_done("tala") and not a.force:
            continue
        ir = b.load()
        if shape is None:
            ir["tala"] = {"source": "unknown", "note": f"no tala shape for {r.taala!r}"}
            b.save(ir)
            b.mark("tala", source="unknown")
            continue
        if len(samas) >= 2:
            beats, index = grid_from_samas(samas, shape[0])
            source, conf = "manual_sama", 1.0
            period = float(np.median(np.diff(samas))) / shape[0]
        elif a.estimate:
            perc, bass, sr = _percussion(b, ir)
            beats, index, period, conf = estimate_grid(perc, bass, sr, shape[0])
            source = "estimated"
        else:
            # the estimator is not reliable enough to label matra (see README), so
            # without annotated samas only the shape and Saraga's laya are kept
            ir["tala"] = {"source": "none", "beats_per_cycle": shape[0], "aksharas_per_beat": shape[1],
                          "laya_bpm": tempo[1] if tempo else None}
            b.save(ir)
            b.mark("tala", source="none")
            continue
        ir["tala"] = {"source": source, "reliable": source == "manual_sama", "phase_conf": round(conf, 3),
                      "beats_per_cycle": shape[0],
                      "aksharas_per_beat": shape[1], "laya_bpm": round(60.0 / period, 2),
                      "beats": np.round(beats, 3).tolist(), "beat_index": index.tolist(),
                      "samas": np.round(beats[index == 0], 3).tolist()}
        b.save(ir)
        b.mark("tala", source=source, laya_bpm=ir["tala"]["laya_bpm"], phase_conf=round(conf, 3))
        print(f"[{i}/{len(recs)}] {r.title}: {source} {ir['tala']['laya_bpm']} bpm, "
              f"{shape[0]} beats/cycle, phase_conf {conf:.2f}", flush=True)
    if a.evaluate and rows:
        for k in ("period_err", "beat_f", "sama_hit"):
            v = np.array([x[k] for x in rows])
            print(f"{k}: median {np.median(v):.3f}, mean {np.mean(v):.3f}"
                  + (f", |err| < 2%: {np.mean(np.abs(v) < 0.02):.2f}" if k == "period_err" else ""), flush=True)


def cmd_export_ir(a):
    """``export-ir``: the ICM-IR v0 dataset (``saregama/export_ir.py``) from every
    bundle that has F0 and swaras, into ``--out``."""
    import numpy as np
    import pandas as pd
    import yaml
    from .export_ir import (IR_SPEC, PHONES, assign_splits, frames, singer_id, swara_rows, syllable_rows)
    from .ir import Bundle
    out = Path(a.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    recs = [r for r in _saraga_recs(a) if Bundle(a.ann_root, r.mbid).stage_done("swaras")]
    split = assign_splits(sorted({r.concert for r in recs}), test=[c for c in a.test_concerts.split(";") if c],
                          val=[c for c in a.val_concerts.split(";") if c])
    rec_rows, syl_rows, sw_rows = [], [], []
    for i, r in enumerate(recs, 1):
        b = Bundle(a.ann_root, r.mbid)
        ir = b.load()
        sections = ir.get("sections") or ir.get("sections_auto") or []
        tala = ir.get("tala") or {}
        lpath = Path(a.lyrics_dir) / f"{r.work_mbid}.yaml"
        ldoc = yaml.safe_load(lpath.read_text(encoding="utf-8")) if lpath.exists() else {}
        verified = bool(ldoc.get("verified"))
        by_singer = {}
        for suffix, singer in (("", "lead"), ("_vocal-s", "second")):
            if b.path(f"lyrics{suffix}.json").exists():
                lyr = json.loads(b.path(f"lyrics{suffix}.json").read_text(encoding="utf-8"))
                by_singer[singer] = syllable_rows(r.mbid, lyr, singer, tala, sections, verified)
                syl_rows += by_singer[singer]
        sw = json.loads(b.path("swaras.json").read_text(encoding="utf-8"))
        sw_rows += swara_rows(r.mbid, sw, ir.get("raaga", ""), tala, sections)
        np.savez_compressed(out / "frames" / f"{r.mbid}.npz", **frames(b, by_singer))
        leads = [singer_id(x) for x in ir.get("lead", [])]
        rec_rows.append({
            "mbid": r.mbid, "title": ir["title"], "concert": r.concert, "split": split[r.concert],
            "singer_ids": ";".join(leads), "duet": "vocal-s_clean" in ir["stems"], "raga": ir.get("raaga", ""),
            "tala": ir.get("taala", ""), "form": ir.get("form", ""), "tonic_hz": ir["tonic_hz"],
            "tonic_source": ir.get("tonic_source", ""), "laya_bpm": tala.get("laya_bpm"),
            "beats_per_cycle": tala.get("beats_per_cycle"), "aksharas_per_beat": tala.get("aksharas_per_beat"),
            "tala_source": tala.get("source", "none"), "sections_source": "manual" if ir.get("sections") else
            ("auto" if ir.get("sections_auto") else "none"), "work_mbid": r.work_mbid,
            "language": " ".join((ldoc.get("musicbrainz") or {}).get("languages") or []),
            "lyrics": "verified" if verified else ("auto" if by_singer else "none"),
            "length_s": ir["length_s"], "status": ir.get("status", "auto"), "flags": ";".join(ir.get("flags", []))})
        print(f"[{i}/{len(recs)}] {r.title}: {sum(len(v) for v in by_singer.values())} syllables", flush=True)
    pd.DataFrame(rec_rows).to_parquet(out / "recordings.parquet", index=False)
    pd.DataFrame(syl_rows).to_parquet(out / "syllables.parquet", index=False)
    pd.DataFrame(sw_rows).to_parquet(out / "swaras.parquet", index=False)
    (out / "phones.json").write_text(json.dumps([p.name for p in PHONES], indent=0), encoding="utf-8")
    (out / "splits.json").write_text(json.dumps(split, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "DATASHEET.md").write_text(_datasheet(IR_SPEC, rec_rows, syl_rows, sw_rows), encoding="utf-8")
    print(f"{len(rec_rows)} recordings, {len(syl_rows)} syllables, {len(sw_rows)} swara segments -> {out}")


def _datasheet(spec: str, recs: list, syls: list, sws: list) -> str:
    """Text of the export's ``DATASHEET.md``: provenance, licence, contents, QC."""
    from collections import Counter
    hours = lambda rows: round(sum(r["length_s"] for r in rows) / 3600, 2)
    by_split = {s: [r for r in recs if r["split"] == s] for s in ("train", "val", "test")}
    lyr = Counter(r["lyrics"] for r in recs)
    return f"""# Saraga Carnatic, {spec}

Built by saregama624 (https://github.com/ramanan849/saregama624) from Saraga 1.5 Carnatic
multitrack recordings. Source data: Saraga, CC BY-NC 4.0 (non-commercial use, attribution
required). Lyrics come from karnatik.com and are for internal annotation only; do not
redistribute them.

| | Recordings | Hours |
|---|---|---|
| all | {len(recs)} | {hours(recs)} |
""" + "".join(f"| {s} | {len(v)} | {hours(v)} |\n" for s, v in by_split.items()) + f"""
Lyrics: {dict(lyr)}. Syllables: {len(syls)}. Swara segments: {len(sws)}.

All times are seconds on the mix clock; frame k is at k / 100 s. Pitch is in cents re the
tonic (10-cent steps in `frames/*.npz`). Splits are by concert (`splits.json`). Read `flags`
and `lyrics` in `recordings.parquet` before training: `lyrics_unverified` alignments were
made against lyrics no person has confirmed, and `low_conf` syllables matched few letters.
See the saregama624 README for the methods and their measured accuracy.
"""


def cmd_export_soulx(a):
    """``export-soulx``: SoulX-Singer training clips (``saregama/export_soulx.py``)
    from every bundle with swaras, for each singer track; splits follow the
    ICM-IR export's ``splits.json`` when it exists (same concerts in each split)."""
    from collections import Counter, defaultdict
    import numpy as np
    import yaml
    from .audio import load_audio, save_audio
    from .export import DATA_PROCESSOR_PATCH
    from .export_ir import assign_splits, singer_id, syllable_rows
    from .export_soulx import (LANGUAGE_NAMES, ClipCfgS, akaram_spans, clip_item, clip_notes, plan_clips,
                               track_notes)
    from .ir import Bundle
    from .paths import SOULX_REPO
    from .phoneset import embedding_init_map, extend_soulx_phoneset
    from .stages.swaras import segment
    cfg = ClipCfgS(min_s=a.min_s, max_s=a.max_s)
    out = Path(a.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    recs = [r for r in _saraga_recs(a) if Bundle(a.ann_root, r.mbid).stage_done("swaras")]
    splits_path = Path(a.splits)
    split = (json.loads(splits_path.read_text(encoding="utf-8")) if splits_path.exists()
             else assign_splits(sorted({r.concert for r in recs})))
    items, stats = defaultdict(list), Counter()
    for i, r in enumerate(recs, 1):
        b = Bundle(a.ann_root, r.mbid)
        ir = b.load()
        sections = ir.get("sections") or ir.get("sections_auto") or []
        lpath = Path(a.lyrics_dir) / f"{r.work_mbid}.yaml"
        ldoc = yaml.safe_load(lpath.read_text(encoding="utf-8")) if lpath.exists() else {}
        langs = (ldoc.get("musicbrainz") or {}).get("languages") or []
        leads = ir.get("lead", [])
        n_clips = 0
        for stem, suffix, singer in VOICES:
            if stem not in ir["stems"] or not b.path(f"f0_100hz{suffix}.npz").exists():
                continue
            z = np.load(b.path(f"f0_100hz{suffix}.npz"))
            f0, cents = z["f0_hz"].astype(np.float64), z["cents"].astype(np.float64)
            rows = []
            if b.path(f"lyrics{suffix}.json").exists():
                lyr = json.loads(b.path(f"lyrics{suffix}.json").read_text(encoding="utf-8"))
                rows = [x for x in syllable_rows(r.mbid, lyr, singer, {}, sections, bool(ldoc.get("verified")))
                        if (a.keep_low_conf or not x["low_conf"]) and x["overlap"] <= cfg.max_overlap]
            if suffix:                                 # the second singer's own swara segments
                segs = [dict(start=s.start, end=s.end, kind=s.kind, swara=s.swara)
                        for s in segment(cents, ir.get("raaga", ""))[0]]
            else:
                segs = json.loads(b.path("swaras.json").read_text(encoding="utf-8"))["segments"]
            extra = akaram_spans(f0 > 0, sections, rows, cfg.akaram_min_s) if a.akaram else []
            n50 = len(f0) // 2
            notes = track_notes(rows, extra, segs, f0, cents, n50)
            spans = plan_clips(notes, f0[::2][:n50] > 0, cfg)
            if not spans:
                continue
            audio = load_audio(b.dir / ir["stems"][stem]["path"], 24000)
            name = leads[1 if suffix and len(leads) > 1 else 0] if leads else singer
            for k, (x, y) in enumerate(spans):
                cid = f"{r.mbid[:8]}{suffix}_{k:04d}"
                cn = clip_notes(notes, x, y)
                if not any(n.kind != 1 for n in cn):
                    continue
                save_audio(out / "wavs" / f"{cid}.wav", audio[x * 480:y * 480])
                sp = split.get(r.concert, "train")
                items[sp].append(clip_item(cn, x, y, f0, {
                    "clip_id": cid, "language": LANGUAGE_NAMES.get(langs[0], langs[0]) if langs else "Carnatic",
                    "tonic_hz": ir["tonic_hz"], "singer_id": singer_id(name), "mbid": r.mbid, "split": sp,
                    "lyrics_verified": bool(ldoc.get("verified")), "wav": f"wavs/{cid}.wav"}))
                stats[f"{sp}_s"] += (y - x) / 50
                stats["akaram_clips" if not rows else "lyric_clips"] += 1
                n_clips += 1
        print(f"[{i}/{len(recs)}] {r.title}: {n_clips} clips", flush=True)
    for sp, lst in items.items():
        (out / f"{sp}.json").write_text(json.dumps(lst, ensure_ascii=False), encoding="utf-8")
        with open(out / f"{sp}.jsonl", "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in lst)
    sx = out / "soulx"
    sx.mkdir(exist_ok=True)
    extend_soulx_phoneset(SOULX_REPO / "soulxsinger" / "utils" / "phoneme" / "phone_set.json",
                          sx / "phone_set_indic.json")
    (sx / "embedding_init_map.json").write_text(json.dumps(embedding_init_map(), indent=1), encoding="utf-8")
    (sx / "data_processor.patch").write_text(DATA_PROCESSOR_PATCH, encoding="utf-8")
    summary = {**{k: round(v / 3600, 3) if k.endswith("_s") else v for k, v in stats.items()},
               "clips": {k: len(v) for k, v in items.items()}}
    (out / "stats.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary), flush=True)


def cmd_review_export(a):
    """``review-export``: Sonic Visualiser layers, sync preview and README into
    each bundle's ``review/``.
    """
    from .ir import Bundle
    from .review_sv import export_review
    recs = _saraga_recs(a)
    for i, r in enumerate(recs, 1):
        b = Bundle(a.ann_root, r.mbid)
        if not b.stage_done("f0"):
            continue
        out = export_review(b, preview=not a.no_preview)
        print(f"[{i}/{len(recs)}] {r.title}: {b.review_dir} ({', '.join(sorted(out))})", flush=True)


def cmd_review_import(a):
    """``review-import``: read corrected layers back and mark the recording reviewed."""
    from .ir import Bundle
    from .review_sv import import_review
    for r in _saraga_recs(a):
        b = Bundle(a.ann_root, r.mbid)
        rep = import_review(b, a.reviewer, Path(a.dir) if a.dir else None)
        print(f"{r.title}: {json.dumps(rep, ensure_ascii=False)}", flush=True)


def cmd_tonic_consensus(a):
    """Enforce one Sa per concert: recordings whose tonic pitch class is off the
    concert consensus get the consensus (octave placed per recording).

    Voting: one vote per recording, not weighted by duration (a single long
    recording with a weak tonic guess must not outvote several confident ones).
    Our raga-aware tonics vote twice; Saraga's ctonic of the concert's other
    recordings votes once.  A recording always votes with its own estimate
    (``tonic_recording_hz`` once a previous run replaced it), so re-runs don't
    feed on their own output, and a recording moved by an earlier run whose own
    estimate agrees with the new consensus gets its own tonic back.
    """
    import numpy as np
    from collections import defaultdict
    from .ir import Bundle
    from .stages.f0_ir import apply_tonic, concert_consensus
    from .stages.tonic import snap_to_tonic_octave
    by_concert = defaultdict(list)
    for r in _saraga_recs(a):
        b = Bundle(a.ann_root, r.mbid)
        if b.stage_done("f0"):
            by_concert[r.concert].append((r, b))
    from .saraga import scan
    saraga_votes = defaultdict(list)                       # every recording of the concert votes
    for r in scan(a.saraga_root):
        if r.tonic():
            saraga_votes[r.concert].append((r.mbid, r.tonic(), 1.0))

    def own(ir):
        """The recording's own tonic estimate, before any consensus replaced it."""
        return ir.get("tonic_recording_hz") or ir["tonic_hz"]

    for concert, items in by_concert.items():
        ours = {r.mbid: (r.mbid, own(b.load()), 2.0) for r, b in items}
        votes = list(ours.values()) + [v for v in saraga_votes[concert] if v[0] not in ours]
        ref = concert_consensus(votes)
        ref = ref / 2 ** np.floor(np.log2(ref / 100.0))     # print in 100-200 Hz
        for r, b in items:
            ir = b.load()
            mine = own(ir)
            off = (1200 * np.log2(mine / ref) + 600) % 1200 - 600
            moved = "tonic_recording_hz" in ir
            if abs(off) <= a.tol and not moved:
                continue
            if abs(off) <= a.tol:                             # an earlier run moved it; undo
                new, src = mine, ir.get("tonic_recording_source", "recording")
                print(f"{concert[:30]} | {r.title}: {ir['tonic_hz']} -> {new:.3f} Hz (own estimate restored)",
                      flush=True)
            else:
                f0 = np.load(b.path("f0_100hz.npz"))["f0_hz"]
                new, src = snap_to_tonic_octave(ref, f0), "concert_consensus"
                if abs(1200 * np.log2(new / ir["tonic_hz"])) < 5:
                    continue                                  # already on this consensus (within 5 c)
                print(f"{concert[:30]} | {r.title}: {ir['tonic_hz']} -> {new:.3f} Hz (off {off:+.0f} c)", flush=True)
            if a.dry_run:
                continue
            if src == "concert_consensus":
                ir["flags"] = sorted(set(ir.get("flags", [])) | {"tonic_from_concert"})
                ir.setdefault("tonic_recording_hz", ir["tonic_hz"])
                ir.setdefault("tonic_recording_source", ir.get("tonic_source", ""))
            else:
                ir["flags"] = sorted(set(ir.get("flags", [])) - {"tonic_from_concert"})
                ir.pop("tonic_recording_hz", None)
                ir.pop("tonic_recording_source", None)
            b.save(ir)
            apply_tonic(b, new, src)
            ir = b.load()
            ir.get("stages", {}).pop("swaras", None)       # swaras depend on the tonic
            b.save(ir)
        print(f"{concert[:30]}: Sa pitch class {ref:.2f} Hz ({len(votes)} votes)", flush=True)


def cmd_report(a):
    """``report``: per-recording QC table (``report.csv``) and one HTML page
    (``report.html``) with stage completion, hours by raga / singer / form and
    outliers, written to ``--out``."""
    from datetime import datetime
    from .report import write_report
    rows = write_report(Path(a.ann_root), Path(a.lyrics_dir), Path(a.out),
                        generated=datetime.now().strftime("%Y-%m-%d %H:%M"))
    print(f"{len(rows)} recordings -> {Path(a.out) / 'report.html'}, {Path(a.out) / 'report.csv'}")


def rescore_svc(a, out, phrases):
    """Re-measure the saved ``NN_{target,generated}.wav`` of a previous
    ``svc-check`` run and rewrite ``results.jsonl`` / ``summary.json``."""
    import numpy as np
    import soundfile as sf
    from .config import F0Cfg
    from .ir import Bundle
    from .stages.f0 import F0Estimator
    from .svc_check import SOULX_FPS, at_50fps, contour_metrics, quantise_like_soulx, summarise
    est = F0Estimator(F0Cfg(backends=["rmvpe", "fcpe", "crepe"], device=a.device), fps=SOULX_FPS)
    rows = []
    for i, p in enumerate(phrases, 1):
        f0 = np.load(Bundle(a.ann_root, p["mbid"]).path("f0_100hz.npz"))["f0_hz"]
        tgt_f0 = at_50fps(f0[int(p["target_start"] * 100):int((p["target_start"] + p["dur"]) * 100)])
        gen, sr = sf.read(out / f"{i:02d}_generated.wav", dtype="float32")
        tgt, _ = sf.read(out / f"{i:02d}_target.wav", dtype="float32")
        row = {**p, "generated": contour_metrics(tgt_f0, est(gen, sr)["f0"]),
               "measure": contour_metrics(tgt_f0, est(tgt, sr)["f0"]),
               "quant": contour_metrics(tgt_f0, quantise_like_soulx(tgt_f0))}
        rows.append(row)
        print(f"[{i}/{len(phrases)}] {p['title'][:28]}: {row['generated']}", flush=True)
    with (out / "results.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summ = summarise(rows)
    (out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


def cmd_svc_check(a):
    """``svc-check``: render probe phrases with SoulX-Singer's SVC model following
    our F0, re-measure the output's F0 and score it (see :mod:`saregama.svc_check`).

    Writes ``<out>/<i>_{target,prompt,generated}.wav`` (24 kHz), ``results.jsonl``
    (phrases and metrics) and ``summary.json`` (medians over phrases).  With
    ``--rescore`` nothing is rendered: the saved wavs of the previous run are
    re-measured and re-scored (e.g. after a metric change).
    """
    import os
    import numpy as np
    import soundfile as sf
    import torch
    from .audio import resample
    from .config import F0Cfg
    from .ir import Bundle
    from .paths import SOULX_REPO
    from .stages.f0 import F0Estimator
    from .svc_check import (SOULX_FPS, PhraseCfg, at_50fps, contour_metrics, pick_phrases, quantise_like_soulx,
                            summarise)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = []
    for r in _saraga_recs(a):
        b = Bundle(a.ann_root, r.mbid)
        if not b.stage_done("f0"):
            continue
        ir = b.load()
        recs.append({"mbid": r.mbid, "title": ir["title"], "raga": ir.get("raaga") or "?",
                     "singer": (ir.get("lead") or ["?"])[0], "tonic_hz": ir["tonic_hz"],
                     "f0": np.load(b.path("f0_100hz.npz"))["f0_hz"]})
    phrases = pick_phrases(recs, PhraseCfg(n=a.n, dur=a.dur, prompt_s=a.prompt_s))
    print(f"{len(phrases)} phrases from {len({p['raga'] for p in phrases})} ragas", flush=True)
    if a.rescore:
        rescore_svc(a, out, phrases)
        return

    sys.path.insert(0, str(SOULX_REPO))
    from cli.inference_svc import build_model                       # SoulX-Singer repo
    from soulxsinger.utils.file_utils import load_config
    cwd = os.getcwd()
    os.chdir(SOULX_REPO)                                             # the SoulX config uses repo-relative paths
    try:
        config = load_config(str(SOULX_REPO / "soulxsinger/config/soulxsinger.yaml"))
        model = build_model(a.model or str(SOULX_REPO / "pretrained_models/SoulX-Singer/model-svc.pt"),
                            config, device=a.device, use_fp16=False)
    finally:
        os.chdir(cwd)
    sr = int(config.audio.sample_rate)
    est = F0Estimator(F0Cfg(backends=["rmvpe", "fcpe", "crepe"], device=a.device), fps=SOULX_FPS)
    rows = []
    with (out / "results.jsonl").open("w", encoding="utf-8") as fh:
        for i, p in enumerate(phrases, 1):
            b = Bundle(a.ann_root, p["mbid"])
            wav, wsr = sf.read(b.path("stems/vocal_clean.flac"), dtype="float32", always_2d=True)
            wav = wav.mean(axis=1)

            def cut(t0, dur):
                """``dur`` seconds of the vocal from ``t0``, at SoulX's rate."""
                return resample(wav[int(t0 * wsr):int((t0 + dur) * wsr)], wsr, sr)

            tgt, pmt = cut(p["target_start"], p["dur"]), cut(p["prompt_start"], p["prompt_s"])
            f0 = np.load(b.path("f0_100hz.npz"))["f0_hz"]
            tgt_f0 = at_50fps(f0[int(p["target_start"] * 100):int((p["target_start"] + p["dur"]) * 100)])
            pmt_f0 = at_50fps(f0[int(p["prompt_start"] * 100):int((p["prompt_start"] + p["prompt_s"]) * 100)])
            dev = a.device
            with torch.no_grad():
                gen, _ = model.infer(pt_wav=torch.from_numpy(pmt)[None].to(dev),
                                     gt_wav=torch.from_numpy(tgt)[None].to(dev),
                                     pt_f0=torch.from_numpy(pmt_f0)[None].to(dev),
                                     gt_f0=torch.from_numpy(tgt_f0)[None].to(dev),
                                     auto_shift=False, pitch_shift=0, n_steps=a.n_steps, cfg=a.cfg)
            gen = gen.squeeze().float().cpu().numpy()
            for name, x in (("target", tgt), ("prompt", pmt), ("generated", gen)):
                sf.write(out / f"{i:02d}_{name}.wav", x, sr)
            row = {**p, "generated": contour_metrics(tgt_f0, est(gen, sr)["f0"]),
                   "measure": contour_metrics(tgt_f0, est(tgt, sr)["f0"]),
                   "quant": contour_metrics(tgt_f0, quantise_like_soulx(tgt_f0))}
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{i}/{len(phrases)}] {p['title'][:28]} ({p['raga']}): generated {row['generated']} | "
                  f"measure floor rmse {row['measure']['rmse_cents']}", flush=True)
    summ = summarise(rows)
    (out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


def main(argv=None):
    """Parse the command line and run the chosen command."""
    p = argparse.ArgumentParser(prog="saregama", description=__doc__)
    p.add_argument("--config", default=None, help="TOML config (defaults if omitted)")
    p.add_argument("--work-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--saraga-root", default=str(SARAGA_ROOT))
    p.add_argument("--ann-root", default=str(ANN_ROOT), help="where per-recording IR bundles live")
    p.add_argument("--lyrics-dir", default=str(LYRICS_DIR), help="per-work lyric YAMLs")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="create a workspace with templates")
    s.add_argument("dir")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("run", help="run all stages on songs.csv")
    s.add_argument("--songs", required=True)
    s.add_argument("--only", default="", help="comma-separated song ids")
    s.add_argument("--force", action="store_true", help="recompute auto annotations (never touches reviewed/gold)")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("draft-lyrics", help="ASR draft lyric sheets for songs without lyrics")
    s.add_argument("--songs", required=True)
    s.add_argument("--asr-model", required=True, help="IndicConformer .nemo file")
    s.add_argument("--out", default="lyrics_drafts")
    s.set_defaults(fn=cmd_draft_lyrics)

    s = sub.add_parser("export-review", help="write Praat TextGrids for flagged (or given) songs")
    s.add_argument("--songs", default="", help="comma-separated ids (default: by status)")
    s.add_argument("--status", default="flagged")
    s.set_defaults(fn=cmd_export_review)

    s = sub.add_parser("audit", help="export a random sample of auto-accepted songs for review")
    s.add_argument("--fraction", type=float, default=0.02)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_audit)

    s = sub.add_parser("import-review", help="read a corrected TextGrid back")
    s.add_argument("--song", required=True)
    s.add_argument("--textgrid", default=None)
    s.add_argument("--gold", action="store_true", help="mark as gold (evaluation + annotator training)")
    s.add_argument("--keep-notes", action="store_true")
    s.add_argument("--reviewer", default="")
    s.set_defaults(fn=cmd_import_review)

    s = sub.add_parser("make-eval-csv", help="songs.csv that re-runs the pipeline on gold songs")
    s.add_argument("--out", default="eval/songs_eval.csv")
    s.set_defaults(fn=cmd_make_eval_csv)

    s = sub.add_parser("evaluate", help="compare predictions with gold")
    s.add_argument("--pred", required=True, help="e.g. work_eval/songs")
    s.add_argument("--gold", required=True, help="e.g. work/songs")
    s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("export", help="build the SoulX-Singer training set")
    s.add_argument("--include-flagged", action="store_true")
    s.add_argument("--soulx-phone-set", default=None, help="SoulX phone_set.json to extend")
    s.add_argument("--val-frac", type=float, default=0.03)
    s.add_argument("--holdout-singers", default="")
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("stats", help="hours by language and status, gold progress")
    s.add_argument("--gold-target", type=float, default=2.5)
    s.set_defaults(fn=cmd_stats)

    s = sub.add_parser("phoneset", help="list the Indic phones, or extend SoulX's phone_set.json")
    s.add_argument("--soulx-phone-set", default=None)
    s.add_argument("--out", default="phone_set_indic.json")
    s.set_defaults(fn=cmd_phoneset)

    s = sub.add_parser("g2p", help="show syllables and phones for a line of lyrics")
    s.add_argument("text")
    s.add_argument("--lang", required=True, choices=["hi", "mr", "ta"])
    s.set_defaults(fn=cmd_g2p)

    s = sub.add_parser("manifest", help="index Saraga Carnatic recordings -> <ann-root>/manifest.csv")
    s.set_defaults(fn=cmd_manifest)

    s = sub.add_parser("sync", help="measure stem offsets, write mix-aligned FLAC stems")
    s.add_argument("--mbids", default="", help="comma-separated mbids or a file of mbids")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_sync, stems_only=True)

    s = sub.add_parser("stems", help="de-bleed and de-reverb the vocal stem(s) (Mel-RoFormer)")
    s.add_argument("--mbids", default="")
    s.add_argument("--device", default="cuda")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_stems, stems_only=True)

    s = sub.add_parser("f0", help="100 Hz F0 ensemble on the cleaned vocal stem(s), tonic, cents")
    s.add_argument("--mbids", default="")
    s.add_argument("--backends", default="rmvpe,fcpe,crepe")
    s.add_argument("--qc-only", action="store_true", help="only re-compare stored F0 with Saraga's tracks")
    s.add_argument("--device", default="cuda")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_f0, stems_only=True)

    s = sub.add_parser("swaras", help="raga-aware swara segments + flat contour")
    s.add_argument("--mbids", default="")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_swaras, stems_only=True)

    s = sub.add_parser("viz", help="contour sheets (review/contour.pdf)")
    s.add_argument("--mbids", default="")
    s.add_argument("--max-pages", type=int, default=0)
    s.set_defaults(fn=cmd_viz, stems_only=True)

    s = sub.add_parser("lyrics-fetch", help="candidate lyrics per Saraga work (MusicBrainz + karnatik.com)")
    s.add_argument("--mbids", default="")
    s.add_argument("--no-second-source", dest="second_source", action="store_false",
                   help="do not search shivkumar.org for works karnatik.com cannot match")
    s.set_defaults(fn=cmd_lyrics_fetch, stems_only=True)

    s = sub.add_parser("lyrics-status", help="rebuild <lyrics-dir>/status.tsv from every work YAML (offline)")
    s.set_defaults(fn=cmd_lyrics_status)

    s = sub.add_parser("lyrics-review-pack", help="works.json + audio excerpts for the lyrics review page")
    s.add_argument("--out", default=str(LYRICS_DIR / "review_pack"))
    s.add_argument("--seconds", type=float, default=20.0)
    s.set_defaults(fn=cmd_lyrics_review_pack, stems_only=True, mbids="")

    s = sub.add_parser("lyrics-apply", help="apply decisions exported from the lyrics review page")
    s.add_argument("--reviews", required=True, help="JSON list of review decisions")
    s.set_defaults(fn=cmd_lyrics_apply)

    s = sub.add_parser("lyrics-verify", help="show a work's lyric candidates, or record the right one")
    s.add_argument("--work", required=True, help="work mbid")
    s.add_argument("--pick", type=int, default=None, help="candidate index to accept")
    s.add_argument("--text", default="", help="lyrics file (section lines end with ':') from another source")
    s.add_argument("--source", default="", help="URL/citation for --text")
    s.add_argument("--reviewer", default="")
    s.add_argument("--notes", default="")
    s.set_defaults(fn=cmd_lyrics_verify)

    s = sub.add_parser("emissions", help="cache MMS_FA emissions of every cleaned vocal stem (GPU)")
    s.add_argument("--mbids", default="")
    s.add_argument("--device", default="cuda")
    s.set_defaults(fn=cmd_emissions, stems_only=True)

    s = sub.add_parser("sections-train", help="evaluate (leave-one-concert-out) and train the section classifier")
    s.add_argument("--mbids", default="")
    s.add_argument("--model", default=str(ANN_ROOT.parent / "models" / "sections.pkl"))
    s.add_argument("--switch-cost", type=float, default=8.0, help="Viterbi cost of a section change")
    s.set_defaults(fn=cmd_sections_train, stems_only=True)

    s = sub.add_parser("sections", help="automatic sections (ir.json sections_auto) where Saraga has none")
    s.add_argument("--mbids", default="")
    s.add_argument("--model", default=str(ANN_ROOT.parent / "models" / "sections.pkl"))
    s.add_argument("--min-region-s", type=float, default=15.0)
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_sections, stems_only=True)

    s = sub.add_parser("tala", help="beat grid, laya and sama (ir.json tala); --evaluate scores the estimator")
    s.add_argument("--mbids", default="")
    s.add_argument("--evaluate", action="store_true", help="score estimation on recordings with manual samas")
    s.add_argument("--estimate", action="store_true",
                   help="also estimate grids where no samas are annotated (unreliable: see README)")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_tala, stems_only=True)

    s = sub.add_parser("export-ir", help="ICM-IR v0 dataset (parquet + 100 Hz frames) from the bundles")
    s.add_argument("--mbids", default="")
    s.add_argument("--out", default=str(ANN_ROOT.parent / "export" / "icm_ir_v0"))
    s.add_argument("--test-concerts", default="", help="';'-separated concert names forced into test")
    s.add_argument("--val-concerts", default="", help="';'-separated concert names forced into val")
    s.set_defaults(fn=cmd_export_ir, stems_only=True)

    s = sub.add_parser("export-soulx", help="SoulX-Singer training clips (melody mode) from the bundles")
    s.add_argument("--mbids", default="")
    s.add_argument("--out", default=str(ANN_ROOT.parent / "export" / "soulx_v0"))
    s.add_argument("--splits", default=str(ANN_ROOT.parent / "export" / "icm_ir_v0" / "splits.json"),
                   help="concert -> split map to follow (the ICM-IR export's)")
    s.add_argument("--min-s", type=float, default=2.0)
    s.add_argument("--max-s", type=float, default=16.0)
    s.add_argument("--no-akaram", dest="akaram", action="store_false", help="skip alapana akaram clips")
    s.add_argument("--keep-low-conf", action="store_true", help="keep low-confidence lyric instances")
    s.set_defaults(fn=cmd_export_soulx, stems_only=True)

    s = sub.add_parser("align", help="line-loop CTC alignment of the (selected) lyrics")
    s.add_argument("--mbids", default="")
    s.add_argument("--device", default="cuda")
    s.add_argument("--filler-cost", type=float, default=-0.1, help="trellis engine only")
    s.add_argument("--engine", default="spikes", choices=["spikes", "trellis"],
                   help="spikes: DP over MMS letter spikes (default); trellis: frame-level CTC")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_align, stems_only=True)

    s = sub.add_parser("review-export", help="Sonic Visualiser layers + sync_preview.flac in review/")
    s.add_argument("--mbids", default="")
    s.add_argument("--no-preview", action="store_true", help="skip sync_preview.flac")
    s.set_defaults(fn=cmd_review_export, stems_only=True)

    s = sub.add_parser("review-import", help="read corrected layers from review/corrected/, mark reviewed")
    s.add_argument("--mbids", required=True)
    s.add_argument("--reviewer", default="")
    s.add_argument("--dir", default="", help="folder with the corrected .svl files")
    s.set_defaults(fn=cmd_review_import, stems_only=True)

    s = sub.add_parser("tonic-consensus", help="one Sa per concert; fix outlier recordings")
    s.add_argument("--mbids", default="")
    s.add_argument("--tol", type=float, default=40.0, help="cents")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_tonic_consensus, stems_only=True)

    s = sub.add_parser("report", help="dataset QC report (report.csv + report.html)")
    s.add_argument("--out", default=str(ANN_ROOT.parent / "report"))
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("svc-check", help="SoulX SVC rendering check on probe phrases (GPU)")
    s.add_argument("--mbids", default="")
    s.add_argument("--out", default=str(ANN_ROOT.parent / "svc_check"))
    s.add_argument("--model", default="", help="model-svc.pt (default: SoulX repo pretrained_models/SoulX-Singer)")
    s.add_argument("--device", default="cuda")
    s.add_argument("--n", type=int, default=20, help="probe phrases")
    s.add_argument("--dur", type=float, default=8.0, help="target phrase length, s")
    s.add_argument("--prompt-s", type=float, default=20.0, help="prompt length, s")
    s.add_argument("--n-steps", type=int, default=32, help="diffusion steps")
    s.add_argument("--cfg", type=float, default=3.0, help="classifier-free guidance scale")
    s.add_argument("--rescore", action="store_true", help="re-measure the saved wavs; render nothing")
    s.set_defaults(fn=cmd_svc_check, stems_only=True)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
