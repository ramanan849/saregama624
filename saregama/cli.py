"""Command line: ``saregama <command> ...`` (or ``python -m saregama``)."""
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
from .saraga import DEFAULT_ROOT as SARAGA_ROOT

SONGS_HEADER = ["song_id", "audio", "language", "lyrics", "style", "singer_id", "gender",
                "vocal_only", "tonic_hz", "source", "license"]


def cmd_init(a):
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
    over = {}
    if getattr(a, "work_dir", None):
        over["work_dir"] = a.work_dir
    if getattr(a, "out_dir", None):
        over["out_dir"] = a.out_dir
    return load_config(a.config, **over)


def cmd_run(a):
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
    need = cfg.align.backend == "annotator" or "annotator" in cfg.f0.backends
    if not need:
        return None
    from annotator_model.infer import AnnotatorRunner
    return AnnotatorRunner(cfg.align.annotator_ckpt, device=cfg.align.device)


def cmd_draft_lyrics(a):
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
    from .evaluate import evaluate
    res = evaluate(a.pred, a.gold, a.out)
    print(json.dumps(res["summary"], indent=1))


def cmd_export(a):
    from .export import export_dataset
    cfg = _cfg(a)
    inc = ("auto", "reviewed", "gold") + (("flagged",) if a.include_flagged else ())
    hold = a.holdout_singers.split(",") if a.holdout_singers else ()
    res = export_dataset(cfg, inc, a.soulx_phone_set, a.val_frac, hold)
    print(json.dumps(res, indent=1, ensure_ascii=False))


def cmd_stats(a):
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
    from .phoneset import PHONES, embedding_init_map, extend_soulx_phoneset
    if a.soulx_phone_set:
        added = extend_soulx_phoneset(a.soulx_phone_set, a.out)
        print(f"added {len(added)} tokens, indices {min(added.values())}-{max(added.values())} -> {a.out}")
    else:
        for p in PHONES:
            print(f"in_{p.name:4s} {p.ipa:5s} {p.kind:9s} init={'+'.join(p.init)}")


def cmd_g2p(a):
    from .g2p import Lexicon, line_to_syllables
    cfg = _cfg(a)
    lex = Lexicon.load(*cfg.g2p.lexicon)
    for s in line_to_syllables(a.text, a.lang, lex):
        print(f"{s.text}\t{'-'.join(s.phones)}")


# ----------------------------------------------------------------------------- Saraga

def _saraga_recs(a):
    from .saraga import scan
    recs = [r for r in scan(a.saraga_root) if r.has_stems or not a.stems_only]
    if a.mbids:
        keep = set(Path(a.mbids).read_text().split()) if Path(a.mbids).exists() else set(a.mbids.split(","))
        recs = [r for r in recs if r.mbid in keep]
    return recs


def cmd_manifest(a):
    from .saraga import scan, write_manifest
    recs = scan(a.saraga_root)
    out = Path(a.ann_root) / "manifest.csv"
    write_manifest(recs, out)
    print(f"{len(recs)} recordings ({sum(r.has_stems for r in recs)} with stems) -> {out}")


def cmd_sync(a):
    from .ir import init_bundle
    from .stages.sync import bleed_scores, estimate_offsets, write_aligned_stems
    recs = _saraga_recs(a)
    for i, r in enumerate(recs, 1):
        b = init_bundle(r, a.ann_root)
        if b.stage_done("sync") and not a.force:
            continue
        res = estimate_offsets(r.mix, r.stems)
        paths = write_aligned_stems(r.mix, r.stems, res.offset_s, b.stems_dir)
        bleed = bleed_scores(paths)
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
    from .config import F0Cfg, SeparationCfg
    from .ir import F0_HZ, Bundle
    from .stages.f0 import F0Estimator
    from .stages.f0_ir import run_f0
    cfg = F0Cfg(backends=a.backends.split(","), device=a.device)
    est = F0Estimator(cfg, SeparationCfg().soulx_repo, fps=F0_HZ)
    recs = _saraga_recs(a)
    for i, r in enumerate(recs, 1):
        b = Bundle(a.ann_root, r.mbid)
        if not b.stage_done("stems"):
            print(f"[{i}/{len(recs)}] {r.title}: run stems first", flush=True)
            continue
        if b.stage_done("f0") and not a.force:
            continue
        rep = run_f0(b, est, r.pitch())
        ir = b.mark("f0", backends=cfg.backends)
        print(f"[{i}/{len(recs)}] {r.title}: tonic {ir['tonic_hz']} ({ir['tonic_source']}, "
              f"histogram {ir.get('tonic_histogram_hz')}) {json.dumps(rep)}", flush=True)


def cmd_swaras(a):
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
    from .lyrics_fetch import Fetcher, fetch_work, karnatik_index
    out_dir = Path(a.lyrics_dir)
    f = Fetcher(out_dir / "_cache")
    index = karnatik_index(f)
    works = {}
    for r in _saraga_recs(a):
        if r.work_mbid and r.work_mbid not in works:
            works[r.work_mbid] = r
    rows = []
    for i, (wm, r) in enumerate(works.items(), 1):
        try:
            doc = fetch_work(f, index, wm, r.work_title or r.title, r.raaga, out_dir)
        except Exception as e:
            print(f"[{i}/{len(works)}] error    {r.work_title or r.title}: {e!r}", flush=True)
            rows.append([wm, r.work_title or r.title, r.raaga, "", "error", "", "", "", ""])
            continue
        best = doc["candidates"][0] if doc["candidates"] else {}
        status = "verified" if doc.get("verified") else ("auto" if doc.get("selected") else "missing")
        rows.append([wm, r.work_title or r.title, r.raaga, "; ".join(doc["musicbrainz"].get("composer", [])),
                     status, best.get("title", ""), best.get("raga", ""), best.get("score", ""), best.get("url", "")])
        print(f"[{i}/{len(works)}] {status:8s} {r.work_title or r.title} ({r.raaga}) -> "
              f"{best.get('title', '-')} / {best.get('raga', '-')} {best.get('score', '')}", flush=True)
    with open(out_dir / "status.tsv", "w", encoding="utf-8") as fh:
        fh.write("work_mbid\ttitle\traga\tcomposer\tstatus\tbest_title\tbest_raga\tscore\turl\n")
        for row in rows:
            fh.write("\t".join(map(str, row)) + "\n")
    print(json.dumps(Counter(r[4] for r in rows)))


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
    if a.text:
        sections, cur = [], None
        for line in Path(a.text).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.endswith(":"):
                cur = {"name": line[:-1], "lines": []}
                sections.append(cur)
            else:
                if cur is None:
                    cur = {"name": "pallavi", "lines": []}
                    sections.append(cur)
                cur["lines"].append(line)
        doc.update(selected=a.source or str(a.text), source="manual", lyrics=sections)
    else:
        c = doc["candidates"][a.pick]
        doc.update(selected=c["url"], source="karnatik", lyrics=c["parsed"]["sections"])
    doc.update(verified=True, verified_by=a.reviewer, notes=a.notes or doc.get("notes", ""))
    path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
    print(f"verified {doc['title']} <- {doc['selected']}")


def cmd_align(a):
    import numpy as np
    import soundfile as sf
    import yaml
    from .audio import resample
    from .ir import Bundle
    from .stages.align_kriti import AlignCfgK, MmsEmitter, align_regions, lyric_lines, to_json
    cfg = AlignCfgK(device=a.device, filler_logp=a.filler_logp)
    em = None
    recs = _saraga_recs(a)
    for i, r in enumerate(recs, 1):
        b = Bundle(a.ann_root, r.mbid)
        if not b.stage_done("stems"):
            print(f"[{i}/{len(recs)}] {r.title}: run stems first", flush=True)
            continue
        if b.stage_done("align") and not a.force:
            continue
        lpath = Path(a.lyrics_dir) / f"{r.work_mbid}.yaml"
        doc = yaml.safe_load(lpath.read_text(encoding="utf-8")) if lpath.exists() else None
        if not doc or not doc.get("lyrics"):
            print(f"[{i}/{len(recs)}] {r.title}: no lyrics selected ({lpath.name})", flush=True)
            continue
        lines = lyric_lines(doc["lyrics"])
        emit_path = b.path("mms_50hz.npz")
        if emit_path.exists():
            logp = np.load(emit_path)["logp"].astype(np.float32)
        else:
            em = em or MmsEmitter(cfg)
            ir = b.load()
            x, sr = sf.read(str(b.dir / ir["stems"]["vocal_clean"]["path"]), dtype="float32")
            logp = em(resample(x, sr, 16000))
            np.savez_compressed(emit_path, logp=logp.astype(np.float16))
        ir = b.load()
        regions = ir.get("sections") or [{"start": 0.0, "end": ir["length_s"], "label": "unknown"}]
        import torchaudio
        vocab = dict(torchaudio.pipelines.MMS_FA.get_dict(star=None))
        inst, rep = align_regions(logp, lines, regions, vocab, vocab["-"], cfg)
        if b.path("f0_100hz.npz").exists():
            from .stages.align_kriti import extend_sustain
            extend_sustain(inst, np.load(b.path("f0_100hz.npz"))["f0_hz"] > 0)
        src = {"work_mbid": r.work_mbid, "selected": doc.get("selected"), "source": doc.get("source"),
               "verified": bool(doc.get("verified"))}
        b.path("lyrics.json").write_text(json.dumps(to_json(lines, inst, rep, src), ensure_ascii=False),
                                         encoding="utf-8")
        lyric_s = sum(x.end - x.start for x in inst)
        summ = {"instances": len(inst), "lyric_s": round(lyric_s, 1),
                "low_conf": sum("low_conf" in x.flags for x in inst),
                "mean_llr": round(float(np.mean([x.llr for x in inst])), 3) if inst else None,
                "lines_seen": sorted({x.line_id for x in inst}), "n_lines": len(lines),
                "lyrics_verified": src["verified"]}
        ir = b.load()
        ir.setdefault("qc", {})["align"] = summ
        if not src["verified"]:
            ir["flags"] = sorted(set(ir.get("flags", [])) | {"lyrics_unverified"})
        b.save(ir)
        b.mark("align", **summ)
        print(f"[{i}/{len(recs)}] {r.title}: {json.dumps(summ)}", flush=True)


def cmd_tonic_consensus(a):
    """Enforce one Sa per concert: recordings whose tonic pitch class is off the
    concert consensus get the consensus (octave placed per recording)."""
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
            saraga_votes[r.concert].append((r.mbid, r.tonic(), r.length_s))
    for concert, items in by_concert.items():
        ours = {r.mbid: (r.mbid, b.load()["tonic_hz"], 2 * r.length_s) for r, b in items}
        votes = list(ours.values()) + [v for v in saraga_votes[concert] if v[0] not in ours]
        ref = concert_consensus(votes)
        ref = ref / 2 ** np.floor(np.log2(ref / 100.0))     # print in 100-200 Hz
        for r, b in items:
            ir = b.load()
            off = (1200 * np.log2(ir["tonic_hz"] / ref) + 600) % 1200 - 600
            if abs(off) <= a.tol:
                continue
            f0 = np.load(b.path("f0_100hz.npz"))["f0_hz"]
            new = snap_to_tonic_octave(ref, f0)
            print(f"{concert[:30]} | {r.title}: {ir['tonic_hz']} -> {new:.3f} Hz (off {off:+.0f} c)", flush=True)
            if a.dry_run:
                continue
            ir["flags"] = sorted(set(ir.get("flags", [])) | {"tonic_from_concert"})
            ir["tonic_recording_hz"] = ir["tonic_hz"]
            b.save(ir)
            apply_tonic(b, new, "concert_consensus")
            ir = b.load()
            ir.get("stages", {}).pop("swaras", None)       # swaras depend on the tonic
            b.save(ir)
        print(f"{concert[:30]}: Sa pitch class {ref:.2f} Hz ({len(votes)} votes)", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(prog="saregama", description=__doc__)
    p.add_argument("--config", default=None, help="TOML config (defaults if omitted)")
    p.add_argument("--work-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--saraga-root", default=str(SARAGA_ROOT))
    p.add_argument("--ann-root", default=str(ANN_ROOT), help="where per-recording IR bundles live")
    p.add_argument("--lyrics-dir", default=str(ANN_ROOT.parent / "lyrics"), help="per-work lyric YAMLs")
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
    s.set_defaults(fn=cmd_lyrics_fetch, stems_only=True)

    s = sub.add_parser("lyrics-verify", help="show a work's lyric candidates, or record the right one")
    s.add_argument("--work", required=True, help="work mbid")
    s.add_argument("--pick", type=int, default=None, help="candidate index to accept")
    s.add_argument("--text", default="", help="lyrics file (section lines end with ':') from another source")
    s.add_argument("--source", default="", help="URL/citation for --text")
    s.add_argument("--reviewer", default="")
    s.add_argument("--notes", default="")
    s.set_defaults(fn=cmd_lyrics_verify)

    s = sub.add_parser("align", help="line-loop CTC alignment of the (selected) lyrics")
    s.add_argument("--mbids", default="")
    s.add_argument("--device", default="cuda")
    s.add_argument("--filler-logp", type=float, default=-3.0)
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_align, stems_only=True)

    s = sub.add_parser("tonic-consensus", help="one Sa per concert; fix outlier recordings")
    s.add_argument("--mbids", default="")
    s.add_argument("--tol", type=float, default=40.0, help="cents")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_tonic_consensus, stems_only=True)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
