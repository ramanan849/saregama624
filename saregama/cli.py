"""Command line: ``svs-annotate <command> ...`` (or ``python -m saregama``)."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .config import load_config

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


def main(argv=None):
    p = argparse.ArgumentParser(prog="svs-annotate", description=__doc__)
    p.add_argument("--config", default=None, help="TOML config (defaults if omitted)")
    p.add_argument("--work-dir", default=None)
    p.add_argument("--out-dir", default=None)
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

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
