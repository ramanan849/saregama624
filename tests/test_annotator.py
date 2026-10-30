"""Smoke test for the unified annotator: train a tiny model for a few steps on a
synthetic exported dataset, then plug it into the pipeline as both an F0 backend
and the alignment backend.  Skips when torch is not installed."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def test_annotator_round_trip():
    try:
        import torch  # noqa: F401
    except Exception:
        print("torch not available; skipping")
        return
    from annotator_model.infer import AnnotatorRunner
    from annotator_model.train import main as train_main
    from saregama.config import load_config
    from saregama.export import export_dataset
    from saregama.pipeline import Pipeline, read_songs
    from test_pipeline import MMS_LIKE_VOCAB, OracleEmitter, build_workspace, oracle_emission

    tmp = Path(tempfile.mkdtemp(prefix="svs_ann_"))
    try:
        synths = build_workspace(tmp)
        cfg = load_config(None, work_dir=str(tmp / "work"), out_dir=str(tmp / "dataset"))
        cfg.separation.backend = "none"
        cfg.f0.backends = ["yin"]
        cfg.align.backend = "external"
        cfg.g2p.lexicon = []
        cfg.clips.min_s = 1.0
        pipe = Pipeline(cfg)
        pipe.aligner.emitter = OracleEmitter()
        rows = read_songs(tmp / "songs.csv")
        for row in rows:
            pipe.run_song(row, force=True, emission=oracle_emission(synths[row["song_id"]], MMS_LIKE_VOCAB))
        export_dataset(cfg, ("auto", "flagged"))
        ck = tmp / "ckpt"
        train_main(["--data", str(tmp / "dataset"), "--out", str(ck), "--d", "64", "--layers", "2",
                    "--epochs", "2", "--batch", "4", "--workers", "0", "--device", "cpu"])
        assert (ck / "best.pt").exists()

        cfg2 = load_config(None, work_dir=str(tmp / "work2"), out_dir=str(tmp / "ds2"))
        cfg2.separation.backend = "none"
        cfg2.f0.backends = ["yin", "annotator"]
        cfg2.align.backend = "annotator"
        cfg2.align.annotator_ckpt = str(ck / "best.pt")
        cfg2.align.device = "cpu"
        cfg2.g2p.lexicon = []
        runner = AnnotatorRunner(cfg2.align.annotator_ckpt, device="cpu")
        ann = Pipeline(cfg2, annotator=runner).run_song(rows[0], force=True)
        assert not ann.validate(), ann.validate()
        print("annotator-driven pipeline ok:", ann.status, len(ann.syllables), "syllables")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_annotator_round_trip()
    print("ALL OK")
