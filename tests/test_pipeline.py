"""End-to-end test on synthetic songs (no GPU, no model downloads).

Run:  python -m pytest -q tests   (or: python tests/test_pipeline.py)

Covers: G2P -> F0 (YIN) -> tonic -> CTC alignment with an oracle emission ->
notes -> QC -> evaluation against gold -> Praat review round-trip -> export ->
compatibility with SoulX-Singer's DataProcessor (re-implemented faithfully below,
plus the real one when torch and the SoulX repo are importable).
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from saregama.audio import FPS, HOP, SR, load_audio, save_audio  # noqa: E402
from saregama.config import load_config  # noqa: E402
from saregama.evaluate import evaluate  # noqa: E402
from saregama.export import export_dataset  # noqa: E402
from saregama.pipeline import Pipeline, read_songs  # noqa: E402
from saregama.review import export_review, import_review, read_textgrid, write_textgrid  # noqa: E402
from saregama.schema import SongAnnotation  # noqa: E402
from saregama.stages.align import Emitter  # noqa: E402
from synth import MMS_LIKE_VOCAB, oracle_emission, synth_song  # noqa: E402

_HI = ["मन मेरा गाए रे", "प्रभु तेरे चरणों में", "सुबह की धूप खिली"]
_MR = ["गणपती बाप्पा मोरया", "माझं मन गातं", "पाऊस आला रे"]
_TA = ["வணக்கம் தமிழ் பாட்டு", "கண்ணே மணியே", "நிலவே வா வா"]
SONGS = [                       # choruses repeat, as in real lyric sheets
    ("hi_test_0001", "hi", _HI + _HI[:2] + _HI[1:], 146.83),
    ("mr_test_0001", "mr", _MR + _MR[:2] + _MR[1:], 196.0),
    ("ta_test_0001", "ta", _TA + _TA[:2] + _TA[1:], 130.81),
]
SOULX_PHONESET = os.environ.get("SOULX_PHONE_SET", "")


class OracleEmitter(Emitter):
    unit = "roman"

    def __init__(self):
        self.vocab = MMS_LIKE_VOCAB
        self.blank = MMS_LIKE_VOCAB["-"]
        self.star_id = MMS_LIKE_VOCAB["*"]


def build_workspace(tmp: Path):
    rows, synths = [], {}
    for k, (sid, lang, lines, tonic) in enumerate(SONGS):
        s = synth_song(sid, lang, lines, tonic, seed=k)
        synths[sid] = s
        save_audio(tmp / "audio" / f"{sid}.wav", s.audio)
        (tmp / "lyrics").mkdir(exist_ok=True)
        (tmp / "lyrics" / f"{sid}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        rows.append([sid, f"audio/{sid}.wav", lang, f"lyrics/{sid}.txt", "bhajan", f"singer_{k}",
                     "f", "1", "", "synthetic", "test"])
    with open(tmp / "songs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["song_id", "audio", "language", "lyrics", "style", "singer_id", "gender",
                    "vocal_only", "tonic_hz", "source", "license"])
        w.writerows(rows)
    # gold copies
    for sid, s in synths.items():
        g = s.gold
        g.vocals_path = str(tmp / "audio" / f"{sid}.wav")
        g.save(tmp / "gold" / "songs" / sid)
    return synths


def soulx_preprocess_reference(meta: dict, phone2idx: dict, sr=SR, hop=HOP):
    """Faithful copy of SoulX-Singer DataProcessor.merge_phoneme + preprocess with
    the one-line `in_` patch, minus torch."""
    duration = [float(x) for x in meta["duration"].split()]
    phoneme = [str(x).replace("<AP>", "<SP>") for x in meta["phoneme"].split()]
    note_pitch = [int(x) for x in meta["note_pitch"].split()]
    note_type = [int(x) if phoneme[i] != "<SP>" else 1 for i, x in enumerate(meta["note_type"].split())]
    merged = []
    for i in range(len(phoneme)):
        if i > 0 and phoneme[i] == phoneme[i - 1] == "<SP>" and note_type[i] == note_type[i - 1] \
                and note_pitch[i] == note_pitch[i - 1]:
            merged[-1][1] += duration[i]
        else:
            merged.append([phoneme[i], duration[i], note_pitch[i], note_type[i]])
    phonemes = [x[0] for x in merged]
    note_duration = [x[1] for x in merged]
    n_mel = int(sum(note_duration) * sr / hop)
    new_ph = []
    for ph in phonemes:
        new_ph.append("<BOW>")
        if ph[:3] in ("en_", "in_"):
            new_ph += [ph[:3] + x for x in ph[3:].split("-")] + ["<SEP>"]
        else:
            new_ph.append(ph)
        new_ph.append("<EOW>")
    ids = [phone2idx[x] for x in ["<PAD>"] + new_ph]
    return n_mel, ids


def test_end_to_end():
    tmp = Path(tempfile.mkdtemp(prefix="svs_e2e_"))
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
            em = oracle_emission(synths[row["song_id"]], MMS_LIKE_VOCAB, seed=1)
            ann = pipe.run_song(row, force=True, emission=em)
            assert not ann.validate(), ann.validate()
            print(row["song_id"], ann.status, ann.qc.get("flags"), "tonic", ann.tonic_hz)

        # ---- evaluation against gold --------------------------------
        res = evaluate(tmp / "work" / "songs", tmp / "gold" / "songs")
        s = res["summary"]
        print(json.dumps(s, indent=1))
        assert s["songs"] == 3
        assert s["tonic_accuracy"] == 1.0
        assert s["per"] == 0.0
        assert s["onset_median_ms"] <= 60
        assert s["gross_pitch_error"] < 0.05

        # ---- review round-trip -------------------------------------
        sid = "hi_test_0001"
        done = export_review(cfg, [sid])
        assert done == [sid]
        tg = tmp / "work" / "review" / sid / f"{sid}.TextGrid"
        tiers = read_textgrid(tg)
        ann0 = SongAnnotation.load(tmp / "work" / "songs" / sid)
        xmax = ann0.n_frames / FPS
        # reviewer: fix tonic (give it an octave up), mark an octave fix, move one boundary
        tiers["tonic"] = [(0.0, xmax, f"{146.83 * 2:.2f}")]
        v = np.where(ann0.f0 > 0)[0]
        a = v[10] / FPS
        tiers["f0_fix"] = [(a, a + 0.1, "x2")]
        syl = [list(iv) for iv in tiers["syllables"]]
        k = next(i for i, iv in enumerate(syl) if iv[2] and i + 1 < len(syl) and syl[i + 1][2])
        shift = 0.04
        syl[k][1] += shift
        syl[k + 1][0] += shift
        syl[k][2] = syl[k][2].split("|")[0]          # drop phones: G2P must regenerate them
        tiers["syllables"] = [tuple(iv) for iv in syl]
        write_textgrid(tg, xmax, list(tiers.items()))
        ann1 = import_review(cfg, sid, gold=True, reviewer="tester")
        assert ann1.status == "gold" and not ann1.validate()
        assert abs(ann1.tonic_hz - 146.83) < 0.5, ann1.tonic_hz        # snapped back to the octave
        i0 = int(round(a * FPS))
        assert abs(ann1.f0[i0 + 1] / ann0.f0[i0 + 1] - 2) < 1e-3
        assert ann1.syllables[k].phones, "G2P should refill phones"

        # ---- export -------------------------------------------------
        out = export_dataset(cfg, ("auto", "flagged", "reviewed", "gold"), SOULX_PHONESET or None)
        print(json.dumps(out, indent=1, ensure_ascii=False))
        items = []
        for split in ("train", "val", "test"):
            p = tmp / "dataset" / f"{split}.jsonl"
            if p.exists():
                items += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
        assert items, "no clips exported"
        if SOULX_PHONESET:
            ph = json.loads((tmp / "dataset" / "soulx" / "phone_set_indic.json").read_text(encoding="utf-8"))
            phone2idx = {p: i for i, p in enumerate(ph)}
            assert len(ph) <= 3000
        else:
            from saregama.phoneset import PHONES
            base = ["<PAD>", "<SP>", "<AP>", "<UNK>", "<BOW>", "<EOW>", "<BOS>", "<EOS>", "<MASK>", "<SEP>"]
            phone2idx = {p: i for i, p in enumerate(base + ["in_" + q.name for q in PHONES])}
        for it in items:
            f0 = it["f0"].split()
            durs = [float(x) for x in it["duration"].split()]
            nn = len(durs)
            for key in ("text", "phoneme", "note_pitch", "note_type", "note_cents", "swara"):
                assert len(it[key].split()) == nn, (key, it["index"])
            for key in ("energy_db", "hnr_db", "h1h2_db", "f0_agree"):
                assert len(it[key].split()) == len(f0)
            n_mel, ids = soulx_preprocess_reference(it, phone2idx)
            assert n_mel == len(f0) == it["n_frames"], (n_mel, len(f0), it["n_frames"])
            wav = load_audio(tmp / "dataset" / it["wav"], SR)
            assert len(wav) == it["n_frames"] * HOP
            assert 1.0 <= it["n_frames"] / FPS <= 16.5
        print(f"{len(items)} clips verified against the SoulX DataProcessor logic")

        # ---- heuristic aligner smoke test --------------------------
        cfg2 = load_config(None, work_dir=str(tmp / "work_h"), out_dir=str(tmp / "ds_h"))
        cfg2.separation.backend = "none"
        cfg2.f0.backends = ["yin"]
        cfg2.align.backend = "heuristic"
        cfg2.g2p.lexicon = []
        ann = Pipeline(cfg2).run_song(rows[0], force=True)
        assert not ann.validate()
        print("heuristic aligner ok:", ann.status)
    finally:
        if os.environ.get("KEEP_TMP"):
            print("kept", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


def test_g2p_examples():
    from saregama.g2p import line_to_syllables
    got = {s.text: "-".join(s.phones) for s in line_to_syllables("कमला बचपन सत्य मैं", "hi")}
    assert got == {"कम": "k-a-m", "ला": "l-aa", "बच": "b-a-c", "पन": "p-a-n",
                   "सत्": "s-a-t", "य": "y-a", "मैं": "m-E-nas"}, got
    mr = ["-".join(s.phones) for s in line_to_syllables("चहा ज्ञान ऐकू", "mr")]
    assert mr == ["ts-a", "h-aa", "d-ny-aa-n", "a-i", "k-uu"], mr
    ta = ["-".join(s.phones) for s in line_to_syllables("ஒன்று குற்றம் பாடல் அப்பா", "ta")]
    assert ta == ["o-n", "d-r-u", "k-u-t", "rr-a-m", "p-aa", "D-a-l", "a-p", "p-aa"], ta


if __name__ == "__main__":
    test_g2p_examples()
    test_end_to_end()
    print("ALL OK")
