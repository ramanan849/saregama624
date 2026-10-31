"""Sonic Visualiser layers round-trip: export -> (edit) -> import rebuilds lyrics."""
from saregama.review_sv import _svl, lyrics_from_layers, read_svl


def _write(tmp_path, name, kind, pts):
    p = tmp_path / f"{name}.svl"
    p.write_text(_svl(kind, name, pts, 10.0, "#000000"), encoding="utf-8")
    return read_svl(p)


def test_roundtrip_rebuilds_hierarchy(tmp_path):
    lines = _write(tmp_path, "lines", "regions",
                   [{"start": 1.0, "end": 3.0, "value": 0, "label": "0|pallavi|ninnu kori"},
                    {"start": 5.0, "end": 6.0, "value": 0, "label": "0|pallavi|ninnu kori"}])
    words = _write(tmp_path, "words", "regions",
                   [{"start": 1.0, "end": 2.0, "value": 0, "label": "ninnu"},
                    {"start": 2.0, "end": 3.0, "value": 0, "label": "kori"},
                    {"start": 5.0, "end": 6.0, "value": 0, "label": "ninnu"}])
    syls = _write(tmp_path, "syllables", "regions",
                  [{"start": s, "end": s + 0.5, "value": 220.0, "label": l}
                   for s, l in [(1.0, "nin"), (1.5, "nu"), (2.0, "ko"), (2.5, "ri"), (5.0, "nin"), (5.5, "nu")]])
    assert abs(syls[1]["start"] - 1.5) < 1e-4 and syls[1]["label"] == "nu"
    insts, problems = lyrics_from_layers(lines, words, syls)
    assert not problems
    assert [i["line_id"] for i in insts] == [0, 0]
    assert [w["text"] for w in insts[0]["words"]] == ["ninnu", "kori"]
    assert [s["text"] for s in insts[0]["words"][1]["syllables"]] == ["ko", "ri"]


def test_instants_and_bad_labels(tmp_path):
    sama = _write(tmp_path, "sama", "timeinstants", [{"start": 2.5, "label": "sama"}])
    assert abs(sama[0]["start"] - 2.5) < 1e-4
    lines = _write(tmp_path, "lines", "regions", [{"start": 0, "end": 1, "value": 0, "label": "oops"}])
    _, problems = lyrics_from_layers(lines, [], [])
    assert problems
