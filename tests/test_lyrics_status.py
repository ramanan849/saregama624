"""status.tsv always lists every work YAML, whatever subset a run touched."""
import yaml

from saregama.lyrics_fetch import write_status


def _doc(mbid, **kw):
    """Minimal work YAML content."""
    return {"work_mbid": mbid, "title": f"t{mbid}", "raga": "r", "musicbrainz": {"composer": ["c"],
            "languages": ["tel"]}, "candidates": [], **kw}


def test_status_covers_every_yaml(tmp_path):
    """One verified, one auto-selected, one missing work -> three rows, right counts."""
    docs = {"a": _doc("a", verified=True, selected="u1"), "b": _doc("b", selected="u2"), "c": _doc("c")}
    for k, d in docs.items():
        (tmp_path / f"{k}.yaml").write_text(yaml.safe_dump(d), encoding="utf-8")
    counts = write_status(tmp_path)
    assert counts == {"verified": 1, "auto": 1, "missing": 1}
    rows = (tmp_path / "status.tsv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 4 and rows[0].startswith("work_mbid\t")
    assert [r.split("\t")[5] for r in rows[1:]] == ["verified", "auto", "missing"]


def test_review_decisions():
    """Each review decision changes the work YAML as documented."""
    import pytest
    from saregama.lyrics_fetch import apply_decision, parse_lyrics_text
    secs = [{"name": "pallavi", "lines": ["ninnu kori"]}]
    cand = {"url": "u1", "site": "shivkumar", "parsed": {"sections": secs}}
    doc = apply_decision(_doc("a", candidates=[cand], selected=None), "candidate", "r1", candidate_url="u1")
    assert doc["verified"] and doc["source"] == "shivkumar" and doc["lyrics"] == secs and doc["verified_by"] == "r1"
    doc = apply_decision(_doc("b"), "text", "r2", text="pallavi:\nsAmaja vara\n# note\ncaraNam:\nvEda", source="book")
    assert [s["name"] for s in doc["lyrics"]] == ["pallavi", "caraNam"] and doc["selected"] == "book"
    doc = apply_decision(_doc("c", selected="u9", lyrics=secs), "wrong", "r3", notes="other song")
    assert doc["selected"] is None and not doc["verified"] and doc["notes"] == "other song"
    assert parse_lyrics_text("one line only") == [{"name": "pallavi", "lines": ["one line only"]}]
    with pytest.raises(ValueError):
        apply_decision(_doc("d"), "candidate", candidate_url="nope")
