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
