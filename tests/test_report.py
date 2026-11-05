"""Dataset report: per-recording rows, missing-stage reasons, outliers, grouping,
and the HTML page; plus the one-vote-per-recording tonic consensus."""
import csv

from saregama.report import collect, hours_by, missing_reason, outliers, render_html, summarise, write_report
from saregama.stages.f0_ir import concert_consensus


def _ir(**kw):
    """A minimal bundle ir.json with sync..f0 done and Saraga's manual sections."""
    ir = {"mbid": "m1", "title": "Song", "concert": "Singer at Arkay by Singer", "lead": ["Singer"],
          "raaga": "Tōḍi", "taala": "Ādi", "form": "Kriti", "length_s": 600.0, "work_mbid": "w1",
          "tonic_hz": 140.0, "tonic_source": "saraga_ctonic", "sections": [{"label": "Pallavi"}],
          "stems": {"vocal_clean": {}}, "flags": [],
          "qc": {"f0": {"vocal_clean": {"voiced": 0.6, "vs_saraga": {"pitch": {"rpa50": 0.9}}}}},
          "stages": {"sync": {"offset_ms": 2.0}, "stems": {}, "f0": {}}}
    ir.update(kw)
    return ir


def test_summarise_and_reasons():
    """Stage gaps get the command's own skip reason; manual sections need no stage."""
    row = summarise(_ir(), {"w1": "auto"})
    assert row["sections_source"] == "manual" and row["minutes"] == 10.0 and row["rpa50"] == 0.9
    assert row["missing"] == "swaras, tala, align"            # sections not missing: manual
    assert "align: not run yet" in row["missing_reason"]
    assert missing_reason("align", _ir(work_mbid=None), "") == "no composition (alapana / tani / RTP)"
    assert missing_reason("align", _ir(), "missing") == "no lyrics found yet"
    assert missing_reason("sections", _ir(stages={"sync": {}}), "") == "waits for swaras"
    assert summarise(_ir(work_mbid=None), {})["lyrics_status"] == "no composition"


def test_outliers():
    """Low pitch agreement, sync flags and big offsets are flagged; a clean row is not."""
    assert outliers(summarise(_ir(), {})) == []
    bad = _ir(flags=["sync:no_tight_stem"], stages={"sync": {"offset_ms": 80.0}, "f0": {}},
              qc={"f0": {"vocal_clean": {"voiced": 0.1, "vs_saraga": {"rpa50": 0.3}}}})
    out = " | ".join(outliers(summarise(bad, {})))
    for part in ("30%", "voiced only 10%", "80 ms", "sync:no_tight_stem"):
        assert part in out


def test_hours_by_and_html(tmp_path):
    """Grouping folds the tail into "other"; files are written; the page escapes text."""
    rows = [summarise(_ir(mbid=f"m{i}", raaga=f"r{i % 3}", title="<b>x</b>"), {}) for i in range(6)]
    g = hours_by(rows, "raga", top=2)
    assert len(g) == 3 and g[-1][0].startswith("other") and abs(sum(t[2] for t in g) - 1.0) < 1e-9
    page = render_html(rows, "now")
    assert "&lt;b&gt;x&lt;/b&gt;" in page and "<b>x</b>" not in page
    import json
    for i in range(2):
        (tmp_path / "ann" / f"m{i}").mkdir(parents=True)
        (tmp_path / "ann" / f"m{i}" / "ir.json").write_text(json.dumps(_ir(mbid=f"m{i}")))
    rows = write_report(tmp_path / "ann", tmp_path / "nolyrics", tmp_path / "out")
    assert len(rows) == 2 == len(list(csv.DictReader((tmp_path / "out" / "report.csv").open())))
    assert (tmp_path / "out" / "report.html").read_text().startswith("<!doctype html>")
    assert collect(tmp_path / "ann", tmp_path / "nolyrics")[0]["lyrics_status"] == "missing"


def test_consensus_counts_recordings_not_minutes():
    """Five recordings at 138 Hz outvote one long recording a fourth away when each
    recording has the same weight (the tonic-consensus voting rule)."""
    votes = [(f"m{i}", 138.0, 2.0) for i in range(5)] + [("long", 206.5, 2.0)]
    ref = concert_consensus(votes)
    off = (1200 * __import__("math").log2(ref / 138.0) + 600) % 1200 - 600
    assert abs(off) < 5
