"""Carnatic-Varnam evaluation: structure parsing with repeated keys, tick spans,
laying notation onto ticks, swara-name conversion and scoring (synthetic data)."""
import numpy as np

from saregama.varnam_eval import Note, expected_notes, parse_structure, score, tick_spans, to_notation


def test_parse_structure_keeps_repeats(tmp_path):
    """Repeated section keys survive, in order (a YAML loader would drop them)."""
    p = tmp_path / "s.yaml"
    p.write_text("artist: 'x'\nraga: 'y'\nstructure:\n  pallavi: 2\n  charanam: 2\n  pallavi: 4\n")
    assert parse_structure(p) == [("pallavi", 2), ("charanam", 2), ("pallavi", 4)]


def test_tick_spans_close_pauses():
    """A long gap ends the tick after one median tick instead of at the next tick."""
    spans = tick_spans(np.array([0.0, 1.0, 2.0, 10.0, 11.0]))
    assert spans[1] == (1.0, 2.0) and spans[2] == (2.0, 3.0) and spans[-1] == (11.0, 12.0)


def test_expected_notes_speeds_and_karvai():
    """Two svaras per tick then four; "," extends the previous note."""
    sections = {"p": ["S", ",", "R", "G"]}
    notes, rep = expected_notes(sections, [("p", 2), ("p", 4)], [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)])
    assert rep == {"ticks_used": 3, "ticks": 3, "skipped": []}
    assert [(n.svara, n.start, n.end) for n in notes[:3]] == [("S", 0.0, 1.0), ("R", 1.0, 1.5), ("G", 1.5, 2.0)]
    assert notes[3].start == 2.0 and notes[3].end == 2.5             # S + karvai at 4 per tick


def test_to_notation():
    """Variant case dropped; octave marks converted."""
    assert [to_notation(x) for x in ["g", ".D", "S'", "N", ""]] == ["G", "D_", "S^", "N", ""]


def test_score():
    """Matching stable segments score full accuracy and on-time onsets; a missing
    segment is unlabelled, not wrong."""
    notes = [Note(0.0, 1.0, "S"), Note(1.0, 2.0, "R"), Note(2.0, 3.0, "G")]
    segs = [{"kind": "stable", "swara": "S", "start": 0.0, "end": 1.0},
            {"kind": "stable", "swara": "R", "start": 1.05, "end": 2.0},
            {"kind": "glide", "swara": "", "start": 2.0, "end": 3.0}]
    sc = score(notes, segs, 300)
    assert sc["label_accuracy"] == 1.0 and sc["unlabelled"] == 1 and sc["labelled_share"] == round(2 / 3, 3)
    assert sc["changes"] == 2 and sc["onset_hit_100ms"] == 0.5 and sc["onset_median_err_ms"] == 50.0
