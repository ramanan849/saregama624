"""Gold evaluation: syllable onset matching, line-instance P/R, sections, sama,
and the prediction snapshot taken by review-import."""
import json

from saregama.gold_eval import evaluate_bundle, instance_scores, sama_scores, section_scores, syllable_scores


def _lyr(shift=0.0, line=0):
    """One line instance with two words of two syllables, shifted by ``shift`` s."""
    sy = lambda t, x: {"text": x, "start": t + shift, "end": t + shift + 0.2}
    return {"instances": [{"line_id": line, "start": 1 + shift, "end": 2 + shift, "words": [
        {"text": "ra ma", "start": 1 + shift, "end": 1.4 + shift, "syllables": [sy(1.0, "ra"), sy(1.2, "ma")]},
        {"text": "sa ri", "start": 1.4 + shift, "end": 2 + shift, "syllables": [sy(1.4, "sa"), sy(1.6, "ri")]}]}]}


def test_syllables_and_instances():
    """A 40 ms shift matches every syllable within 50 ms; a wrong line id fails P/R."""
    from saregama.gold_eval import syllables
    s = syllable_scores(syllables(_lyr(0.04)), syllables(_lyr()))
    assert s["recall"] == 1.0 and s["within_50ms"] == 1.0 and abs(s["median_abs_err_ms"] - 40) < 0.01
    assert instance_scores(_lyr(0.04), _lyr())["f1"] == 1.0
    assert instance_scores(_lyr(line=3), _lyr())["f1"] == 0.0


def test_sections_and_sama():
    """Section accuracy over gold-labelled seconds; sama hits within 70 ms."""
    gold = [{"start": 0, "end": 10, "label": "Pallavi"}, {"start": 10, "end": 20, "label": "Tani Avartanam"}]
    pred = [{"start": 0, "end": 15, "label": "Pallavi"}, {"start": 15, "end": 20, "label": "Tani Avartanam"}]
    assert section_scores(pred, gold, 20)["accuracy"] == 0.75
    assert sama_scores([1.0, 2.05, 3.5], [1.0, 2.0, 3.0])["hit_70ms"] == round(2 / 3, 3)


def test_snapshot_and_evaluate(tmp_path):
    """review-import keeps the predictions once; evaluate_bundle scores them."""
    from saregama.ir import Bundle
    from saregama.review_sv import snapshot_predictions
    d = tmp_path / "m1"
    d.mkdir()
    ir = {"mbid": "m1", "title": "t", "length_s": 20, "sections": [], "sama": [1.0], "tonic_hz": 140.0}
    (d / "ir.json").write_text(json.dumps(ir))
    (d / "lyrics.json").write_text(json.dumps(_lyr(0.04)))
    b = Bundle(tmp_path, "m1")
    assert snapshot_predictions(b, ir) is True and snapshot_predictions(b, ir) is False
    (d / "lyrics.json").write_text(json.dumps(_lyr()))                 # the corrections
    ir.update(status="reviewed", sama=[1.03], review={"reviewer": "x", "layers": ["lines", "words", "syllables",
                                                                                  "sama"]})
    (d / "ir.json").write_text(json.dumps(ir))
    r = evaluate_bundle(d)
    assert r["syllables"]["within_50ms"] == 1.0 and r["sama"]["hit_70ms"] == 1.0
