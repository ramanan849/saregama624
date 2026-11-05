"""Film-song pitch-contour sargam report (``viz.song_contour_pdf``) on a tiny
synthetic legacy song: side files, swara positions and page count."""
import json
import re

import numpy as np

from saregama.viz import _dot_name, song_contour_pdf


def test_dot_names():
    """Mandra gets a dot below, tara a dot above, madhya none."""
    assert _dot_name("D", -1) == "Ḍ" and _dot_name("S", 1) == "Ṡ" and _dot_name("P", 0) == "P"


def test_song_contour_pdf(tmp_path):
    """A 40 s song held on Sa then Pa: Pa is found ~700 cents up, and the PDF
    has an overview page plus two 30 s pages."""
    fps, tonic = 50.0, 200.0
    hz = np.zeros(int(40 * fps), dtype=np.float32)
    hz[100:900] = tonic                                   # Sa, 2-18 s
    hz[1000:1800] = tonic * 2 ** (700 / 1200)             # Pa, 20-36 s
    d = tmp_path / "song"
    d.mkdir()
    np.save(d / "f0.npy", hz)
    syl = [{"text": "x", "phones": ["s", "aa"], "start": 100, "line": 0, "word": 0},
           {"text": "y", "phones": ["r", "e"], "start": 150, "line": 0, "word": 0},
           {"text": "z", "phones": ["p", "aa"], "start": 1000, "line": 1, "word": 0}]
    (d / "annotation.json").write_text(json.dumps({"song_id": "t", "tonic_hz": tonic, "syllables": syl}))
    out = song_contour_pdf(d, tmp_path / "out" / "t.pdf", title="Test")
    sw = json.loads(out.with_suffix(".swaras.json").read_text())
    assert abs(sw["positions_cents"]["P"] - 700) < 10
    assert np.load(out.with_suffix(".f0.npz"))["cents"].shape == hz.shape
    assert len(re.findall(rb"/Type\s*/Page[^s]", out.read_bytes())) == 3     # overview + 2 windows
