"""Automatic sections: label mapping, smoothing, region merging, rules baseline."""
import numpy as np

from saregama.stages.sections import (CLASSES, FEATURE_NAMES, class_of_label, frame_labels, rules,
                                      to_regions, viterbi_smooth)


def test_saraga_labels_map_to_classes():
    """Every Saraga label seen on the 46 labelled recordings lands in the right class."""
    assert class_of_label("Vocal ālāp") == "vocal_alapana"
    assert class_of_label("Violin ālāp") == "violin_alapana"
    assert class_of_label("Ślōka") == "vocal_alapana"
    assert class_of_label("Caraṇam hamīr kalyāṇi") == "kriti"
    assert class_of_label("Nereval") == "kriti"
    assert class_of_label("Kalpanā svara") == "svara"
    assert class_of_label("Ciṭṭa svara") == "svara"
    assert class_of_label("Tani āvartana") == "tani"
    assert class_of_label("-") is None
    y = frame_labels([{"start": 0, "end": 3, "label": "Pallavi"}, {"start": 3, "end": 5, "label": "-"}], 6)
    assert y.tolist() == [2, 2, 2, -1, -1, -1]


def test_viterbi_removes_flicker():
    """A one-second blip of another class does not survive a switching cost."""
    p = np.full((30, len(CLASSES)), 0.05)
    p[:, 2] = 0.8
    p[15] = 0.05
    p[15, 4] = 0.8                                   # one noisy "tani" second inside kriti
    assert (viterbi_smooth(p, switch_cost=8.0) == 2).all()


def test_short_regions_merge_into_neighbours():
    """A 5 s region inside long ones disappears; region labels are aligner labels."""
    cls = np.array([0] * 60 + [2] * 5 + [0] * 40 + [4] * 30)
    regs = to_regions(cls, min_region_s=15.0, length_s=135.4)
    assert [r["class"] for r in regs] == ["vocal_alapana", "tani"]
    assert regs[0]["label"] == "Vocal ālāp" and regs[-1]["end"] == 135.4


def test_rules_baseline():
    """Percussion + voice -> kriti, percussion alone -> tani, voice alone -> alapana."""
    n = len(FEATURE_NAMES)
    x = np.zeros((3, 3 * n))
    perc, voiced = 2 * n + FEATURE_NAMES.index("perc_db"), 2 * n + FEATURE_NAMES.index("voiced")
    x[:, perc] = [-5, -5, -60]
    x[:, voiced] = [0.8, 0.0, 0.8]
    assert [CLASSES[k] for k in rules(x)] == ["kriti", "tani", "vocal_alapana"]
