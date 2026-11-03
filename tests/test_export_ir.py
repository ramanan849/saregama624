"""ICM-IR export pieces: swara naming, svarasthana, splits, quantisation, phones."""
import numpy as np

from saregama.export_ir import (PHONE_ID, UNVOICED, assign_splits, instance_phones, ir_swara_name, phone_track,
                                quantise_cents, singer_id, svarasthana)


def test_swara_names_and_svarasthana():
    """IR names with octaves; ambiguous svarasthanas resolved by the raga."""
    assert ir_swara_name(".D") == ("Dha", -1) and ir_swara_name("M'") == ("Ma_t", 1)
    assert svarasthana("g", "S R g m P D n".split()) == "G2"        # Kharaharapriya
    assert svarasthana("g", "S g G m P n N".split()) == "R3"        # Nata: R3 and G3
    assert svarasthana("R", "S R G m P D N".split()) == "R2"
    assert svarasthana("n", "S R g m P D N".split()) == "N2"
    assert svarasthana("n", "S r g m P d N".split()) == "N2"        # N3 present but so is D1
    assert svarasthana("G", None) == "G3"


def test_splits_are_by_concert_and_stable():
    """Forced concerts land where asked; the rest is deterministic."""
    cs = [f"concert {k}" for k in range(8)]
    a = assign_splits(cs, test=["concert 5"])
    assert a["concert 5"] == "test" and list(a.values()).count("test") == 2
    assert list(a.values()).count("val") == 2 and a == assign_splits(cs, test=["concert 5"])
    assert singer_id("Cherthala Ranganatha Sharma") == "cherthala_ranganatha_sharma"


def test_quantise_and_phone_track():
    """10-cent ints with an unvoiced code; consonants short, vowel fills the syllable."""
    q = quantise_cents(np.array([0.0, 1203.0, np.nan, -95.0]))
    assert q.tolist() == [0, 120, UNVOICED, -10]
    t = phone_track(30, [{"phones": "m-aa-n", "start": 0.0, "end": 0.2}])
    assert t[0] == PHONE_ID["m"] and t[10] == PHONE_ID["aa"] and t[19] == PHONE_ID["n"] and t[25] == -1


def test_instance_phones_follow_the_line_text():
    """Phones come from the karnatik.com spelling of the line, fitted per word."""
    lines = {0: {"id": 0, "text": "dAsuDan pErAsacE", "words": ["dasudan", "perasace"]}}
    inst = {"line_id": 0, "section": "pallavi",
            "words": [{"text": "perasace", "syllables": [{"text": "pe"}, {"text": "ra"}, {"text": "sa"},
                                                         {"text": "ce"}]}]}
    ph = instance_phones(inst, lines)
    assert ph == [[["p", "ee"], ["r", "aa"], ["s", "a"], ["c", "ee"]]]
