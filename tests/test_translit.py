"""karnatik.com transliteration -> ISO-15919 and phones."""
from saregama.translit import (fit_syllables, is_notation, line_phones, notation_syllables, syllabify_phones,
                               to_iso, to_phones, word_tokens)


def test_iso_spelling():
    """Long vowels, retroflexes, sibilants, clusters, Tamil letters, vocalic r."""
    assert to_iso("dAsuDanE") == "dāsuḍanē"
    assert to_iso("karuNAmrta") == "karuṇāmṛta"
    assert to_iso("shrI") == "śrī"
    assert to_iso("lakSmIm") == "lakṣmīm"
    assert to_iso("jnAna") == "jñāna"
    assert to_iso("vinaigaL") == "vinaigaḷ"
    assert to_iso("mazhai") == "maḻai"
    assert to_iso("Bhakthi") == "bhakthi"                  # capital B is not a scheme letter
    assert to_iso("karadhrta-parvata") == "karadhṛtaparvata"


def test_phones():
    """Phones follow South Indian pronunciation (ṛ = ru, anusvara assimilates)."""
    assert to_phones("dAsuDanE") == ["d", "aa", "s", "u", "D", "a", "n", "ee"]
    assert to_phones("amrta") == ["a", "m", "r", "u", "t", "a"]
    assert to_phones("saMgIta") == ["s", "a", "ng", "g", "ii", "t", "a"]
    assert to_phones("kaimkarya")[:2] == ["k", "ai"]
    assert to_phones("namaH") == ["n", "a", "m", "a", "h"]


def test_syllables_and_fitting():
    """Phone syllables use the aligner's rule; counts are fitted to the alignment."""
    assert syllabify_phones(to_phones("manasaramati")) == [["m", "a"], ["n", "a"], ["s", "a"], ["r", "a"],
                                                            ["m", "a"], ["t", "i"]]
    syls = syllabify_phones(to_phones("amrta"))           # a | mru | ta  (3) vs aligner am | rta (2)
    assert len(syls) == 3 and len(fit_syllables(syls, 2)) == 2
    assert sum(len(s) for s in fit_syllables(syls, 2)) == 6
    assert len(fit_syllables([["k", "a"]], 3)) == 3        # melisma: more aligned syllables than phones


def test_notation_and_tokens():
    """Svara notation lines become sargam; words split like the aligner splits them."""
    assert is_notation("pdsrsmpmdpSndRS,|") and is_notation("S , , R G")
    assert not is_notation("sa ri ga ma")                 # written-out sargam is ordinary text
    assert notation_syllables("srG,") == ["sa", "ri", "ga"]
    assert word_tokens("karadhrta-parvata tyAgarAja") == ["karadhrta", "parvata", "tyAgarAja"]
    assert line_phones("S , R G") is None
