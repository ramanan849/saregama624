"""karnatik.com page parsing: sections, repeat markers and wrapped line tails."""
from saregama.lyrics_fetch import parse_karnatik

PAGE = """<html><script>var Language: x;</script>
<b>Song:</b> dinamaNi vamsha<br>
raagam: <a href="r.shtml">harikaambhOji</a><br>
taaLam: aadi<br>
Composer: <a href="c.shtml">Tyaagaraaja</a><br>
Language: Telugu<br>
Telugu<br>
<p>pallavi<br>dinamaNi vamsha tilaka lAvaNya dIna sharanya<br>(dinamaNi)<br>
<p>anupallavi<br>manavini bAguga madini dalancucu canavuna nElu cAlugA jAlu<br>
caraNam<br>sharva vinuta nanu samrakSincanu garvamu Ela gAcu vArevarE<br>
nirvikAra guNa nirmala karadhrta-parvata tyAgarAja<br>sarvasvamau<br>
ciTTaswaram<br>sa ri ga ma pa da ni<br>
<hr>Meaning: ...</html>"""


def test_parse_sections_markers_and_wrapped_tail():
    """Header fields; sections (one header mid-block, one in the "sw" spelling);
    the language name spilling into the lyrics dropped; the repeat marker kept
    aside; the one-word wrapped tail joined to its line."""
    out = parse_karnatik(PAGE)
    assert out["raga"] == "harikaambhOji" and out["tala"] == "aadi" and out["language"] == "Telugu"
    names = [s["name"] for s in out["sections"]]
    assert names == ["pallavi", "anupallavi", "caraNam", "ciTTaswaram"]   # no "Telugu" pseudo-section
    assert out["sections"][0]["repeat_markers"] == ["dinamaNi"]
    car = out["sections"][2]["lines"]
    assert len(car) == 2 and car[1].endswith("tyAgarAja sarvasvamau")
