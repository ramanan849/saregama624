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


SHIV = """<p>Ragam : Charukesi<br>Talam : Adi<br>Composer: Tyagaraja<br>
Pallavi :<br>Adamodi Galade Ramayya Maata<br>
Anupallavi :<br>Thodunida Nive Yanuchu Bhaktito<br>Gudina Nee Paadamu Battina<br>Naato Maata<br>
Charanam :<br>Chaduvulanni Delisi Shankaraantshu<br>Dai<br>
Meaning:<br>O Ramayya ! ...<br>P ; D- P ; M G R | G- gr S ||<br>"""


def test_parse_shivkumar():
    """Header fields, sections up to 'Meaning', lower-cased lines, wrapped tails joined."""
    from saregama.lyrics_fetch import parse_shivkumar
    out = parse_shivkumar(SHIV)
    assert out["raga"] == "Charukesi" and out["tala"] == "Adi" and out["composer"] == "Tyagaraja"
    assert [s["name"] for s in out["sections"]] == ["pallavi", "anupallavi", "caraNam"]
    assert out["sections"][0]["lines"] == ["adamodi galade ramayya maata"]
    assert out["sections"][1]["lines"][-2:] == ["gudina nee paadamu battina", "naato maata"]  # 2-word tail stays
    assert out["sections"][2]["lines"] == ["chaduvulanni delisi shankaraantshu dai"]
