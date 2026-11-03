"""Candidate lyrics for Saraga works, fetched from the web for human review.

For each MusicBrainz work (``work_mbid`` in the Saraga metadata):

1. MusicBrainz gives the composer and language (1 request/s, as its policy asks).
2. karnatik.com's alphabetical index (one page listing ~8000 songs as
   "title - raaga - composer") is fuzzy-matched on title, raga and composer.
3. The best candidate pages are parsed into sections (pallavi, anupallavi,
   caranam, ...) of lyric lines in karnatik's transliteration.
4. ``<lyrics_dir>/<work_mbid>.yaml`` is written with every candidate,
   ``selected: null`` and ``verified: false``.  A person picks the right
   candidate (``lyrics-verify``) or pastes lyrics from another source.

``auto_select`` pre-selects a candidate whose title and raga match closely
(``AUTO_SCORE``, ``AUTO_RAGA``); it is used for alignment but the recording is
flagged ``lyrics_unverified`` until a person confirms it.  The ``lyrics-fetch``
command writes ``status.tsv`` (auto / missing / verified per work); missing
works need lyrics from another source via ``lyrics-verify --text``.  The
lyrics are for internal annotation only and are not redistributed.

Works karnatik.com cannot match are then searched on shivkumar.org's krithi
archive (``shivkumar_index`` / ``parse_shivkumar``) with the same scoring and
auto-selection; that site spells in plain English-style capitals, so its
lines are lower-cased.

Requests carry a generic User-Agent naming this repository (no personal
contact details) and are rate-limited per host (shivkumar.org: 10 s, as its
robots.txt asks).
"""
from __future__ import annotations

import html
import json
import re
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional

import yaml

USER_AGENT = "saregama624/0.1 (+https://github.com/ramanan849/saregama624)"
SHIVKUMAR = "https://www.shivkumar.org/music/"
HOST_INTERVAL = {"www.shivkumar.org": 10.0}           # its robots.txt: crawl-delay 10
KARNATIK = "https://www.karnatik.com/"
MUSICBRAINZ = "https://musicbrainz.org/ws/2/"
LANGUAGES = {"telugu", "tamil", "kannada", "sanskrit", "malayalam", "hindi", "marathi", "manipravalam"}
SECTION_NAMES = ("pallavi", "anupallavi", "caraNam", "samaShTi caraNam", "madhyama kAla sAhityam",
                 "madhyama kAlam", "madhyama kaalam", "madhyamakAla sAhityam",
                 "ciTTasvaram", "ciTTa svaram", "svara sAhityam", "muktAyi svaram", "ettugaDa svaram",
                 "caraNam 1", "caraNam 2", "caraNam 3", "caraNam 4", "caraNam 5", "caraNam 6")


class Fetcher:
    """HTTP GET with an on-disk cache and a per-host minimum interval."""

    def __init__(self, cache_dir: Path, min_interval_s: float = 1.0):
        """``cache_dir`` holds one file per URL; ``min_interval_s`` is the
        minimum gap between two requests to the same host.
        """
        import requests
        self.s = requests.Session()
        self.s.headers["User-Agent"] = USER_AGENT
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval_s
        self.last: Dict[str, float] = {}
        self.host_interval = dict(HOST_INTERVAL)      # robots.txt crawl delays

    def get(self, url: str, encoding: Optional[str] = None) -> str:
        """Body of ``url``, from the cache if present.

        Otherwise waits for the host's interval, retries up to 5 times with
        exponential backoff on network errors and 429/5xx, and caches the text.
        ``encoding`` forces the decoding (karnatik.com pages are latin-1).
        """
        key = re.sub(r"[^A-Za-z0-9._-]+", "_", url)[-180:]
        path = self.cache / key
        if path.exists():
            return path.read_text(encoding="utf-8")
        host = url.split("/")[2]
        interval = max(self.min_interval, self.host_interval.get(host, 0.0))
        wait = interval - (time.time() - self.last.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        for attempt in range(5):
            try:
                r = self.s.get(url, timeout=30)
                self.last[host] = time.time()
                if r.status_code in (429, 500, 502, 503, 504):
                    raise IOError(f"HTTP {r.status_code}")
                break
            except Exception:
                self.last[host] = time.time()
                if attempt == 4:
                    raise
                time.sleep(2.0 * 2 ** attempt)
        r.raise_for_status()
        if encoding:
            r.encoding = encoding
        path.write_text(r.text, encoding="utf-8")
        return r.text


# ----------------------------------------------------------------------------- matching

def norm(s: str) -> str:
    """Loose phonetic key for transliterated Indian titles: case, diacritics,
    vowel length, aspiration and common spelling variants are ignored."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z ]+", " ", s)
    for a, b in (("aa", "a"), ("ee", "i"), ("ii", "i"), ("oo", "u"), ("uu", "u"), ("ai", "ai"),
                 ("zh", "l"), ("sh", "s"), ("ch", "c"), ("kh", "k"), ("gh", "g"), ("th", "t"),
                 ("dh", "d"), ("ph", "p"), ("bh", "b"), ("w", "v"), ("ou", "au"), ("ksh", "ks")):
        s = s.replace(a, b)
    s = re.sub(r"(.)\1+", r"\1", s)                 # doubled consonants
    return re.sub(r"\s+", "", s)


def sim(a: str, b: str) -> float:
    """Similarity 0..1 of two titles after ``norm``: the better of the whole
    string ratio and (slightly discounted) a prefix ratio, because Saraga titles
    are often just the first words of the opening line.
    """
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    full = SequenceMatcher(None, a, b).ratio()
    n = min(len(a), len(b))                          # titles are often a prefix of the opening line
    prefix = SequenceMatcher(None, a[:n], b[:n]).ratio() if n >= 6 else 0.0
    return max(full, 0.95 * prefix)


def karnatik_index(f: Fetcher) -> List[dict]:
    """karnatik.com's song index as [{url, title, raga, composer}], parsed
    from the ``<OPTION>`` list on lyrics.shtml (one request, cached).
    """
    page = f.get(KARNATIK + "lyrics.shtml", encoding="latin-1")
    out = []
    for url, label in re.findall(r'<OPTION VALUE="(c\d+\.shtml)">([^<\n]+)', page):
        parts = [p.strip() for p in html.unescape(label).split(" - ")]
        title = parts[0]
        raga = parts[1] if len(parts) > 2 else ""
        composer = parts[-1] if len(parts) > 1 else ""
        out.append({"url": KARNATIK + url, "title": title, "raga": raga, "composer": composer})
    return out


# raga names that differ between Saraga/MusicBrainz and karnatik.com
RAGA_ALIASES = [("sanmukhapriya", "camaram"), ("nata", "nattai"), ("harikambhoji", "harikambodi"),
                ("kamavardani", "pantuvarali"), ("sankarabharanam", "dhirasankarabharanam"),
                ("todi", "hanumatodi"), ("mayamalavagaula", "malavagaula"), ("sriranjani", "sri ranjani"),
                ("simhendramadhyamam", "simhendra madhyamam"), ("hamsadhvani", "hamsadhwani"),
                ("karnataka devagandhari", "abheri")]


def raga_sim(a: str, b: str) -> float:
    """``sim`` for raga names, also trying the known spelling aliases."""
    best = sim(a, b)
    for x, y in RAGA_ALIASES:
        for p, q in ((x, y), (y, x)):
            if sim(a, p) > 0.85:
                best = max(best, sim(q, b))
    return best


def match(index: List[dict], title: str, raga: str = "", composer: str = "", k: int = 5) -> List[dict]:
    """Top ``k`` index entries for a work, scored 0.7 title + 0.2 raga + 0.1
    composer similarity (0.5 when a field is missing on either side).
    """
    t0 = norm(title)[:4]
    scored = []
    for e in index:
        if t0 and norm(e["title"])[:2] != t0[:2]:   # cheap prefilter on the first letters
            continue
        ts = sim(title, e["title"])
        if ts < 0.6:
            continue
        rs = max((raga_sim(r, e["raga"]) for r in raga.split("/")), default=0.5) if raga and e["raga"] else 0.5
        cs = sim(composer, e["composer"]) if composer and e["composer"] else 0.5
        scored.append({**e, "score": round(0.7 * ts + 0.2 * rs + 0.1 * cs, 3),
                       "title_sim": round(ts, 3), "raga_sim": round(rs, 3), "composer_sim": round(cs, 3)})
    return sorted(scored, key=lambda e: -e["score"])[:k]


# ----------------------------------------------------------------------------- parsing

def _text(fragment: str) -> List[str]:
    """Visible text lines of an HTML fragment (``<br>`` = new line)."""
    fragment = re.sub(r"(?i)<br\s*/?>", "\n", fragment)
    fragment = re.sub(r"(?s)<[^>]+>", " ", fragment)
    lines = [re.sub(r"\s+", " ", html.unescape(l)).strip() for l in fragment.split("\n")]
    return [l for l in lines if l]


def parse_karnatik(page: str) -> dict:
    """Header fields and lyric sections of a karnatik.com song page."""
    out = {"raga": "", "tala": "", "composer": "", "language": "", "sections": []}
    # the lyric region starts at the song header; earlier text is navigation / scripts
    i = page.find("raagam:")
    page = page[i:] if i >= 0 else page[page.find("Song:"):]
    m = re.search(r"(?is)raagam:\s*(?:<a[^>]*>)?([^<\n]+)", page)
    out["raga"] = m.group(1).strip() if m else ""
    m = re.search(r"(?is)taaLam:\s*([^<\n]+)", page)
    out["tala"] = m.group(1).strip() if m else ""
    m = re.search(r"(?is)Composer:\s*(?:<a[^>]*>)?([^<\n]+)", page)
    out["composer"] = m.group(1).strip() if m else ""
    m = re.search(r"(?s)Language:\s*([^<\n]+)", page)
    out["language"] = m.group(1).strip() if m else ""
    start = m.end() if m else 0          # lyrics follow the "Language:" header line ...
    end = page.find("<hr>", start)       # ... up to the next rule (meaning, notation follow)
    body = page[start:end if end > 0 else None]
    blocks = re.split(r"(?i)<p>", body)
    sections, current = [], None
    head_key = lambda l: re.sub(r"\s+", "", l.strip().rstrip(":").lower()).replace("sw", "sv")
    names = {head_key(n) for n in SECTION_NAMES}           # spaces and sw/sv spellings ignored
    is_head = lambda l: (head_key(l) in names
                         or re.fullmatch(r"(caraNam|charanam)\s*\d*:?", l.strip(), re.I) is not None)
    # section headers ("pallavi", "caraNam 2", ...) may appear on any line, not only
    # at the start of a <p> block
    for blk in blocks:
        for l in _text(blk):
            if is_head(l):
                current = {"name": l.strip().rstrip(":"), "lines": []}
                sections.append(current)
                continue
            if current is None:
                current = {"name": "pallavi", "lines": []}
                sections.append(current)
            if re.fullmatch(r"\(.{1,30}\)", l):        # "(kOTi)" = repeat the pallavi
                current.setdefault("repeat_markers", []).append(l.strip("()"))
                continue
            if l.lower().startswith(("meaning", "other information", "notation", "variations")):
                continue
            if l.strip().lower() in LANGUAGES:              # header text spilling into the lyrics
                continue
            if re.match(r"(?i)^\s*(aa|av|aro|ava|arohanam|avarohanam)\s*:", l):
                continue                                  # the raga's scale ("Aa: S R2 G2 ..."), not lyrics
            if len(l.split()) == 1 and current["lines"]:
                # a one-word "line" is the wrapped tail of the previous one
                # ("... tyAgarAja" / "sarvasvamau"); kriti lines are never one word
                current["lines"][-1] += " " + l
                continue
            current["lines"].append(l)
    out["sections"] = [s for s in sections if s["lines"]]
    return out


# ----------------------------------------------------------------------------- shivkumar.org

def shivkumar_index(f: Fetcher) -> List[dict]:
    """shivkumar.org's krithi archive as [{url, title, raga, composer, site}],
    parsed from the index rows ``<li><b> Title: <a>Raga</a>; Tala; Composer ...``
    and their notation-page link."""
    page = f.get(SHIVKUMAR + "index.html", encoding="latin-1")
    out = []
    row = re.compile(r"(?is)<li>\s*<b>\s*([^:<]+?)\s*:\s*<a[^>]*>\s*([^<]+?)\s*</a>\s*;([^<]*?)</b>(.*?)(?=<li>|$)")
    for title, raga, rest, tail in row.findall(page):
        link = re.search(r'href="\./([^"]+\.htm)"', tail)
        if not link:
            continue
        parts = [x.strip() for x in rest.split(";")]
        out.append({"url": SHIVKUMAR + link.group(1), "title": html.unescape(title).strip(),
                    "raga": html.unescape(raga).strip(), "composer": parts[1] if len(parts) > 1 else "",
                    "site": "shivkumar"})
    return out


SHIVKUMAR_HEAD = re.compile(r"(?i)^(pallavi|anupallavi|charanam|caranam|madhyama\s*kala(?:m)?|chittaswaram|"
                            r"samashti\s*charanam)\s*(\d*)\s*:?\s*(.*)$")


def parse_shivkumar(page: str) -> dict:
    """Lyric sections of a shivkumar.org notation page: the block from the first
    section header ("Pallavi :") up to "Meaning" or the first notation line.
    The site spells in plain English-style capitals (``Adamodi Galade``), which
    carry no length / retroflex information, so the lines are lower-cased."""
    text = re.sub(r"(?i)<br\s*/?>|</p>|<p[^>]*>|</tr>|</div>", "\n", page)
    text = html.unescape(re.sub(r"(?s)<[^>]+>", " ", text))
    lines = [l for l in (re.sub(r"\s+", " ", x).strip() for x in text.splitlines()) if l]
    out = {"raga": "", "tala": "", "composer": "", "language": "", "sections": []}
    for l in lines[:80]:
        m = re.match(r"(?i)(ragam|talam|composer)\s*:\s*(.+)", l)
        if m:
            out[{"ragam": "raga", "talam": "tala", "composer": "composer"}[m.group(1).lower()]] = m.group(2).strip()
    sections, current, started = [], None, False
    for l in lines:
        if started and (l.lower().startswith("meaning") or re.search(r"[;|]", l)):
            break                                     # the meaning / notation part follows the lyrics
        m = SHIVKUMAR_HEAD.match(l)
        if m:
            started = True
            name = m.group(1).lower()
            name = "caraNam" if name in ("charanam", "caranam") else name
            current = {"name": (name + " " + m.group(2)).strip(), "lines": []}
            sections.append(current)
            if m.group(3):
                current["lines"].append(m.group(3).lower())
            continue
        if not started:
            continue
        l = l.lower()
        if len(l.split()) == 1 and current["lines"]:  # wrapped tail of the previous line
            current["lines"][-1] += " " + l
        else:
            current["lines"].append(l)
    out["sections"] = [x for x in sections if x["lines"]]
    return out


# ----------------------------------------------------------------------------- MusicBrainz

def musicbrainz_work(f: Fetcher, mbid: str) -> dict:
    """Title, composer, lyricist, languages, aliases, raga and tala of a
    MusicBrainz work (one cached API request).
    """
    data = json.loads(f.get(f"{MUSICBRAINZ}work/{mbid}?fmt=json&inc=artist-rels+aliases"))
    rel = lambda t: [r["artist"]["name"] for r in data.get("relations", []) if r.get("type") == t]
    attrs = {a.get("type", ""): a.get("value", "") for a in data.get("attributes", [])}
    return {"title": data.get("title", ""), "composer": rel("composer"), "lyricist": rel("lyricist"),
            "languages": data.get("languages", []), "aliases": [a["name"] for a in data.get("aliases", [])],
            "raga": attrs.get("Rāga (Carnatic)", ""), "tala": attrs.get("Tāla (Carnatic)", "")}


# ----------------------------------------------------------------------------- driver

def fetch_work(f: Fetcher, index: List[dict], work_mbid: str, title: str, raga: str,
               out_dir: Path, n_pages: int = 3, second_index: Optional[List[dict]] = None) -> dict:
    """Build or refresh ``<out_dir>/<work_mbid>.yaml`` for one composition.

    Verified files are returned untouched.  Otherwise: MusicBrainz metadata, a
    karnatik.com search under every known title alias, the top ``n_pages``
    candidate pages parsed, and ``auto_select`` applied.  When nothing is
    auto-selected and ``second_index`` (shivkumar.org) is given, its two best
    candidates scoring at least 0.75 are added and selection is tried again.
    """
    out = out_dir / f"{work_mbid}.yaml"
    if out.exists():
        doc = yaml.safe_load(out.read_text(encoding="utf-8"))
        if doc.get("verified"):
            return doc
    mb = musicbrainz_work(f, work_mbid) if work_mbid else {}
    composer = (mb.get("composer") or [""])[0]
    names = [title] + [a for a in mb.get("aliases", []) if a != title]
    cands: Dict[str, dict] = {}
    for name in names:
        for c in match(index, name, raga or mb.get("raga", ""), composer):
            if c["url"] not in cands or c["score"] > cands[c["url"]]["score"]:
                cands[c["url"]] = c
    ranked = sorted(cands.values(), key=lambda c: -c["score"])
    for c in ranked[:n_pages]:
        c.setdefault("site", "karnatik")
        try:
            c["parsed"] = parse_karnatik(f.get(c["url"], encoding="latin-1"))
        except Exception as e:                       # keep the candidate, note the failure
            c["error"] = repr(e)
    doc = auto_select({"work_mbid": work_mbid, "title": title, "raga": raga, "musicbrainz": mb,
                       "candidates": ranked, "selected": None, "verified": False,
                       "source": None, "lyrics": None, "notes": ""})
    if not doc.get("selected") and second_index:
        extra: Dict[str, dict] = {}
        for name in names:
            for c in match(second_index, name, raga or mb.get("raga", ""), composer, k=2):
                if c["url"] not in extra or c["score"] > extra[c["url"]]["score"]:
                    extra[c["url"]] = c
        for c in sorted(extra.values(), key=lambda c: -c["score"])[:2]:
            if c["score"] < 0.75:                    # do not spend 10 s requests on poor matches
                continue
            try:
                c["parsed"] = parse_shivkumar(f.get(c["url"], encoding="latin-1"))
            except Exception as e:
                c["error"] = repr(e)
            doc["candidates"].append(c)
        doc["candidates"].sort(key=lambda c: -c["score"])
        doc = auto_select(doc)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
    return doc


AUTO_SCORE = 0.9             # combined match score needed for auto-selection ...
AUTO_RAGA = 0.85             # ... and raga similarity (same title, other raga = other song)


def parse_lyrics_text(text: str) -> List[dict]:
    """Sections from typed or pasted lyrics: a line ending in ``:`` starts a
    section ("pallavi:"), other lines are its lines, ``#`` starts a comment;
    lines before any header belong to the pallavi."""
    sections, cur = [], None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(":"):
            cur = {"name": line[:-1].strip(), "lines": []}
            sections.append(cur)
            continue
        if cur is None:
            cur = {"name": "pallavi", "lines": []}
            sections.append(cur)
        cur["lines"].append(line)
    return [s for s in sections if s["lines"]]


DECISIONS = ("candidate", "text", "wrong", "missing")


def apply_decision(doc: dict, decision: str, reviewer: str = "", candidate_url: str = "", text: str = "",
                   source: str = "", notes: str = "") -> dict:
    """Record a person's decision on a work's lyrics (in place; returns ``doc``).

    * ``candidate``: the candidate at ``candidate_url`` is right -> verified
    * ``text``: lyrics typed or pasted (``parse_lyrics_text``), from ``source`` -> verified
    * ``wrong``: the auto-selected candidate is not this song -> unselected, unverified
    * ``missing``: no candidate is right and no text yet -> stays unverified, noted
    """
    if decision not in DECISIONS:
        raise ValueError(f"unknown decision {decision!r}; expected one of {DECISIONS}")
    if decision == "candidate":
        c = next((c for c in doc.get("candidates", []) if c["url"] == candidate_url), None)
        if c is None or not (c.get("parsed") or {}).get("sections"):
            raise ValueError(f"{doc.get('title')}: no parsed candidate {candidate_url!r}")
        doc.update(selected=c["url"], source=c.get("site", "karnatik"), lyrics=c["parsed"]["sections"],
                   verified=True)
    elif decision == "text":
        sections = parse_lyrics_text(text)
        if not sections:
            raise ValueError(f"{doc.get('title')}: the pasted lyrics have no lines")
        doc.update(selected=source or "manual", source="manual", lyrics=sections, verified=True)
    elif decision == "wrong":
        doc.update(selected=None, source=None, lyrics=None, verified=False)
    doc["review"] = {"decision": decision, "reviewer": reviewer, "notes": notes}
    if decision in ("candidate", "text"):
        doc["verified_by"] = reviewer
    if notes:
        doc["notes"] = notes
    return doc


STATUS_HEADER = ["work_mbid", "title", "raga", "composer", "language", "status", "source",
                 "best_title", "best_raga", "score", "url"]


def work_status(doc: dict) -> str:
    """``verified`` (a person confirmed it), ``auto`` (pre-selected, unconfirmed)
    or ``missing`` (no usable lyrics yet)."""
    if doc.get("verified"):
        return "verified"
    return "auto" if doc.get("selected") else "missing"


def write_status(lyrics_dir: Path) -> Dict[str, int]:
    """Rebuild ``<lyrics_dir>/status.tsv`` from *every* work YAML in the folder
    (not just the works a run touched, so a partial ``lyrics-fetch`` never
    shrinks the table).  Returns the count per status."""
    lyrics_dir = Path(lyrics_dir)
    rows, counts = [], {}
    for path in sorted(lyrics_dir.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        mb = doc.get("musicbrainz") or {}
        best = (doc.get("candidates") or [{}])[0]
        status = work_status(doc)
        counts[status] = counts.get(status, 0) + 1
        rows.append([doc.get("work_mbid", path.stem), doc.get("title", ""), doc.get("raga", ""),
                     "; ".join(mb.get("composer") or []), " ".join(mb.get("languages") or []), status,
                     doc.get("source") or "", best.get("title", ""), best.get("raga", ""),
                     best.get("score", ""), best.get("url", "")])
    with open(lyrics_dir / "status.tsv", "w", encoding="utf-8") as fh:
        fh.write("\t".join(STATUS_HEADER) + "\n")
        for row in rows:
            fh.write("\t".join(str(x).replace("\t", " ") for x in row) + "\n")
    return counts


def auto_select(doc: dict) -> dict:
    """Pre-select a candidate whose title and raga match well; it still needs a
    human ``verified: true``."""
    for c in doc.get("candidates", []):
        if c["score"] >= AUTO_SCORE and c["raga_sim"] >= AUTO_RAGA and c.get("parsed", {}).get("sections"):
            doc["selected"] = c["url"]
            doc["source"] = "auto"
            doc["site"] = c.get("site", "karnatik")
            doc["lyrics"] = c["parsed"]["sections"]
            return doc
    return doc
