"""Grapheme-to-phoneme and syllabification for Hindi, Marathi (Devanagari) and Tamil.

Rule-based, deterministic, and overridable: any word found in a lexicon file is
taken from the lexicon instead of the rules.  The rules cover the cases that
matter most for singing; the lexicon is where reviewers record sung
pronunciations (retained schwas, loanwords, Marathi च/ज exceptions, Tamil
allophony exceptions).

Lexicon format (UTF-8 TSV, '#' starts a comment)::

    मन	m a . n a          # sung variant: schwa kept
    சந்தோஷம்	s a n . d oo . Sh a m

Phones are names from :mod:`saregama.phoneset`; '.' marks syllable boundaries.

Every phone remembers which written akshara it came from, so each syllable also
gets its native-script text (used for the ``text`` field and in review tools).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .phoneset import BY_NAME, CONSONANTS, VOWELS

LANGS = ("hi", "mr", "ta")
Ph = Tuple[str, int]          # (phone name, akshara index)


@dataclass
class Syllable:
    text: str                 # native-script text of the syllable
    phones: List[str]         # phone names
    word_index: int           # index of the word in its line
    word_final: bool = False

    @property
    def nucleus(self) -> Optional[str]:
        for p in self.phones:
            if p in VOWELS:
                return p
        return None


# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------
class Lexicon:
    def __init__(self) -> None:
        self.entries: Dict[str, List[List[str]]] = {}

    def add(self, word: str, syllables: List[List[str]]) -> None:
        for syl in syllables:
            for p in syl:
                if p not in BY_NAME:
                    raise KeyError(f"lexicon entry {word!r}: unknown phone {p!r}")
        self.entries[unicodedata.normalize("NFC", word)] = syllables

    def get(self, word: str) -> Optional[List[List[str]]]:
        return self.entries.get(unicodedata.normalize("NFC", word))

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def load(cls, *paths) -> "Lexicon":
        lex = cls()
        for path in paths:
            if not path or not Path(path).exists():
                continue
            for raw in Path(path).read_text(encoding="utf-8").splitlines():
                line = raw.split("#", 1)[0].strip()
                if not line:
                    continue
                word, _, pron = line.partition("\t")
                if not pron:
                    raise ValueError(f"{path}: expected 'word<TAB>phones': {raw!r}")
                syls = [s.split() for s in pron.split(".")]
                lex.add(word.strip(), [s for s in syls if s])
        return lex


def _is_matra(ch: str) -> bool:
    return unicodedata.category(ch) in ("Mn", "Mc")


# ---------------------------------------------------------------------------
# Devanagari (Hindi, Marathi)
# ---------------------------------------------------------------------------
VIRAMA_D = "्"
NUKTA_D = "़"
ANUSVARA_D = "ं"
CHANDRABINDU_D = "ँ"
VISARGA_D = "ः"

DEV_CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "c", "छ": "ch", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "T", "ठ": "Th", "ड": "D", "ढ": "Dh", "ण": "N",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "sh", "ष": "Sh", "स": "s", "ह": "h",
    "ळ": "L", "ऴ": "zh", "ऩ": "n", "ऱ": "r",
}
DEV_NUKTA = {"k": "q", "kh": "x", "g": "gg", "j": "z", "ph": "f", "D": "R", "Dh": "Rh"}

DEV_VOWELS_COMMON = {
    "अ": ["a"], "आ": ["aa"], "इ": ["i"], "ई": ["ii"], "उ": ["u"], "ऊ": ["uu"],
    "ए": ["ee"], "ओ": ["oo"], "ऍ": ["ae"], "ॲ": ["ae"], "ऑ": ["O"], "ऎ": ["e"], "ऒ": ["o"],
    "ा": ["aa"], "ि": ["i"], "ी": ["ii"], "ु": ["u"], "ू": ["uu"],
    "े": ["ee"], "ो": ["oo"], "ॅ": ["ae"], "ॉ": ["O"], "ॆ": ["e"], "ॊ": ["o"],
}
DEV_VOWELS_HI = {"ऐ": ["E"], "औ": ["O"], "ै": ["E"], "ौ": ["O"],
                 "ऋ": ["r", "i"], "ृ": ["r", "i"], "ॠ": ["r", "ii"], "ॄ": ["r", "ii"]}
DEV_VOWELS_MR = {"ऐ": ["a", "i"], "औ": ["a", "u"], "ै": ["a", "i"], "ौ": ["a", "u"],
                 "ऋ": ["r", "u"], "ृ": ["r", "u"], "ॠ": ["r", "uu"], "ॄ": ["r", "uu"]}

HOMORGANIC_NASAL: Dict[str, str] = {}
for _grp, _nas in ((("k", "kh", "g", "gh", "q", "x", "gg"), "ng"),
                   (("c", "ch", "j", "jh", "ts", "dz", "z"), "ny"),
                   (("T", "Th", "D", "Dh", "R", "Rh"), "N"),
                   (("t", "th", "d", "dh"), "n"),
                   (("p", "ph", "b", "bh", "f"), "m")):
    for _c in _grp:
        HOMORGANIC_NASAL[_c] = _nas

BACK_VOWELS = {"a", "aa", "u", "uu", "o", "oo", "O"}


@dataclass
class _Seg:
    phone: str
    src: int
    inherent: bool = False      # an inherent schwa (deletion candidate)


def _dev_parse(word: str, lang: str) -> Tuple[List[_Seg], List[str]]:
    """Return phone segments and the akshara strings they index into."""
    vowels = dict(DEV_VOWELS_COMMON)
    vowels.update(DEV_VOWELS_HI if lang == "hi" else DEV_VOWELS_MR)
    w = unicodedata.normalize("NFC", word)     # NFC splits क़ into क + ़
    segs: List[_Seg] = []
    aks: List[str] = []
    i, n = 0, len(w)
    prev_virama = False
    while i < n:
        ch = w[i]
        start = i
        if ch in DEV_CONS:
            aks.append("")                       # one text unit per consonant
            ak = len(aks) - 1
            if ch == "ज" and w[i + 1:i + 3] == VIRAMA_D + "ञ":
                segs += ([_Seg("g", ak), _Seg("y", ak)] if lang == "hi"
                         else [_Seg("d", ak), _Seg("ny", ak)])
                i += 3
            else:
                ph = DEV_CONS[ch]
                if lang == "hi" and ph == "Sh":
                    ph = "sh"                        # modern Hindi ष ≈ श
                i += 1
                if i < n and w[i] == NUKTA_D:
                    ph = DEV_NUKTA.get(ph, ph)
                    i += 1
                segs.append(_Seg(ph, ak))
            prev_virama = False
            if i < n and w[i] == VIRAMA_D:
                i += 1
                prev_virama = True
            elif i < n and w[i] in vowels and _is_matra(w[i]):
                segs += [_Seg(p, ak) for p in vowels[w[i]]]
                i += 1
            else:
                segs.append(_Seg("a", ak, inherent=True))
        elif ch in vowels and not _is_matra(ch):
            aks.append("")
            segs += [_Seg(p, len(aks) - 1) for p in vowels[ch]]
            i += 1
            prev_virama = False
        elif ch in (ANUSVARA_D, CHANDRABINDU_D, VISARGA_D) and aks:
            ph = {"ं": "<anusvara>", "ँ": "nas", "ः": "h"}[ch]
            segs.append(_Seg(ph, len(aks) - 1))
            i += 1
        else:
            if aks:                                   # ZWJ, avagraha, etc.
                aks[-1] += ch
            i += 1
            continue
        aks[-1] += w[start:i]
    return segs, aks


def _delete_schwas(segs: List[_Seg], medial: bool, keep_final: bool) -> List[_Seg]:
    """Schwa deletion: word-final rule plus right-to-left V C _ C V rule."""

    def is_v(k: int) -> bool:
        return 0 <= k < len(segs) and segs[k].phone in VOWELS

    def is_c(k: int) -> bool:
        return 0 <= k < len(segs) and segs[k].phone in CONSONANTS

    drop = set()
    last = len(segs) - 1
    n_vowels = sum(1 for s in segs if s.phone in VOWELS)
    if not keep_final and last >= 1 and segs[last].inherent and n_vowels > 1:
        cluster, k = 0, last - 1
        while is_c(k):
            cluster += 1
            k -= 1
        # keep final schwa after clusters ending in y/r/v: सत्य, मित्र, पत्र
        if not (cluster >= 2 and segs[last - 1].phone in ("y", "r", "v")):
            drop.add(last)
    if medial:
        for k in range(last - 1, 0, -1):
            if not segs[k].inherent:
                continue
            nxt = [j for j in range(k + 1, len(segs)) if j not in drop]
            prv = [j for j in range(k - 1, -1, -1) if j not in drop]
            if len(nxt) < 2 or len(prv) < 2:
                continue
            c1, v0, c2, v2 = prv[0], prv[1], nxt[0], nxt[1]
            if is_c(c1) and is_v(v0) and v0 + 1 == c1 and is_c(c2) and is_v(v2):
                drop.add(k)
    return [s for k, s in enumerate(segs) if k not in drop]


def _resolve_anusvara(segs: List[_Seg]) -> List[Ph]:
    out: List[Ph] = []
    for k, s in enumerate(segs):
        if s.phone != "<anusvara>":
            out.append((s.phone, s.src))
            continue
        nxt = segs[k + 1].phone if k + 1 < len(segs) else None
        if nxt in HOMORGANIC_NASAL:
            out.append((HOMORGANIC_NASAL[nxt], s.src))
        elif nxt == "m":
            out.append(("m", s.src))
        else:
            out.append(("nas", s.src))
    return out


def devanagari_word(word: str, lang: str, *, keep_final_schwa: bool = False,
                    delete_medial_schwa: bool = True,
                    marathi_affricates: bool = True) -> Tuple[List[Ph], List[str]]:
    segs, aks = _dev_parse(word, lang)
    segs = _delete_schwas(segs, delete_medial_schwa, keep_final_schwa)
    phones = _resolve_anusvara(segs)
    if lang == "mr" and marathi_affricates:
        phones = [(("ts" if p == "c" else "dz"), s)
                  if p in ("c", "j") and k + 1 < len(phones) and phones[k + 1][0] in BACK_VOWELS
                  else (p, s) for k, (p, s) in enumerate(phones)]
    return _clean(phones), aks


# ---------------------------------------------------------------------------
# Tamil
# ---------------------------------------------------------------------------
PULLI = "்"
AYTHAM = "ஃ"
TA_CONS = {
    "க": "K", "ங": "ng", "ச": "C", "ஞ": "ny", "ட": "TT", "ண": "N",
    "த": "TD", "ந": "n", "ப": "P", "ம": "m", "ய": "y", "ர": "r",
    "ல": "l", "வ": "v", "ழ": "zh", "ள": "L", "ற": "RR", "ன": "nn",
    "ஜ": "j", "ஷ": "Sh", "ஸ": "s", "ஹ": "h", "ஶ": "sh",
}
TA_VOWELS = {
    "அ": ["a"], "ஆ": ["aa"], "இ": ["i"], "ஈ": ["ii"], "உ": ["u"], "ஊ": ["uu"],
    "எ": ["e"], "ஏ": ["ee"], "ஐ": ["a", "i"], "ஒ": ["o"], "ஓ": ["oo"], "ஔ": ["a", "u"],
    "ா": ["aa"], "ி": ["i"], "ீ": ["ii"], "ு": ["u"], "ூ": ["uu"],
    "ெ": ["e"], "ே": ["ee"], "ை": ["a", "i"], "ொ": ["o"], "ோ": ["oo"], "ௌ": ["a", "u"],
}
# Stop archiphonemes: (word-initial/geminate/after another stop, after a nasal,
# after a vowel or sonorant).  Tamil script does not mark voicing.
TA_ARCHI = {"K": ("k", "g", "g"), "C": ("c", "j", "s"), "TT": ("T", "D", "D"),
            "TD": ("t", "d", "d"), "P": ("p", "b", "b"), "RR": ("rr", "rr", "rr")}
NASALS = {"ng", "ny", "N", "n", "m", "nn"}


def tamil_word(word: str, *, initial_c_as_s: bool = True,
               intervocalic_k: str = "g") -> Tuple[List[Ph], List[str]]:
    w = unicodedata.normalize("NFC", word)
    raw: List[List] = []
    aks: List[str] = []
    i, n = 0, len(w)
    while i < n:
        ch, start = w[i], i
        if ch == AYTHAM and i + 1 < n and w[i + 1] == "ப":       # ஃப = f
            aks.append("")
            raw.append(["f", len(aks) - 1])
            i += 2
            if i < n and w[i] == PULLI:
                i += 1
            elif i < n and w[i] in TA_VOWELS and _is_matra(w[i]):
                raw += [[p, len(aks) - 1] for p in TA_VOWELS[w[i]]]
                i += 1
            else:
                raw.append(["a", len(aks) - 1])
        elif ch in TA_CONS:
            aks.append("")
            ak = len(aks) - 1
            if ch == "க" and w[i + 1:i + 3] == PULLI + "ஷ":
                raw += [["k", ak], ["Sh", ak]]
                i += 3
            else:
                raw.append([TA_CONS[ch], ak])
                i += 1
            if i < n and w[i] == PULLI:
                i += 1
            elif i < n and w[i] in TA_VOWELS and _is_matra(w[i]):
                raw += [[p, ak] for p in TA_VOWELS[w[i]]]
                i += 1
            else:
                raw.append(["a", ak])
        elif ch in TA_VOWELS and not _is_matra(ch):
            aks.append("")
            raw += [[p, len(aks) - 1] for p in TA_VOWELS[ch]]
            i += 1
        else:
            if aks:
                aks[-1] += ch
            i += 1
            continue
        aks[-1] += w[start:i]

    out: List[Ph] = []
    for k, (p, s) in enumerate(raw):
        if p not in TA_ARCHI:
            out.append((p, s))
            continue
        voiceless, after_nasal, interv = TA_ARCHI[p]
        prev = raw[k - 1][0] if k > 0 else None
        nxt = raw[k + 1][0] if k + 1 < len(raw) else None
        if prev is None:
            ph = "s" if (p == "C" and initial_c_as_s) else voiceless
        elif prev == p or nxt == p or prev in TA_ARCHI:
            ph = voiceless
        elif prev in NASALS:
            ph = after_nasal
        else:
            ph = intervocalic_k if p == "K" else interv
        if p == "RR" and prev == "RR":                       # ற்ற -> t r
            out[-1] = ("t", out[-1][1])
            ph = "rr"
        elif p == "RR" and prev == "nn":                     # ன்ற -> n d r
            out[-1] = ("n", out[-1][1])
            out.append(("d", s))
            ph = "r"
        out.append((ph, s))
    return _clean(out), aks


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------
def _clean(phones: List[Ph]) -> List[Ph]:
    out: List[Ph] = []
    for p, s in phones:
        if p not in BY_NAME:
            raise KeyError(f"G2P produced unknown phone {p!r}")
        if p == "nas" and (not out or out[-1][0] not in VOWELS):
            continue
        out.append((p, s))
    return out


def syllabify(phones: Sequence[str], srcs: Optional[Sequence[int]] = None) -> List[List[int]]:
    """Split a word's phones into syllables; returns lists of phone indices.

    A vowel (plus a following 'nas') is a nucleus; two adjacent vowels from the
    same written letter (Marathi ऐ, Tamil ஐ) form one diphthong nucleus.  Between
    nuclei a single consonant starts the next syllable; with two or more, the
    first closes the previous one (geminates split k|k).  Word-initial consonants
    join the first syllable and word-final ones the last.
    """
    srcs = list(srcs) if srcs is not None else list(range(len(phones)))
    nuclei: List[Tuple[int, int]] = []          # (first index, last index)
    k = 0
    while k < len(phones):
        if phones[k] in VOWELS:
            j = k
            if j + 1 < len(phones) and phones[j + 1] in VOWELS and srcs[j + 1] == srcs[k]:
                j += 1
            if j + 1 < len(phones) and phones[j + 1] == "nas":
                j += 1
            nuclei.append((k, j))
            k = j + 1
        else:
            k += 1
    if not nuclei:
        return [list(range(len(phones)))] if phones else []
    out: List[List[int]] = []
    start = 0
    for idx, (_, nend) in enumerate(nuclei):
        if idx + 1 < len(nuclei):
            nxt = nuclei[idx + 1][0]
            cluster = nxt - (nend + 1)
            cut = nend + 1 if cluster <= 1 else nend + 2
            out.append(list(range(start, cut)))
            start = cut
        else:
            out.append(list(range(start, len(phones))))
    return out


_DEV_RANGE = re.compile(r"[ऀ-ॿ]")
_TA_RANGE = re.compile(r"[஀-௿]")
_WORD_SPLIT = re.compile(r"[\s।॥,;:!?\.\"'()\[\]{}\-–—…|]+")


def split_words(line: str) -> List[str]:
    return [w for w in _WORD_SPLIT.split(unicodedata.normalize("NFC", line)) if w]


def word_to_syllables(word: str, lang: str, lexicon: Optional[Lexicon] = None,
                      **opts) -> List[Tuple[str, List[str]]]:
    """One written word -> [(syllable text, phones), ...]."""
    entry = lexicon.get(word) if lexicon else None
    if entry is not None:
        return [(word if k == 0 else "~", list(s)) for k, s in enumerate(entry)]
    if _TA_RANGE.search(word):
        phs, aks = tamil_word(word, **{k: v for k, v in opts.items()
                                       if k in ("initial_c_as_s", "intervocalic_k")})
    elif _DEV_RANGE.search(word):
        phs, aks = devanagari_word(word, "mr" if lang == "mr" else "hi",
                                   **{k: v for k, v in opts.items()
                                      if k in ("keep_final_schwa", "delete_medial_schwa",
                                               "marathi_affricates")})
    else:
        raise ValueError(f"word {word!r} is not Devanagari or Tamil; add it to the lexicon")
    names = [p for p, _ in phs]
    srcs = [s for _, s in phs]
    groups = syllabify(names, srcs)
    # each akshara goes to the syllable holding its first surviving phone
    ak_to_syl: Dict[int, int] = {}
    for si, g in enumerate(groups):
        for pi in g:
            ak_to_syl.setdefault(srcs[pi], si)
    texts = [""] * len(groups)
    cur = 0
    for ai, ak in enumerate(aks):
        cur = ak_to_syl.get(ai, cur)
        texts[cur] += ak
    return [(t or "~", [names[i] for i in g]) for t, g in zip(texts, groups)]


def line_to_syllables(line: str, lang: str, lexicon: Optional[Lexicon] = None,
                      **opts) -> List[Syllable]:
    """Lyrics line (native script) -> syllables with phones and text."""
    if lang not in LANGS:
        raise ValueError(f"lang must be one of {LANGS}, got {lang!r}")
    out: List[Syllable] = []
    for wi, word in enumerate(split_words(line)):
        sylls = word_to_syllables(word, lang, lexicon, **opts)
        for si, (txt, ph) in enumerate(sylls):
            out.append(Syllable(text=txt, phones=ph, word_index=wi,
                                word_final=(si == len(sylls) - 1)))
    return out


def lyrics_to_syllables(lines: Iterable[str], lang: str, lexicon: Optional[Lexicon] = None,
                        **opts) -> List[List[Syllable]]:
    return [line_to_syllables(l, lang, lexicon, **opts) for l in lines if l.strip()]
