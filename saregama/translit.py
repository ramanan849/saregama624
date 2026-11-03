"""karnatik.com transliteration -> ISO-15919 text and Indic phones.

karnatik.com writes Telugu / Sanskrit / Kannada / Tamil lyrics in a
Harvard-Kyoto-like ASCII scheme, with spelling variants from other sources
mixed in.  This module turns each word into

* ISO-15919 text (``dAsuDanE`` -> ``dāsuḍanē``) for the IR and for people, and
* phones of ``saregama.phoneset`` (``d aa s u D a n ee``) for SoulX-Singer,

in one greedy pass over the scheme's units (longest match first):

======================  ==========  =============================================
karnatik                ISO-15919   phones
======================  ==========  =============================================
A / aa, I / ii / ee,    ā ī ū       aa ii uu      (ee / oo: Tamil-style spellings)
U / uu / oo
E, O                    ē ō         ee oo         (e, o are the short vowels)
ai, au                  ai au       ai au         (diphthongs)
T Th D Dh N L           ṭ ṭh ḍ ḍh   T Th D Dh N L
                        ṇ ḷ
sh, S, kS / ksh         ś ṣ kṣ      sh Sh, k Sh
zh, R                   ḻ ṟ         zh rr         (Tamil)
jn, ny, ng, G           jñ ñ ṅg ṅ   j ny, ny, ng g, ng
M, H                    ṁ ḥ         homorganic nasal (m before a vowel / word end), h
r between consonants    ṛ           r u           (sung "ru" in the South)
======================  ==========  =============================================

Other capitals (``Bhakthi``, ``Jaya``) are just capitalised plain letters.
Hyphens (sandhi marks) and punctuation are dropped.  Unknown letters raise
``ValueError`` instead of producing a wrong phone silently.

Svara notation lines (``pdsrsmpmdpSndRS,|``, ``S , , R G``) are not words:
``is_notation`` spots them and ``notation_syllables`` turns them into the
sargam syllables that are actually sung.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Sequence, Tuple

from .phoneset import CONSONANTS, VOWELS

# (karnatik unit, ISO text, phones, kind) -- longest units first within each kind
_VOWELS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("ai", "ai", ("ai",)), ("au", "au", ("au",)),
    ("aa", "ā", ("aa",)), ("ii", "ī", ("ii",)), ("ee", "ī", ("ii",)), ("uu", "ū", ("uu",)), ("oo", "ū", ("uu",)),
    ("A", "ā", ("aa",)), ("I", "ī", ("ii",)), ("U", "ū", ("uu",)), ("E", "ē", ("ee",)), ("O", "ō", ("oo",)),
    ("a", "a", ("a",)), ("i", "i", ("i",)), ("u", "u", ("u",)), ("e", "e", ("e",)), ("o", "o", ("o",)),
]
_CONSONANTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("ksh", "kṣ", ("k", "Sh")), ("chh", "ch", ("ch",)),
    ("kS", "kṣ", ("k", "Sh")), ("jn", "jñ", ("j", "ny")),
    ("kh", "kh", ("kh",)), ("gh", "gh", ("gh",)), ("ch", "c", ("c",)), ("jh", "jh", ("jh",)),
    ("Th", "ṭh", ("Th",)), ("Dh", "ḍh", ("Dh",)), ("th", "th", ("th",)), ("dh", "dh", ("dh",)),
    ("ph", "ph", ("ph",)), ("bh", "bh", ("bh",)), ("sh", "ś", ("sh",)), ("zh", "ḻ", ("zh",)),
    ("ny", "ñ", ("ny",)), ("ng", "ṅg", ("ng", "g")),
    ("T", "ṭ", ("T",)), ("D", "ḍ", ("D",)), ("N", "ṇ", ("N",)), ("L", "ḷ", ("L",)), ("S", "ṣ", ("Sh",)),
    ("R", "ṟ", ("rr",)), ("G", "ṅ", ("ng",)),
    ("k", "k", ("k",)), ("g", "g", ("g",)), ("c", "c", ("c",)), ("j", "j", ("j",)), ("t", "t", ("t",)),
    ("d", "d", ("d",)), ("n", "n", ("n",)), ("p", "p", ("p",)), ("b", "b", ("b",)), ("m", "m", ("m",)),
    ("y", "y", ("y",)), ("r", "r", ("r",)), ("l", "l", ("l",)), ("v", "v", ("v",)), ("w", "v", ("v",)),
    ("s", "s", ("s",)), ("h", "h", ("h",)), ("f", "f", ("f",)), ("z", "z", ("z",)),
]
_MARKS = {"M": "ṁ", "H": "ḥ"}
_SCHEME_CAPITALS = set("AIUEOTDNLSRGMH")       # capitals that mean something in the scheme
# nasal that anusvara becomes before a consonant of each place of articulation
_HOMORGANIC = {**{c: "ng" for c in ("k", "kh", "g", "gh")}, **{c: "ny" for c in ("c", "ch", "j", "jh")},
               **{c: "N" for c in ("T", "Th", "D", "Dh")}, **{c: "n" for c in ("t", "th", "d", "dh")},
               **{c: "m" for c in ("p", "ph", "b", "bh")}}


def _clean(word: str) -> str:
    """Letters only (sandhi hyphens, punctuation, digits and diacritics dropped);
    capitals the scheme does not use are lower-cased (``Bhakthi`` -> ``bhakthi``)."""
    s = unicodedata.normalize("NFKD", word)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^A-Za-z]", "", s)
    return "".join(ch if ch.islower() or ch in _SCHEME_CAPITALS else ch.lower() for ch in s)


def _units(word: str) -> List[Tuple[str, str, Tuple[str, ...], str]]:
    """Greedy segmentation into (unit, ISO, phones, kind) with kind V / C / M."""
    out, i = [], 0
    while i < len(word):
        for table, kind in ((_CONSONANTS, "C"), (_VOWELS, "V")):
            hit = next((u for u in table if word.startswith(u[0], i)), None)
            if hit:
                break
        if hit is None and word[i] in _MARKS:
            hit, kind = (word[i], _MARKS[word[i]], ()), "M"
        if hit is None:
            raise ValueError(f"cannot transliterate {word[i]!r} in {word!r}")
        out.append((hit[0], hit[1], hit[2], kind))
        i += len(hit[0])
    # vocalic r: an r with a consonant (or the word start) before and a consonant
    # (or the word end) after carries the vowel itself (amrta -> amṛta)
    for k, (u, iso, ph, kind) in enumerate(out):
        if u == "r" and (k == 0 or out[k - 1][3] == "C") and (k + 1 == len(out) or out[k + 1][3] == "C"):
            if len(out) > 1:
                out[k] = ("r", "ṛ", ("r", "u"), "V")
    return out


def to_iso(word: str) -> str:
    """ISO-15919 spelling of a karnatik.com word."""
    return "".join(u[1] for u in _units(_clean(word)))


def to_phones(word: str) -> List[str]:
    """Phones (``saregama.phoneset`` names) of a karnatik.com word."""
    units = _units(_clean(word))
    phones: List[str] = []
    for k, (u, iso, ph, kind) in enumerate(units):
        if kind == "M" and u == "M":                  # anusvara: nasal of the next consonant's place
            nxt = next((x[2][0] for x in units[k + 1:] if x[2]), None)
            phones.append(_HOMORGANIC.get(nxt, "m") if nxt in CONSONANTS else "m")
        elif kind == "M":                             # visarga
            phones.append("h")
        else:
            phones.extend(ph)
    unknown = [p for p in phones if p not in VOWELS and p not in CONSONANTS]
    if unknown:
        raise ValueError(f"{word!r}: phones {unknown} are not in the phone set")
    return phones


def syllabify_phones(phones: Sequence[str]) -> List[List[str]]:
    """Split phones into syllables at vowel nuclei, with the same rule as
    ``align_kriti.syllabify``: a single consonant between vowels starts the
    next syllable; of a cluster, the first consonant closes the previous one."""
    nuclei = [i for i, p in enumerate(phones) if p in VOWELS]
    if len(nuclei) <= 1:
        return [list(phones)] if phones else []
    cuts = []
    for a, b in zip(nuclei, nuclei[1:]):
        n_c = b - a - 1
        cuts.append(b if n_c == 0 else (b - 1 if n_c == 1 else a + 2))
    out, prev = [], 0
    for c in cuts:
        out.append(list(phones[prev:c]))
        prev = c
    out.append(list(phones[prev:]))
    return [s for s in out if s]


def fit_syllables(phone_syls: List[List[str]], n: int) -> List[List[str]]:
    """Fit phone syllables onto ``n`` aligned syllables: when they differ in
    number (``ṛ`` adds a vowel the aligner's spelling lacks, a diphthong may be
    one nucleus or two), merge the shortest adjacent pair, or split the longest
    syllable after its first phone, until the counts agree."""
    syls = [list(s) for s in phone_syls]
    while len(syls) > n and len(syls) > 1:
        k = min(range(len(syls) - 1), key=lambda i: len(syls[i]) + len(syls[i + 1]))
        syls[k:k + 2] = [syls[k] + syls[k + 1]]
    while len(syls) < n:
        k = max(range(len(syls)), key=lambda i: len(syls[i]))
        if len(syls[k]) < 2:
            syls.append(list(syls[-1]))               # nothing left to split: repeat (melisma)
            continue
        syls[k:k + 1] = [syls[k][:1], syls[k][1:]]
    return syls


# ----------------------------------------------------------------------------- svara notation

SARGAM_SYLLABLE = {"s": "sa", "r": "ri", "g": "ga", "m": "ma", "p": "pa", "d": "dha", "n": "ni"}


def is_notation(line: str) -> bool:
    """True for a svara-notation line (letters s r g m p d n in either case,
    commas, semicolons, bars), e.g. ``pdsrsmpmdpSndRS,|`` or ``S , , R G``."""
    body = re.sub(r"[\s,;|.()\[\]*]", "", line)
    return len(body) >= 3 and re.fullmatch(r"[srgmpdnSRGMPDN]+", body) is not None


def notation_syllables(line: str) -> List[str]:
    """The sargam syllables a notation line is sung as (``srG`` -> sa ri ga)."""
    return [SARGAM_SYLLABLE[ch.lower()] for ch in line if ch.lower() in SARGAM_SYLLABLE]


def word_tokens(text: str) -> List[str]:
    """Words of a lyric line split exactly like ``align_kriti.romanise`` splits
    them (non-letters separate words), keeping the karnatik.com case."""
    s = unicodedata.normalize("NFKD", text)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^A-Za-z ]+", " ", s).split()


def line_phones(text: str) -> Optional[List[List[List[str]]]]:
    """Words -> syllables -> phones of one lyric line (None for notation lines)."""
    if is_notation(text):
        return None
    return [syllabify_phones(to_phones(w)) for w in word_tokens(text)]
