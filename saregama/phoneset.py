"""Shared Indic phoneme inventory for Hindi, Marathi and Tamil.

Every phone is an ASCII name (no '-' allowed, because SoulX-Singer splits a note's
phoneme string on '-').  In SoulX metadata a note looks like ``in_k-aa-m``: the
``in_`` prefix once, then phone names joined with '-'.  This mirrors the English
convention (``en_P-R-IH1-T-IY0``) that SoulX-Singer's ``DataProcessor`` already
understands once its ``en_`` branch is generalised to ``in_``.

Each entry carries:
  ipa      - the IPA value (documentation only)
  kind     - 'vowel', 'consonant' or 'mark'
  roman    - a lowercase a-z spelling used to drive multilingual CTC aligners
             (e.g. torchaudio MMS_FA, which works on romanised characters)
  init     - the closest existing SoulX token(s); their mean initialises the new
             embedding row when the vocabulary is extended
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

PREFIX = "in_"


@dataclass(frozen=True)
class Phone:
    name: str
    ipa: str
    kind: str
    roman: str
    init: tuple


_P = Phone
PHONES: List[Phone] = [
    # ---- vowels -------------------------------------------------------
    _P("a", "ə", "vowel", "a", ("en_AH0",)),
    _P("aa", "aː", "vowel", "aa", ("en_AA1",)),
    _P("i", "ɪ", "vowel", "i", ("en_IH1",)),
    _P("ii", "iː", "vowel", "ii", ("en_IY1",)),
    _P("u", "ʊ", "vowel", "u", ("en_UH1",)),
    _P("uu", "uː", "vowel", "uu", ("en_UW1",)),
    _P("e", "e", "vowel", "e", ("en_EH1",)),          # Tamil short e
    _P("ee", "eː", "vowel", "ee", ("en_EY1",)),
    _P("o", "o", "vowel", "o", ("en_AO1",)),          # Tamil short o
    _P("oo", "oː", "vowel", "oo", ("en_OW1",)),
    _P("E", "ɛ", "vowel", "ai", ("en_EH1",)),         # Hindi ऐ
    _P("O", "ɔ", "vowel", "au", ("en_AO1",)),         # Hindi औ, Marathi ऑ
    _P("ae", "æ", "vowel", "ae", ("en_AE1",)),        # Marathi ॲ/ऍ
    _P("nas", "̃", "mark", "n", ("en_N",)),            # nasalisation of the preceding vowel
    # ---- velar --------------------------------------------------------
    _P("k", "k", "consonant", "k", ("en_K",)),
    _P("kh", "kʰ", "consonant", "kh", ("en_K", "en_HH")),
    _P("g", "g", "consonant", "g", ("en_G",)),
    _P("gh", "gʱ", "consonant", "gh", ("en_G", "en_HH")),
    _P("ng", "ŋ", "consonant", "ng", ("en_NG",)),
    # ---- palatal ------------------------------------------------------
    _P("c", "tʃ", "consonant", "ch", ("en_CH",)),
    _P("ch", "tʃʰ", "consonant", "chh", ("en_CH", "en_HH")),
    _P("j", "dʒ", "consonant", "j", ("en_JH",)),
    _P("jh", "dʒʱ", "consonant", "jh", ("en_JH", "en_HH")),
    _P("ny", "ɲ", "consonant", "ny", ("en_N", "en_Y")),
    # ---- retroflex ----------------------------------------------------
    _P("T", "ʈ", "consonant", "t", ("en_T",)),
    _P("Th", "ʈʰ", "consonant", "th", ("en_T", "en_HH")),
    _P("D", "ɖ", "consonant", "d", ("en_D",)),
    _P("Dh", "ɖʱ", "consonant", "dh", ("en_D", "en_HH")),
    _P("N", "ɳ", "consonant", "n", ("en_N",)),
    # ---- dental -------------------------------------------------------
    _P("t", "t̪", "consonant", "t", ("en_T",)),
    _P("th", "t̪ʰ", "consonant", "th", ("en_TH",)),
    _P("d", "d̪", "consonant", "d", ("en_D",)),
    _P("dh", "d̪ʱ", "consonant", "dh", ("en_DH",)),
    _P("n", "n", "consonant", "n", ("en_N",)),
    # ---- labial -------------------------------------------------------
    _P("p", "p", "consonant", "p", ("en_P",)),
    _P("ph", "pʰ", "consonant", "ph", ("en_P", "en_HH")),
    _P("b", "b", "consonant", "b", ("en_B",)),
    _P("bh", "bʱ", "consonant", "bh", ("en_B", "en_HH")),
    _P("m", "m", "consonant", "m", ("en_M",)),
    # ---- approximants, fricatives ------------------------------------
    _P("y", "j", "consonant", "y", ("en_Y",)),
    _P("r", "ɾ", "consonant", "r", ("en_R",)),
    _P("l", "l", "consonant", "l", ("en_L",)),
    _P("v", "ʋ", "consonant", "v", ("en_V", "en_W")),
    _P("sh", "ʃ", "consonant", "sh", ("en_SH",)),
    _P("Sh", "ʂ", "consonant", "sh", ("en_SH",)),
    _P("s", "s", "consonant", "s", ("en_S",)),
    _P("h", "ɦ", "consonant", "h", ("en_HH",)),
    # ---- flaps, laterals, Tamil specials ------------------------------
    _P("R", "ɽ", "consonant", "r", ("en_R", "en_D")),
    _P("Rh", "ɽʱ", "consonant", "rh", ("en_R", "en_HH")),
    _P("L", "ɭ", "consonant", "l", ("en_L",)),
    _P("zh", "ɻ", "consonant", "zh", ("en_R", "en_ZH")),     # Tamil ழ
    _P("rr", "r", "consonant", "rr", ("en_R",)),             # Tamil ற (trill)
    _P("nn", "n̠", "consonant", "n", ("en_N",)),              # Tamil ன
    # ---- Perso-Arabic / loan sounds (nukta) ---------------------------
    _P("q", "q", "consonant", "k", ("en_K",)),
    _P("x", "x", "consonant", "kh", ("en_K", "en_HH")),
    _P("gg", "ɣ", "consonant", "g", ("en_G",)),
    _P("z", "z", "consonant", "z", ("en_Z",)),
    _P("f", "f", "consonant", "f", ("en_F",)),
    # ---- Marathi dental affricates -----------------------------------
    _P("ts", "ts", "consonant", "ts", ("en_T", "en_S")),
    _P("dz", "dz", "consonant", "dz", ("en_D", "en_Z")),
]

BY_NAME: Dict[str, Phone] = {p.name: p for p in PHONES}
VOWELS = {p.name for p in PHONES if p.kind == "vowel"}
CONSONANTS = {p.name for p in PHONES if p.kind == "consonant"}

assert len(BY_NAME) == len(PHONES), "duplicate phone names"
assert all("-" not in p.name and " " not in p.name for p in PHONES)


def token(name: str) -> str:
    """SoulX vocabulary token for a phone, e.g. 'aa' -> 'in_aa'."""
    return PREFIX + name


def note_phoneme_string(phones: Sequence[str]) -> str:
    """Phones of one note -> SoulX note string, e.g. ['k','aa'] -> 'in_k-aa'."""
    for p in phones:
        if p not in BY_NAME:
            raise KeyError(f"unknown phone {p!r}")
    return PREFIX + "-".join(phones)


def parse_note_phoneme_string(s: str) -> List[str]:
    if s in ("<SP>", "<AP>"):
        return []
    if not s.startswith(PREFIX):
        raise ValueError(f"not an Indic note string: {s!r}")
    return s[len(PREFIX):].split("-")


def romanize(phones: Sequence[str]) -> str:
    """a-z spelling for multilingual CTC aligners (MMS_FA)."""
    return "".join(BY_NAME[p].roman for p in phones)


def extend_soulx_phoneset(phone_set_in: str | Path, phone_set_out: str | Path,
                          vocab_size: int = 3000) -> Dict[str, int]:
    """Append the Indic tokens to SoulX-Singer's phone_set.json.

    The released table has 2,820 tokens and the embedding has 3,000 rows, so the
    new tokens land in unused rows and no tensor needs resizing.  Returns the
    token -> index map of the added tokens.  Existing indices never move.
    """
    phones = json.loads(Path(phone_set_in).read_text(encoding="utf-8"))
    existing = set(phones)
    added: Dict[str, int] = {}
    for p in PHONES:
        t = token(p.name)
        if t in existing:
            continue
        phones.append(t)
        added[t] = len(phones) - 1
    if len(phones) > vocab_size:
        raise ValueError(f"{len(phones)} tokens exceed vocab_size={vocab_size}")
    Path(phone_set_out).write_text(json.dumps(phones, ensure_ascii=False, indent=0),
                                   encoding="utf-8")
    return added


def embedding_init_map() -> Dict[str, List[str]]:
    """New token -> existing SoulX tokens whose mean embedding initialises it."""
    return {token(p.name): list(p.init) for p in PHONES}
