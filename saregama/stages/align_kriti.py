"""Stage 6 - lyric alignment for Carnatic kritis with repeats.

A kriti is not sung as written: lines repeat with variations (sangati), one
line is looped for minutes (neraval), the pallavi returns after every
section, and long stretches carry no lyrics at all (alapana, kalpana svara,
akaram, tani).  A plain left-to-right forced alignment cannot follow that, so
the CTC Viterbi here runs over a **line-loop graph**:

* every lyric line is a CTC chain over romanised letters (MMS_FA emissions);
* after each line ``l`` there is a filler state ``F_l`` that absorbs anything
  that is not lyrics (blank or akaram vowels, at ``filler_cost`` per frame);
* from the end of line ``l`` (directly, or after ``F_l``) the singer may enter
  line ``l'`` with a log-penalty: next line 0, same line ``repeat``, back to
  the pallavi or the start of the current section ``return``, anything else
  ``jump``;
* a line may be entered at a later word (``mid_entry``: sangatis and neraval
  often start mid-line), left at a word boundary (``mid_exit``) and, after filler,
  resumed at a later word of the same line (``resume``): long melismas and
  breaths inside a line become filler instead of stretching one syllable;
* frames in a line's gaps cost slightly more than filler (``hold_extra``;
  MMS is >90% blank on singing), so long holds leave the line for filler;
* letters may be skipped (``del_cost`` each): inside melisma MMS hears only
  about a quarter of the sung letters, so the line's letter order is enforced
  but not every letter must be heard;
* the gaps between a line's letters also absorb the vowels and glides that
  melisma inserts ("a-ya-ya" between the written letters) at ``insert_cost``
  per frame.

Saraga's section annotations restrict which lines can occur where (pallavi
lines in the pallavi, anupallavi lines plus the returning pallavi in the
anupallavi, ...); alapana and tani are not aligned at all.  Every lyric
occurrence ("instance") carries word and syllable spans and a confidence:
the mean log-likelihood ratio of its letter frames over the filler.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FPS_EMIT = 50                     # MMS_FA: 20 ms frames
NEG = -1e30
DIGRAPHS = ("kh", "gh", "ch", "jh", "th", "dh", "ph", "bh", "sh", "zh", "ng", "ny")
VOWELS = set("aeiou")


@dataclass
class AlignCfgK:
    filler_cost: float = -0.1     # per-frame cost of "not lyrics" (blank or akaram vowels)
    repeat: float = -1.0
    ret: float = -2.0
    jump: float = -8.0
    mid_exit: float = -3.0        # leave a line at a word end ...
    resume: float = 0.0           # ... and come back at a later word of it (after filler)
    mid_entry: float = -3.0       # enter a line at a later word (sangati / neraval eduppu)
    del_cost: float = -1.0        # per letter skipped (MMS misses most letters inside melisma)
    hold_extra: float = -0.02     # in-line gap frames cost filler_cost + this: holds > ~3 s go to filler
    merge_gap_s: float = 12.0     # resumed pieces of one line closer than this form one instance
    insert_letters: str = "ayh"   # melismatic insertions absorbed between a line's letters ...
    insert_cost: float = -1.0     # ... at this cost per frame (a-ya-ya articulation in gamakas)
    min_conf_llr: float = 1.0     # instances whose letter frames beat filler by less are flagged
    chunk_s: float = 30.0
    overlap_s: float = 2.0
    device: str = "cuda"


# ----------------------------------------------------------------------------- text

def romanise(s: str) -> str:
    """karnatik.com / ISO-15919-ish transliteration -> MMS letters (a-z)."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    return re.sub(r"[^a-z ]+", " ", s)


def _units(word: str) -> List[str]:
    out, i = [], 0
    while i < len(word):
        if word[i:i + 2] in DIGRAPHS:
            out.append(word[i:i + 2])
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


def syllabify(word: str) -> List[str]:
    """Split a romanised word at vowel nuclei: a single consonant between
    vowels starts the next syllable; of a cluster, the first unit closes the
    previous syllable (kar.na, man.da.ram, kri.ti)."""
    u = _units(word)
    kinds = ["V" if x[0] in VOWELS else "C" for x in u]
    nuclei = []
    i = 0
    while i < len(u):                       # vowel runs are one nucleus (ai, au, aa)
        if kinds[i] == "V":
            j = i
            while j + 1 < len(u) and kinds[j + 1] == "V" and len(u[i:j + 2]) <= 2:
                j += 1
            nuclei.append((i, j))
            i = j + 1
        else:
            i += 1
    if len(nuclei) <= 1:
        return [word]
    cuts = []
    for (a1, b1), (a2, b2) in zip(nuclei, nuclei[1:]):
        n_c = a2 - b1 - 1
        cuts.append(a2 if n_c == 0 else (a2 - 1 if n_c == 1 else b1 + 2))
    pieces, prev = [], 0
    for c in cuts:
        pieces.append("".join(u[prev:c]))
        prev = c
    pieces.append("".join(u[prev:]))
    return [p for p in pieces if p]


@dataclass
class Line:
    id: int
    section: str
    text: str
    words: List[str]
    syllables: List[List[str]]


def lyric_lines(sections: List[dict]) -> List[Line]:
    out = []
    for sec in sections:
        for text in sec["lines"]:
            words = [w for w in romanise(text).split() if w]
            if words:
                out.append(Line(len(out), sec["name"], text, words, [syllabify(w) for w in words]))
    return out


# ----------------------------------------------------------------------------- graph

@dataclass
class Graph:
    lab: np.ndarray               # (S,) emission column per chain state
    line_of: np.ndarray           # (S,) line index
    pos: np.ndarray               # (S,) state index within its line chain (0 .. 2n)
    tok: np.ndarray               # (S,) token index within line (-1 for blanks)
    word: np.ndarray              # (S,) word index (-1 blank)
    syl: np.ndarray               # (S,) syllable index within word (-1 blank)
    starts: np.ndarray            # (L,) first chain state of each line
    ends: np.ndarray              # (L,) last chain state (final blank) of each line
    word_end_states: List[np.ndarray]   # per line: token states that end a word (mid exits)
    word_start_states: np.ndarray       # all lines: blank + first token of every word but the first
    allow2: np.ndarray            # (S,) skip transition allowed (CTC)
    lines: List[Line]


def build_graph(lines: List[Line], vocab: Dict[str, int], blank: int) -> Graph:
    lab, line_of, pos, tok, word, syl, allow2 = [], [], [], [], [], [], []
    starts, ends, wend, wstart = [], [], [], []
    for li, ln in enumerate(lines):
        tokens = []                                   # (column, word, syllable)
        for wi, sy in enumerate(ln.syllables):
            for si, piece in enumerate(sy):
                for ch in piece:
                    if ch in vocab:
                        tokens.append((vocab[ch], wi, si))
        starts.append(len(lab))
        last_word_tok, first_word_tok = {}, {}
        for k in range(2 * len(tokens) + 1):
            if k % 2 == 0:
                lab.append(blank)
                tok.append(-1), word.append(-1), syl.append(-1)
                allow2.append(False)
            else:
                t = (k - 1) // 2
                col, wi, si = tokens[t]
                lab.append(col)
                tok.append(t), word.append(wi), syl.append(si)
                allow2.append(t > 0 and tokens[t - 1][0] != col)
                last_word_tok[wi] = len(lab) - 1
                first_word_tok.setdefault(wi, len(lab) - 1)
            line_of.append(li)
            pos.append(k)
        ends.append(len(lab) - 1)
        wend.append(np.array([s for w, s in sorted(last_word_tok.items())][:-1], dtype=np.int64))
        for w, st in sorted(first_word_tok.items())[1:]:
            wstart.extend([st - 1, st])
    return Graph(np.array(lab), np.array(line_of), np.array(pos), np.array(tok), np.array(word),
                 np.array(syl), np.array(starts), np.array(ends), wend, np.array(wstart, dtype=np.int64),
                 np.array(allow2), lines)


SARGAM = ("sa", "ri", "ga", "ma", "pa", "da", "ni")


def with_sargam(lines: List[Line]) -> List[Line]:
    """Lyric lines plus one pseudo-line per solfege syllable (kalpana svara)."""
    n = len(lines)
    return list(lines) + [Line(n + k, "sargam", sy, [sy], [[sy]]) for k, sy in enumerate(SARGAM)]


def transition_penalties(lines: List[Line], allowed: Sequence[int], cfg: AlignCfgK,
                         free: Sequence[int] = ()) -> np.ndarray:
    """(L+1, L): row l = after line l (row L = start of the region).  Lines
    in ``free`` (sargam syllables) can follow anything at no cost."""
    L = len(lines)
    P = np.full((L + 1, L), NEG)
    allowed = sorted(set(allowed))
    if not allowed:
        return P
    sec_first = {}
    for l in allowed:
        sec_first.setdefault(lines[l].section, l)
    returns = set(sec_first.values()) | {allowed[0]}
    for src in range(L + 1):
        for dst in allowed:
            if src == L:
                p = 0.0 if dst == allowed[0] else (cfg.ret if dst in returns else cfg.jump)
            elif dst == src + 1:
                p = 0.0
            elif dst == src:
                p = cfg.repeat
            elif dst in returns:
                p = cfg.ret
            else:
                p = cfg.jump
            P[src, dst] = p
    for dst in free:
        P[:, dst] = 0.0
    return P


# ----------------------------------------------------------------------------- viterbi

def filler_scores(logp: np.ndarray, cfg: AlignCfgK, blank: int, insert_cols: Sequence[int] = ()) -> np.ndarray:
    """Per-frame score of the filler: blank, or a vowel/glide (akaram), minus filler_cost."""
    f = logp[:, blank].copy()
    if len(insert_cols):
        f = np.maximum(f, logp[:, list(insert_cols)].max(axis=1) + cfg.insert_cost)
    return f + cfg.filler_cost


def _mid_exits(g: Graph):
    """Per mid-line exit point: (line, exit token state, resume states)."""
    lines_, states, resume = [], [], []
    for l, we in enumerate(g.word_end_states):
        for st in we:
            nxt = st + 1                                  # blank after the word, then next word's first token
            while nxt <= g.ends[l] and g.tok[nxt] < 0:
                nxt += 1
            if nxt <= g.ends[l]:
                lines_.append(l)
                states.append(int(st))
                resume.append((nxt - 1, nxt))
    return np.array(lines_, dtype=np.int64), np.array(states, dtype=np.int64), np.array(resume, dtype=np.int64)


def loop_viterbi(logp: np.ndarray, g: Graph, P: np.ndarray, cfg: AlignCfgK, blank: int,
                 insert_cols: Sequence[int] = ()):
    """Best path through the line-loop graph.  Returns per-frame line and chain
    state (-1 for filler frames) and the path score.

    Fillers: ``E_l`` after line l ends, ``M_f`` after mid-line exit f (resumes
    only at the next word, or leaves the line like ``E_l``), ``I`` before
    anything (region start)."""
    T = logp.shape[0]
    S, L = len(g.lab), len(g.starts)
    mline, mstate, mres = _mid_exits(g)
    Fm = len(mline)
    filler_emit = filler_scores(logp, cfg, blank, insert_cols)
    emit = logp[:, g.lab].copy()                              # (T, S)
    gap = g.tok < 0
    if len(insert_cols):                                      # gaps may also carry inserted vowels
        ins = logp[:, list(insert_cols)].max(axis=1) + cfg.insert_cost
        emit[:, gap] = np.maximum(emit[:, gap], ins[:, None])
    emit[:, gap] += cfg.filler_cost + cfg.hold_extra
    same1 = np.concatenate([[False], g.line_of[1:] == g.line_of[:-1]])
    same2 = np.concatenate([[False, False], g.line_of[2:] == g.line_of[:-2]]) & g.allow2
    # deletions: token j may follow any earlier token i < j-1 of the same line at
    # del_cost per skipped token.  With k = token index, the best source is
    # max_i (alpha_i - del*k_i) + del*(k_j - 1), a running max along each line.
    tokstates = [np.where((g.line_of == l) & (g.tok >= 0))[0] for l in range(L)]
    ends, last_tok = g.ends, g.ends - 1
    # source rows of P for every entry source: E_0..E_{L-1}, I, direct line ends, M_f
    src_rows = np.concatenate([np.arange(L), [L], np.arange(L), mline])
    res_of_state = np.full(S, -1, dtype=np.int64)
    for f in range(Fm):
        res_of_state[mres[f]] = f
    alpha = np.full(S, NEG)
    fe = np.full(L, NEG)
    fi = float(filler_emit[0])
    fm = np.full(Fm, NEG)
    for l in range(L):
        if P[L, l] > NEG / 2:
            alpha[g.starts[l]] = P[L, l] + emit[0, g.starts[l]]
            alpha[g.starts[l] + 1] = P[L, l] + emit[0, g.starts[l] + 1]
    bp_chain = np.zeros((T, S), dtype=np.int8)                # 0 stay, 1 s-1, 2 s-2, 3 entry, 4 resume, 5 s-4
    bp_entry = np.zeros((T, L), dtype=np.int32)               # index into the source list
    bp_fe = np.zeros((T, L), dtype=np.int32)                  # -1 self, else exited chain state
    bp_fm = np.zeros((T, max(Fm, 1)), dtype=np.int8)          # 0 self, 1 from exit state
    bp_lend = np.zeros((T, L), dtype=np.int32)
    bp_del = np.zeros((T, S), dtype=np.int32)                 # source token state of a deletion jump
    bp_del_src = np.zeros(S, dtype=np.int32)
    for t in range(1, T):
        line_end = np.maximum(alpha[ends], alpha[last_tok])
        line_end_state = np.where(alpha[ends] >= alpha[last_tok], ends, last_tok)
        src = np.concatenate([fe, [fi], line_end, fm])
        cand = src[:, None] + P[src_rows]
        best = np.argmax(cand, axis=0)
        entry = cand[best, np.arange(L)]
        a1 = np.where(same1, np.concatenate([[NEG], alpha[:-1]]), NEG)
        a2 = np.where(same2, np.concatenate([[NEG, NEG], alpha[:-2]]), NEG)
        a3 = np.full(S, NEG)
        a3[g.starts] = entry
        a3[g.starts + 1] = entry
        if len(g.word_start_states):
            ws = g.word_start_states
            a3[ws] = entry[g.line_of[ws]] + cfg.mid_entry
        a4 = np.full(S, NEG)
        if Fm:
            a4[mres[:, 0]] = fm + cfg.resume
            a4[mres[:, 1]] = fm + cfg.resume
        a5 = np.full(S, NEG)
        for ts in tokstates:
            if len(ts) < 3:
                continue
            k = g.tok[ts].astype(float)
            run = np.maximum.accumulate(alpha[ts] - cfg.del_cost * k)      # best source up to i
            a5[ts[2:]] = run[:-2] + cfg.del_cost * (k[2:] - 1)
            bp_del_src[ts[2:]] = ts[np.maximum.accumulate(np.where(
                alpha[ts] - cfg.del_cost * k == run, np.arange(len(ts)), 0))[:-2]]
        stack = np.stack([alpha, a1, a2, a3, a4, a5])
        choice = np.argmax(stack, axis=0)
        new_alpha = stack[choice, np.arange(S)] + emit[t]
        bp_chain[t], bp_entry[t], bp_lend[t] = choice, best, line_end_state
        bp_del[t] = bp_del_src
        take = line_end > fe
        bp_fe[t] = np.where(take, line_end_state, -1)
        new_fe = np.where(take, line_end, fe) + filler_emit[t]
        if Fm:
            ex = alpha[mstate] + cfg.mid_exit
            takem = ex > fm
            bp_fm[t] = takem
            fm = np.where(takem, ex, fm) + filler_emit[t]
        fi = fi + filler_emit[t]
        alpha, fe = new_alpha, new_fe
    final = np.concatenate([fe, [fi], alpha[ends], alpha[last_tok], fm])
    k = int(np.argmax(final))
    score = float(final[k])
    line_path = np.full(T, -1, dtype=np.int64)
    state_path = np.full(T, -1, dtype=np.int64)
    # decode the final source: ("E", l) | ("I",) | ("S", state) | ("M", f)
    if k < L:
        node = ("E", k)
    elif k == L:
        node = ("I", 0)
    elif k < 2 * L + 1:
        node = ("S", int(ends[k - L - 1]))
    elif k < 3 * L + 1:
        node = ("S", int(last_tok[k - 2 * L - 1]))
    else:
        node = ("M", k - 3 * L - 1)
    for t in range(T - 1, -1, -1):
        kind, x = node
        if kind == "S":
            line_path[t], state_path[t] = g.line_of[x], x
        elif kind == "E":
            line_path[t] = x
        elif kind == "M":
            line_path[t] = mline[x]
        if t == 0:
            break
        if kind == "S":
            c = bp_chain[t, x]
            if c == 4:
                node = ("M", int(res_of_state[x]))
            elif c == 3:
                sidx = int(bp_entry[t, g.line_of[x]])
                if sidx < L:
                    node = ("E", sidx)
                elif sidx == L:
                    node = ("I", 0)
                elif sidx < 2 * L + 1:
                    node = ("S", int(bp_lend[t, sidx - L - 1]))
                else:
                    node = ("M", sidx - 2 * L - 1)
            elif c == 5:
                node = ("S", int(bp_del[t, x]))
            else:
                node = ("S", x - int(c))
        elif kind == "E":
            prev = bp_fe[t, x]
            if prev >= 0:
                node = ("S", int(prev))
        elif kind == "M":
            if bp_fm[t, x]:
                node = ("S", int(mstate[x]))
    return line_path, state_path, score


# ----------------------------------------------------------------------------- results

@dataclass
class SylSpan:
    text: str
    start: float
    end: float


@dataclass
class WordSpan:
    text: str
    start: float
    end: float
    syllables: List[SylSpan] = field(default_factory=list)


@dataclass
class Instance:
    line_id: int
    section: str
    start: float
    end: float
    llr: float                    # mean log-likelihood ratio of the letter frames vs filler
    complete: bool                # reached the end of the line
    region: str = ""              # Saraga section label it was found in
    words: List[WordSpan] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)


def instances_from_path(line_path, state_path, logp, g: Graph, cfg: AlignCfgK, blank: int,
                        t0: float = 0.0, region: str = "", insert_cols: Sequence[int] = ()) -> List[Instance]:
    T = len(line_path)
    filler_emit = filler_scores(logp, cfg, blank, insert_cols)
    out = []
    t = 0
    while t < T:
        if state_path[t] < 0:
            t += 1
            continue
        l = line_path[t]
        a = t
        while t < T and state_path[t] >= 0 and line_path[t] == l and \
                (t == a or g.pos[state_path[t]] >= g.pos[state_path[t - 1]]):
            t += 1
        b = t
        states = state_path[a:b]
        frames = np.arange(a, b)
        letters = g.tok[states] >= 0
        lp = logp[frames[letters], g.lab[states[letters]]]
        llr = float(np.mean(lp - filler_emit[frames[letters]])) if letters.any() else 0.0
        ln = g.lines[l]
        words = []
        for wi, wtext in enumerate(ln.words):
            m = g.word[states] == wi
            if not m.any():
                continue
            ws, we = frames[m].min(), frames[m].max() + 1
            sy = []
            for si, stext in enumerate(ln.syllables[wi]):
                ms = m & (g.syl[states] == si)
                if ms.any():
                    sy.append(SylSpan(stext, t0 + frames[ms].min() / FPS_EMIT, t0 + (frames[ms].max() + 1) / FPS_EMIT))
            words.append(WordSpan(wtext, t0 + ws / FPS_EMIT, t0 + we / FPS_EMIT, sy))
        complete = bool(g.pos[states[-1]] >= g.pos[g.ends[l]] - 1)
        inst = Instance(int(l), ln.section, t0 + a / FPS_EMIT, t0 + b / FPS_EMIT, round(llr, 3), complete,
                        region, words)
        inst._first_pos, inst._last_pos, inst._frames = int(g.pos[states[0]]), int(g.pos[states[-1]]), b - a
        inst._llr_sum = llr * (b - a)
        prev = out[-1] if out else None
        if (prev is not None and prev.line_id == inst.line_id and inst._first_pos > prev._last_pos
                and not prev.complete and inst.start - prev.end <= cfg.merge_gap_s and inst._first_pos > 1):
            prev.end = inst.end
            prev.words.extend(inst.words)
            prev._last_pos = inst._last_pos
            prev._llr_sum += inst._llr_sum
            prev._frames += inst._frames
            prev.llr = round(prev._llr_sum / prev._frames, 3)
            prev.complete = inst.complete
            prev.flags = [f for f in prev.flags if f != "low_conf"] + ["resumed"] * ("resumed" not in prev.flags)
        else:
            out.append(inst)
    for inst in out:
        if inst.llr < cfg.min_conf_llr:
            inst.flags.append("low_conf")
        for k in ("_first_pos", "_last_pos", "_frames", "_llr_sum"):
            inst.__dict__.pop(k, None)
    return out


def extend_sustain(instances: List[Instance], voiced: np.ndarray, fps: float = 100.0,
                   max_gap_s: float = 0.12) -> None:
    """CTC puts a syllable's letters near its onset; the sung vowel continues
    after them.  Extend every syllable to the next syllable's start through
    continuous voicing (stopping at the first unvoiced gap >= ``max_gap_s``),
    then refit word and instance ends.  In place."""
    syls = sorted((sy for inst in instances for w in inst.words for sy in w.syllables), key=lambda x: x.start)
    gap = max(1, int(round(max_gap_s * fps)))
    n = len(voiced)
    for k, sy in enumerate(syls):
        limit = syls[k + 1].start if k + 1 < len(syls) else n / fps
        e, run = int(sy.end * fps), 0
        stop = min(n, int(limit * fps))
        while e < stop:
            run = 0 if voiced[e] else run + 1
            if run >= gap:
                e -= run - 1
                break
            e += 1
        sy.end = max(sy.end, min(e, stop) / fps)
    for inst in instances:
        for w in inst.words:
            if w.syllables:
                w.end = max(w.end, w.syllables[-1].end)
        if inst.words:
            inst.end = max(inst.end, inst.words[-1].end)


# ----------------------------------------------------------------------------- sections

def _label(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower().strip()


def region_lines(label: str, lines: List[Line]) -> Tuple[str, List[int]]:
    """(kind, allowed line ids) for a Saraga section label."""
    lab = _label(label)
    sec = lambda name: [l.id for l in lines if _label(l.section).startswith(name)]
    pallavi = sec("pallavi")
    if any(k in lab for k in ("alap", "tani", "violin", "sloka", "viruttam", "tanam")):
        return "no_lyrics", []
    if "kalpana" in lab or "svara" in lab and "sahitya" not in lab:
        return "swara", [l.id for l in lines]       # swaras return to the eduppu line
    if lab.startswith("pallavi"):
        return "lyrics", pallavi or [l.id for l in lines]
    if lab.startswith("anupallavi"):
        return "lyrics", sec("anupallavi") + pallavi
    if lab.startswith("caranam") or lab.startswith("charanam"):
        return "lyrics", [l.id for l in lines if _label(l.section).startswith(("caranam", "charanam",
                          "samasti", "madhyama"))] + pallavi
    return "lyrics", [l.id for l in lines]           # nereval, muktayi, unknown


# ----------------------------------------------------------------------------- emissions

class MmsEmitter:
    """MMS_FA log-probs over letters at 50 fps (star column dropped: the filler
    is modelled explicitly)."""

    def __init__(self, cfg: AlignCfgK):
        import torch
        import torchaudio  # type: ignore
        self.torch, self.cfg = torch, cfg
        bundle = torchaudio.pipelines.MMS_FA
        self.model = bundle.get_model(with_star=False).to(cfg.device).eval()
        self.vocab = dict(bundle.get_dict(star=None))
        self.blank = self.vocab["-"]

    def __call__(self, x16: np.ndarray) -> np.ndarray:
        hop, chunk, ov = 320, int(self.cfg.chunk_s * 16000), int(self.cfg.overlap_s * 16000)
        n = int(np.ceil(len(x16) / hop))
        pieces, pos = [], 0
        while pos < len(x16):
            a, b = max(0, pos - ov), min(len(x16), pos + chunk + ov)
            with self.torch.inference_mode():
                w = self.torch.from_numpy(x16[a:b]).float()[None].to(self.cfg.device)
                e, _ = self.model(w)
            e = self.torch.log_softmax(e[0].float(), dim=-1).cpu().numpy()
            fa = (pos - a) // hop
            fb = fa + int(np.ceil(min(chunk, len(x16) - pos) / hop))
            pieces.append(e[fa:fb])
            pos += chunk
        e = np.concatenate(pieces, axis=0)
        return e[:n] if len(e) >= n else np.concatenate([e, np.repeat(e[-1:], n - len(e), 0)])


def align_regions(logp: np.ndarray, lines: List[Line], regions: List[dict], vocab: Dict[str, int],
                  blank: int, cfg: AlignCfgK) -> Tuple[List[Instance], List[dict]]:
    """Align each region (dict with start, end, label) separately."""
    g = build_graph(lines, vocab, blank)
    lines_sw = with_sargam(lines)
    g_sw = build_graph(lines_sw, vocab, blank)
    sargam_ids = list(range(len(lines), len(lines_sw)))
    out, report = [], []
    for reg in regions:
        kind, allowed = region_lines(reg["label"], lines)
        a, b = int(reg["start"] * FPS_EMIT), min(len(logp), int(np.ceil(reg["end"] * FPS_EMIT)))
        rep = {"label": reg["label"], "kind": kind, "start": reg["start"], "end": reg["end"], "instances": 0}
        if kind == "no_lyrics" or b - a < 10 or not allowed:
            report.append(rep)
            continue
        if kind == "swara":
            P = transition_penalties(lines_sw, allowed, cfg, free=sargam_ids)
            gg = g_sw
        else:
            P = transition_penalties(lines, allowed, cfg)
            gg = g
        ins = [vocab[c] for c in cfg.insert_letters if c in vocab]
        lp, sp, score = loop_viterbi(logp[a:b], gg, P, cfg, blank, ins)
        inst = instances_from_path(lp, sp, logp[a:b], gg, cfg, blank, t0=a / FPS_EMIT, region=reg["label"],
                                   insert_cols=ins)
        rep["sargam"] = sum(i.section == "sargam" for i in inst)
        rep.update(instances=len(inst), lyric_s=round(sum(i.end - i.start for i in inst), 2),
                   mean_llr=round(float(np.mean([i.llr for i in inst])), 3) if inst else None)
        out.extend(inst)
        report.append(rep)
    return out, report


def to_json(lines: List[Line], instances: List[Instance], report: List[dict], source: dict) -> dict:
    return {"source": source,
            "lines": [asdict(l) for l in lines],
            "instances": [asdict(i) for i in instances],
            "regions": report}
