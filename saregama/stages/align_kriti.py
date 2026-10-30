"""Stage 6 - lyric alignment for Carnatic kritis with repeats.

A kriti is not sung as written: lines repeat with variations (sangati), one
line is looped for minutes (neraval), the pallavi returns after every
section, and long stretches carry no lyrics at all (alapana, kalpana svara,
akaram, tani).  A plain left-to-right forced alignment cannot follow that, so
the CTC Viterbi here runs over a **line-loop graph**:

* every lyric line is a CTC chain over romanised letters (MMS_FA emissions);
* after each line ``l`` there is a filler state ``F_l`` that absorbs anything
  that is not lyrics (scored at a constant ``filler_logp`` per frame, or the
  blank, whichever is better);
* from the end of line ``l`` (directly, or after ``F_l``) the singer may enter
  line ``l'`` with a log-penalty: next line 0, same line ``repeat``, back to
  the pallavi or the start of the current section ``return``, anything else
  ``jump``;
* a line may be left early at a word boundary (``mid_exit``), for half-line
  repeats.

Saraga's section annotations restrict which lines can occur where (pallavi
lines in the pallavi, anupallavi lines plus the returning pallavi in the
anupallavi, ...); alapana and tani are not aligned at all.  Every lyric
occurrence ("instance") carries word and syllable spans and a confidence:
the mean per-frame log-likelihood ratio of the lyric path over the filler.
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
    filler_logp: float = -3.0     # per-frame score of "not lyrics"
    repeat: float = -1.0
    ret: float = -2.0
    jump: float = -8.0
    mid_exit: float = -3.0
    min_conf_llr: float = 0.5     # instances below this per-frame LLR are flagged
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
    allow2: np.ndarray            # (S,) skip transition allowed (CTC)
    lines: List[Line]


def build_graph(lines: List[Line], vocab: Dict[str, int], blank: int) -> Graph:
    lab, line_of, pos, tok, word, syl, allow2 = [], [], [], [], [], [], []
    starts, ends, wend = [], [], []
    for li, ln in enumerate(lines):
        tokens = []                                   # (column, word, syllable)
        for wi, sy in enumerate(ln.syllables):
            for si, piece in enumerate(sy):
                for ch in piece:
                    if ch in vocab:
                        tokens.append((vocab[ch], wi, si))
        starts.append(len(lab))
        last_word_tok = {}
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
            line_of.append(li)
            pos.append(k)
        ends.append(len(lab) - 1)
        wend.append(np.array([s for w, s in sorted(last_word_tok.items())][:-1], dtype=np.int64))
    return Graph(np.array(lab), np.array(line_of), np.array(pos), np.array(tok), np.array(word),
                 np.array(syl), np.array(starts), np.array(ends), wend, np.array(allow2), lines)


def transition_penalties(lines: List[Line], allowed: Sequence[int], cfg: AlignCfgK) -> np.ndarray:
    """(L+1, L): row l = after line l (row L = start of the region)."""
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
    return P


# ----------------------------------------------------------------------------- viterbi

def loop_viterbi(logp: np.ndarray, g: Graph, P: np.ndarray, cfg: AlignCfgK, blank: int):
    """Best path through the line-loop graph.  Returns per-frame (line, chain
    state) with -1 chain state for filler frames, and the path score."""
    T = logp.shape[0]
    S, L = len(g.lab), len(g.starts)
    filler_emit = np.maximum(logp[:, blank], cfg.filler_logp)
    emit = logp[:, g.lab]                                     # (T, S)
    # neighbours inside a chain: predecessor s-1 / s-2 must be on the same line
    same1 = np.concatenate([[False], g.line_of[1:] == g.line_of[:-1]])
    same2 = np.concatenate([[False, False], g.line_of[2:] == g.line_of[:-2]]) & g.allow2
    entry_ok = np.zeros(S, dtype=bool)
    entry_ok[g.starts] = True
    entry_ok[g.starts + 1] = True
    alpha = np.full(S, NEG)
    fill = np.full(L + 1, NEG)                                # F_0..F_{L-1}, F_init
    fill[L] = filler_emit[0]
    src0 = P[L]                                               # entering from region start
    for l in range(L):
        if src0[l] > NEG / 2:
            alpha[g.starts[l]] = src0[l] + emit[0, g.starts[l]]
            alpha[g.starts[l] + 1] = src0[l] + emit[0, g.starts[l] + 1]
    bp_chain = np.zeros((T, S), dtype=np.int8)                # 0 stay, 1 s-1, 2 s-2, 3 entry
    bp_entry = np.zeros((T, L), dtype=np.int32)               # source id (0..L filler, L+1+l direct)
    bp_fill = np.zeros((T, L + 1), dtype=np.int32)            # -1 self, else chain state exited
    bp_lend = np.zeros((T, L), dtype=np.int32)                # end state of each line used at t-1
    ends = g.ends
    last_tok = ends - 1
    for t in range(1, T):
        # sources for entering a line at t: filler F_l(t-1) or line-end states of l at t-1
        line_end = np.maximum(alpha[ends], alpha[last_tok])               # (L,)
        line_end_state = np.where(alpha[ends] >= alpha[last_tok], ends, last_tok)
        src_scores = np.concatenate([fill, line_end])                     # (2L+1,)
        src_rows = np.concatenate([np.arange(L + 1), np.arange(L)])       # row of P
        cand = src_scores[:, None] + P[src_rows]                          # (2L+1, L)
        best_src = np.argmax(cand, axis=0)
        entry = cand[best_src, np.arange(L)]                              # (L,)
        # chain recursion
        a0 = alpha
        a1 = np.where(same1, np.concatenate([[NEG], alpha[:-1]]), NEG)
        a2 = np.where(same2, np.concatenate([[NEG, NEG], alpha[:-2]]), NEG)
        a3 = np.full(S, NEG)
        a3[g.starts] = entry
        a3[g.starts + 1] = entry
        stack = np.stack([a0, a1, a2, a3])
        choice = np.argmax(stack, axis=0)
        new_alpha = stack[choice, np.arange(S)] + emit[t]
        bp_chain[t] = choice
        bp_entry[t] = best_src
        bp_lend[t] = line_end_state
        # fillers: self loop, or exit from line l's end / a word end of line l
        new_fill = np.full(L + 1, NEG)
        fb = np.full(L + 1, -1, dtype=np.int32)
        for l in range(L):
            best, arg = fill[l], -1
            if line_end[l] > best:
                best, arg = line_end[l], int(line_end_state[l])
            we = g.word_end_states[l]
            if len(we):
                k = int(np.argmax(alpha[we]))
                if alpha[we[k]] + cfg.mid_exit > best:
                    best, arg = alpha[we[k]] + cfg.mid_exit, int(we[k])
            new_fill[l], fb[l] = best, arg
        new_fill[L] = fill[L]
        new_fill += filler_emit[t]
        bp_fill[t] = fb
        alpha, fill = new_alpha, new_fill
    # termination: any filler or any line end
    final = np.concatenate([fill, alpha[ends], alpha[last_tok]])
    k = int(np.argmax(final))
    score = float(final[k])
    # backtrace
    line_path = np.full(T, -1, dtype=np.int64)
    state_path = np.full(T, -1, dtype=np.int64)
    if k <= L:
        mode, cur = "fill", k
    else:
        mode, cur = "chain", int(ends[k - L - 1] if k - L - 1 < L else last_tok[k - 2 * L - 1])
    for t in range(T - 1, -1, -1):
        if mode == "fill":
            line_path[t] = cur if cur < L else -1
            state_path[t] = -1
            if t == 0:
                break
            prev = bp_fill[t, cur]
            if prev >= 0:
                mode, cur = "chain", int(prev)
        else:
            line_path[t] = g.line_of[cur]
            state_path[t] = cur
            if t == 0:
                break
            c = bp_chain[t, cur]
            if c == 3:
                l = g.line_of[cur]
                src = bp_entry[t, l]
                if src <= L:
                    mode, cur = "fill", int(src)
                else:
                    cur = int(bp_lend[t, src - L - 1])
            else:
                cur -= int(c)
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
    llr: float                    # mean per-frame log-likelihood ratio, lyric path vs filler
    complete: bool                # reached the end of the line
    region: str = ""              # Saraga section label it was found in
    words: List[WordSpan] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)


def instances_from_path(line_path, state_path, logp, g: Graph, cfg: AlignCfgK, blank: int,
                        t0: float = 0.0, region: str = "") -> List[Instance]:
    T = len(line_path)
    filler_emit = np.maximum(logp[:, blank], cfg.filler_logp)
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
        lp = logp[frames, g.lab[states]]
        llr = float(np.mean(lp - filler_emit[frames]))
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
        if llr < cfg.min_conf_llr:
            inst.flags.append("low_conf")
        out.append(inst)
    return out


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
    out, report = [], []
    for reg in regions:
        kind, allowed = region_lines(reg["label"], lines)
        a, b = int(reg["start"] * FPS_EMIT), min(len(logp), int(np.ceil(reg["end"] * FPS_EMIT)))
        rep = {"label": reg["label"], "kind": kind, "start": reg["start"], "end": reg["end"], "instances": 0}
        if kind == "no_lyrics" or b - a < 10 or not allowed:
            report.append(rep)
            continue
        P = transition_penalties(lines, allowed, cfg)
        lp, sp, score = loop_viterbi(logp[a:b], g, P, cfg, blank)
        inst = instances_from_path(lp, sp, logp[a:b], g, cfg, blank, t0=a / FPS_EMIT, region=reg["label"])
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
