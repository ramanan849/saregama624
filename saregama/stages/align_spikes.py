"""Lyric alignment on MMS letter spikes (the default for Carnatic singing).

On sung Carnatic music MMS_FA is blank on ~98% of frames and emits short
letter spikes elsewhere, roughly a quarter of the sung letters, with "a"/"ya"
insertions from melisma.  A frame-level CTC trellis cannot win against a
filler model on such evidence, so alignment here happens on the spike
sequence instead:

1. ``spikes``: collapse the MMS posteriors to (time, letter posteriors among
   non-blank outputs) at every non-blank run.
2. A dynamic program over spikes and the same line-loop structure as
   :mod:`align_kriti` (repeat, next line, return to pallavi/section start,
   mid-line entry and exit, sargam pseudo-lines in kalpana svara):

   * a spike either **matches** the next lyric letter (gain
     ``log p(letter | non-blank) + bonus``; letters in between are skipped at
     ``del_cost`` each),
   * or is an **insertion** inside the line (melisma vowels cheap, other
     letters dearer),
   * or is absorbed by the **filler** (not singing lyrics).

   Staying inside a line across a long silence costs ``gap_cost`` per second
   beyond ``free_gap_s``.

Matched spikes timestamp their letters; syllable onsets come from their first
matched letter (unmatched syllables are interpolated between neighbours) and
:func:`align_kriti.extend_sustain` then extends them through voicing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .align_kriti import (FPS_EMIT, NEG, Graph, Instance, Line, SylSpan, WordSpan, build_graph,
                          region_lines, transition_penalties, with_sargam)

VOWELS = set("aeiou")


@dataclass
class SpikeCfg:
    bonus_cons: float = 2.5       # reward for matching a consonant ...
    bonus_vowel: float = 1.0      # ... and a vowel (vowels are everywhere in melisma)
    del_cost: float = -0.7        # per lyric letter skipped
    ins_vowel: float = -0.2       # unmatched vowel / y / h spike inside a line
    ins_other: float = -1.5       # unmatched consonant spike inside a line
    fill_vowel: float = -0.2      # spike absorbed by the filler
    fill_other: float = -0.8
    mid_entry: float = -2.0       # enter a line at a later word
    mid_exit: float = -1.0        # leave a line before its end
    free_gap_s: float = 2.5       # silence inside a line that costs nothing ...
    gap_cost: float = -0.3        # ... and per second beyond it
    min_matches: int = 3          # instances with fewer matched letters are dropped
    min_match_frac: float = 0.25  # matched / covered letters below this -> low_conf
    repeat: float = -0.5
    ret: float = -1.5
    jump: float = -6.0


@dataclass
class Spikes:
    t: np.ndarray                 # (N,) time of the spike (s)
    rel: np.ndarray               # (N, C) log p(letter | non-blank)
    top: np.ndarray               # (N,) argmax letter column
    soft: np.ndarray              # (N,) True when the top letter is a vowel / y / h


def spikes(logp: np.ndarray, blank: int, inv: Dict[int, str], t0: float = 0.0) -> Spikes:
    am = logp.argmax(axis=1)
    k, rows, times = 0, [], []
    T = len(am)
    while k < T:
        if am[k] == blank:
            k += 1
            continue
        j = k
        while j < T and am[j] == am[k]:
            j += 1
        f = k + int(np.argmax(logp[k:j, am[k]]))
        p = np.exp(logp[f])
        nb = max(1e-6, 1.0 - p[blank])
        rel = np.log(np.maximum(p / nb, 1e-8))
        rel[blank] = NEG
        rows.append(rel)
        times.append(t0 + f / FPS_EMIT)
        k = j
    if not rows:
        return Spikes(np.zeros(0), np.zeros((0, logp.shape[1])), np.zeros(0, int), np.zeros(0, bool))
    rel = np.array(rows)
    top = rel.argmax(axis=1)
    soft = np.array([inv[c] in VOWELS or inv[c] in "yh" for c in top])
    return Spikes(np.array(times), rel, top, soft)


def spike_dp(sp: Spikes, g: Graph, P: np.ndarray, cfg: SpikeCfg, inv: Dict[int, str]):
    """Best assignment of spikes to (line token | filler).  Returns per-spike
    token state (-1 filler, -2 insertion inside the line of the previous
    match) and the line it belongs to."""
    N = len(sp.t)
    S, L = len(g.lab), len(g.starts)
    tok_states = np.where(g.tok >= 0)[0]
    K = len(tok_states)
    line_of = g.line_of[tok_states]
    tidx = g.tok[tok_states].astype(float)
    letter = g.lab[tok_states]
    is_vowel = np.array([inv[c] in VOWELS for c in letter])
    bonus = np.where(is_vowel, cfg.bonus_vowel, cfg.bonus_cons)
    first_of_line = np.zeros(K, bool)
    last_of_line = np.zeros(K, bool)
    for l in range(L):
        idx = np.where(line_of == l)[0]
        if len(idx):
            first_of_line[idx[0]] = True
            last_of_line[idx[-1]] = True
    ws = set(g.word_start_states.tolist())
    word_start = np.array([(s in ws) for s in tok_states]) & ~first_of_line
    word_end = np.zeros(K, bool)
    for l, we in enumerate(g.word_end_states):
        word_end[np.isin(tok_states, we)] = True
    lines_idx = [np.where(line_of == l)[0] for l in range(L)]
    # DP values after spike n: V[k] = best score with the last matched letter at token k
    V = np.full(K, NEG)
    E = np.full(L, NEG)            # filler after leaving line l
    I = 0.0                        # filler before anything
    bpV = np.zeros((N, K), dtype=np.int64)   # >=0 previous token (match) ; -1 insertion ; -2-src entry
    bpE = np.zeros((N, L), dtype=np.int64)   # -1 stay ; else exiting token (index into the token list)
    entry_src = np.zeros((N, L), dtype=np.int64)
    prev_t = sp.t[0] if N else 0.0
    for n in range(N):
        dt = sp.t[n] - prev_t
        prev_t = sp.t[n]
        gap = cfg.gap_cost * max(0.0, dt - cfg.free_gap_s)
        fill = cfg.fill_vowel if sp.soft[n] else cfg.fill_other
        ins = cfg.ins_vowel if sp.soft[n] else cfg.ins_other
        match = sp.rel[n, letter] + bonus                                  # (K,)
        # entering a line now (from fillers, region start, or a line end via E)
        exits_end = np.full(L, NEG)
        for l in range(L):
            idx = lines_idx[l]
            if len(idx):
                exits_end[l] = V[idx[-1]]
        srcs = np.concatenate([E, [I], exits_end])
        rows = np.concatenate([np.arange(L), [L], np.arange(L)])
        cand = srcs[:, None] + P[rows]                                     # (2L+1, L)
        best_src = np.argmax(cand, axis=0)
        enter = cand[best_src, np.arange(L)]
        entry_src[n] = best_src
        # arrivals by match: from an earlier token of the same line (skips cost del)
        arrive = np.full(K, NEG)
        arr_bp = np.full(K, -1, dtype=np.int64)
        for l in range(L):
            idx = lines_idx[l]
            if not len(idx):
                continue
            k = tidx[idx]
            val = V[idx] - cfg.del_cost * k + gap
            run = np.maximum.accumulate(val)
            arg = np.maximum.accumulate(np.where(val == run, np.arange(len(idx)), 0))
            if len(idx) > 1:
                arrive[idx[1:]] = run[:-1] + cfg.del_cost * (k[1:] - 1)
                arr_bp[idx[1:]] = idx[arg[:-1]]
            # entering: at the line's first letter, or at a word start (mid entry); letters
            # before the entry point are not charged as deletions
            e = enter[l]
            starts = idx[first_of_line[idx] | word_start[idx]]
            ecost = np.where(first_of_line[starts], 0.0, cfg.mid_entry) + e
            better = ecost > arrive[starts]
            arrive[starts] = np.where(better, ecost, arrive[starts])
            arr_bp[starts] = np.where(better, -2 - l, arr_bp[starts])
        new_match = arrive + match
        new_ins = V + ins + gap
        take_ins = new_ins > new_match
        newV = np.where(take_ins, new_ins, new_match)
        bpV[n] = np.where(take_ins, -1, arr_bp)
        # fillers: stay, or leave line l from any token (mid_exit unless at a word end / line end)
        newE = np.full(L, NEG)
        for l in range(L):
            idx = lines_idx[l]
            if not len(idx):
                continue
            cost = np.where(last_of_line[idx], 0.0, np.where(word_end[idx], cfg.mid_exit, NEG))
            ex = V[idx] + cost
            j = int(np.argmax(ex))
            if ex[j] > E[l]:
                newE[l], bpE[n, l] = ex[j], idx[j]
            else:
                newE[l], bpE[n, l] = E[l], -1
        newE += fill
        I += fill
        V, E = newV, newE
    # termination
    exits_final = np.array([V[idx[-1]] if len(idx) else NEG for idx in lines_idx])
    final = np.concatenate([E, [I], exits_final])
    k = int(np.argmax(final))
    assign = np.full(N, -1, dtype=np.int64)          # token state or -1 filler / -2 insertion
    a_line = np.full(N, -1, dtype=np.int64)
    if k < L:
        node = ("E", k)
    elif k == L:
        node = ("I", 0)
    else:
        node = ("V", int(lines_idx[k - L - 1][-1]))
    for n in range(N - 1, -1, -1):
        kind, x = node
        if kind == "V":
            b = bpV[n, x]
            a_line[n] = line_of[x]
            if b == -1:                                # insertion: stay on token x
                assign[n] = -2
                node = ("V", x)
                continue
            assign[n] = tok_states[x]
            if b >= 0:
                node = ("V", int(b))
            else:                                      # entered line (-2 - l) from a source
                l = -2 - int(b)
                s = int(entry_src[n, l])
                if s < L:
                    node = ("E", s)
                elif s == L:
                    node = ("I", 0)
                else:
                    node = ("V", int(lines_idx[s - L - 1][-1]))
        elif kind == "E":
            b = bpE[n, x]
            assign[n] = -1
            # spike n was absorbed by E_x; if E_x was entered at n from token b (an index
            # into the token list), b was the state before spike n
            node = ("E", x) if b < 0 else ("V", int(b))
        else:
            assign[n] = -1
    return assign, a_line


def instances_from_spikes(sp: Spikes, assign: np.ndarray, a_line: np.ndarray, g: Graph,
                          cfg: SpikeCfg, region: str = "") -> List[Instance]:
    """Group consecutive spikes of one line (in increasing letter order) into
    instances with word/syllable onsets."""
    out: List[Instance] = []
    n, N = 0, len(assign)
    while n < N:
        if assign[n] < 0 or a_line[n] < 0:
            n += 1
            continue
        l = a_line[n]
        members = []
        last_tok = -1
        m = n
        while m < N and (assign[m] == -2 or (assign[m] >= 0 and a_line[m] == l and g.tok[assign[m]] > last_tok)):
            if assign[m] >= 0:
                members.append(m)
                last_tok = g.tok[assign[m]]
            m += 1
        n = m
        ln = g.lines[l]
        if ln.section == "sargam":                    # "sa".."ni": the consonant must be heard
            if not any(g.tok[assign[x]] == 0 for x in members):
                continue
        elif len(members) < cfg.min_matches:
            continue
        st = assign[members]
        times = sp.t[members]
        # syllable onsets: first matched letter of each syllable
        syl_on: Dict[Tuple[int, int], float] = {}
        for s_, t_ in zip(st, times):
            key = (int(g.word[s_]), int(g.syl[s_]))
            syl_on.setdefault(key, float(t_))
        w0, w1 = int(g.word[st[0]]), int(g.word[st[-1]])
        keys = [(w, s) for w in range(w0, w1 + 1) for s in range(len(ln.syllables[w]))]
        keys = [k for k in keys if not (k[0] == w0 and k[1] < g.syl[st[0]]) and
                not (k[0] == w1 and k[1] > g.syl[st[-1]])]
        known = [i for i, k in enumerate(keys) if k in syl_on]
        on = np.interp(np.arange(len(keys)), known, [syl_on[keys[i]] for i in known])
        on = np.maximum.accumulate(on)
        end_t = float(times[-1]) + 1.0 / FPS_EMIT
        words: List[WordSpan] = []
        for i, (w, s) in enumerate(keys):
            nxt = on[i + 1] if i + 1 < len(keys) else end_t
            sy = SylSpan(ln.syllables[w][s], float(on[i]), float(max(nxt, on[i] + 0.02)))
            if not words or words[-1].text != ln.words[w] or (i and keys[i - 1][0] != w):
                words.append(WordSpan(ln.words[w], sy.start, sy.end, []))
            words[-1].syllables.append(sy)
            words[-1].end = sy.end
        rel = sp.rel[members, g.lab[st]]
        complete = bool(g.tok[st[-1]] >= g.tok[g.ends[l] - 1] - 1)
        inst = Instance(int(l), ln.section, float(on[0]), end_t, round(float(np.mean(rel)), 3), complete, region,
                        words)
        span = int(g.tok[st[-1]] - g.tok[st[0]]) + 1
        inst.flags.append(f"matched_{len(members)}_of_{int(g.tok[g.ends[l] - 1]) + 1}")
        if ln.section != "sargam" and len(members) / span < cfg.min_match_frac:
            inst.flags.append("low_conf")
        out.append(inst)
    return out


def align_regions_spikes(logp: np.ndarray, lines: List[Line], regions: List[dict], vocab: Dict[str, int],
                         blank: int, cfg: SpikeCfg) -> Tuple[List[Instance], List[dict]]:
    inv = {i: c for c, i in vocab.items()}
    g = build_graph(lines, vocab, blank)
    lines_sw = with_sargam(lines)
    g_sw = build_graph(lines_sw, vocab, blank)
    sargam_ids = list(range(len(lines), len(lines_sw)))
    out, report = [], []
    from .align_kriti import AlignCfgK
    tp = AlignCfgK(repeat=cfg.repeat, ret=cfg.ret, jump=cfg.jump)
    for reg in regions:
        kind, allowed = region_lines(reg["label"], lines)
        a, b = int(reg["start"] * FPS_EMIT), min(len(logp), int(np.ceil(reg["end"] * FPS_EMIT)))
        rep = {"label": reg["label"], "kind": kind, "start": reg["start"], "end": reg["end"], "instances": 0}
        if kind == "no_lyrics" or b - a < 10 or not allowed:
            report.append(rep)
            continue
        if kind == "swara":
            gg, P = g_sw, transition_penalties(lines_sw, allowed, tp, free=sargam_ids)
        else:
            gg, P = g, transition_penalties(lines, allowed, tp)
        sp = spikes(logp[a:b], blank, inv, t0=a / FPS_EMIT)
        if len(sp.t) == 0:
            report.append(rep)
            continue
        assign, a_line = spike_dp(sp, gg, P, cfg, inv)
        inst = instances_from_spikes(sp, assign, a_line, gg, cfg, region=reg["label"])
        rep.update(instances=len(inst), spikes=int(len(sp.t)),
                   matched=int((assign >= 0).sum()), sargam=sum(i.section == "sargam" for i in inst),
                   lyric_s=round(sum(i.end - i.start for i in inst if i.section != "sargam"), 2))
        out.extend(inst)
        report.append(rep)
    return out, report
