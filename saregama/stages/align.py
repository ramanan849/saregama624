"""Stage 4 - lyrics-to-audio alignment at syllable level.

Known lyrics are force-aligned to the separated vocal with a CTC acoustic model.
The model gives frame posteriors (50 fps); a Viterbi pass (pure numpy, below)
finds the best monotonic path through the lyric tokens.  Token spans are then
grouped back into syllables, and the gaps between syllables are classified as
sustain of the previous syllable, rest (silence), or unlabelled vocalising
(aalap, humming, tarana not in the lyric sheet).

Emission backends
-----------------
mms_fa     torchaudio's MMS_FA bundle (wav2vec2 trained on 23k h in 1,100+
           languages).  Tokens are romanised letters (phoneset.Phone.roman), and a
           '*' star token between lines absorbs non-lyric vocalising.
hf_ctc     Any Hugging Face CTC model over romanised letters, e.g. the same MMS
           forced-aligner weights converted to transformers.
annotator  The unified annotator trained in ``annotator_model``; its CTC head is
           over the Indic phone set itself (best once trained).
heuristic  No acoustic model: spreads syllables over voiced phrases, snapping to
           energy onsets.  Only for smoke tests and emergencies.
external   Caller sets ``Aligner.emitter`` to any :class:`Emitter` (tests, custom models).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..audio import FPS, SR, resample
from ..config import AlignCfg
from ..g2p import Syllable
from ..phoneset import BY_NAME, VOWELS


# ---------------------------------------------------------------------------
# token sequence
# ---------------------------------------------------------------------------
@dataclass
class TokenSeq:
    ids: np.ndarray            # token ids into the emission columns
    syl: np.ndarray            # syllable index per token, -1 for star tokens
    is_star: np.ndarray


def build_tokens(lines: List[List[Syllable]], unit: str, vocab: Dict[str, int],
                 star_id: Optional[int]) -> Tuple[TokenSeq, List[Tuple[int, int]]]:
    """unit = 'roman' (letters) or 'phone' (Indic phone names).
    Returns the token sequence and, per syllable, (line, index in line)."""
    ids, syl, star = [], [], []
    index: List[Tuple[int, int]] = []

    def add_star():
        if star_id is not None:
            ids.append(star_id)
            syl.append(-1)
            star.append(True)

    add_star()
    for li, line in enumerate(lines):
        for si, s in enumerate(line):
            sid = len(index)
            index.append((li, si))
            units = (list("".join(BY_NAME[p].roman for p in s.phones)) if unit == "roman"
                     else list(s.phones))
            units = [u for u in units if u in vocab]
            if not units:                      # keep every syllable alignable
                units = [next(iter(k for k in vocab if k not in ("*", "<blank>", "-", "|")))]
            for u in units:
                ids.append(vocab[u])
                syl.append(sid)
                star.append(False)
        add_star()
    return TokenSeq(np.array(ids), np.array(syl), np.array(star, dtype=bool)), index


# ---------------------------------------------------------------------------
# CTC Viterbi forced alignment (numpy)
# ---------------------------------------------------------------------------
def ctc_viterbi(logp: np.ndarray, tokens: np.ndarray, blank: int = 0
                ) -> Tuple[np.ndarray, float]:
    """Best CTC path.  logp: (T, C) log-probabilities; tokens: (L,).
    Returns the token index (0..L-1) emitted at each frame, -1 for blank frames,
    and the path log-probability."""
    T = logp.shape[0]
    L = len(tokens)
    S = 2 * L + 1
    if T < L:
        raise ValueError(f"{T} frames cannot hold {L} tokens")
    lab = np.full(S, blank)
    lab[1::2] = tokens
    allow2 = np.zeros(S, dtype=bool)
    allow2[3::2] = tokens[1:] != tokens[:-1]
    NEG = -1e30
    alpha = np.full(S, NEG)
    alpha[0] = logp[0, blank]
    alpha[1] = logp[0, lab[1]]
    bp = np.zeros((T, S), dtype=np.int8)
    for t in range(1, T):
        a0 = alpha
        a1 = np.concatenate([[NEG], alpha[:-1]])
        a2 = np.concatenate([[NEG, NEG], alpha[:-2]])
        a2 = np.where(allow2, a2, NEG)
        stack = np.stack([a0, a1, a2])
        choice = np.argmax(stack, axis=0)
        alpha = stack[choice, np.arange(S)] + logp[t, lab]
        bp[t] = choice
    s = S - 1 if alpha[S - 1] >= alpha[S - 2] else S - 2
    score = float(alpha[s])
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(bp[t, s])
    tok = np.where(path % 2 == 1, path // 2, -1)
    return tok, score


def token_spans(tok_path: np.ndarray, logp: np.ndarray, tokens: np.ndarray, L: int
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per token: first frame, last frame + 1, mean log-prob over its frames."""
    start = np.full(L, -1)
    end = np.full(L, -1)
    score = np.zeros(L)
    cnt = np.zeros(L)
    for t, k in enumerate(tok_path):
        if k < 0:
            continue
        if start[k] < 0:
            start[k] = t
        end[k] = t + 1
        score[k] += logp[t, tokens[k]]
        cnt[k] += 1
    return start, end, score / np.maximum(cnt, 1)


# ---------------------------------------------------------------------------
# emission backends
# ---------------------------------------------------------------------------
class Emitter:
    """Produces (T, C) log-probs at 50 fps plus the vocabulary."""
    unit = "roman"
    blank = 0
    star_id: Optional[int] = None
    vocab: Dict[str, int] = {}

    def __call__(self, vocals24k: np.ndarray, n_frames: int) -> np.ndarray:
        raise NotImplementedError


def _chunked(fn: Callable[[np.ndarray], np.ndarray], x16: np.ndarray, n_frames: int,
             chunk_s: float, overlap_s: float) -> np.ndarray:
    """Run a 16 kHz, 20 ms-frame model over long audio in overlapping chunks."""
    hop = 320
    chunk = int(chunk_s * 16000)
    ov = int(overlap_s * 16000)
    if len(x16) <= chunk + ov:
        e = fn(x16)
        return _fit_rows(e, n_frames)
    pieces = []
    pos = 0
    while pos < len(x16):
        a = max(0, pos - ov)
        b = min(len(x16), pos + chunk + ov)
        e = fn(x16[a:b])
        fa = (pos - a) // hop
        fb = fa + min(chunk, len(x16) - pos) // hop
        pieces.append(e[fa:fb])
        pos += chunk
    return _fit_rows(np.concatenate(pieces, axis=0), n_frames)


def _fit_rows(e: np.ndarray, n: int) -> np.ndarray:
    if len(e) >= n:
        return e[:n]
    return np.concatenate([e, np.repeat(e[-1:], n - len(e), axis=0)], axis=0)


def _log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


class MmsFaEmitter(Emitter):
    unit = "roman"

    def __init__(self, cfg: AlignCfg):
        import torch
        import torchaudio  # type: ignore
        self.torch, self.cfg = torch, cfg
        bundle = torchaudio.pipelines.MMS_FA
        self.model = bundle.get_model(with_star=cfg.use_star).to(cfg.device).eval()
        d = bundle.get_dict(star="*" if cfg.use_star else None)
        self.vocab = dict(d)
        self.blank = self.vocab.get("-", 0)
        self.star_id = self.vocab.get("*") if cfg.use_star else None

    def __call__(self, vocals24k: np.ndarray, n_frames: int) -> np.ndarray:
        x16 = resample(vocals24k, SR, 16000)

        def run(seg):
            with self.torch.inference_mode():
                w = self.torch.from_numpy(seg).float()[None].to(self.cfg.device)
                e, _ = self.model(w)
            return _log_softmax(e[0].float().cpu().numpy())

        return _chunked(run, x16, n_frames, self.cfg.chunk_s, self.cfg.chunk_overlap_s)


class HfCtcEmitter(Emitter):
    unit = "roman"

    def __init__(self, cfg: AlignCfg):
        import torch
        from transformers import AutoModelForCTC, AutoTokenizer  # type: ignore
        self.torch, self.cfg = torch, cfg
        self.model = AutoModelForCTC.from_pretrained(cfg.hf_model).to(cfg.device).eval()
        tok = AutoTokenizer.from_pretrained(cfg.hf_model)
        self.vocab = {k.lower(): v for k, v in tok.get_vocab().items()}
        self.blank = tok.pad_token_id if tok.pad_token_id is not None else 0
        self.star_id = self.vocab.get("<star>", self.vocab.get("*")) if cfg.use_star else None

    def __call__(self, vocals24k: np.ndarray, n_frames: int) -> np.ndarray:
        x16 = resample(vocals24k, SR, 16000)

        def run(seg):
            seg = (seg - seg.mean()) / (seg.std() + 1e-7)
            with self.torch.inference_mode():
                w = self.torch.from_numpy(seg).float()[None].to(self.cfg.device)
                logits = self.model(w).logits[0]
            return _log_softmax(logits.float().cpu().numpy())

        return _chunked(run, x16, n_frames, self.cfg.chunk_s, self.cfg.chunk_overlap_s)


class AnnotatorEmitter(Emitter):
    unit = "phone"

    def __init__(self, cfg: AlignCfg, annotator):
        self.annotator = annotator
        self.vocab = annotator.phone_vocab()          # phone name -> column, blank = 0
        self.blank = 0
        self.star_id = self.vocab.get("*") if cfg.use_star else None

    def __call__(self, vocals24k: np.ndarray, n_frames: int) -> np.ndarray:
        return self.annotator.phone_logprobs(vocals24k, n_frames)


# ---------------------------------------------------------------------------
# heuristic aligner (no acoustic model)
# ---------------------------------------------------------------------------
def heuristic_onsets(lines: List[List[Syllable]], f0: np.ndarray, energy_db: np.ndarray
                     ) -> np.ndarray:
    """Assign each syllable an onset frame by spreading syllables over voiced
    phrases (one lyric line per phrase when the counts allow) and snapping to the
    strongest energy rise nearby.  Returns onset frames, one per syllable."""
    voiced = f0 > 0
    phrases = _phrases(voiced, min_gap=int(0.25 * FPS), min_len=int(0.2 * FPS))
    sylls = [s for line in lines for s in line]
    if not phrases:
        return np.linspace(0, len(f0) - 1, len(sylls)).astype(int)
    # distribute syllables to phrases proportionally to phrase length
    lens = np.array([b - a for a, b in phrases], dtype=float)
    if len(phrases) == len(lines):
        counts = [len(l) for l in lines]
    else:
        counts = np.floor(lens / lens.sum() * len(sylls)).astype(int)
        counts[np.argmax(lens)] += len(sylls) - counts.sum()
    rise = np.maximum(np.diff(energy_db, prepend=energy_db[0]), 0)
    onsets = []
    for (a, b), c in zip(phrases, counts):
        if c <= 0:
            continue
        grid = np.linspace(a, b, c + 1)[:-1]
        for g in grid:
            lo, hi = int(max(a, g - 4)), int(min(b - 1, g + 4))
            onsets.append(lo + int(np.argmax(rise[lo:hi + 1])) if hi > lo else int(g))
    onsets = np.maximum.accumulate(np.array(onsets[:len(sylls)]))
    for k in range(1, len(onsets)):
        if onsets[k] <= onsets[k - 1]:
            onsets[k] = onsets[k - 1] + 1
    return onsets


def _phrases(mask: np.ndarray, min_gap: int, min_len: int) -> List[Tuple[int, int]]:
    runs, k, n = [], 0, len(mask)
    while k < n:
        if mask[k]:
            j = k
            while j < n and mask[j]:
                j += 1
            runs.append([k, j])
            k = j
        else:
            k += 1
    merged: List[List[int]] = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < min_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b in merged if b - a >= min_len]


# ---------------------------------------------------------------------------
# from token onsets to syllable intervals
# ---------------------------------------------------------------------------
@dataclass
class AlignedSyllable:
    line: int
    index: int                  # index within line
    start: int
    end: int
    conf: float
    kind: str = "lyric"         # lyric | vocalise


def _runs_of(mask: np.ndarray, a: int, b: int, min_len: int = 1) -> List[Tuple[int, int]]:
    out, k = [], a
    while k < b:
        if mask[k]:
            j = k
            while j < b and mask[j]:
                j += 1
            if j - k >= min_len:
                out.append((k, j))
            k = j
        else:
            k += 1
    return out


def syllable_intervals(onsets: np.ndarray, last_token_end: np.ndarray, confs: np.ndarray,
                       index: List[Tuple[int, int]], f0: np.ndarray, energy_db: np.ndarray,
                       cfg: AlignCfg, star_spans: Optional[List[Tuple[int, int]]] = None
                       ) -> List[AlignedSyllable]:
    """Turn per-syllable onsets into tiled intervals with rests and vocalise spans.

    Between syllable k's last token and syllable k+1's onset:
      * voicing continuous with syllable k (gaps <= 3 frames) is its sustain;
      * then silence runs >= min_gap_rest_s become rests (left uncovered here);
      * a later voiced run >= vocalise_min_s is unlabelled vocalising.
    """
    n = len(f0)
    voiced = f0 > 0
    floor = np.percentile(energy_db, 20)
    quiet = (~voiced) & (energy_db < floor + 12)
    min_rest = max(1, int(round(cfg.min_gap_rest_s * FPS)))
    min_voc = max(1, int(round(cfg.vocalise_min_s * FPS)))
    out: List[AlignedSyllable] = []
    K = len(onsets)
    for k in range(K):
        a = int(onsets[k])
        nxt = int(onsets[k + 1]) if k + 1 < K else n
        tail = int(max(last_token_end[k], a + 1))
        # sustain: extend through voicing continuous with the syllable
        e = tail
        gap_run = 0
        while e < nxt:
            if voiced[e]:
                gap_run = 0
            else:
                gap_run += 1
                if gap_run > 3:
                    e -= gap_run - 1
                    break
            e += 1
        else:
            e = nxt
        e = max(e, a + 1)
        syl = AlignedSyllable(index[k][0], index[k][1], a, min(e, nxt), float(confs[k]))
        out.append(syl)
        if e >= nxt:
            continue
        # what lies between the sustain and the next onset?
        long_voiced = [(p, q) for p, q in _runs_of(voiced, e, nxt, 1) if q - p >= min_voc]
        for p, q in long_voiced:
            out.append(AlignedSyllable(index[k][0], -1, p, q, 0.0, kind="vocalise"))
        covered = np.zeros(nxt - e, dtype=bool)
        for p, q in long_voiced:
            covered[p - e:q - e] = True
        rests = [(p, q) for p, q in _runs_of(~voiced, e, nxt, min_rest)
                 if not covered[p - e:q - e].any() and quiet[p:q].mean() > 0.5]
        firsts = [r[0] for r in rests[:1]] + [v[0] for v in long_voiced[:1]]
        # short bits before the first rest/vocalise are decay or breath: keep them
        # on the syllable; anything not covered later becomes rest in the notes stage
        syl.end = min(firsts) if firsts else nxt
        syl.end = max(syl.end, syl.start + 1)
    out.sort(key=lambda s: (s.start, s.kind != "lyric"))
    # remove overlaps created by vocalise spans
    clean: List[AlignedSyllable] = []
    for s in out:
        if clean and s.start < clean[-1].end:
            if s.kind == "vocalise":
                s.start = clean[-1].end
            else:
                clean[-1].end = s.start
        if s.end > s.start:
            clean.append(s)
    # pull lyric onsets back over audible unvoiced frames (fricative/plosive onsets)
    covered = np.zeros(n, dtype=bool)
    for s in clean:
        covered[s.start:s.end] = True
    max_pull = int(0.25 * FPS)
    for s in clean:
        if s.kind != "lyric":
            continue
        k = s.start
        while k - 1 >= 0 and not covered[k - 1] and not quiet[k - 1] and s.start - k < max_pull:
            k -= 1
        covered[k:s.start] = True
        s.start = k
    return clean


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
class Aligner:
    def __init__(self, cfg: AlignCfg, annotator=None):
        self.cfg = cfg
        self.emitter: Optional[Emitter] = None
        if cfg.backend == "mms_fa":
            self.emitter = MmsFaEmitter(cfg)
        elif cfg.backend == "hf_ctc":
            self.emitter = HfCtcEmitter(cfg)
        elif cfg.backend == "annotator":
            self.emitter = AnnotatorEmitter(cfg, annotator)
        elif cfg.backend == "external":
            pass                      # caller assigns .emitter (tests, custom models)
        elif cfg.backend != "heuristic":
            raise ValueError(f"unknown align backend {cfg.backend!r}")

    def align(self, vocals: np.ndarray, lines: List[List[Syllable]], f0: np.ndarray,
              energy_db: np.ndarray, emission: Optional[np.ndarray] = None
              ) -> Tuple[List[AlignedSyllable], Dict]:
        n = len(f0)
        sylls_index = [(li, si) for li, line in enumerate(lines) for si in range(len(line))]
        info: Dict = {"backend": self.cfg.backend}
        if self.cfg.backend == "heuristic" and emission is None:
            onsets = heuristic_onsets(lines, f0, energy_db)
            ends = onsets + 1
            confs = np.full(len(onsets), np.nan)
            info["path_logp"] = None
        else:
            em = self.emitter
            logp = emission if emission is not None else em(vocals, n)
            seq, index = build_tokens(lines, em.unit, em.vocab, em.star_id)
            path, score = ctc_viterbi(logp, seq.ids, em.blank)
            st, en, sc = token_spans(path, logp, seq.ids, len(seq.ids))
            K = len(index)
            onsets = np.zeros(K, dtype=int)
            ends = np.zeros(K, dtype=int)
            confs = np.zeros(K)
            for k in range(K):
                m = seq.syl == k
                onsets[k] = st[m].min()
                ends[k] = en[m].max()
                confs[k] = sc[m].mean()
            info["path_logp_per_frame"] = score / max(n, 1)
            info["mean_token_logp"] = float(np.mean(sc[~seq.is_star])) if (~seq.is_star).any() else None
            info["star_frames"] = int(sum(en[k] - st[k] for k in np.where(seq.is_star)[0] if st[k] >= 0))
        out = syllable_intervals(onsets, ends, confs, sylls_index, f0, energy_db, self.cfg)
        return out, info
