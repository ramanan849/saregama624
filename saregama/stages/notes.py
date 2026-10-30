"""Stage 5 - notes, note types and swaras.

SoulX-Singer's unit is the *note*: a stretch of one syllable at one pitch.  The
first note of a syllable is type 2 (lyric); further notes of the same syllable
(melisma, meend landing on a new swara) are type 3 (slur) and repeat the
syllable's phonemes; silences are type 1 (rest).  Notes tile the song exactly.

Pitch is segmented on a 12-tone grid relative to the tonic (so note boundaries
fall where the singer moves to a new swara), but each note also stores its exact
median pitch in cents above Sa: the 12-tone MIDI number is only for SoulX's score
mode, the cents value is the truth.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..audio import FPS
from ..config import NotesCfg
from ..schema import NOTE_LYRIC, NOTE_REST, NOTE_SLUR, NoteAnn, SyllableAnn

SWARAS = ["S", "r", "R", "g", "G", "m", "M", "P", "d", "D", "n", "N"]


def swara_name(semitones_above_sa: int) -> str:
    """0 -> 'S', 7 -> 'P', 12 -> "S'", -1 -> '.N' (mandra), 13 -> "r'" (taar)."""
    octave, pc = divmod(int(semitones_above_sa), 12)
    name = SWARAS[pc]
    if octave < 0:
        return "." * (-octave) + name
    return name + "'" * octave


def _segment_runs(q: np.ndarray, min_len: int) -> List[List[int]]:
    """Run-length encode q, then merge runs shorter than min_len into the
    neighbour with the closest value.  Returns [start, end, value] lists."""
    runs: List[List[int]] = []
    for k, v in enumerate(q):
        if runs and runs[-1][2] == v:
            runs[-1][1] = k + 1
        else:
            runs.append([k, k + 1, int(v)])
    changed = True
    while changed and len(runs) > 1:
        changed = False
        lens = [r[1] - r[0] for r in runs]
        i = int(np.argmin(lens))
        if lens[i] >= min_len:
            break
        left = runs[i - 1] if i > 0 else None
        right = runs[i + 1] if i + 1 < len(runs) else None
        if left is None:
            tgt = i + 1
        elif right is None:
            tgt = i - 1
        else:
            dl, dr = abs(left[2] - runs[i][2]), abs(right[2] - runs[i][2])
            tgt = i - 1 if dl <= dr else i + 1
        if tgt < i:
            runs[tgt][1] = runs[i][1]
        else:
            runs[tgt][0] = runs[i][0]
        del runs[i]
        # merge equal neighbours created by the merge
        j = 0
        while j + 1 < len(runs):
            if runs[j][2] == runs[j + 1][2]:
                runs[j][1] = runs[j + 1][1]
                del runs[j + 1]
            else:
                j += 1
        changed = True
    return runs


def build_notes(syllables: Sequence[SyllableAnn], f0: np.ndarray, tonic_hz: float,
                n_frames: int, cfg: NotesCfg) -> List[NoteAnn]:
    f0 = np.asarray(f0, dtype=np.float64)
    voiced = f0 > 0
    cents = np.full(n_frames, np.nan)
    cents[voiced] = 1200 * np.log2(f0[voiced] / tonic_hz)
    min_len = max(1, int(round(cfg.min_note_s * FPS)))
    notes: List[NoteAnn] = []
    pos = 0

    def add_rest(a: int, b: int):
        if b > a:
            if notes and notes[-1].type == NOTE_REST and notes[-1].end == a:
                notes[-1].end = b
            else:
                notes.append(NoteAnn(a, b, NOTE_REST, -1, 0, None, ""))

    for si, s in enumerate(syllables):
        if s.start > pos:
            add_rest(pos, s.start)
        a, b = s.start, s.end
        c = cents[a:b]
        v = ~np.isnan(c)
        if not v.any():
            # unvoiced syllable (whispered, or F0 failure): one note, pitch of neighbours
            prev = [n for n in notes if n.type != NOTE_REST]
            semi = int(round(prev[-1].cents / 100)) if prev and prev[-1].cents is not None else 0
            notes.append(NoteAnn(a, b, NOTE_LYRIC, si, _midi(tonic_hz, semi * 100), None,
                                 swara_name(semi)))
            pos = b
            continue
        q = _stable_grid(c, cfg.smooth_frames, min_len)
        runs = _segment_runs(q, min_len)
        for ri, (ra, rb, semi) in enumerate(runs):
            seg = c[ra:rb]
            seg = seg[~np.isnan(seg)]
            med = float(np.median(seg)) if len(seg) else float(semi * 100)
            notes.append(NoteAnn(a + ra, a + rb, NOTE_LYRIC if ri == 0 else NOTE_SLUR, si,
                                 _midi(tonic_hz, med), round(med, 1), swara_name(int(round(med / 100)))))
        pos = b
    if pos < n_frames:
        add_rest(pos, n_frames)
    return notes


def _stable_grid(c: np.ndarray, win: int, min_len: int) -> np.ndarray:
    """Semitone index per frame that only changes where the voice *settles*.

    A frame is stable when the pitch range over a ``win``-frame window is under
    50 cents and it sits within 40 cents of a semitone.  Stable runs shorter than
    ``min_len`` are ignored.  Unstable frames (gamak, kampita, andolan, glides) take
    the value of the preceding stable run, so an ornament stays inside the note it
    decorates instead of being chopped into many short notes; a meend becomes a
    new note only where it lands and holds.
    """
    filled = _ffill_bfill(c)
    w = max(3, win | 1)
    pad = np.pad(filled, (w // 2, w // 2), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(pad, w)
    spread = windows.max(axis=1) - windows.min(axis=1)
    near = np.abs(filled - 100 * np.round(filled / 100)) < 40
    stable = (spread < 50) & near & ~np.isnan(c)
    q = np.full(len(c), np.nan)
    q[stable] = np.round(filled[stable] / 100)
    # drop stable runs that are too short to count as a held swara
    k = 0
    while k < len(q):
        if np.isnan(q[k]):
            k += 1
            continue
        j = k
        while j < len(q) and q[j] == q[k]:
            j += 1
        if j - k < min_len:
            q[k:j] = np.nan
        k = j
    if np.all(np.isnan(q)):
        v = c[~np.isnan(c)]
        return np.full(len(c), int(np.round(np.median(v) / 100)) if len(v) else 0)
    return _ffill_bfill(q).astype(int)


def _midi(tonic_hz: float, cents: float) -> int:
    f = tonic_hz * 2 ** (cents / 1200)
    return int(np.clip(round(69 + 12 * np.log2(f / 440.0)), 1, 127))


def _ffill_bfill(c: np.ndarray) -> np.ndarray:
    c = c.copy()
    idx = np.where(~np.isnan(c))[0]
    if len(idx) == 0:
        return np.zeros_like(c)
    c[:idx[0]] = c[idx[0]]
    for k in range(idx[0] + 1, len(c)):
        if np.isnan(c[k]):
            c[k] = c[k - 1]
    return c
