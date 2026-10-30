"""Synthetic 'singing' with exact ground truth, for testing the pipeline offline.

A source-filter toy singer: harmonic source following an F0 contour with meend
(glides between swaras), kampita-style oscillation and vibrato, shaped by vowel
formants; unvoiced consonants are noise bursts.  It is not realistic singing, but
every label (syllable spans, F0, tonic, notes) is known exactly, so the pipeline's
logic can be checked end to end without neural models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from saregama.audio import FPS, HOP, SR
from saregama.config import NotesCfg
from saregama.g2p import line_to_syllables
from saregama.phoneset import BY_NAME, VOWELS
from saregama.schema import SongAnnotation, SyllableAnn
from saregama.stages.notes import build_notes

FORMANTS = {"a": (700, 1200), "aa": (800, 1250), "i": (400, 2000), "ii": (300, 2300),
            "u": (400, 900), "uu": (320, 800), "e": (500, 1800), "ee": (450, 2000),
            "o": (550, 1000), "oo": (450, 850), "E": (600, 1800), "O": (600, 950), "ae": (700, 1700)}
UNVOICED = {"k", "kh", "c", "ch", "T", "Th", "t", "th", "p", "ph", "s", "sh", "Sh", "h", "f", "x", "q", "ts"}
SCALE = [0, 200, 400, 500, 700, 900, 1100, 1200]           # S R G m P D N S' in cents
# Sa and Pa dominate, as in most Indian melodies (this is what tonic detection relies on)
SCALE_P = np.array([0.26, 0.09, 0.11, 0.08, 0.2, 0.09, 0.07, 0.10])


@dataclass
class SynthSong:
    audio: np.ndarray
    gold: SongAnnotation
    lines: List[str]
    onsets: List[int]            # syllable onset frames (for an oracle emission)


def synth_song(song_id: str, lang: str, lines: List[str], tonic: float, seed: int = 0) -> SynthSong:
    rng = np.random.default_rng(seed)
    sylls_by_line = [line_to_syllables(l, lang) for l in lines]
    frames_f0: List[float] = []
    syl_ann: List[SyllableAnn] = []
    onsets: List[int] = []
    voiced_kind: List[str] = []           # per frame: 'rest' | 'cons' | vowel name
    t = int(0.6 * FPS)
    frames_f0 += [0.0] * t
    voiced_kind += ["rest"] * t
    prev_c = SCALE[rng.integers(0, 5)]
    for li, sylls in enumerate(sylls_by_line):
        for si_, s in enumerate(sylls):
            start = len(frames_f0)
            onsets.append(start)
            cons = [p for p in s.phones if p not in VOWELS and p != "nas"]
            vowel = next((p for p in s.phones if p in VOWELS), "a")
            n_unv = 3 if cons and cons[0] in UNVOICED else 0
            dur = int(rng.integers(int(0.30 * FPS), int(0.9 * FPS)))
            target = SCALE[int(rng.choice(len(SCALE), p=SCALE_P))]
            if si_ == len(sylls) - 1:                  # phrases resolve to Sa
                target = 0 if rng.random() < 0.7 else 1200
            meend = rng.random() < 0.35 and dur > 25
            gamak = (not meend) and rng.random() < 0.3
            c = np.full(dur, float(target))
            if meend:                                  # glide to another swara mid-syllable
                tgt2 = SCALE[int(rng.integers(0, len(SCALE)))]
                if tgt2 == target:
                    tgt2 = target + 200
                mid = dur // 2
                glide = 8
                c[mid:] = tgt2
                c[mid - glide // 2: mid + glide // 2] = np.linspace(target, tgt2, glide)
            if gamak:                                  # kampita-like oscillation
                k = np.arange(dur)
                c += 60 * np.sin(2 * np.pi * 6.0 * k / FPS)
            elif dur > 30:                             # gentle vibrato on long notes
                k = np.arange(dur)
                c += 15 * np.sin(2 * np.pi * 5.5 * k / FPS)
            c[:3] = np.linspace(prev_c, c[3], 3)       # portamento from previous note
            f = tonic * 2 ** (c / 1200)
            f[:n_unv] = 0.0
            frames_f0 += list(f)
            voiced_kind += ["cons"] * n_unv + [vowel] * (dur - n_unv)
            syl_ann.append(SyllableAnn(s.text, list(s.phones), start, len(frames_f0), li,
                                       s.word_index, 0.0, []))
            prev_c = c[-1]
        gap = int(rng.integers(int(0.35 * FPS), int(0.8 * FPS)))
        frames_f0 += [0.0] * gap
        voiced_kind += ["rest"] * gap
    n = len(frames_f0)
    f0 = np.array(frames_f0, dtype=np.float32)

    # ---- render audio --------------------------------------------------
    ns = n * HOP
    tf = np.arange(n) * HOP
    ts = np.arange(ns)
    f_s = np.interp(ts, tf, np.where(f0 > 0, f0, np.nan))
    vmask = np.interp(ts, tf, (f0 > 0).astype(float)) > 0.5
    f_s = np.where(np.isnan(f_s), 0, f_s)
    phase = 2 * np.pi * np.cumsum(f_s) / SR
    kinds = np.array(voiced_kind)
    F1 = np.array([FORMANTS.get(k, (600, 1200))[0] for k in kinds], dtype=float)
    F2 = np.array([FORMANTS.get(k, (600, 1200))[1] for k in kinds], dtype=float)
    F1s, F2s = np.interp(ts, tf, F1), np.interp(ts, tf, F2)
    y = np.zeros(ns)
    for h in range(1, 30):
        fh = h * f_s
        # glottal source tilt (~ -6 dB/octave) boosted around the two formants
        env = (1.0 / h) * (1 + 3 * np.exp(-((fh - F1s) / 150) ** 2)
                           + 2 * np.exp(-((fh - F2s) / 200) ** 2))
        env[fh > 9000] = 0
        y += env * np.sin(h * phase)
    y *= vmask
    # attack/decay envelope per syllable to make onsets audible
    amp = np.ones(n)
    for s in syl_ann:
        k = np.arange(s.end - s.start)
        amp[s.start:s.end] = np.minimum(1, (k + 1) / 4) * np.minimum(1, (s.end - s.start - k) / 6 + 0.4)
    y *= np.interp(ts, tf, amp)
    noise = rng.normal(0, 1, ns)
    cons_mask = np.interp(ts, tf, (kinds == "cons").astype(float)) > 0.5
    y += 0.25 * noise * cons_mask
    y += 0.003 * noise                                    # room noise
    y = (0.3 * y / (np.abs(y).max() + 1e-9)).astype(np.float32)

    gold = SongAnnotation(song_id=song_id, language=lang, style="bhajan", singer_id="synth",
                          gender="f", n_frames=n, tonic_hz=tonic, tonic_source="given",
                          tonic_conf=1.0, lines=list(lines), syllables=syl_ann, status="gold",
                          f0=f0, f0_agree=np.ones(n, dtype=np.float32))
    gold.notes = build_notes(gold.syllables, f0, tonic, n, NotesCfg())
    z = np.zeros(n, dtype=np.float32)
    gold.energy_db, gold.hnr_db, gold.h1h2_db = z, z, z
    return SynthSong(y, gold, lines, onsets)


def oracle_emission(song: SynthSong, vocab: Dict[str, int], jitter: int = 2, seed: int = 0
                    ) -> np.ndarray:
    """CTC-like posteriors that fire each syllable's romanised letters near its
    true onset (with jitter), blank elsewhere - stands in for MMS_FA in tests."""
    rng = np.random.default_rng(seed)
    n = song.gold.n_frames
    C = max(vocab.values()) + 1
    logp = np.full((n, C), -8.0)
    logp[:, vocab["-"]] = -0.05
    if "*" in vocab:
        logp[:, vocab["*"]] = -4.0
    for s, on in zip(song.gold.syllables, song.onsets):
        letters = [ch for ch in "".join(BY_NAME[p].roman for p in s.phones) if ch in vocab]
        t = max(0, on + int(rng.integers(-jitter, jitter + 1)))
        for ch in letters:
            if t >= n:
                break
            logp[t, vocab[ch]] = -0.1
            logp[t, vocab["-"]] = -4.0
            t += 1
    logp += rng.normal(0, 0.2, logp.shape)
    return logp - np.log(np.exp(logp).sum(axis=1, keepdims=True))


MMS_LIKE_VOCAB = {ch: i for i, ch in enumerate(["-"] + list("abcdefghijklmnopqrstuvwxyz'") + ["*"])}
