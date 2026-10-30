"""Line-loop alignment recovers repeats, returns and filler on synthetic emissions."""
import numpy as np

from saregama.stages.align_kriti import (AlignCfgK, build_graph, instances_from_path, loop_viterbi,
                                         lyric_lines, syllabify, transition_penalties)

VOCAB = {c: i + 1 for i, c in enumerate("abcdeghijklmnoprstuvy")}
BLANK = 0


def _emissions(seq, n_cols, rng, frames_per_tok=3, filler_frames=12):
    """seq: list of letter strings or None (= filler).  Returns (T, C) log-probs
    and the (start, end) frame of every lyric item."""
    rows, spans = [], []
    def frame(col, p=0.9):
        v = np.full(n_cols, (1 - p) / (n_cols - 1))
        v[col] = p
        rows.append(np.log(v))
    for item in seq:
        if item is None:
            for _ in range(filler_frames):          # vowel-ish noise: no letter is confident
                v = rng.dirichlet(np.ones(n_cols))
                rows.append(np.log(v * 0.3 + 0.7 / n_cols))
            continue
        a = len(rows)
        for ch in item.replace(" ", ""):
            for _ in range(frames_per_tok):
                frame(VOCAB[ch])
            frame(BLANK, 0.6)
        spans.append((a, len(rows)))
    return np.array(rows), spans


def test_syllabify():
    assert syllabify("manasaramati") == ["ma", "na", "sa", "ra", "ma", "ti"]
    assert syllabify("karnataka") == ["kar", "na", "ta", "ka"]
    assert syllabify("bhakti") == ["bhak", "ti"]


def test_loop_alignment_recovers_order():
    rng = np.random.default_rng(0)
    sections = [{"name": "pallavi", "lines": ["sarigama padani", "ninnu kori"]},
                {"name": "anupallavi", "lines": ["mohana rama"]}]
    lines = lyric_lines(sections)
    texts = [" ".join(l.words) for l in lines]
    sung = [None, texts[0], texts[0], None, texts[1], texts[2], None, texts[0], None]
    logp, spans = _emissions(sung, len(VOCAB) + 1, rng)
    g = build_graph(lines, VOCAB, BLANK)
    cfg = AlignCfgK(filler_logp=-2.5)
    P = transition_penalties(lines, [0, 1, 2], cfg)
    lp, sp, _ = loop_viterbi(logp, g, P, cfg, BLANK)
    inst = instances_from_path(lp, sp, logp, g, cfg, BLANK)
    assert [i.line_id for i in inst] == [0, 0, 1, 2, 0]
    for i, (a, b) in zip(inst, spans):
        assert abs(i.start * 50 - a) <= 2 and abs(i.end * 50 - b) <= 2
        assert i.complete and not i.flags
    assert [w.text for w in inst[0].words] == ["sarigama", "padani"]
    assert [s.text for s in inst[0].words[0].syllables] == ["sa", "ri", "ga", "ma"]


def test_long_hold_inside_a_line_is_resumed():
    rng = np.random.default_rng(1)
    lines = lyric_lines([{"name": "pallavi", "lines": ["sarigama padani", "ninnu kori"]}])
    sung = [None, "sarigama", None, None, None, None, None, None, None, None, "padani", None, "ninnu kori"]
    logp, _ = _emissions(sung, len(VOCAB) + 1, rng, filler_frames=30)
    g = build_graph(lines, VOCAB, BLANK)
    cfg = AlignCfgK(filler_logp=-2.5)
    lp, sp, _ = loop_viterbi(logp, g, transition_penalties(lines, [0, 1], cfg), cfg, BLANK)
    inst = instances_from_path(lp, sp, logp, g, cfg, BLANK)
    assert [i.line_id for i in inst] == [0, 1]
    assert inst[0].complete and "resumed" in inst[0].flags
    assert [w.text for w in inst[0].words] == ["sarigama", "padani"]
