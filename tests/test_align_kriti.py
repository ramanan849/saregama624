"""Line-loop alignment recovers repeats, returns and filler on synthetic emissions."""
import numpy as np

from saregama.stages.align_kriti import (AlignCfgK, build_graph, instances_from_path, loop_viterbi,
                                         lyric_lines, syllabify, transition_penalties)

VOCAB = {c: i + 1 for i, c in enumerate("abcdeghijklmnoprstuvy")}
BLANK = 0
INS = [VOCAB[c] for c in "ayh"]


def _emissions(seq, n_cols, rng, frames_per_tok=3, filler_frames=12):
    """seq: list of letter strings or None (= filler).  Returns (T, C) log-probs
    and the (start, end) frame of every lyric item."""
    rows, spans = [], []
    def frame(col, p=0.9):  # noqa: E306
        """Append one frame peaked (probability ``p``) on column ``col``."""
        v = np.full(n_cols, (1 - p) / (n_cols - 1))
        v[col] = p
        rows.append(np.log(v))
    for item in seq:
        if item is None:
            for _ in range(filler_frames):          # like MMS on akaram: mostly blank, some "a"
                frame(VOCAB["a"], 0.6) if rng.random() < 0.15 else frame(BLANK, 0.9)
            continue
        a = len(rows)
        for ch in item.replace(" ", ""):
            for _ in range(frames_per_tok):
                frame(VOCAB[ch])
            frame(BLANK, 0.6)
        spans.append((a, len(rows)))
    return np.array(rows), spans


def test_syllabify():
    """Single consonants start the next syllable; clusters split after their first unit."""
    assert syllabify("manasaramati") == ["ma", "na", "sa", "ra", "ma", "ti"]
    assert syllabify("karnataka") == ["kar", "na", "ta", "ka"]
    assert syllabify("bhakti") == ["bhak", "ti"]


def test_loop_alignment_recovers_order():
    """Repeats, the next line and a return to the pallavi are recovered with spans."""
    rng = np.random.default_rng(0)
    sections = [{"name": "pallavi", "lines": ["sarigama padani", "ninnu kori"]},
                {"name": "anupallavi", "lines": ["mohana rama"]}]
    lines = lyric_lines(sections)
    texts = [" ".join(l.words) for l in lines]
    sung = [None, texts[0], texts[0], None, texts[1], texts[2], None, texts[0], None]
    logp, spans = _emissions(sung, len(VOCAB) + 1, rng)
    g = build_graph(lines, VOCAB, BLANK)
    cfg = AlignCfgK()
    P = transition_penalties(lines, [0, 1, 2], cfg)
    lp, sp, _ = loop_viterbi(logp, g, P, cfg, BLANK, INS)
    inst = instances_from_path(lp, sp, logp, g, cfg, BLANK, insert_cols=INS)
    assert [i.line_id for i in inst] == [0, 0, 1, 2, 0]
    for i, (a, b) in zip(inst, spans):
        assert abs(i.start * 50 - a) <= 2 and abs(i.end * 50 - b) <= 2
        assert i.complete and not i.flags
    assert [w.text for w in inst[0].words] == ["sarigama", "padani"]
    assert [s.text for s in inst[0].words[0].syllables] == ["sa", "ri", "ga", "ma"]


def test_long_hold_inside_a_line_is_resumed():
    """A long filler inside a line is left and resumed, not stretched over."""
    rng = np.random.default_rng(1)
    lines = lyric_lines([{"name": "pallavi", "lines": ["sarigama padani", "ninnu kori"]}])
    sung = [None, "sarigama", None, None, None, None, None, None, None, None, "padani", None, "ninnu kori"]
    logp, _ = _emissions(sung, len(VOCAB) + 1, rng, filler_frames=30)
    g = build_graph(lines, VOCAB, BLANK)
    cfg = AlignCfgK()
    lp, sp, _ = loop_viterbi(logp, g, transition_penalties(lines, [0, 1], cfg), cfg, BLANK, INS)
    inst = instances_from_path(lp, sp, logp, g, cfg, BLANK, insert_cols=INS)
    assert [i.line_id for i in inst] == [0, 1]
    assert inst[0].complete and "resumed" in inst[0].flags
    assert [w.text for w in inst[0].words] == ["sarigama", "padani"]


def test_mark_overlap_share_of_other_singer():
    """overlap = share of an instance's frames in which the other voice is voiced."""
    from saregama.stages.align_kriti import Instance, mark_overlap
    other = np.zeros(1000, bool)
    other[100:150] = True                                   # other singer sings 1.0-1.5 s
    a = Instance(0, "pallavi", 1.0, 2.0, 0.0, True)         # half of it overlaps
    b = Instance(0, "pallavi", 5.0, 6.0, 0.0, True)         # solo
    mark_overlap([a, b], other, fps=100)
    assert a.overlap == 0.5 and b.overlap == 0.0
    assert a.singer == "lead"


def test_other_section_start_costs_a_jump():
    """From inside caranam 3, its own start and the pallavi are cheap returns;
    the start of caranam 1 is a full jump."""
    sections = [{"name": "pallavi", "lines": ["ninnu kori"]},
                {"name": "caraNam 1", "lines": ["pankaja vadana", "sarasa"]},
                {"name": "caraNam 3", "lines": ["raja bhanga", "sujana"]}]
    lines = lyric_lines(sections)
    cfg = AlignCfgK()
    P = transition_penalties(lines, [0, 1, 2, 3, 4], cfg)
    assert P[4, 3] == cfg.ret and P[4, 0] == cfg.ret        # own section start, pallavi
    assert P[4, 1] == cfg.jump                              # another caranam's start
    assert P[5, 1] == cfg.ret                               # region start may begin any section
