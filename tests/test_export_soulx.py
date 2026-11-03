"""SoulX clip export: notes per syllable, short-note merging, clips, item format."""
import numpy as np

from saregama.export_soulx import (NOTE_LYRIC, NOTE_REST, NOTE_SLUR, ClipCfgS, Note, clip_item, plan_clips,
                                   syllable_notes, track_notes)


def _curves(n100, hz=220.0):
    """Constant-pitch 100 Hz F0 / cents curves (Sa = 220 Hz)."""
    return np.full(n100, hz), np.zeros(n100)


def test_syllable_split_at_segments_and_short_notes_merge():
    """A syllable spanning two segments becomes lyric + vowel-only slur; a cut
    leaving a note too short for its phones is merged away."""
    f0, cents = _curves(400)
    notes = syllable_notes(10, 40, "ma", ["m", "a"], [25], f0, cents, lambda t: "S")
    assert [(n.a, n.b, n.kind, n.phones) for n in notes] == [(10, 25, NOTE_LYRIC, ["m", "a"]),
                                                              (25, 40, NOTE_SLUR, ["a"])]
    short = syllable_notes(10, 40, "ma", ["m", "a"], [12], f0, cents, lambda t: "S")
    assert len(short) == 1 and short[0].kind == NOTE_LYRIC and short[0].phones == ["m", "a"]
    assert notes[0].midi == 57                                   # 220 Hz = A3


def test_track_notes_rests_and_clips():
    """Syllables with rests between them; clips cut in long rests, sized 2-16 s."""
    n50 = 50 * 40
    f0, cents = _curves(n50 * 2)
    syl = [{"start": t, "end": t + 0.4, "syllable": "sa", "phones": "s-a"} for t in np.arange(1, 10, 0.5)]
    syl += [{"start": t, "end": t + 0.4, "syllable": "ri", "phones": "r-i"} for t in np.arange(20, 26, 0.5)]
    notes = track_notes(syl, [], [], f0, cents, n50)
    assert notes[0].kind == NOTE_REST and sum(n.b - n.a for n in notes) == n50
    voiced = np.zeros(n50, bool)
    for s in syl:
        voiced[int(s["start"] * 50):int(s["end"] * 50)] = True
    clips = plan_clips(notes, voiced, ClipCfgS())
    assert len(clips) == 2 and all(2 * 50 <= b - a <= 16 * 50 for a, b in clips)
    voiced[int(15 * 50):int(15.5 * 50)] = True                   # unlabelled singing inside a rest ...
    syl2 = syl[:4]
    notes2 = track_notes(syl2 + [{"start": 15.6, "end": 16.0, "syllable": "ga", "phones": "g-a"}], [], [], f0,
                         cents, n50)
    assert all(not (a <= 15 * 50 < b) for a, b in plan_clips(notes2, voiced, ClipCfgS()))   # ... is never in a clip


def test_clip_item_durations_cover_the_frames():
    """Durations sum to the clip's frames as SoulX counts them; F0 is 50 fps."""
    f0, _ = _curves(400)
    notes = [Note(0, 10, NOTE_REST), Note(10, 30, NOTE_LYRIC, "ma", ["m", "a"], 57, 0.0, "S"),
             Note(30, 40, NOTE_REST)]
    it = clip_item(notes, 0, 40, f0, {"clip_id": "x", "language": "Telugu", "tonic_hz": 220.0, "singer_id": "s",
                                     "mbid": "m", "split": "train", "lyrics_verified": False, "wav": "w"})
    assert int(sum(float(d) for d in it["duration"].split()) * 50) == 40 == len(it["f0"].split())
    assert it["phoneme"] == "<SP> in_m-a <SP>" and it["note_type"] == "1 2 1" and it["note_pitch"] == "0 57 0"
