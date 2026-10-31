# saregama624

Time-synced annotation of Carnatic concert recordings (Saraga 1.5) for the IE624 project:
separated stems, F0 contour, swara segments and lyrics, all on one clock (the mix's),
exported as an ICM-IR bundle per recording and as SoulX-Singer training clips.

Status: steps 1–7 run on a 10-recording pilot (`manifest`, `sync`, `stems`, `f0`,
`tonic-consensus`, `swaras`, `viz`, `lyrics-fetch`, `lyrics-verify`, `align`,
`review-export`, `review-import`).  The ICM-IR / SoulX exporters are next.

Review: `review/` in each bundle has Sonic Visualiser layers (`f0`, `flat`, `swaras`,
`lines`, `words`, `syllables`, `sections`, `sama` as `.svl`), `sync_preview.flac` (left:
cleaned vocal; right: a sine on the F0, ticks at syllable onsets, a knock per sama) and
a `README.txt` with the exact commands.  Corrected layers exported into
`review/corrected/` are read back by `review-import`, which marks the recording
`reviewed` so `align` no longer touches it.

Findings so far: stems sit 0–9 ms off the mix (one offset per recording, no drift);
Saraga's `ctonic` is sometimes Pa (a fifth off), so Sa is chosen per concert; Saraga's
`pitch.txt` timestamps lead the mix by ~13 ms; MMS hears only ~25% of the letters in
melismatic singing, so lyrics are aligned on its letter spikes (`stages/align_spikes.py`),
and sung sargam in kalpana svara matches the pitch track ~45% of the time (chance 16%),
peaking within 50 ms of zero shift.  Against Saraga's vocal pitch track our F0 agrees
95–98% (50 cents) at 0–16 ms lag; some Saraga pitch files belong to another recording
(wrong length) or follow the violin, so `f0` scores each track separately and skips
mismatched ones (`f0 --qc-only` re-scores without recomputing).

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[test,web]'
uv pip install --python .venv/bin/python -e '.[models]' \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
.venv/bin/python -m pytest -q tests
```

Separation and RMVPE use SoulX-Singer's preprocessing models:

```bash
hf download Soul-AILab/SoulX-Singer-Preprocess \
    --local-dir ../SoulX-Singer/pretrained_models/SoulX-Singer-Preprocess
```

## Planned pipeline (Saraga Carnatic, 168 multitrack recordings first)

1. `manifest` — index Saraga recordings, stems and annotations
2. `sync` — measure each stem's offset to the mix; everything is stored in mix time
3. `stems` — de-bleed the vocal stem(s) with Mel-RoFormer; offset-corrected FLAC stems
4. `f0` / `swaras` — F0 ensemble at 100 Hz, tonic-relative cents, raga-aware swara segments
5. `lyrics-fetch` / `lyrics-verify` — candidate lyrics from the web, human-verified
6. `align` — CTC alignment over a line-loop graph (sangatis, neraval), kalpanaswara as sargam
7. `review-export` / `review-import` — Sonic Visualiser layers and a sync-preview audio
8. `export` — ICM-IR v0 bundle and SoulX-Singer clips (50 fps)
