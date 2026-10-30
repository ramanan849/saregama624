# saregama624

Time-synced annotation of Carnatic concert recordings (Saraga 1.5) for the IE624 project:
separated stems, F0 contour, swara segments and lyrics, all on one clock (the mix's),
exported as an ICM-IR bundle per recording and as SoulX-Singer training clips.

Status: **Phase 0** — the draft `indic-svs-annotator` pipeline (separation, F0 ensemble,
tonic, CTC alignment, notes, QC, Praat review, SoulX export) has been ported here as the
`saregama` package and its synthetic tests pass. Saraga-specific stages (loader, stem
sync, raga-aware swaras, lyric fetching, repetition-aware alignment, Sonic Visualiser
review) are being added next.

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
