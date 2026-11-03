# saregama624

Time-synced annotation of Carnatic concert recordings (Saraga 1.5) for the IE624
project. For every multitrack recording, the pipeline produces:

- offset-corrected stems: each singer and each instrument;
- a 100 Hz F0 contour;
- the tonic (Sa);
- raga-aware swara segments;
- time-aligned lyrics, down to syllables.

Everything is on **one clock, the mix's**, and is stored as an ICM-IR bundle per
recording. The bundles are the source for ICM-IR training data and for
SoulX-Singer fine-tuning clips.

- [Status](#status)
- [Setup](#setup)
- [Running the pipeline](#running-the-pipeline)
- [What each stage does](#what-each-stage-does)
- [Output: the IR bundle](#output-the-ir-bundle)
- [Lyrics](#lyrics)
- [Review in Sonic Visualiser](#review-in-sonic-visualiser)
- [Flags and statuses](#flags-and-statuses)
- [Code map](#code-map)
- [Conventions](#conventions)
- [Tests](#tests)
- [Known limitations and next steps](#known-limitations-and-next-steps)

## Status

Steps 1–7 run end to end on a 10-recording pilot
(`annotations/carnatic/pilot_mbids.txt`).

| Step | Command(s) | Pilot result |
|---|---|---|
| 1 Index | `manifest` | every Saraga Carnatic recording indexed |
| 2 Sync | `sync` | one offset per recording, 0–9 ms, no drift |
| 3 Stems | `stems` | vocal bleed in silent stretches: −52 dB → digital silence |
| 4 Pitch | `f0`, `tonic-consensus`, `swaras`, `viz` | F0 agrees 95–98 % with Saraga's vocal pitch (50 cents); one Sa per concert |
| 5 Lyrics | `lyrics-fetch`, `lyrics-verify` | 72 / 133 compositions matched automatically, none verified by a person yet |
| 6 Alignment | `align` | 7 pilot songs aligned; sync validated to within ~50 ms |
| 7 Review | `review-export`, `review-import` | Sonic Visualiser layers + sync preview; lossless round trip |
| 8 Export | `export-ir`, `export-soulx` | ICM-IR v0 dataset; SoulX clips (pilot: 243 clips, 0.65 h, all pass SoulX's frame/phone checks) |

The 168 multitrack recordings have not been run yet; only the pilot has.

## Setup

### Folder layout

The repository is expected to sit next to its data:

```
<project>/                     e.g. ~/Gokul/624
    saregama624/               this repository
    SoulX-Singer/              git clone https://github.com/Soul-AILab/SoulX-Singer
        pretrained_models/SoulX-Singer-Preprocess/
    Datasets/Saraga/files/saraga1.5_carnatic/
    annotations/               created by the pipeline
```

With this layout nothing needs configuring. If the data lives elsewhere, set any
of these environment variables (see `saregama/paths.py`):

| Variable | Default |
|---|---|
| `SAREGAMA_PROJECT` | the repository's parent folder |
| `SAREGAMA_SOULX` | `<project>/SoulX-Singer` |
| `SAREGAMA_SARAGA` | `<project>/Datasets/Saraga/files/saraga1.5_carnatic` |
| `SAREGAMA_ANNOTATIONS` | `<project>/annotations` |

The CLI options `--saraga-root`, `--ann-root` and `--lyrics-dir` override these
for a single run.

### Environment

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[test,web]'
# GPU stages (stems, f0, align).  On a machine without CUDA, drop the --index-url line.
uv pip install --python .venv/bin/python -e '.[models]' \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
.venv/bin/python -m pytest -q tests
```

Keep torchaudio below 2.9, which still ships the `MMS_FA` pipeline used for alignment.

### Models

The separation and RMVPE checkpoints come from SoulX-Singer's preprocessing release:

```bash
hf download Soul-AILab/SoulX-Singer-Preprocess \
    --local-dir ../SoulX-Singer/pretrained_models/SoulX-Singer-Preprocess
```

- FCPE and CREPE weights ship with their pip packages.
- MMS_FA (about 1.3 GB) downloads itself into `~/.cache/torch` on first use.

## Running the pipeline

Every command is resumable per recording. A recording whose stage is already
recorded in its `ir.json` is skipped unless `--force` is given.
`--mbids` limits a command to a comma-separated list of recording ids, or to a file
with one id per line.

```bash
S="python -m saregama"          # or the installed `saregama` command
P=../annotations/carnatic/pilot_mbids.txt

$S manifest                                   # 1  index -> annotations/carnatic/manifest.csv
$S sync            --mbids $P                 # 2  offsets + mix-aligned FLAC stems
$S stems           --mbids $P                 # 3  de-bleed the vocal stem(s)        GPU
$S f0              --mbids $P                 # 4  F0, tonic, QC vs Saraga            GPU
$S tonic-consensus --mbids $P                 #    one Sa per concert (--dry-run to preview)
$S swaras          --mbids $P                 #    swara segments + flat contour
$S lyrics-fetch    --mbids $P                 # 5  candidate lyrics per composition (web)
$S lyrics-verify   --work <work_mbid>         #    show candidates / record a choice
$S lyrics-status                              #    rebuild lyrics/status.tsv (offline)
$S emissions       --mbids $P                 #    cache MMS emissions per voice      GPU
$S sections-train                             #    evaluate + train the section model
$S sections        --mbids $P                 #    automatic sections where Saraga has none
$S tala            --mbids $P                 #    beat grid, laya, sama (--evaluate to score)
$S align           --mbids $P                 # 6  time-aligned lyrics
$S viz             --mbids $P                 #    review/contour.pdf
$S review-export   --mbids $P                 # 7  Sonic Visualiser layers + sync preview
$S review-import   --mbids <id> --reviewer <name>
$S export-ir                                  # 8  ICM-IR v0 dataset -> annotations/export/icm_ir_v0
$S export-soulx                               #    SoulX clips (same splits) -> annotations/export/soulx_v0
```

`scripts/run_stages.sh <mbids-file> <gpu> [stage ...]` runs stages in order over a
list (default: `sync stems f0 emissions`, which need neither lyrics nor sections).

Long runs go in a tmux window or under `nohup`, with output teed to
`annotations/logs/`. Use `CUDA_VISIBLE_DEVICES` to choose GPUs.

Where the time goes:
- `sync` decodes each mp3 once (about 0.03× real time: a 5-minute recording takes
  about 7 s).
- `stems` and `f0` are the GPU stages.
- `align` runs MMS once per recording; re-aligning from the cache takes about 0.1 s.
- `review-export` mostly writes the preview FLAC (`--no-preview` skips it).

## What each stage does

### 1. `manifest` — `saregama/saraga.py`

Walks `saraga1.5_carnatic/<concert>/<recording>/`.

- **Matching files.** Saraga's file prefixes are not uniform, so files are matched
  by suffix only:
  - the mix: `.mp3.mp3` or `.mp3`;
  - stems: `.multitrack-<stem>.mp3`;
  - annotations: `.pitch.txt`, `.pitch-vocal.txt`, `.ctonic.txt`,
    `.sections-manual-p.txt`, `.sama-manual.txt`, …;
  - the metadata JSON, which gives raga, tala, form, composition (`work`) and lead
    artists.
- **Output.** `manifest.csv` lists every recording with flags such as `has_stems`,
  `has_tonic` and `has_sections`.

### 2. `sync` — `saregama/stages/sync.py`

The stems and the mix are separate mp3s whose starts differ by a few milliseconds.

**Measuring the offset.** `sync_recording` decodes the mix and each stem once at
44.1 kHz and derives the 16 kHz (GCC-PHAT) and 4 kHz (bleed) copies in memory.
- Each stem's lag against the mix is measured with GCC-PHAT on 10 s windows every
  20 s, at 16 kHz.
- The close-mic melodic stems (vocal, violin) agree to the sample.
- The mridangam and ghatam stems often lock onto a different path in the mix.
- So the recording's single offset is the median lag of the *tight* melodic stems,
  and every stem is shifted by it. The stems therefore stay mutually aligned.

**Output.**
- `stems/*.flac` at 44.1 kHz, exactly the mix's length, plus `mridangam` (both heads
  summed).
- `ir.json['sync']`.
- Per-stem bleed scores: the correlation of each stem's energy envelope with every
  other stem's.

Pilot finding: offsets are 0–9 ms, with no drift.

### 3. `stems` — `saregama/stages/separate.py` (`clean_stem`)

The close mics bleed: violin, mridangam and the other singer are audible in the
vocal stem. SoulX-Singer's Mel-RoFormer karaoke and de-reverb models are run **on the
vocal stem**, not on the mix. The result is `vocal_clean` (and `vocal-s_clean` for
duos), at the stem's own rate, length and level, so it stays sample-aligned.

### 4. `f0`, `tonic-consensus`, `swaras`, `viz` — `stages/f0.py`, `stages/f0_ir.py`, `stages/swaras.py`, `viz.py`

**F0 ensemble.**
- RMVPE (chunked; peak salience is used as confidence), FCPE and CREPE run on
  `vocal_clean` at 100 Hz.
- Each frame is octave-aligned to RMVPE and takes the median.
- `agree` stores the share of members within 50 cents of the median.
- Voiced frames are never smoothed, so gamakas survive.
- The members agree on about 98 % of voiced frames.

**Tonic.** Saraga's `ctonic` is sometimes Pa or Ma rather than Sa.
- `choose_tonic` scores Saraga's value, the pitch-histogram estimate, and their
  ×3/2 and ×4/3 relatives.
- The score is how well each explains the stable pitches of the raga, plus the
  time spent on Sa and Pa.
- The octave is placed just below the singer's median pitch.
- `tonic-consensus` then sets one Sa per concert. Every Saraga recording of the
  concert votes, and ours vote ×2.

**QC against Saraga.** Each of Saraga's pitch tracks is scored separately, with the
lag searched over ±30 ms.
- Some `pitch-vocal` files belong to another recording (wrong length) and are
  skipped.
- The mix-based `pitch` track often follows the violin, so a low score there isn't
  our error.
- Re-score without recomputing F0 using `f0 --qc-only`.

**Swaras.** The cents track (0 = Sa) is cut into four kinds of segment:

| Kind | Definition |
|---|---|
| stable | holds a svarasthana within 35 cents for ≥ 80 ms; *sustained* if ≥ 0.25 s |
| glide (meend) | moves ≥ 150 cents over ≥ 60 ms |
| kampita | ≥ 3 quick reversals within 500 cents |
| transit | any other movement |

The raga's swaras come from a table (`RAGAS`). Each position is then refined to this
recording's own histogram peak, which keeps, for example, a low Ga. Unknown ragas use
the histogram alone. The *flat* contour (stable swaras joined by straight lines) is
saved for flat→expressive gamaka modelling.

**Contour sheets.** `viz` writes `review/contour.pdf`:
- page 1: an overview and the stable-time histogram per swara;
- then 30 s panels showing swara gridlines, sustained swaras, meends, kampitas,
  sections, sama and lyric syllables.

### 5. `lyrics-fetch`, `lyrics-verify` — `saregama/lyrics_fetch.py`

See [Lyrics](#lyrics).

### Sections — `stages/sections.py`

Only 46 of the 168 multitrack recordings have Saraga's manual sections, and the
aligner needs them. `sections` labels every second of a recording as one of five
classes:

| Class | What is happening | Region label |
|---|---|---|
| vocal alapana | voice, no percussion (alapana, sloka) | `Vocal ālāp` |
| violin alapana | violin alone | `Violin ālāp` |
| kriti | voice + percussion (composition, neraval) | `Kriti` (all lines allowed) |
| svara | voice + percussion singing sargam | `Kalpanā svara` |
| tani | percussion solo | `Tani āvartana` |

- **Features, per second:** stem levels (vocal, violin, percussion) relative to their
  loud level, percussion onset rate, voicing of each singer, the MMS letter-spike rate
  and its consonant / sargam-consonant shares, and swara movement. Each is also
  averaged over ±2 s and ±10 s.
- **Model:** gradient boosting trained on the labelled recordings
  (`sections-train`, which first reports leave-one-concert-out accuracy against a
  rules baseline), then Viterbi smoothing with a switching cost and a 15 s minimum
  region.
- **Output:** `ir.json['sections_auto']`. Manual sections always take precedence;
  `align` and `review-export` use `sections`, falling back to `sections_auto`. The
  model is saved to `annotations/models/sections.pkl`.

### Tala — `stages/tala.py`

ICM-IR needs the laya and a matra index / sam flag on every syllable and swara.
Saraga annotates sama for 63 recordings and tempo for 60. Its tempo file is
`apm, bpm, cycle_s, aksharas_per_cycle, aksharas_per_beat`, which I checked against
the sama annotations; Ādi, for example, is `…, 32, 4`, i.e. 8 beats of 4 aksharas.

- **With annotated samas:** beats are subdivided evenly between consecutive samas.
- **Without:** the grid is estimated from the clean percussion stems.
  1. Spectral-flux onset strength at exactly 100 fps (audio at 16 kHz, hop 160).
  2. The beat period is the autocorrelation peak, weighted by a log-normal tempo prior
     at 95 bpm (against tempo-octave errors) and refined to sub-frame precision on
     the cycle-lag peak.
  3. Beats come from dynamic-programming tracking.
  4. Sama is the beat phase with the strongest bass-head (`mridangam-left`) strokes;
     `phase_conf` is its margin over the runner-up.
- **Output:** `ir.json['tala']` with `source`, `laya_bpm`, `beats_per_cycle`,
  `aksharas_per_beat`, `beats`, `beat_index` (0 = sama), `samas` and `phase_conf`.
- **Evaluation:** `tala --evaluate` scores the estimator against the 58 annotated
  recordings it can run on.
  - Beat period within 2 %: only 28 % of recordings; errors cluster at ×¾, ×½ and ×2.
  - Median beat F-measure: 0.29.
  - Median sama hit rate (within 70 ms): 6 %.

  Real mridangam accents and subdivisions defeat it, although it works on synthetic
  clicks. So by default `tala` writes grids only from annotated samas (`source:
  manual_sama`, `reliable: true`; 63 recordings, 13 h). Elsewhere it keeps the tala
  shape and Saraga's laya (`source: none`). `--estimate` writes estimated grids marked
  `reliable: false`, and the exports fill matra / sam only from reliable grids. Tala
  tracking for the other recordings needs a dedicated meter-tracking model; it is
  listed under next steps.

### 6. `align` — `stages/align_spikes.py` (default), `stages/align_kriti.py`

A kriti isn't sung as written:
- lines repeat with variations (sangati);
- one line can loop for minutes (neraval);
- the pallavi returns after every section;
- alapana, kalpana svara and tani carry no lyric text.

**Why spikes.** On melismatic singing, MMS_FA is blank on about 98 % of frames and
hears roughly a quarter of the letters. A frame-level CTC trellis (`--engine
trellis`, kept for comparison) therefore loses to the "not lyrics" filler. The
default engine works on MMS letter **spikes** instead:

1. Each run of a non-blank letter becomes one spike, renormalised over the non-blank
   letters.
2. A dynamic program assigns each spike one of three things:
   - the next lyric letter, skipping missed letters at a cost;
   - a melisma insertion;
   - filler (not lyrics).
3. The program runs over a **line-loop graph**:
   - from each line it can repeat the line, go to the next line, return to the
     pallavi or to its own section's start, or jump elsewhere (starting a
     *different* section from inside one, say caraṇam 1 from caraṇam 3, costs a
     full jump);
   - it can enter and leave a line mid-way, at word boundaries;
   - in kalpana svara it can also sing the sargam syllables sa…ni, which need only
     their consonant heard.
4. Saraga's section labels decide which lines are allowed in each region.
5. Syllables are then extended through continuous voicing (`extend_sustain`).

**Cached emissions.** MMS emissions are cached in `mms_50hz.npz`, so re-aligning
takes about 0.1 s per recording.

**Validation.** In kalpana svara, the sung sargam syllable matches the pitch track's
swara 45 % of the time (chance is 16 %). The match peaks within 50 ms of zero time
shift, which confirms the stems, F0 and lyrics share one clock to about 50 ms.

### 8. `export-ir` — `saregama/export_ir.py`

This writes the ICM-IR v0 dataset (the Phase 1 Model Landscape's spec) from the
bundles into `annotations/export/icm_ir_v0/`. The bundles stay the editable record.

| File | Level | Contents |
|---|---|---|
| `recordings.parquet` | global | tonic, raga, tala, laya (bpm), form, singer ids, duet, lyric status, section and tala source, split, flags |
| `syllables.parquet` | syllable | aligned text, ISO-15919, phones, start / duration, line / word / section, singer, beat / akshara / sam, `low_conf`, duet `overlap` |
| `swaras.parquet` | swara | kind, IR name (`Sa Re_k Re Ga_k Ga Ma Ma_t Pa Dha_k Dha Ni_k Ni`) + octave, svarasthana resolved for the raga (R2 vs G1, G2 vs R3, …), cents, beat / akshara / sam |
| `frames/<mbid>.npz` | 100 Hz | `cents_q` (cents re Sa in 10-cent steps, int16, −32768 unvoiced), `voiced`, `loudness_db`, `conf`, `phone_id`, `flat_cents_q` (the S2 flat → expressive pair); duets add `*_second` |
| `phones.json`, `splits.json`, `DATASHEET.md` | — | phone ids, the concert-level split, provenance and licence (Saraga: CC BY-NC 4.0) |

- **Splits** are by concert, so no singer appears in two splits. `--test-concerts` and
  `--val-concerts` force concerts into a split; the rest are assigned
  deterministically.
- **Phone timing.** `phone_id` places the consonants at a syllable's edges and lets the
  vowel fill the rest. Only the syllable span is aligned, not each phone.
- **Matra.** The matra fields are filled only inside kriti / svara / tani sections.

### 8. `export-soulx` — `saregama/export_soulx.py`

This writes SoulX-Singer training clips in melody mode (lyrics + F0). SoulX ships no
training code, so the data follows the metadata format its preprocessing writes and its
`DataProcessor` reads: 24 kHz audio, 50 fps, with `duration` / `text` / `phoneme` /
`note_pitch` / `note_type` / `f0` per clip. Extras: tonic, per-note cents and swara,
singer id, recording, split, lyric status.

**Notes**
- Each syllable is split into notes at swara-segment boundaries. The first note
  (type 2) carries the phones; the rest are type-3 slurs carrying only the vowel,
  because SoulX cycles a note's phones over its frames.
- A note needs phones + 3 frames (`<BOW>`, phones, `<SEP>`, `<EOW>`); shorter notes
  merge into a neighbour.
- In alapana sections, voiced stretches with no syllables become akaram notes (`in_a`).

**Clips**
- Clips are 2–16 s, cut in rests of at least 0.3 s, and padded only into silence.
- Voiced audio that nothing labels ends a phrase, so it never sits inside a clip
  labelled as a rest.
- Instances flagged `low_conf`, and duet stretches with more than 30 % overlap, are
  left out.
- Splits follow the ICM-IR export's `splits.json`.

**Output:** `wavs/`, `{train,val,test}.json(l)`, `stats.json`, and `soulx/`: the phone set
extended with the `in_` tokens (indices after SoulX's 2,820), the embedding
initialisation map, and the one-branch `data_processor.patch`.

**Pilot yield** (share of voiced time that ends up in clips): alapana 99 %, kalpana
svara 39 %, kriti 21 %. The kriti figure is limited by lyric alignment coverage
(42–70 % of voiced kriti time).

### 7. `review-export`, `review-import` — `saregama/review_sv.py`

See [Review in Sonic Visualiser](#review-in-sonic-visualiser).

## Output: the IR bundle

There is one folder per recording: `annotations/carnatic/<recording mbid>/`. Times are
seconds on the mix clock (t = 0 is the first sample of the Saraga mix), and frame k
of every 100 Hz array is at t = k / 100 s.

| File | Contents |
|---|---|
| `ir.json` | index: metadata, stems, tonic, sections, sama, stage log, QC, flags, status |
| `stems/*.flac` | `vocal`, `vocal-s` (duos), `violin`, `mridangam-left/right`, `mridangam`, `ghatam`, `vocal_clean`, `vocal-s_clean` |
| `f0_100hz.npz` | `fps`, `f0_hz` (0 = unvoiced), `cents` (re Sa, NaN unvoiced), `agree`, `conf`, `energy_db`, `track_rmvpe/fcpe/crepe` |
| `f0_100hz_vocal-s.npz` | the same for the second singer (duos) |
| `flat_100hz.npz` | `fps`, `flat_cents` |
| `swaras.json` | `raga`, `raga_swaras`, `positions_cents`, `summary`, `segments[]` |
| `sections_feat.npz` | cached per-second section features (`x`, `names`) |
| `lyrics.json` | `source`, `lines[]`, `instances[]`, `regions[]` |
| `lyrics_vocal-s.json` | the same for the second singer (duets) |
| `mms_50hz.npz`, `mms_50hz_vocal-s.npz` | `logp`: cached MMS_FA log-probabilities (float16) |
| `review/` | `contour.pdf`, `*.svl`, `sync_preview.flac`, `README.txt`, `corrected/` |

### `ir.json` keys

**Identity**
- `version`, `mbid`, `title`, `concert`, `lead`.
- `raaga`, `taala`, `form`, `work_mbid`, `work_title`, `length_s`.
- `mix`, `source_dir`: absolute Saraga paths, kept for provenance only.

**Tonic**
- `tonic_hz`, `tonic_source`: for example `saraga_ctonic*4/3`, `histogram` or
  `concert_consensus`.
- `tonic_saraga_hz`, `tonic_candidates[]`.
- `tonic_recording_hz`: set when the concert consensus overrode the recording's own
  tonic.

**Structure**
- `sections[]` as `{start, end, label}` (Saraga's manual sections), `sections_auto[]`
  (the `sections` stage: adds `class`), and `sama[]`.
- `tala`: see [Tala](#tala--stagestalapy).

**Stems**
- `stems.<name>`:
  - `path`: relative to the bundle;
  - `source` (original mp3) and `bleed` (correlation with each other stem);
  - for clean stems, `from`, `in_db`, `out_db` and `removed_db` instead.
- `sync`: `offset_s`, `consensus_stems`, `flags`, and per-stem `lag_s`, `iqr_ms`,
  `n_windows`, `drift_ms_per_min`, `tight`.

**Pipeline**
- `stages.<stage>`: a summary of each stage that has run (this drives resumability).
- `qc.f0.vocal_clean` holds:
  - `voiced` and `agree_voiced`;
  - `vs_saraga.<track>`: `lag_ms`, `rpa50`, `octave_err`, `both_voiced`, or
    `skipped`.
- `qc.swaras` (the swara summary), and `qc.align` / `qc.align_vocal-s` (duets):
  - `instances`, `lyric_s`, `low_conf`, `sargam`;
  - `lines_seen`, `n_lines`, `lyrics_verified`.
- `flags[]`, `status` (`auto` or `reviewed`), and `review` (reviewer, layers).

### `swaras.json` segment

```json
{"start": 12.34, "end": 12.71, "kind": "stable", "swara": "G", "svarasthana": "G3",
 "cents": 392.5, "cents_end": 0.0, "span": 18.0, "sustained": true}
```

`kind` is `stable`, `glide`, `kampita` or `transit`. Swara names carry octave marks:
`.D` is mandra, `S'` is tara.

### `lyrics.json`

```json
{"source": {"work_mbid": "...", "selected": "https://www.karnatik.com/c2227.shtml",
            "source": "auto", "verified": false},
 "lines": [{"id": 0, "section": "pallavi", "text": "bhuvini dAsuDan ...",
            "words": ["bhuvini", ...], "syllables": [["bhu", "vi", "ni"], ...]}],
 "instances": [{"line_id": 0, "section": "pallavi", "start": 7.44, "end": 16.43,
                "llr": -0.51, "complete": false, "region": "Pallavi",
                "flags": ["matched_9_of_57"], "singer": "lead", "overlap": 0.0,
                "words": [{"text": "dasudan", "start": 7.44, "end": 13.9,
                           "syllables": [{"text": "da", "start": 7.44, "end": 12.54}, ...]}]}],
 "regions": [{"label": "Pallavi", "kind": "lyrics", "start": 0.07, "end": 85.36,
              "instances": 5, "spikes": 168, "matched": 123, "sargam": 0, "lyric_s": 51.4}]}
```

**Instances.** An instance is one sung occurrence of a lyric line, so a line usually
has many.

**Duets.** Each cleaned vocal stem is aligned on its own: `lyrics.json` for the lead,
`lyrics_vocal-s.json` for the second singer. The two usually sing the same line in
heterophonic unison (pitch within 200 cents on about 80 % of shared frames, best at
0 ms lag) and take turns in kalpana svara and alapana. `overlap` is the share of an
instance during which the *other* singer is voiced. Exporters should prefer
`overlap ≈ 0` stretches, where a stem carries no unison bleed. `qc.align_vocal-s`
and the `solo_s` totals summarise this.

**Sargam.** Sargam syllables are instances whose `section` is `"sargam"`. Their
`line_id` is a pseudo-line numbered after the lyric lines.

## Lyrics

Saraga has no lyric text, so lyrics are fetched per **composition** (MusicBrainz work)
and stored in `annotations/lyrics/`.

**Fetching** (`lyrics-fetch`). For each composition:
1. MusicBrainz gives the composer, language and title aliases.
2. karnatik.com's index (about 8,000 songs) is fuzzy-matched on title, raga and
   composer.
3. The top candidate pages are parsed into pallavi / anupallavi / caraṇam lines.
   Section headers are recognised on any line, ignoring spaces and the sw/sv
   spelling (`ciTTaswaram`). A one-word "line" is joined to the line before it
   (karnatik.com wraps long lines), and stray language names are dropped.
4. A candidate with a match score ≥ 0.9 and raga similarity ≥ 0.85 is pre-selected
   (`source: auto`).

**Files written.**
- `<work_mbid>.yaml`: MusicBrainz info, every candidate with its scores and parsed
  sections, `selected`, `lyrics`, `verified`, `source`, `notes`.
- `status.tsv`: `auto`, `missing` or `verified` per composition, with its language, source
  and best match. It is rebuilt from every YAML after each `lyrics-fetch` and
  `lyrics-verify`, so a run on a subset never shrinks it; `lyrics-status` rebuilds it
  offline.
- `_cache/`: raw pages. Requests are rate-limited (1/s per host) and sent with a
  generic User-Agent.

**Transliteration** (`saregama/translit.py`). karnatik.com writes lyrics in a
Harvard-Kyoto-like ASCII scheme. `to_iso` gives ISO-15919 (`dAsuDanE` → `dāsuḍanē`) and
`to_phones` gives phones of `saregama/phoneset.py` with South Indian pronunciation: ṛ
is sung "ru", anusvara takes the next consonant's nasal, visarga is h, and ai/au are
diphthongs. Svara-notation lines (`pdsrsm…`) are recognised and sung as sargam. All
3,315 words of the selected lyrics convert; unknown letters raise an error rather than
producing a wrong phone. The aligner keeps its plain romanisation; `fit_syllables`
reconciles phone syllables with aligned syllables when their counts differ.

**Verifying** (`lyrics-verify`). Auto-selected lyrics are used for alignment, but the
recording is flagged `lyrics_unverified` until a person confirms them.

```bash
$S lyrics-verify --work <work_mbid>                       # show the candidates side by side
$S lyrics-verify --work <work_mbid> --pick 0 --reviewer <name>
$S lyrics-verify --work <work_mbid> --text lyrics.txt --source <url> --reviewer <name>
```

For `--text`, write a section name ending in `:` (for example `pallavi:`) on its own
line, followed by that section's lines; `#` starts a comment. After verifying, re-run
`align --force` for the recordings of that composition.

Lyrics are for internal annotation only and are not redistributed.

## Review in Sonic Visualiser

`review-export` writes into each bundle's `review/` folder:

| File | Layer |
|---|---|
| `f0.svl`, `flat.svl` | time-value curves (Hz) |
| `swaras.svl` | stable / kampita regions at their pitch, labelled with the swara |
| `lines.svl` | regions labelled `<line_id>\|<section>\|<text>` |
| `words.svl`, `syllables.svl` | regions (syllables drawn at their median F0) |
| `sections.svl`, `sama.svl` | regions, time instants |
| `*_vocal-s.svl` | duets: the second singer's F0, lines, words, syllables (view only) |
| `sync_preview.flac` | left: cleaned vocal; right: a sine on the F0, a tick per syllable onset, a knock per sama |
| `README.txt` | the exact open and import commands for this recording |

**Listening.** Listen to `sync_preview.flac` on headphones. Wherever the sine or the
ticks drift away from the voice, sync or alignment is off.

**Correcting.**
1. Open the audio and the layers: `sonic-visualiser <vocal_clean.flac> *.svl`, or
   *File › Import Annotation Layer*.
2. Edit the lines, words, syllables, sections or sama.
3. Export each edited layer under the same name into `review/corrected/`.
4. Run `review-import`.

**How import works.**
- Lyrics are rebuilt by containment: a syllable belongs to the word region that holds
  its midpoint, and a word to the line region that holds its midpoint. Regions may
  therefore be added, moved or deleted freely; keep the `line_id` in line labels.
- The recording becomes `status: reviewed`, and `align` never overwrites it again.

## Flags and statuses

| Flag (in `ir.json`) | Meaning |
|---|---|
| `sync:no_tight_stem` | no stem gave a consistent lag; the offset is a best guess |
| `sync:tight_stems_disagree_<x>ms` | the melodic stems disagree by more than 0.5 ms |
| `sync:vocal_not_tight` | the vocal stem's lag was inconsistent (heavy bleed or silence) |
| `tonic_ambiguous` | the top two tonic candidates scored within 0.05 |
| `tonic_from_concert` | the concert consensus replaced the recording's own tonic |
| `lyrics_unverified` | aligned against auto-selected lyrics no person has confirmed |

| Instance flag (in `lyrics.json`) | Meaning |
|---|---|
| `matched_X_of_Y` | X letters matched out of the Y in the line |
| `low_conf` | matched less than 25 % of the letters the instance spans |
| `resumed` | trellis engine: the line was left for filler and resumed |
| `reviewed` | came from a human-corrected layer |

`status` is `auto` until `review-import`, then `reviewed`.

## Code map

**Saraga pipeline**

| Module | Role |
|---|---|
| `saregama/paths.py` | data locations (layout defaults, `SAREGAMA_*` overrides) |
| `saregama/saraga.py` | Saraga loader, manifest |
| `saregama/ir.py` | the IR bundle (`Bundle`, `init_bundle`) |
| `saregama/stages/sync.py` | stem offsets (GCC-PHAT), aligned FLAC stems, bleed scores |
| `saregama/stages/separate.py` | `clean_stem`: vocal-stem de-bleed / de-reverb |
| `saregama/stages/f0.py` | F0 backends (RMVPE, FCPE, CREPE, YIN) and the ensemble |
| `saregama/stages/f0_ir.py` | stage `f0`: tonic choice, QC vs Saraga, concert consensus |
| `saregama/stages/tonic.py` | histogram tonic estimate, octave placement |
| `saregama/stages/swaras.py` | raga table, swara segmentation, flat contour |
| `saregama/viz.py` | contour sheets (PDF) |
| `saregama/lyrics_fetch.py` | MusicBrainz + karnatik.com lyric candidates |
| `saregama/translit.py` | karnatik.com scheme → ISO-15919 and Indic phones; svara notation |
| `saregama/stages/sections.py` | automatic sections (features, classifier, smoothing) |
| `saregama/stages/tala.py` | beat grid, laya, sama (from annotations or estimated) |
| `saregama/stages/align_spikes.py` | default lyric aligner (spike DP) |
| `saregama/stages/align_kriti.py` | text, line graph, section rules, MMS emitter, trellis engine |
| `saregama/review_sv.py` | Sonic Visualiser export / import, sync preview |
| `saregama/export_ir.py` | ICM-IR v0 dataset export |
| `saregama/export_soulx.py` | SoulX-Singer clip export (notes, akaram, clip planning) |
| `saregama/cli.py` | all commands |

**Legacy** (ported from `indic-svs-annotator`, for film songs: songs.csv workspaces,
Praat review). These modules are kept for the SoulX exporter, G2P and the phone set,
which step 8 will adapt:

`pipeline.py`, `schema.py`, `review.py` (Praat), `export.py` (SoulX clips),
`evaluate.py`, `g2p.py`, `phoneset.py`, `audio.py`, `config.py`,
`stages/{align,lyrics,notes,quality,qc}.py`, `annotator_model/` (optional unified
annotator network), `configs/default.toml`, `lexicons/common.tsv`.

## Conventions

- **One clock.** Every time is seconds on the mix clock. Stems are shifted onto it
  before any analysis; nothing is stored on a stem's own clock.
- **100 Hz master grid** (`ir.F0_HZ`). The SoulX 50 fps grid is derived only at
  export time.
- **Resumable stages.** A stage records itself in `ir.json['stages']`. Human-reviewed
  data is never overwritten.
- **No smoothing of voiced F0.** Gamakas are the signal, not noise.
- **Documentation is enforced.** Every module, class and function has a docstring,
  and non-obvious logic gets an inline comment. `tests/test_docs.py` fails on any
  undocumented definition or hard-coded machine path. Update this README together
  with any change to commands, outputs or file formats.

## Tests

```bash
.venv/bin/python -m pytest -q tests
```

| File | Covers |
|---|---|
| `test_sync.py` | a known stem shift is recovered and undone |
| `test_align_kriti.py` | syllabification; line-loop order (repeats, returns); resuming a held line |
| `test_review_sv.py` | Sonic Visualiser layers round-trip and re-nest by containment |
| `test_sections.py` | label mapping, smoothing, region merging, rules baseline |
| `test_tala.py` | tala shapes, grids from samas, matra lookup, metrics, estimation on a synthetic track |
| `test_lyrics_parse.py`, `test_lyrics_status.py` | karnatik.com parsing; status table |
| `test_translit.py` | ISO-15919 spelling, phones, syllable fitting, notation lines |
| `test_export_ir.py` | swara naming, svarasthana, splits, quantisation, phone track |
| `test_export_soulx.py` | notes per syllable, short-note merging, clip planning, item format |
| `test_pipeline.py` | legacy end to end on synthetic songs (G2P → … → SoulX format) |
| `test_annotator.py` | legacy annotator train / load / run smoke test |
| `test_docs.py` | docstrings everywhere, no hard-coded paths |

No test needs a GPU or a model download.

## Known limitations and next steps

**Limitations**
- **Lyrics.** 61 of 133 compositions have no lyrics yet. None of the 72 auto-matched
  ones have been verified by a person.
- **Alignment.**
  - It covers 160–450 s of lyrics per recording; heavy melisma still drops out.
  - It has not been measured against hand labels.
  - Known errors on the pilot: one false line match in Mati Matiki (around 1193 s)
    and a parse artifact in Dinamani Vamsa ("sarvasvamau").
- **Tonic.** Recordings flagged `tonic_ambiguous` need a listen.
- **Tala.** Matra and sam exist only for the 63 recordings with annotated samas (13 h);
  the percussion-based estimator is not reliable (median beat F 0.29).

**Next steps**
1. The Probe-40 Carnatic set, and an SVC rendering check: does SoulX's F0 conditioning
   keep gamakas?
2. Run all 168 multitrack recordings.
3. About 5 hand-corrected gold recordings, plus a Carnatic-Varnam onset evaluation.
4. Phase 2: the 81 recordings without stems (mix separation), then Hindustani.
