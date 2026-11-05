"""Dataset QC report: one row per recording bundle, plus totals, as CSV and HTML.

``saregama report`` reads every ``<ann-root>/<mbid>/ir.json`` and the lyric
status table (``<lyrics-dir>/status.tsv``) and writes:

- ``report.csv``: one row per recording (:func:`summarise`);
- ``report.html``: one self-contained page (no external files) with
  headline numbers, stage completion, hours by raga / singer / form, the
  outliers that need a look, and the full per-recording table (sortable,
  filterable).

Every recording that is missing a stage gets a stated reason (:func:`missing_reason`),
so the report shows where each recording stopped, and why.
"""
from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from pathlib import Path

# Pipeline stages in run order, as recorded in ``ir.json['stages']``.
STAGES = ["sync", "stems", "f0", "swaras", "sections", "tala", "align"]

# Outlier thresholds (see :func:`outliers`).
MIN_RPA50 = 0.5          # F0 vs Saraga's pitch track: share within 50 cents
MIN_VOICED = 0.2         # share of frames the lead vocal is voiced
MAX_OFFSET_MS = 50.0     # stem-to-mix sync offset
MIN_LYRIC_SHARE = 0.10   # aligned lyric minutes / recording minutes, for aligned recordings

CSV_FIELDS = ["mbid", "title", "concert", "singers", "raga", "tala", "form", "minutes", "duet",
              "tonic_hz", "tonic_source", "sections_source", "tala_source", "laya_bpm",
              "lyrics_status", "lyric_min", "align_instances", "align_mean_llr",
              "voiced", "rpa50", "sync_offset_ms", "flags", "stages_done", "missing", "missing_reason"]


def load_lyric_status(lyrics_dir: Path) -> dict[str, str]:
    """``work_mbid -> status`` (auto / verified / missing) from ``status.tsv``;
    empty if the table is absent."""
    path = Path(lyrics_dir) / "status.tsv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {r["work_mbid"]: r["status"] for r in csv.DictReader(fh, delimiter="\t")}


def missing_reason(stage: str, ir: dict, lyric_status: str) -> str:
    """Why ``stage`` has not run on this recording, as far as the bundle can tell.

    The reasons follow each command's own skip rules (see ``cli.py``): a stage
    needs the previous one, ``align`` needs selected lyrics, ``tala`` needs a
    known tala shape, ``sections`` only runs where Saraga has no manual sections.
    """
    done = ir.get("stages", {})
    prev = {"stems": "sync", "f0": "stems", "swaras": "f0", "sections": "swaras", "align": "stems"}
    if stage in prev and prev[stage] not in done:
        return f"waits for {prev[stage]}"
    if stage == "sections" and ir.get("sections"):
        return "manual sections (not needed)"
    if stage == "align":
        if not ir.get("work_mbid"):
            return "no composition (alapana / tani / RTP)"
        if lyric_status in ("", "missing"):
            return "no lyrics found yet"
    if stage == "tala" and (ir.get("tala") or {}).get("source") in ("none", "unknown"):
        return "no manual sama"
    return "not run yet"


def summarise(ir: dict, lyric_status: dict[str, str]) -> dict:
    """One report row for a bundle's ``ir.json`` (see ``CSV_FIELDS``)."""
    st = ir.get("stages", {})
    qc = ir.get("qc", {})
    f0 = (qc.get("f0") or {}).get("vocal_clean") or {}
    vs = f0.get("vs_saraga") or {}
    vs = vs.get("pitch", vs)                       # older bundles store the scores one level up
    al = qc.get("align") or {}
    tala = ir.get("tala") or {}
    ls = lyric_status.get(ir.get("work_mbid") or "", "")
    if ir.get("sections"):
        sec = "manual"
    elif ir.get("sections_auto"):
        sec = "auto"
    else:
        sec = "none"
    missing = [s for s in STAGES if s not in st and not (s == "sections" and ir.get("sections"))]
    reasons = [missing_reason(s, ir, ls) for s in missing]
    return {
        "mbid": ir.get("mbid", ""), "title": ir.get("title", ""),
        "concert": (ir.get("concert") or "").split(" by ")[0],
        "singers": "; ".join(ir.get("lead") or []), "raga": ir.get("raaga") or "",
        "tala": ir.get("taala") or "", "form": ir.get("form") or "",
        "minutes": round((ir.get("length_s") or 0) / 60, 2),
        "duet": "vocal-s_clean" in (ir.get("stems") or {}),
        "tonic_hz": ir.get("tonic_hz"), "tonic_source": ir.get("tonic_source") or "",
        "sections_source": sec, "tala_source": tala.get("source", ""), "laya_bpm": tala.get("laya_bpm"),
        "lyrics_status": ("no composition" if not ir.get("work_mbid") else (ls or "missing")),
        "lyric_min": round((al.get("lyric_s") or 0) / 60, 2), "align_instances": al.get("instances"),
        "align_mean_llr": al.get("mean_llr"),
        "voiced": f0.get("voiced"), "rpa50": vs.get("rpa50"),
        "sync_offset_ms": (st.get("sync") or {}).get("offset_ms"),
        "flags": "; ".join(map(str, ir.get("flags") or [])),
        "stages_done": ", ".join(s for s in STAGES if s in st),
        "missing": ", ".join(missing),
        "missing_reason": "; ".join(f"{s}: {r}" for s, r in zip(missing, reasons)),
    }


def outliers(row: dict) -> list[str]:
    """Problems worth a look in one report row (empty list if none)."""
    out = []
    if row["rpa50"] is not None and row["rpa50"] < MIN_RPA50:
        out.append(f"F0 agrees with Saraga only {row['rpa50']:.0%} within 50 c")
    if row["voiced"] is not None and row["voiced"] < MIN_VOICED:
        out.append(f"lead voiced only {row['voiced']:.0%} of the time")
    if row["sync_offset_ms"] is not None and abs(row["sync_offset_ms"]) > MAX_OFFSET_MS:
        out.append(f"sync offset {row['sync_offset_ms']:.0f} ms")
    for f in row["flags"].split("; "):
        if f.startswith("sync:") or f == "tonic_ambiguous":
            out.append(f)
    if "align" in row["stages_done"] and row["minutes"] and row["lyric_min"] / row["minutes"] < MIN_LYRIC_SHARE:
        out.append(f"little aligned lyric time ({row['lyric_min']:.1f} of {row['minutes']:.0f} min)")
    return out


def collect(ann_root: Path, lyrics_dir: Path) -> list[dict]:
    """Report rows for every bundle under ``ann_root``, sorted by concert and title."""
    status = load_lyric_status(lyrics_dir)
    rows = [summarise(json.loads(p.read_text(encoding="utf-8")), status)
            for p in sorted(Path(ann_root).glob("*/ir.json"))]
    return sorted(rows, key=lambda r: (r["concert"], r["title"]))


def hours_by(rows: list[dict], key: str, top: int = 0) -> list[tuple[str, int, float]]:
    """``(value, recordings, hours)`` grouped by ``rows[key]``, most hours first;
    with ``top`` > 0 the tail is folded into one "other" row."""
    agg = defaultdict(lambda: [0, 0.0])
    for r in rows:
        for v in (r[key].split("; ") if key == "singers" else [r[key]]):
            agg[v or "(none)"][0] += 1
            agg[v or "(none)"][1] += r["minutes"] / 60
    out = sorted(((k, n, h) for k, (n, h) in agg.items()), key=lambda t: -t[2])
    if top and len(out) > top:
        rest = out[top:]
        out = out[:top] + [(f"other ({len(rest)})", sum(t[1] for t in rest), sum(t[2] for t in rest))]
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    """Write the report rows to ``path`` as CSV."""
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _esc(v) -> str:
    """HTML-escaped text of a cell value ("" for None, 2 decimals for floats)."""
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.2f}"
    return html.escape(str(v))


def _table(head: list[str], body: list[list], cls: str = "") -> str:
    """A plain HTML table; numeric cells are right-aligned via the ``num`` class."""
    th = "".join(f"<th>{_esc(h)}</th>" for h in head)
    tr = "".join("<tr>" + "".join(
        f'<td class="{"num" if isinstance(c, (int, float)) and not isinstance(c, bool) else ""}">{_esc(c)}</td>'
        for c in row) + "</tr>" for row in body)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>'


# Page style: colour tokens on :root with a dark-mode set; one column on phones.
CSS = """
:root{--bg:#f6f7f5;--panel:#fff;--ink:#1d2321;--ink2:#4f5955;--muted:#7b8580;--line:#dde2de;--accent:#0d6b63;--bar:#9fcfc8;--warn:#8f6300;--warnbg:#f7ecd2}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#111614;--panel:#19201d;--ink:#e5ebe8;--ink2:#b2bcb7;--muted:#86918c;--line:#2a3330;--accent:#55b8ac;--bar:#2f6d65;--warn:#e2b452;--warnbg:#33291a}}
:root[data-theme="dark"]{--bg:#111614;--panel:#19201d;--ink:#e5ebe8;--ink2:#b2bcb7;--muted:#86918c;--line:#2a3330;--accent:#55b8ac;--bar:#2f6d65;--warn:#e2b452;--warnbg:#33291a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1400px;margin:0 auto;padding:20px 16px 48px}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 10px}
.sub{color:var(--ink2);margin:0 0 16px}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px}.tile b{display:block;font-size:22px;font-variant-numeric:tabular-nums}
.tile span{color:var(--muted);font-size:13px}.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.scroll{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:5px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{position:sticky;top:0;background:var(--panel);color:var(--ink2);font-weight:600;cursor:pointer;white-space:nowrap}
td.num{text-align:right;font-variant-numeric:tabular-nums}.barcell{width:40%}.bar{height:10px;background:var(--bar);border-radius:0 3px 3px 0}
.warn{background:var(--warnbg);color:var(--warn);border-radius:4px;padding:1px 6px;display:inline-block;margin:1px 2px 1px 0;font-size:12px}
input{font:inherit;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:6px 10px;width:min(360px,100%);margin-bottom:8px}
"""

# Sort the full table on header click; filter its rows by the search box.
JS = """
document.querySelectorAll('table.sortable th').forEach((th,i)=>th.addEventListener('click',()=>{
 const tb=th.closest('table').tBodies[0],rows=[...tb.rows],asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';
 const val=r=>{const t=r.cells[i].textContent.trim(),n=parseFloat(t);return t!==''&&!isNaN(n)&&/^[-\\d.]/.test(t)?n:t.toLowerCase()};
 rows.sort((a,b)=>{const x=val(a),y=val(b);return (x>y?1:x<y?-1:0)*(asc?1:-1)});rows.forEach(r=>tb.appendChild(r))}));
const q=document.getElementById('q');if(q)q.addEventListener('input',()=>{const s=q.value.toLowerCase();
 document.querySelectorAll('#all tbody tr').forEach(r=>r.hidden=s&&!r.textContent.toLowerCase().includes(s))});
"""


def render_html(rows: list[dict], generated: str = "") -> str:
    """The whole report page as one HTML string (inline CSS and JS)."""
    hours = sum(r["minutes"] for r in rows) / 60
    stage_rows = []
    for s in STAGES:
        done = [r for r in rows if s in r["stages_done"].split(", ")]
        if s == "sections":
            done = [r for r in rows if r["sections_source"] in ("manual", "auto")]
        reasons = defaultdict(int)
        for r in rows:
            for part in r["missing_reason"].split("; "):
                if part.startswith(s + ":"):
                    reasons[part.split(": ", 1)[1]] += 1
        stage_rows.append([s, len(done), round(sum(r["minutes"] for r in done) / 60, 1),
                           len(rows) - len(done), ", ".join(f"{k} ({v})" for k, v in sorted(reasons.items()))])
    lyric_h = sum(r["lyric_min"] for r in rows) / 60
    tiles = [("recordings", len(rows)), ("hours of music", f"{hours:.1f}"),
             ("concerts", len({r["concert"] for r in rows})), ("ragas", len({r["raga"] for r in rows})),
             ("duets", sum(r["duet"] for r in rows)), ("hours of aligned lyrics", f"{lyric_h:.1f}"),
             ("with manual sama", sum(r["tala_source"] == "manual_sama" for r in rows)),
             ("need a look", sum(bool(outliers(r)) for r in rows))]

    def group(key, top, label):
        """One "hours by <label>" table with inline bars (bar length = share of the top row)."""
        g = hours_by(rows, key, top)
        mx = max((t[2] for t in g), default=1) or 1
        body = "".join(f'<tr><td>{_esc(k)}</td><td class="num">{n}</td><td class="num">{h:.1f}</td>'
                       f'<td class="barcell"><div class="bar" style="width:{100 * h / mx:.0f}%"></div></td></tr>'
                       for k, n, h in g)
        return (f'<div><h2>Hours by {label}</h2><div class="scroll"><table><thead><tr><th>{label}</th>'
                f'<th>recordings</th><th>hours</th><th></th></tr></thead><tbody>{body}</tbody></table></div></div>')

    odd = [[r["title"], r["concert"], " ".join(f'<span class="warn">{_esc(o)}</span>' for o in outliers(r))]
           for r in rows if outliers(r)]
    odd_html = "".join(f"<tr><td>{_esc(a)}</td><td>{_esc(b)}</td><td>{c}</td></tr>" for a, b, c in odd)
    cols = ["title", "concert", "raga", "tala", "form", "minutes", "duet", "tonic_hz", "tonic_source",
            "sections_source", "tala_source", "lyrics_status", "lyric_min", "rpa50", "voiced",
            "sync_offset_ms", "stages_done", "missing_reason", "flags"]
    full = _table(cols, [[("yes" if r[c] else "") if c == "duet" else r[c] for c in cols] for r in rows], "sortable")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Saraga Dataset Report</title>
<style>{CSS}</style></head><body><div class="wrap">
<h1>Saraga Carnatic: dataset report</h1>
<p class="sub">saregama624 annotation bundles{(" · generated " + _esc(generated)) if generated else ""}.
Each stage counts the recordings that finished it; missing ones say why.</p>
<div class="tiles">{"".join(f'<div class="tile"><b>{_esc(v)}</b><span>{_esc(k)}</span></div>' for k, v in tiles)}</div>
<h2>Stages</h2><div class="scroll">{_table(["stage", "recordings done", "hours done", "not done", "why not"], stage_rows)}</div>
<div class="cols">{group("raga", 20, "raga")}{group("singers", 20, "singer")}{group("form", 0, "form")}{group("lyrics_status", 0, "lyrics_status")}</div>
<h2>Needs a look ({len(odd)})</h2><div class="scroll"><table><thead><tr><th>recording</th><th>concert</th><th>why</th></tr></thead><tbody>{odd_html}</tbody></table></div>
<h2>All recordings</h2><input id="q" type="search" placeholder="Filter rows" aria-label="Filter recordings">
<div class="scroll" id="all">{full}</div>
</div><script>{JS}</script></body></html>"""


def write_report(ann_root: Path, lyrics_dir: Path, out_dir: Path, generated: str = "") -> list[dict]:
    """Write ``report.csv`` and ``report.html`` into ``out_dir``; return the rows."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect(ann_root, lyrics_dir)
    write_csv(rows, out_dir / "report.csv")
    (out_dir / "report.html").write_text(render_html(rows, generated), encoding="utf-8")
    return rows
