# Visual report style (confirmed preference)

Optional: if you're building a shareable visual (HTML/slide) report from
`profile.py`'s output for a two-run comparison, use this layout and styling
by default rather than designing from scratch. It was validated against
`cpu_gpu_etl_report.png` (the original hand-made report) and confirmed again
after a from-scratch alternative was tried and rejected in favor of this one.

## Structure, top to bottom

1. **Header** — title as a short name (e.g. "CPU vs GPU ETL Job Profile"),
   one dense subtitle line packing in the key identifying facts (job counts,
   allocation mode, executor counts) rather than a separate stats table.
   No badge/chip row under the header.
2. **Stat tiles** (3, in individually-bordered cards, no outer wrapper) —
   naive speedup, true speedup, and one "share of total" callout (e.g.
   startup %). This is the only place plain numbers get color-coded as the
   headline.
3. **Timeline section** — one card containing a stacked horizontal bar per
   run, each bar independently scaled to its own total (not a shared time
   axis). Segment legend goes in small bordered "legend boxes" directly
   under each bar, not a separate legend row. Use a diagonal-hatch fill
   (`repeating-linear-gradient(45deg, ...)`) for any segment that's dead
   time (idle gap, inferred pre-log startup) — never a plain color for
   "nothing happened here."
4. **Naive vs. true comparison** — one card with a small vertical bar pair
   (not horizontal), bar height roughly proportional to the multiplier,
   naive in neutral gray/ink, true in a dedicated "good result" color
   (green) that is used *only* for this concept, never as a run identity.
   A small breakdown table under the bars showing how "true" was derived
   (e.g. matched SQL executions) keeps the number auditable.
5. **Supporting bare tables** (config diff, execution volume, GPU-only
   metrics) — no card border, no box. Plain hairline-divided rows, muted
   uppercase column headers. These carry the "deeper breakdown" content
   (Spark config, SQL-execution timing, RAPIDS accumulators) that the
   original report didn't have room for, in the same restrained style so
   they don't visually compete with the sections above.
6. **Heaviest stages** — two bare tables side by side (one per run), no
   pill/chip for skew — just bold the number itself in an amber/orange
   "hot" color past the severity threshold (~5×), plain otherwise. A
   one-line italic footnote below explains any stage/job-ID numbering
   confound (see `interpreting-results.md`).
7. **Key takeaways** — one bordered card, numbered list, bold lead clause
   per item followed by the explanation in regular weight.
8. **Footer** — one muted line citing the source event log paths. No
   heavier treatment than the body text.

## Color roles (fixed assignment, not per-report choice)

- One run = orange, the other = blue — pick consistently per report (e.g.
  CPU=orange, GPU=blue) and hold it through every section: timeline bar,
  table value cells, stage-column headings. Don't let identity colors leak
  into the naive/true comparison.
- Green is reserved for "the corrected/fair number" — never a run identity.
- Neutral gray/ink for "naive" or baseline.
- Everything else (body text, table headers, borders) stays in a quiet
  warm-neutral ink/gray scale — no other hues.

## What NOT to do

- Don't wrap every section in a bordered card — only stat tiles, the
  timeline, the naive/true comparison, and takeaways get card chrome. Bare
  tables everywhere else is what keeps the page from feeling like a
  dashboard-builder default.
- Don't use pill/chip badges for skew severity or run identity — bold text
  color is enough, and matches the original's minimalism.
- Don't invent a shared time axis across both runs' timeline bars unless
  asked — the original scales each bar to its own total, which is what
  reads as "clean" rather than "trying to prove a point."
