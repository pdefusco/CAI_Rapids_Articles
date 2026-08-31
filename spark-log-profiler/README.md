# Spark Log Profiler

Parse raw Spark (including Spark-RAPIDS/GPU) event logs into structured
job/stage/task data, and get a fair speedup number when comparing two runs
(e.g. CPU vs GPU) instead of a raw wall-clock ratio that can be skewed by a
cold vs. warm start.

Pure standard library, no dependencies. Python 3.7+.

## What it does

- Parses event logs (v2 rolling directories, or legacy single-file logs,
  gzip'd or not) into job/stage/task timing, executor timeline, shuffle/spill/GC,
  and startup-vs-execution split.
- Pulls out Spark config, SQL-execution-level (query) timing, and RAPIDS GPU
  accumulator metrics (`gpuTime`, `gpuSemaphoreWait`, decode/encode time) that
  the job/stage/task summary alone doesn't cover.
- For a two-run comparison, reports both the naive speedup (raw totals) and
  the true speedup (execution-only, startup excluded) — see
  [`reference/interpreting-results.md`](reference/interpreting-results.md) for
  why these can differ a lot and which one you should actually be quoting.

## Usage

Single run, human-readable summary:

```bash
python3 profile.py /path/to/eventlog_v2_spark-<id> --summary
```

Two runs (e.g. CPU then GPU) — naive + true speedup, config diff:

```bash
python3 profile.py /path/to/cpu_run /path/to/gpu_run --summary
```

Raw JSON (for scripting / piping into something else):

```bash
python3 profile.py /path/to/eventlog_v2_spark-<id> -o run.json --pretty
```

`<path>` is either a v2 rolling event log directory (containing
`events_<n>_<appid>` files) or a single legacy event log file.

### Flags

| Flag | Meaning |
|---|---|
| `--summary` | Print a human-readable report instead of JSON |
| `-o / --output FILE` | Write JSON to a file instead of stdout |
| `--pretty` | Pretty-print JSON |
| `--top-stages N` | Cap the stages list to the top-N by duration (JSON mode) |
| `--all-properties` | Include the full `spark.*` config dump, not just the curated subset |
| `--label-a` / `--label-b` | Override the CPU/GPU-inferred run labels in `--summary` output |

### Using the two scripts directly

`profile.py` is a thin wrapper. You can also run the two parsers
independently if you only need one piece:

```bash
# job/stage/task summary only
python3 parse_eventlog.py /path/to/eventlog_v2_spark-<id> -o summary.json

# config, SQL-execution timing, GPU metrics only
python3 extract_details.py /path/to/eventlog_v2_spark-<id> --no-all-properties -o details.json
```

Both accept multiple paths (emits a JSON array) and `--pretty`.

## Interpreting the output

Read [`reference/interpreting-results.md`](reference/interpreting-results.md)
before drawing conclusions from the numbers — it covers non-obvious judgment
calls: what counts as "startup" on an in-progress log, how to spot a
warm-session confound when comparing two runs, what a skew number actually
implies, and RAPIDS accumulator quirks (some GPU metrics come through as bare
nanosecond integers, not the pre-formatted `HH:MM:SS.mmm` strings most others
use).

[`reference/report-style.md`](reference/report-style.md) documents a
validated layout/color scheme if you want to turn a comparison into a
shareable visual report rather than reading the summary in a terminal.

## Origin

This was originally built as a Claude Code skill
(`spark-log-profiler`) for profiling Spark-RAPIDS benchmark runs. Pulled out
here as a standalone tool with no Claude Code dependency, so it can be run
directly.
