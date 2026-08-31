#!/usr/bin/env python3
"""Profile one Spark event log, or compare two (e.g. CPU vs GPU).

Combines parse_eventlog.py (job/stage/task summary) and extract_details.py
(config, SQL-execution timing, RAPIDS GPU metrics) into a single command, and
adds a --summary mode that prints a human-readable report instead of raw JSON.

Usage:
  profile.py <path>                       single-run profile
  profile.py <path_a> <path_b>             A/B comparison (naive + true speedup)

  <path> may be a v2 rolling event log directory (events_<n>_<appid> files) or
  a legacy single-file event log.

Examples:
  profile.py /logs/eventlog_v2_spark-cpu-run --summary
  profile.py /logs/eventlog_v2_spark-cpu-run /logs/eventlog_v2_spark-gpu-run --summary
  profile.py /logs/eventlog_v2_spark-cpu-run -o run.json --pretty
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_eventlog import parse
from extract_details import extract


def fmt_ms(ms):
    if ms is None:
        return "n/a"
    secs = ms / 1000
    if secs < 60:
        return f"{secs:.1f}s"
    return f"{secs / 60:.2f}m ({secs:.1f}s)"


def fmt_bytes(n):
    if n is None:
        return "n/a"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def fmt_pct(part, whole):
    if not whole:
        return "n/a"
    return f"{100.0 * part / whole:.1f}%"


def profile_run(path, top_stages=0, all_properties=False):
    run = dict(parse(path))
    details = extract(path)
    run["environment"] = details["environment"]
    if not all_properties:
        run["environment"].pop("all_properties", None)
    run["sql_executions"] = details["sql_executions"]
    run["sql_queries"] = details["sql_queries"]
    run["gpu_metrics"] = details["gpu_metrics"]
    if top_stages:
        run["stages"] = run["stages"][:top_stages]
    return run


def is_gpu_run(run):
    return any(rp.get("executor_gpus") for rp in run.get("resource_profiles", {}).values())


def label_run(run):
    return "GPU run" if is_gpu_run(run) else "CPU run"


def summarize_run(run, label=None, top_n=5):
    label = label or label_run(run)
    t = run["timing"]
    totals = run["totals"]
    lines = []
    lines.append(f"=== {label}: {run.get('app_name')} ({run.get('app_id')}) ===")
    lines.append(f"Spark version: {run.get('spark_version')}  |  source: {run.get('source')}")
    lines.append(
        f"Total: {fmt_ms(t['total_ms'])}  "
        f"(startup {fmt_ms(t['startup_ms'])} / execution {fmt_ms(t['execution_ms'])})"
    )
    if t.get("app_in_progress"):
        lines.append("NOTE: log is in-progress (.inprogress) -- total_ms is an estimate.")

    lines.append(
        f"Tasks: {totals['num_tasks']:,} ({totals['num_failed_tasks']} failed)  "
        f"Stages: {totals['num_stages']}  Jobs: {totals['num_jobs']}"
    )
    lines.append(
        f"Shuffle write: {fmt_bytes(totals['total_shuffle_write_bytes'])}  "
        f"Shuffle read: {fmt_bytes(totals['total_shuffle_read_bytes'])}  "
        f"Spill (mem/disk): {fmt_bytes(totals['total_mem_spill_bytes'])} / "
        f"{fmt_bytes(totals['total_disk_spill_bytes'])}  "
        f"GC: {fmt_ms(totals['total_gc_time_ms'])}"
    )

    lines.append(f"\nTop {top_n} stages by duration:")
    for s in run["stages"][:top_n]:
        skew = f"{s['skew']}x" if s.get("skew") else "n/a"
        lines.append(
            f"  stage {s['stage_id']}: {fmt_ms(s['duration_ms'])}  "
            f"tasks={s['task_count']}  skew={skew}  "
            f"shuffle_write={fmt_bytes(s['shuffle_write_bytes'])}"
        )

    if run["gpu_metrics"]:
        lines.append("\nGPU metrics:")
        for name, m in run["gpu_metrics"].items():
            unit = m["unit"] or ""
            val = m["value"]
            if m["unit"] == "ms":
                val = fmt_ms(m["value"])
                unit = ""
            elif m["unit"] == "bytes":
                val = fmt_bytes(m["value"])
                unit = ""
            lines.append(f"  {name}: {val}{unit}")

    curated = run["environment"]["curated"]
    if curated:
        lines.append("\nConfig (curated):")
        for k, v in curated.items():
            lines.append(f"  {k} = {v}")

    return "\n".join(lines)


def summarize_comparison(run_a, run_b, label_a=None, label_b=None):
    label_a = label_a or label_run(run_a)
    label_b = label_b or label_run(run_b)
    ta, tb = run_a["timing"], run_b["timing"]

    lines = []
    lines.append(f"\n=== Comparison: {label_a} vs {label_b} ===")
    lines.append(f"{label_a} total: {fmt_ms(ta['total_ms'])}  ({label_b} total: {fmt_ms(tb['total_ms'])})")

    if ta["total_ms"] and tb["total_ms"]:
        naive = ta["total_ms"] / tb["total_ms"]
        lines.append(f"Naive speedup ({label_a} / {label_b}, raw totals): {naive:.2f}x")
    if ta["execution_ms"] and tb["execution_ms"]:
        true_speedup = ta["execution_ms"] / tb["execution_ms"]
        lines.append(
            f"True speedup ({label_a} / {label_b}, execution-only, startup excluded): "
            f"{true_speedup:.2f}x"
        )
    lines.append(
        f"Startup share of total: {label_a}={fmt_pct(ta['startup_ms'], ta['total_ms'])}  "
        f"{label_b}={fmt_pct(tb['startup_ms'], tb['total_ms'])}"
    )

    curated_a = run_a["environment"]["curated"]
    curated_b = run_b["environment"]["curated"]
    diffs = {
        k: (curated_a.get(k), curated_b.get(k))
        for k in sorted(set(curated_a) | set(curated_b))
        if curated_a.get(k) != curated_b.get(k)
    }
    if diffs:
        lines.append("\nConfig differences:")
        for k, (va, vb) in diffs.items():
            lines.append(f"  {k}: {label_a}={va}  {label_b}={vb}")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="one path to profile, or two paths to compare (A/B)")
    ap.add_argument("-o", "--output", help="write JSON to this file instead of stdout")
    ap.add_argument("--pretty", action="store_true", help="pretty-print JSON")
    ap.add_argument("--summary", action="store_true", help="print a human-readable summary instead of JSON")
    ap.add_argument("--top-stages", type=int, default=0, help="cap stages list to top-N by duration (JSON mode)")
    ap.add_argument("--all-properties", action="store_true", help="include the full spark.* config dump")
    ap.add_argument("--label-a", help="display label for the first path (default: inferred CPU/GPU run)")
    ap.add_argument("--label-b", help="display label for the second path (default: inferred CPU/GPU run)")
    args = ap.parse_args()

    if len(args.paths) > 2:
        ap.error("pass at most two paths (single-run profile, or A/B comparison)")

    runs = [profile_run(p, top_stages=args.top_stages, all_properties=args.all_properties) for p in args.paths]

    if args.summary:
        out = "\n\n".join(summarize_run(r, label=(args.label_a, args.label_b)[i] if i < 2 else None)
                           for i, r in enumerate(runs))
        if len(runs) == 2:
            out += "\n" + summarize_comparison(runs[0], runs[1], args.label_a, args.label_b)
        print(out)
        return

    payload = runs[0] if len(runs) == 1 else {"runs": runs}
    text = json.dumps(payload, indent=2 if args.pretty else None)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
