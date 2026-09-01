#!/usr/bin/env python3
"""Compare the most recent CPU vs GPU Spark ETL runs in this project.

Auto-discovers event logs under spark-event-logs-dir (the directory every
ETL script here points spark.eventLog.dir at), profiles the two most
recent ones with ../spark-log-profiler, and prints a text summary with
naive/true speedup, per-run stage tables, and key observations. Run it
after the CPU ETL script (02_etl_v3.py / 02_etl_v4.py) and its GPU
counterpart (04_spark_rapids_etl.py / 04_spark_rapids_etl_v4.py) have
both run.

In a Cloudera AI Workbench session (or any IPython-backed console/
notebook), a matplotlib chart is also displayed inline automatically --
no flag needed. In a plain terminal there's no display surface, so use
--chart to save the same chart as a PNG instead.

Usage:
  python3 05_compare_cpu_gpu.py                latest two runs, text summary
                                                (+ inline chart, in a Workbench/notebook session)
  python3 05_compare_cpu_gpu.py --chart         also save a PNG bar chart to disk
  python3 05_compare_cpu_gpu.py --no-chart      suppress the automatic inline chart
  python3 05_compare_cpu_gpu.py --single        analyze only the single latest run
  python3 05_compare_cpu_gpu.py --list          list discovered event logs, newest first
  python3 05_compare_cpu_gpu.py --dir <path>    look in a different directory

No required arguments -- works the same whether launched from a terminal
or run directly in a Cloudera AI Workbench session.

Author: Brandon Antone
"""
import argparse
import importlib.util
import os
import sys

DEFAULT_EVENT_LOG_DIR = "/home/cdsw/spark-rapids-qualification-tool/spark-event-logs-dir"

try:
    # __file__ isn't defined when this code runs in a notebook cell
    # (as opposed to `python3 05_compare_cpu_gpu.py` from a terminal) --
    # fall back to this project's known layout under /home/cdsw.
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _SCRIPT_DIR = "/home/cdsw/spark-rapids-qualification-tool"

PROFILER_SCRIPT = os.path.join(_SCRIPT_DIR, "..", "spark-log-profiler", "profile.py")

# Prefixes Spark uses for event log files/directories: v2 rolling
# (eventlog_v2_spark-<appid>), legacy single-file (local-<ts>,
# spark-application-<ts>, app-<ts>).
EVENT_LOG_PREFIXES = ("eventlog_v2_", "local-", "spark-application-", "app-")


def load_profiler():
    spec = importlib.util.spec_from_file_location(
        "_spark_log_profiler", PROFILER_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_event_logs(event_log_dir):
    if not os.path.isdir(event_log_dir):
        return []

    candidates = [
        os.path.join(event_log_dir, name)
        for name in os.listdir(event_log_dir)
        if name.startswith(EVENT_LOG_PREFIXES)
    ]
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates


def render_time_breakdown(run, label, profiler, width=50):
    t = run["timing"]
    total_ms = t.get("total_ms")
    if not total_ms:
        return f"{label}: n/a"

    startup_ms = t.get("startup_ms") or 0
    execution_ms = t.get("execution_ms") or 0
    startup_chars = min(width, round(width * startup_ms / total_ms))
    exec_chars = width - startup_chars
    bar = "." * startup_chars + "#" * exec_chars

    lines = [f"{label}  ({profiler.fmt_ms(total_ms)} total)", f"  [{bar}]"]
    lines.append(
        f"  startup {profiler.fmt_ms(startup_ms)} "
        f"({profiler.fmt_pct(startup_ms, total_ms)})   "
        f"execution {profiler.fmt_ms(execution_ms)} "
        f"({profiler.fmt_pct(execution_ms, total_ms)})"
    )
    return "\n".join(lines)


def render_speedup_bars(naive, true_speedup, width=30):
    pairs = [("naive", naive), ("true", true_speedup)]
    present = [(name, v) for name, v in pairs if v is not None]
    if not present:
        return "  n/a"

    max_v = max(v for _, v in present)
    lines = []
    for name, v in present:
        bar_len = max(1, round((v / max_v) * width)) if max_v else 0
        bar = "#" * bar_len + "." * (width - bar_len)
        lines.append(f"  {name:<6} [{bar}] {v:.2f}x")
    return "\n".join(lines)


def render_stage_table(stages, profiler, top_n=8):
    header = f"  {'stage':>6} {'duration':>10} {'tasks':>6} {'skew':>7} {'shuffle_w':>10}"
    rule = "  " + "-" * (len(header) - 2)
    lines = [header, rule]
    for s in stages[:top_n]:
        skew = f"{s['skew']}x" if s.get("skew") else "n/a"
        lines.append(
            f"  {s['stage_id']:>6} {profiler.fmt_ms(s['duration_ms']):>10} "
            f"{s['task_count']:>6} {skew:>7} "
            f"{profiler.fmt_bytes(s['shuffle_write_bytes']):>10}"
        )
    return "\n".join(lines)


def derive_key_observations(run_a, run_b, label_a, label_b, profiler):
    obs = []
    ta, tb = run_a["timing"], run_b["timing"]

    share_a = ta["startup_ms"] / ta["total_ms"] if ta.get("total_ms") else None
    share_b = tb["startup_ms"] / tb["total_ms"] if tb.get("total_ms") else None
    if share_a is not None and share_b is not None and abs(share_a - share_b) > 0.15:
        lower_label = label_a if share_a < share_b else label_b
        obs.append(
            f"Startup share differs a lot between runs ({label_a}="
            f"{profiler.fmt_pct(ta['startup_ms'], ta['total_ms'])}, {label_b}="
            f"{profiler.fmt_pct(tb['startup_ms'], tb['total_ms'])}). {lower_label} "
            "may have run against an already-warm session (e.g. other jobs run "
            "earlier in the same notebook/session) -- trust the true speedup "
            "(execution-only) over the naive one here."
        )

    for label, run in ((label_a, run_a), (label_b, run_b)):
        worst = max(
            (s for s in run["stages"] if s.get("skew")),
            key=lambda s: s["skew"],
            default=None,
        )
        if worst and worst["skew"] >= 5:
            obs.append(
                f"{label}: stage {worst['stage_id']} shows {worst['skew']}x task skew "
                "-- a handful of partitions are doing disproportionate work; check "
                "the join/repartition key feeding that stage."
            )

    for label, run in ((label_a, run_a), (label_b, run_b)):
        totals = run["totals"]
        if totals.get("total_mem_spill_bytes") or totals.get("total_disk_spill_bytes"):
            obs.append(
                f"{label}: spill detected (mem="
                f"{profiler.fmt_bytes(totals['total_mem_spill_bytes'])}, disk="
                f"{profiler.fmt_bytes(totals['total_disk_spill_bytes'])}) -- "
                "executor memory may be undersized for this data volume."
            )
        if totals.get("num_failed_tasks"):
            obs.append(
                f"{label}: {totals['num_failed_tasks']} failed task(s) -- check "
                "for retried/OOM'd tasks in that run's executor logs."
            )

    if not obs:
        obs.append("No skew, spill, startup-share, or failed-task issues detected on either run.")

    return obs


def print_comparison_report(run_a, run_b, label_a, label_b, profiler):
    ta, tb = run_a["timing"], run_b["timing"]
    naive = (
        ta["total_ms"] / tb["total_ms"]
        if ta["total_ms"] and tb["total_ms"] else None
    )
    true_speedup = (
        ta["execution_ms"] / tb["execution_ms"]
        if ta["execution_ms"] and tb["execution_ms"] else None
    )

    print(f"=== CPU vs GPU comparison: {label_a} vs {label_b} ===\n")

    print("Where the time went")
    print(render_time_breakdown(run_a, label_a, profiler))
    print()
    print(render_time_breakdown(run_b, label_b, profiler))

    print("\nNaive vs true speedup")
    print(render_speedup_bars(naive, true_speedup))

    print(f"\n{label_a} -- top stages by duration")
    print(render_stage_table(run_a["stages"], profiler))
    print(f"\n{label_b} -- top stages by duration")
    print(render_stage_table(run_b["stages"], profiler))

    print("\nKey observations")
    for i, line in enumerate(derive_key_observations(run_a, run_b, label_a, label_b, profiler), 1):
        print(f"  {i}. {line}")


def in_notebook():
    """True when running under an IPython kernel/console -- a real Jupyter
    notebook, or a CML Workbench session (which runs a plain .py file
    through an embedded IPython console, not just genuine .ipynb files)."""
    try:
        return get_ipython() is not None
    except NameError:
        return False


def build_chart_figure(run_a, run_b, label_a, label_b):
    """Build (but don't save or show) the comparison figure. Caller decides
    whether to display it inline (notebook/Workbench) or save it to disk."""
    import matplotlib.pyplot as plt

    ta, tb = run_a["timing"], run_b["timing"]
    naive = (
        ta["total_ms"] / tb["total_ms"]
        if ta["total_ms"] and tb["total_ms"] else None
    )
    true_speedup = (
        ta["execution_ms"] / tb["execution_ms"]
        if ta["execution_ms"] and tb["execution_ms"] else None
    )

    labels = [label_a, label_b]
    totals_s = [ta["total_ms"] / 1000, tb["total_ms"] / 1000]
    startup_s = [(ta["startup_ms"] or 0) / 1000, (tb["startup_ms"] or 0) / 1000]
    execution_s = [(ta["execution_ms"] or 0) / 1000, (tb["execution_ms"] or 0) / 1000]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].bar(labels, totals_s, color=["tab:blue", "tab:green"])
    axes[0].set_ylabel("Total time (s)")
    axes[0].set_title("Total runtime")

    axes[1].bar(labels, startup_s, label="startup", color="tab:gray")
    axes[1].bar(labels, execution_s, bottom=startup_s, label="execution", color="tab:orange")
    axes[1].set_ylabel("Time (s)")
    axes[1].set_title("Startup vs execution")
    axes[1].legend()

    speedup_pairs = [("naive", naive), ("true", true_speedup)]
    speedup_labels = [n for n, v in speedup_pairs if v is not None]
    speedups = [v for _, v in speedup_pairs if v is not None]

    axes[2].bar(speedup_labels, speedups, color="tab:orange")
    axes[2].set_ylabel("Speedup (x)")
    axes[2].set_title(f"{label_a} / {label_b} speedup")
    for i, v in enumerate(speedups):
        axes[2].text(i, v, f"{v:.2f}x", ha="center", va="bottom")

    fig.suptitle("CPU vs GPU comparison")
    fig.tight_layout()
    return fig


def save_chart(run_a, run_b, label_a, label_b, out_path):
    import matplotlib
    matplotlib.use("Agg")
    fig = build_chart_figure(run_a, run_b, label_a, label_b)
    fig.savefig(out_path, dpi=150)
    print(f"\nChart saved to {out_path}")


def show_chart_inline(run_a, run_b, label_a, label_b):
    from IPython.display import display

    fig = build_chart_figure(run_a, run_b, label_a, label_b)
    display(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--dir",
        default=DEFAULT_EVENT_LOG_DIR,
        help="event log directory to search (default: %(default)s)",
    )
    ap.add_argument(
        "--single",
        action="store_true",
        help="analyze only the single most recent run instead of comparing two",
    )
    ap.add_argument(
        "--chart",
        action="store_true",
        help="also save a matplotlib PNG bar chart of the comparison to disk",
    )
    ap.add_argument(
        "--chart-out",
        default="cpu_vs_gpu_comparison.png",
        help="path to save the chart PNG (default: %(default)s)",
    )
    ap.add_argument(
        "--no-chart",
        action="store_true",
        help="in a notebook/Workbench session, skip the automatic inline chart",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="list discovered event logs (newest first) and exit",
    )
    # parse_known_args instead of parse_args: a Jupyter/notebook kernel
    # populates sys.argv with its own launcher args (e.g. -f
    # <connection-file>), which parse_args would reject outright. Ignore
    # anything we don't recognize instead of failing on it.
    args, _unknown = ap.parse_known_args()

    logs = find_event_logs(args.dir)
    if not logs:
        sys.exit(f"No event logs found under {args.dir}")

    if args.list:
        for path in logs:
            print(path)
        return

    profiler = load_profiler()

    if args.single:
        run = profiler.profile_run(logs[0])
        print(profiler.summarize_run(run))
        return

    if len(logs) < 2:
        sys.exit(
            f"Need two event logs to compare, only found one under {args.dir} "
            "(use --single to analyze it alone)"
        )

    # Oldest of the two first so labels/ordering read naturally
    # (e.g. CPU run before GPU run, if CPU ran first).
    path_a, path_b = logs[1], logs[0]
    run_a = profiler.profile_run(path_a)
    run_b = profiler.profile_run(path_b)
    label_a, label_b = profiler.label_run(run_a), profiler.label_run(run_b)

    print(profiler.summarize_run(run_a, label=label_a))
    print()
    print(profiler.summarize_run(run_b, label=label_b))
    print(profiler.summarize_comparison(run_a, run_b, label_a, label_b))
    print()
    print_comparison_report(run_a, run_b, label_a, label_b, profiler)

    if in_notebook() and not args.no_chart:
        try:
            show_chart_inline(run_a, run_b, label_a, label_b)
        except ImportError:
            print(
                "\nmatplotlib not installed -- skipping inline chart. "
                "Install with: pip install matplotlib"
            )

    if args.chart:
        try:
            save_chart(run_a, run_b, label_a, label_b, args.chart_out)
        except ImportError:
            print(
                "\nmatplotlib not installed -- skipping chart. "
                "Install with: pip install matplotlib"
            )


if __name__ == "__main__":
    main()
