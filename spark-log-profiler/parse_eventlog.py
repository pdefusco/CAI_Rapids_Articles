#!/usr/bin/env python3
"""Parse Spark event log(s) into a JSON summary of jobs, stages, tasks, and executors.

Supports:
  - v2 "rolling" event logs: a directory containing events_<n>_<appid>[.gz] files
  - legacy single-file event logs (plain or .gz)

Usage:
  parse_eventlog.py <path> [<path> ...] [-o out.json] [--pretty]

<path> may be a directory (v2 rolling log) or a single file (legacy log).
Multiple paths are parsed independently and emitted as a JSON array.
"""
import argparse
import gzip
import json
import os
import statistics
import sys
from collections import defaultdict


def _open(path):
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def _event_files(path):
    if os.path.isdir(path):
        candidates = [
            f for f in os.listdir(path)
            if f.startswith("events_") and not f.endswith(".crc")
        ]
        def sort_key(fname):
            # events_<n>_<appid>[.gz] -> sort by rolling segment number n
            parts = fname.split("_")
            try:
                return int(parts[1])
            except (IndexError, ValueError):
                return fname
        candidates.sort(key=sort_key)
        if not candidates:
            raise FileNotFoundError(f"no events_* files found in {path}")
        return [os.path.join(path, f) for f in candidates]
    return [path]


def _iter_events(path):
    for fpath in _event_files(path):
        with _open(fpath) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # truncated/in-progress final line, skip


def _pct(part, whole):
    return round(100.0 * part / whole, 1) if whole else None


def parse(path):
    app_id = app_name = spark_version = user = None
    app_start_ts = app_end_ts = None
    driver_bm_ts = None
    resource_profiles = {}

    jobs = {}          # job_id -> dict
    stages = {}         # (stage_id, attempt) -> dict
    stage_tasks = defaultdict(list)  # (stage_id, attempt) -> list of task dicts

    executors_added = []
    executors_removed = []

    last_event_ts = None

    for ev in _iter_events(path):
        etype = ev.get("Event")
        ts = ev.get("Timestamp")
        if isinstance(ts, int):
            last_event_ts = ts if last_event_ts is None else max(last_event_ts, ts)

        if etype == "SparkListenerLogStart":
            spark_version = ev.get("Spark Version")

        elif etype == "SparkListenerApplicationStart":
            app_name = ev.get("App Name")
            app_id = ev.get("App ID")
            user = ev.get("User")
            app_start_ts = ev.get("Timestamp")

        elif etype == "SparkListenerApplicationEnd":
            app_end_ts = ev.get("Timestamp")

        elif etype == "SparkListenerBlockManagerAdded":
            bm = ev.get("Block Manager ID", {})
            if bm.get("Executor ID") == "driver" and driver_bm_ts is None:
                driver_bm_ts = ev.get("Timestamp")

        elif etype == "SparkListenerResourceProfileAdded":
            rp_id = ev.get("Resource Profile Id")
            ereq = ev.get("Executor Resource Requests", {})
            treq = ev.get("Task Resource Requests", {})
            resource_profiles[rp_id] = {
                "executor_cores": (ereq.get("cores") or {}).get("Amount"),
                "executor_gpus": (ereq.get("gpu") or {}).get("Amount"),
                "executor_memory_mb": (ereq.get("memory") or {}).get("Amount"),
                "task_cpus": (treq.get("cpus") or {}).get("Amount"),
                "task_gpus": (treq.get("gpu") or {}).get("Amount"),
            }

        elif etype == "SparkListenerExecutorAdded":
            info = ev.get("Executor Info", {})
            executors_added.append({
                "executor_id": ev.get("Executor ID"),
                "ts": ev.get("Timestamp"),
                "host": info.get("Host"),
                "cores": info.get("Total Cores"),
                "resource_profile_id": info.get("Resource Profile Id"),
            })

        elif etype == "SparkListenerExecutorRemoved":
            executors_removed.append({
                "executor_id": ev.get("Executor ID"),
                "ts": ev.get("Timestamp"),
                "reason": ev.get("Removed Reason"),
            })

        elif etype == "SparkListenerJobStart":
            jid = ev.get("Job ID")
            jobs[jid] = {
                "job_id": jid,
                "submission_time": ev.get("Submission Time"),
                "completion_time": None,
                "result": None,
                "stage_ids": ev.get("Stage IDs", []),
            }

        elif etype == "SparkListenerJobEnd":
            jid = ev.get("Job ID")
            j = jobs.setdefault(jid, {"job_id": jid, "submission_time": None, "stage_ids": []})
            j["completion_time"] = ev.get("Completion Time")
            j["result"] = (ev.get("Job Result") or {}).get("Result")

        elif etype == "SparkListenerStageSubmitted":
            si = ev.get("Stage Info", {})
            key = (si.get("Stage ID"), si.get("Stage Attempt ID", 0))
            stages[key] = {
                "stage_id": si.get("Stage ID"),
                "attempt": si.get("Stage Attempt ID", 0),
                "name": si.get("Stage Name"),
                "num_tasks": si.get("Number of Tasks"),
                "submission_time": si.get("Submission Time"),
                "completion_time": None,
                "failure_reason": None,
            }

        elif etype == "SparkListenerStageCompleted":
            si = ev.get("Stage Info", {})
            key = (si.get("Stage ID"), si.get("Stage Attempt ID", 0))
            s = stages.setdefault(key, {
                "stage_id": si.get("Stage ID"),
                "attempt": si.get("Stage Attempt ID", 0),
                "name": si.get("Stage Name"),
                "num_tasks": si.get("Number of Tasks"),
                "submission_time": si.get("Submission Time"),
            })
            s["completion_time"] = si.get("Completion Time")
            s["failure_reason"] = si.get("Failure Reason")

        elif etype == "SparkListenerTaskEnd":
            key = (ev.get("Stage ID"), ev.get("Stage Attempt ID", 0))
            tinfo = ev.get("Task Info", {})
            tmetrics = ev.get("Task Metrics") or {}
            reason = (ev.get("Task End Reason") or {}).get("Reason")
            shuffle_read = tmetrics.get("Shuffle Read Metrics") or {}
            shuffle_write = tmetrics.get("Shuffle Write Metrics") or {}
            input_m = tmetrics.get("Input Metrics") or {}
            output_m = tmetrics.get("Output Metrics") or {}
            launch = tinfo.get("Launch Time")
            finish = tinfo.get("Finish Time")
            stage_tasks[key].append({
                "duration_ms": (finish - launch) if (launch and finish) else None,
                "executor_run_time_ms": tmetrics.get("Executor Run Time"),
                "executor_deserialize_time_ms": tmetrics.get("Executor Deserialize Time"),
                "gc_time_ms": tmetrics.get("JVM GC Time"),
                "mem_spill_bytes": tmetrics.get("Memory Bytes Spilled", 0) or 0,
                "disk_spill_bytes": tmetrics.get("Disk Bytes Spilled", 0) or 0,
                "shuffle_read_bytes": (shuffle_read.get("Remote Bytes Read", 0) or 0)
                                      + (shuffle_read.get("Local Bytes Read", 0) or 0),
                "shuffle_write_bytes": shuffle_write.get("Shuffle Bytes Written", 0) or 0,
                "input_bytes": input_m.get("Bytes Read", 0) or 0,
                "output_bytes": output_m.get("Bytes Written", 0) or 0,
                "failed": bool(tinfo.get("Failed")) or reason not in (None, "Success"),
                "reason": reason,
            })

    # ---- derive stage-level summaries ----
    stage_list = []
    for key, s in stages.items():
        tasks = stage_tasks.get(key, [])
        durations = sorted(t["duration_ms"] for t in tasks if t["duration_ms"] is not None)
        sub, comp = s.get("submission_time"), s.get("completion_time")
        entry = dict(s)
        entry["task_count"] = len(tasks)
        entry["failed_tasks"] = sum(1 for t in tasks if t["failed"])
        entry["duration_ms"] = (comp - sub) if (sub and comp) else None
        if durations:
            median = statistics.median(durations)
            entry["task_duration_ms"] = {
                "min": durations[0],
                "max": durations[-1],
                "median": median,
                "mean": round(statistics.mean(durations), 1),
            }
            entry["skew"] = round(durations[-1] / median, 2) if median else None
        else:
            entry["task_duration_ms"] = None
            entry["skew"] = None
        entry["shuffle_read_bytes"] = sum(t["shuffle_read_bytes"] for t in tasks)
        entry["shuffle_write_bytes"] = sum(t["shuffle_write_bytes"] for t in tasks)
        entry["input_bytes"] = sum(t["input_bytes"] for t in tasks)
        entry["output_bytes"] = sum(t["output_bytes"] for t in tasks)
        entry["mem_spill_bytes"] = sum(t["mem_spill_bytes"] for t in tasks)
        entry["disk_spill_bytes"] = sum(t["disk_spill_bytes"] for t in tasks)
        gc_times = [t["gc_time_ms"] for t in tasks if t["gc_time_ms"] is not None]
        entry["gc_time_ms"] = sum(gc_times) if gc_times else 0
        entry["gc_time_pct_of_runtime"] = _pct(
            entry["gc_time_ms"],
            sum(t["executor_run_time_ms"] or 0 for t in tasks),
        )
        stage_list.append(entry)
    stage_list.sort(key=lambda s: (s["duration_ms"] or 0), reverse=True)

    job_list = sorted(jobs.values(), key=lambda j: j["job_id"] if j["job_id"] is not None else -1)
    for j in job_list:
        sub, comp = j.get("submission_time"), j.get("completion_time")
        j["duration_ms"] = (comp - sub) if (sub and comp) else None

    all_tasks = [t for tl in stage_tasks.values() for t in tl]
    totals = {
        "num_jobs": len(job_list),
        "num_stages": len(stage_list),
        "num_tasks": len(all_tasks),
        "num_failed_tasks": sum(1 for t in all_tasks if t["failed"]),
        "total_shuffle_write_bytes": sum(t["shuffle_write_bytes"] for t in all_tasks),
        "total_shuffle_read_bytes": sum(t["shuffle_read_bytes"] for t in all_tasks),
        "total_input_bytes": sum(t["input_bytes"] for t in all_tasks),
        "total_output_bytes": sum(t["output_bytes"] for t in all_tasks),
        "total_mem_spill_bytes": sum(t["mem_spill_bytes"] for t in all_tasks),
        "total_disk_spill_bytes": sum(t["disk_spill_bytes"] for t in all_tasks),
        "total_gc_time_ms": sum(t["gc_time_ms"] or 0 for t in all_tasks),
    }

    first_job_start = min((j["submission_time"] for j in job_list if j.get("submission_time")), default=None)
    last_job_end = max((j["completion_time"] for j in job_list if j.get("completion_time")), default=None)
    last_stage_end = max((s["completion_time"] for s in stage_list if s.get("completion_time")), default=None)

    # Most event types (job/stage/task end) have no top-level "Timestamp" field,
    # so last_event_ts alone underestimates activity end for in-progress logs.
    candidates = [t for t in (last_event_ts, last_job_end, last_stage_end) if t is not None]
    app_end_effective = app_end_ts or (max(candidates) if candidates else None)

    timing = {
        "app_start_ts": app_start_ts,
        "app_end_ts": app_end_ts,
        "driver_block_manager_ts": driver_bm_ts,
        "first_job_start_ts": first_job_start,
        "last_job_end_ts": last_job_end,
        "last_stage_end_ts": last_stage_end,
        "total_ms": (app_end_effective - app_start_ts) if (app_end_effective and app_start_ts) else None,
        "startup_ms": (first_job_start - app_start_ts) if (first_job_start and app_start_ts) else None,
        "execution_ms": (last_job_end - first_job_start) if (last_job_end and first_job_start) else None,
        "app_in_progress": app_end_ts is None,
    }

    max_concurrent_executors = len(executors_added) - len(executors_removed)
    # crude high-water mark: count adds not yet matched by a remove, scanning chronologically
    events_seq = sorted(
        [(e["ts"], 1) for e in executors_added if e["ts"]] +
        [(e["ts"], -1) for e in executors_removed if e["ts"]]
    )
    running = 0
    peak = 0
    for _, delta in events_seq:
        running += delta
        peak = max(peak, running)

    return {
        "source": path,
        "app_id": app_id,
        "app_name": app_name,
        "spark_version": spark_version,
        "user": user,
        "resource_profiles": resource_profiles,
        "timing": timing,
        "executors": {
            "added": executors_added,
            "removed": executors_removed,
            "peak_concurrent": peak,
            "dynamic_allocation_likely": bool(executors_removed) and app_end_ts is not None,
        },
        "jobs": job_list,
        "stages": stage_list,
        "totals": totals,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="event log directory (v2 rolling) or file (legacy)")
    ap.add_argument("-o", "--output", help="write JSON to this file instead of stdout")
    ap.add_argument("--pretty", action="store_true", help="pretty-print JSON")
    ap.add_argument("--top-stages", type=int, default=0,
                     help="if set, only include the top-N stages by duration in output (default: all)")
    args = ap.parse_args()

    results = []
    for p in args.paths:
        r = parse(p)
        if args.top_stages:
            r["stages"] = r["stages"][: args.top_stages]
        results.append(r)

    out = results[0] if len(results) == 1 else results
    text = json.dumps(out, indent=2 if args.pretty else None)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
