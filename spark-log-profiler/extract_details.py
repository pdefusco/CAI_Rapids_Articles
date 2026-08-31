#!/usr/bin/env python3
"""Extract environment/config, SQL-execution boundaries, and RAPIDS GPU metrics
from a Spark event log — the detail parse_eventlog.py's job/stage/task summary
doesn't cover.

Complements parse_eventlog.py (run both; this one answers "what config was this
run under" and "how long did the actual write / count / etc. SQL query take",
not "which stage was slow").

Usage:
  extract_details.py <path> [<path> ...] [-o out.json] [--pretty]
"""
import argparse
import json
import re
import sys
from collections import defaultdict

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from parse_eventlog import _iter_events  # reuses v2-rolling/gzip/file handling

# Config keys worth surfacing for a CPU-vs-GPU / before-vs-after diff. The full
# "Spark Properties" map is in the output too, under "environment.all_properties",
# for anything not in this curated list.
INTERESTING_PROPS = [
    "spark.master",
    "spark.app.name",
    "spark.dynamicAllocation.enabled",
    "spark.dynamicAllocation.maxExecutors",
    "spark.dynamicAllocation.minExecutors",
    "spark.executor.instances",
    "spark.executor.cores",
    "spark.executor.memory",
    "spark.executor.memoryOverhead",
    "spark.driver.memory",
    "spark.driver.cores",
    "spark.sql.shuffle.partitions",
    "spark.sql.adaptive.enabled",
    "spark.default.parallelism",
    "spark.rapids.sql.enabled",
    "spark.rapids.sql.concurrentGpuTasks",
    "spark.executor.resource.gpu.amount",
    "spark.task.resource.gpu.amount",
    "spark.rapids.memory.pinnedPool.size",
    "spark.rapids.sql.batchSizeBytes",
]

# Well-known RAPIDS Accelerator accumulator names (as they appear in Stage Info
# "Accumulables"). Names containing "Max"/"Peak" are watermarks -> take max
# across stages; everything else here is cumulative -> sum across stages.
GPU_METRIC_NAMES = {
    "gpuTime": "sum",
    "gpuSemaphoreWait": "sum",
    "GPU decode time": "sum",
    "GPU encode and buffer time": "sum",
    "gpuDiskWriteSavedBytes": "sum",
    "gpuMaxTaskFootprint": "max",
    "gpuMaxDeviceMemoryBytes": "max",
    "gpuMaxHostMemoryBytes": "max",
    "gpuMaxPinnedMemoryBytes": "max",
    "gpuMaxConcurrentGpuTasks": "max",
    "gpuOnGpuTasksWaitingGPUAvgCount": "max",
}

# A handful of RAPIDS metrics don't come through the log pre-formatted like
# "00:00:00.024" — their Accumulable "Value" is a bare integer. For those we
# need to know what the integer actually is (nanoseconds vs. already-bytes)
# since a bare int is otherwise indistinguishable from a plain counter.
BARE_INT_KIND = {
    "GPU decode time": "ns",
    "GPU encode and buffer time": "ns",
    "gpuDiskWriteSavedBytes": "bytes",
}

_TIME_RE = re.compile(r"^(\d+):(\d{2}):(\d{2})\.(\d{3})$")
_SIZE_RE = re.compile(r"\(([\d,]+) bytes\)")


def _parse_accumulable_value(raw, name=None):
    """Return (numeric_value_in_ms_or_bytes_or_count, unit) for a Spark
    Accumulable value. Most are pre-formatted as human-readable strings, e.g.
    "00:00:00.024" (duration) or "192.87MB (202235392 bytes)" (size); a few
    RAPIDS metrics come through as bare integers whose meaning depends on the
    metric name (see BARE_INT_KIND).
    """
    m = _TIME_RE.match(raw)
    if m:
        h, mi, s, ms = (int(x) for x in m.groups())
        return (h * 3600 + mi * 60 + s) * 1000 + ms, "ms"
    m = _SIZE_RE.search(raw)
    if m:
        return int(m.group(1).replace(",", "")), "bytes"
    try:
        value = int(raw.replace(",", ""))
    except ValueError:
        return None, None
    kind = BARE_INT_KIND.get(name)
    if kind == "ns":
        return round(value / 1e6, 3), "ms"
    if kind == "bytes":
        return value, "bytes"
    return value, "count"


def extract(path):
    environment = {}
    executions = {}          # execution_id -> dict
    job_execution_id = {}    # job_id -> execution_id (from spark.sql.execution.id job property)
    job_span = {}            # job_id -> (submission_time, completion_time)
    gpu_metric_totals = defaultdict(lambda: {"agg": 0, "stage_count": 0, "unit": None})
    stage_seen_for_metric = defaultdict(set)  # metric_name -> set of (stage_id, attempt) already counted

    for ev in _iter_events(path):
        etype = ev.get("Event")

        if etype == "SparkListenerEnvironmentUpdate":
            props = ev.get("Spark Properties", {})
            environment.update(props)  # later update wins if conf changed at runtime

        elif etype == "org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionStart":
            eid = ev.get("executionId")
            executions[eid] = {
                "execution_id": eid,
                "root_execution_id": ev.get("rootExecutionId"),
                "description": ev.get("description"),
                "start_time": ev.get("time"),
                "end_time": None,
                "duration_ms": None,
                "job_ids": [],
            }

        elif etype == "org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionEnd":
            eid = ev.get("executionId")
            e = executions.setdefault(eid, {
                "execution_id": eid, "root_execution_id": None, "description": None,
                "start_time": None, "job_ids": [],
            })
            e["end_time"] = ev.get("time")
            if e.get("start_time") and e["end_time"]:
                e["duration_ms"] = e["end_time"] - e["start_time"]

        elif etype == "SparkListenerJobStart":
            jid = ev.get("Job ID")
            props = ev.get("Properties") or {}
            eid = props.get("spark.sql.execution.id")
            if eid is not None:
                job_execution_id[jid] = int(eid)
            job_span[jid] = [ev.get("Submission Time"), None]

        elif etype == "SparkListenerJobEnd":
            jid = ev.get("Job ID")
            if jid in job_span:
                job_span[jid][1] = ev.get("Completion Time")

        elif etype == "SparkListenerStageCompleted":
            si = ev.get("Stage Info", {})
            key = (si.get("Stage ID"), si.get("Stage Attempt ID", 0))
            for acc in si.get("Accumulables", []):
                name = acc.get("Name")
                if name not in GPU_METRIC_NAMES:
                    continue
                if key in stage_seen_for_metric[name]:
                    continue  # avoid double count on duplicate StageCompleted (e.g. retries)
                stage_seen_for_metric[name].add(key)
                value, unit = _parse_accumulable_value(str(acc.get("Value", "")), name)
                if value is None:
                    continue
                m = gpu_metric_totals[name]
                m["unit"] = unit
                m["stage_count"] += 1
                if GPU_METRIC_NAMES[name] == "max":
                    m["agg"] = max(m["agg"], value)
                else:
                    m["agg"] += value

    # roll up job ids per SQL execution, and root-execution (nested query) totals
    for jid, eid in job_execution_id.items():
        if eid in executions:
            executions[eid]["job_ids"].append(jid)

    for e in executions.values():
        e["job_ids"].sort()
        spans = [job_span[j] for j in e["job_ids"] if j in job_span]
        starts = [s[0] for s in spans if s[0] is not None]
        ends = [s[1] for s in spans if s[1] is not None]
        e["job_span_ms"] = (max(ends) - min(starts)) if starts and ends else None

    execution_list = sorted(executions.values(), key=lambda e: (e["start_time"] or 0))

    # group nested executions under their root for a "logical query" rollup
    roots = defaultdict(list)
    for e in execution_list:
        roots[e["root_execution_id"] if e["root_execution_id"] is not None else e["execution_id"]].append(e)
    root_summary = []
    for root_id, members in sorted(roots.items(), key=lambda kv: kv[0]):
        starts = [m["start_time"] for m in members if m["start_time"] is not None]
        ends = [m["end_time"] for m in members if m["end_time"] is not None]
        top = next((m for m in members if m["execution_id"] == root_id), members[0])
        root_summary.append({
            "root_execution_id": root_id,
            "description": top["description"],
            "duration_ms": (max(ends) - min(starts)) if starts and ends else None,
            "member_execution_ids": sorted(m["execution_id"] for m in members),
        })

    gpu_metrics = {}
    for name, agg in sorted(gpu_metric_totals.items()):
        gpu_metrics[name] = {
            "value": agg["agg"],
            "unit": agg["unit"],
            "aggregation": GPU_METRIC_NAMES[name],
            "stage_count": agg["stage_count"],
        }

    return {
        "source": path,
        "environment": {
            "curated": {k: environment.get(k) for k in INTERESTING_PROPS if k in environment},
            "all_properties": environment,
        },
        "sql_executions": execution_list,
        "sql_queries": root_summary,  # nested executions rolled up under their root query
        "gpu_metrics": gpu_metrics,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-o", "--output")
    ap.add_argument("--pretty", action="store_true")
    ap.add_argument("--no-all-properties", action="store_true",
                     help="omit the full Spark Properties dump, keep only the curated subset")
    args = ap.parse_args()

    results = []
    for p in args.paths:
        r = extract(p)
        if args.no_all_properties:
            r["environment"].pop("all_properties", None)
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
