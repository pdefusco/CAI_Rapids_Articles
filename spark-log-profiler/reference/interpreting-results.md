# Interpreting Spark event log summaries

These are the judgment calls that turn the raw JSON into a correct performance
story. Skipping them produces numbers that are technically computed but
practically misleading.

## Startup vs execution

`timing.startup_ms` = `first_job_start_ts - app_start_ts`. This is a reasonable
default but has two failure modes:

- **In-progress / driver-only logs**: if `driver_block_manager_ts` is much
  closer to `first_job_start_ts` than `app_start_ts` is, most of "startup" was
  actually cluster/pod scheduling and CUDA/plugin init that happened *before*
  logging began (the log only starts once the driver JVM is up). Call this out
  as "pre-log startup, inferred" rather than presenting it as measured — you
  can bound it (e.g. from a known submission timestamp or scheduler event) but
  can't split it precisely from `app_start_ts` alone.
- **Warm sessions**: if the user is timing "job N" specifically (not the whole
  app) and jobs 0..N-1 already ran unrelated work in the same session, the
  session was already warm by the time job N started. In that case `startup_ms`
  computed against `app_start_ts` overstates the true one-time cost — the
  correct startup number for that job is close to zero, because JVMs,
  executors, and caches were already hot. Check `jobs[].submission_time` for a
  gap: a cluster of jobs finishing, then an idle period, then the job(s) the
  user actually cares about starting — that idle gap is the tell.

Always check `jobs` for this pattern before quoting a startup number, especially
when comparing two runs where one had a warm head start and the other didn't.

## Naive vs "true" speedup (A/B comparisons)

Never just divide reported wall-clock totals. If one run's clock started warm
(mid-session) and the other's started cold (fresh cluster/executor/plugin init),
the raw ratio conflates "which engine is faster" with "which run got a warm
start." Compute both numbers and label them:

- **Naive speedup** = `total_ms(A) / total_ms(B)` — what a stopwatch would show.
- **True speedup** = `execution_ms(A) / execution_ms(B)` — startup excluded from
  both sides, comparing only the work that's repeated on every run. This is the
  number that matters for a job that runs in a long-lived cluster or on a
  schedule, where startup is paid once and amortized.

State explicitly which one you're reporting and why they differ — the gap
itself (e.g. "GPU startup is 53% of the reported total") is often the most
useful finding, not just the final ratio.

## Stage skew

`skew` = max task duration / median task duration within a stage, from
`stages[].task_duration_ms`. Rule of thumb:

- `< 2x`: normal variance, not worth flagging.
- `2x–5x`: mild imbalance, mention only if the stage is also a top-N contributor
  to wall clock.
- `> 5x` (the report used ~11–15x as "severe"): a handful of partitions are
  doing disproportionate work — almost always a join/repartition/group-by key
  with skewed cardinality. Recommend checking the partition key feeding that
  stage, not tuning executor count or memory.

Skew is orthogonal to CPU vs GPU or spill/GC — a stage can be skewed on both
engines simultaneously (as in the original CPU/GPU comparison), which is a
signal the fix is upstream in the data/query, not the execution engine.

## Spill and GC

- `mem_spill_bytes` / `disk_spill_bytes` > 0 anywhere means a task didn't fit
  its data in the memory budget for that stage — flag it as a sizing issue
  distinct from skew, even if it's the same stage.
- `gc_time_pct_of_runtime` > ~10% on a stage is worth calling out; under that,
  don't mention GC — it's noise.
- If both are zero/low across all stages, say so explicitly when comparing
  runs — it rules out "the gap is memory pressure" as an explanation and points
  back to startup amortization or skew instead (this was the case in the
  original CPU vs GPU report).

## Executor timeline

`executors.added` sorted by `ts` shows the ramp-up curve.
`executors.peak_concurrent` vs the count of `Number of Tasks` per stage tells
you if a stage was actually parallelized across all available executors or
bottlenecked on a slow ramp-up. `executors.removed` non-empty mid-run (before
`app_end_ts`) indicates dynamic allocation; empty `removed` with all `added`
near the start indicates a fixed-size cluster. This distinction matters when
comparing a dynamic-allocation CPU run against a fixed-executor GPU run (or
vice versa) — the executor count isn't constant across the run in the dynamic
case, so "total capacity" comparisons need to account for that.

## `total_ms` on in-progress logs with dynamic allocation

`timing.total_ms` falls back to the latest timestamp seen anywhere in the log
when `app_end_ts` is null (log still `.inprogress`). If the run uses dynamic
allocation, idle executors can be removed minutes after the last real job
finished — `executors.removed[].ts` stretching well past `last_job_end_ts`.
That pulls `total_ms` up with it, so it stops meaning "how long the workload
took" and starts meaning "how long the session has existed so far." For an
in-progress log, prefer `last_job_end_ts - app_start_ts` (or `- first_job_start_ts`
for execution-only) as the "real work" duration, and only quote `total_ms` when
you've checked it isn't inflated by trailing executor teardown.

## SQL execution IDs are nested, not flat

A single user-level call like `df.write.saveAsTable(...)` on a RAPIDS/GPU run
often produces *two* `SparkListenerSQLExecutionStart` events: one for the
outer `saveAsTable` call and one for a nested `run at
GpuExecutedCommandExec.scala` that does the actual work, both sharing the same
`rootExecutionId`. If you report the outer execution's duration alone it's
usually right anyway (it starts before and ends after the child), but if you
sum every execution's duration independently you'll double count. Use
`extract_details.py`'s `sql_queries` output (grouped by root execution ID) for
"how long did this logical query take," and only drop to `sql_executions`
(ungrouped) if you need to see the nested breakdown.

## RAPIDS GPU accumulator quirks

`Stage Info.Accumulables` values are pre-formatted strings, but not
consistently: most are `"HH:MM:SS.mmm"` (durations) or `"192.87MB (N
bytes)"` (sizes), but a few — notably `GPU decode time` and `GPU encode and
buffer time` — come through as bare integers in **nanoseconds**, not
milliseconds. `gpuDiskWriteSavedBytes` comes through as a bare integer that's
already bytes. Don't treat an unformatted integer accumulable as a plain
count without checking the metric name first — `extract_details.py` handles
the known cases, but a metric name not in its `GPU_METRIC_NAMES` allowlist
will fall through as a raw, unit-less "count" and needs the same scrutiny
before quoting it.

Metrics named with `Max`/`Peak` (`gpuMaxTaskFootprint`, `gpuMaxDeviceMemoryBytes`,
etc.) are watermarks — aggregate across stages with `max`, never `sum`.
Everything else RAPIDS-specific (`gpuTime`, `gpuSemaphoreWait`, decode/encode
time, `gpuDiskWriteSavedBytes`) is cumulative work and aggregates with `sum`.
Note `gpuTime` summed across stages can exceed wall-clock execution time —
that's expected, since multiple tasks run concurrently on the GPU
(`spark.rapids.sql.concurrentGpuTasks`); it's total GPU-busy time, not a
timeline.

## Config diffing between two runs

When comparing an A/B pair, the `environment.curated` fields matter more than
the raw job/stage numbers for explaining *why* the two runs differ — e.g. one
run using `spark.dynamicAllocation.enabled=true` with a much lower steady-state
executor count than the other's fixed pool, or a mismatched
`spark.sql.shuffle.partitions` inflating task count and per-task overhead on
one side. Check these before attributing a runtime gap purely to CPU-vs-GPU or
any other headline difference — an uncontrolled config difference between the
two runs can explain part of the gap on its own.

## Stage/job ID numbering across a shared session

If two phases of work ran in the same Spark session (e.g. validation queries
then the real ETL job), stage and job IDs are monotonically increasing across
*all* activity in that session, not per-phase. A later phase will have much
higher stage IDs than an equivalent phase run alone in a fresh session — this
is expected and not evidence of doing more work, just evidence of running
later in a longer-lived session. Don't compare stage/job ID magnitudes across
runs as a proxy for work done; compare `stages[].duration_ms` and
`totals.num_tasks` instead.
