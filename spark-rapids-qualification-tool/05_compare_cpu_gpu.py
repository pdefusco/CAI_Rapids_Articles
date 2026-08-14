#****************************************************************************
# (C) Cloudera, Inc. 2020-2026
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# #  Author(s): Brandon Antone
#***************************************************************************/

#****************************************************************************
# Spark RAPIDS Benchmark - CPU vs GPU Comparison
#
# Native Cloudera AI Workbench counterpart to 05_compare_cpu_gpu.ipynb.
# Reads the runtimes recorded by 02_etl_v3.py (CPU) and
# 04_spark_rapids_etl.py (GPU) and renders the same bar chart inline in
# a Workbench session console.
#
#****************************************************************************


import json
import os

import matplotlib.pyplot as plt


RUNTIME_METRICS_PATH = "/home/cdsw/spark-rapids-qualification-tool/runtime_metrics.json"


def load_metrics():

    if not os.path.exists(RUNTIME_METRICS_PATH):
        raise FileNotFoundError(
            "runtime_metrics.json not found. Run 02_etl_v3.py and 04_spark_rapids_etl.py first."
        )

    with open(RUNTIME_METRICS_PATH, "r") as f:
        metrics = json.load(f)

    missing = [mode for mode in ("cpu", "gpu") if mode not in metrics]

    if missing:
        raise KeyError(
            f"Missing runtime entries for: {missing}. "
            "Run the corresponding ETL script(s) before comparing."
        )

    return metrics


def plot_comparison(cpu_seconds, gpu_seconds):

    speedup = cpu_seconds / gpu_seconds

    labels = ["CPU", "GPU (Spark RAPIDS)"]
    values = [cpu_seconds, gpu_seconds]
    colors = ["#5b7db1", "#76b041"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}s",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Runtime (seconds)")
    ax.set_title(f"ETL v3 Runtime: CPU vs GPU  ({speedup:.2f}x speedup)")

    plt.tight_layout()
    plt.show()

    return speedup


def main():

    metrics = load_metrics()

    cpu_seconds = metrics["cpu"]["seconds"]
    gpu_seconds = metrics["gpu"]["seconds"]

    print(f"CPU runtime: {cpu_seconds:.2f}s")
    print(f"GPU runtime: {gpu_seconds:.2f}s")

    speedup = plot_comparison(cpu_seconds, gpu_seconds)

    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":

    main()
