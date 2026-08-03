# Spark RAPIDS Qualification Tool Demo

This repository demonstrates how to use the **NVIDIA Spark RAPIDS Qualification Tool** to evaluate Apache Spark workloads for GPU acceleration and compare CPU execution against GPU-accelerated execution using the Spark RAPIDS Accelerator.

The project walks through a complete end-to-end workflow:

1. Generate a synthetic data warehouse
2. Execute representative Spark ETL workloads
3. Analyze the Spark Event Logs with the Spark RAPIDS Qualification Tool
4. Re-run the same workloads with Spark RAPIDS enabled to measure performance improvements

---

# What is Spark RAPIDS?

The **Spark RAPIDS Accelerator** is an open-source plugin for Apache Spark that enables Spark SQL and DataFrame operations to execute on NVIDIA GPUs instead of CPUs.

Rather than requiring applications to be rewritten, Spark RAPIDS replaces many Spark physical operators with GPU-accelerated implementations while preserving the existing Spark APIs. This allows many Spark applications to achieve significant reductions in execution time with minimal code changes.

Before enabling GPU acceleration, it is useful to determine whether an application is actually a good candidate for acceleration. This is the purpose of the **Spark RAPIDS Qualification Tool**.

The Qualification Tool analyzes Spark Event Logs and estimates:

- Whether an application is a good candidate for GPU acceleration
- Which Spark operators are GPU compatible
- Estimated execution time improvements
- Potential GPU utilization
- Recommended GPU cluster sizing

---

# Repository Workflow

The repository is organized into four stages.

## Step 1 — Create the Demo Tables

Run the scripts whose filenames begin with **`1_`**.

These scripts generate the synthetic datasets used throughout the demo, including fact tables and dimension tables that resemble a small analytical data warehouse.

Typical datasets include:

- Customers
- Accounts
- Transactions
- Branches
- Products
- Additional supporting dimension tables

These datasets provide enough scale to exercise Spark joins, aggregations, filters, and shuffle operations.

---

## Step 2 — Execute the ETL Workloads

Run the scripts whose filenames begin with **`2_`**.

These scripts execute increasingly complex Spark ETL pipelines against the generated data.

The workloads are designed to exercise operations that are commonly accelerated by Spark RAPIDS, including:

- Large joins
- Wide aggregations
- GroupBy operations
- Sorting
- Window functions
- Shuffle-intensive transformations

Running these jobs also generates the Spark Event Logs that will later be analyzed by the Qualification Tool.

---

## Step 3 — Run the Spark RAPIDS Qualification Tool

Run the script whose filename begins with **`3_`**.

This step analyzes the Spark Event Logs produced during the ETL runs.

The Qualification Tool generates reports describing:

- GPU compatibility
- Estimated acceleration
- SQL operator analysis
- Recommended GPU configuration
- Estimated runtime improvements

The generated report helps determine whether the workloads are good candidates for GPU acceleration before any infrastructure changes are made.

---

## Step 4 — Re-run the ETL Jobs with Spark RAPIDS

Finally, enable the Spark RAPIDS Accelerator and execute the same ETL applications again.

The goal of this step is to compare:

- CPU execution time
- GPU execution time
- Overall speedup
- Resource utilization

Because the application code remains unchanged, the comparison highlights the performance gains obtained simply by enabling Spark RAPIDS.

---

# Prerequisites

Before running the Qualification Tool, ensure that Spark Event Logging is enabled and configured to write logs into the project directory.

In your **Cloudera AI Workbench** project settings, configure the Spark Event Log directory as:

```text
/home/cdsw/spark-rapids-qualification-tool
```

This allows the Qualification Tool to locate and analyze the generated Spark Event Logs.

---

# Repository Structure

```text
1_*    Generate demo datasets

2_*    Execute Spark ETL workloads

3_*    Run the Spark RAPIDS Qualification Tool

4_*    Execute the same workloads with Spark RAPIDS enabled
```

---

# Expected Outcome

After completing this workflow, you will have:

- Generated a realistic Spark analytics workload
- Produced Spark Event Logs
- Evaluated the workload using the Spark RAPIDS Qualification Tool
- Identified GPU acceleration opportunities
- Measured the performance improvements achieved by enabling Spark RAPIDS

This repository provides a practical introduction to evaluating and benchmarking Spark GPU acceleration using NVIDIA Spark RAPIDS.
