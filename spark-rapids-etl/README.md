# Spark Rapids ETL in Cloudera AI

![alt text](../img/xgboost-deployment.png)

## Objective

In this tutorial you will learn how to leverage Spark Rapids with GPU acceleration in Cloudera AI.

## Motivation

Using Spark RAPIDS acceleration with GPUs in Cloudera AI enables organizations to dramatically speed up data processing and ETL workloads by offloading compute-intensive operations to GPUs. When combined with the Spark-on-Kubernetes runtime provided by Cloudera AI, users gain the flexibility of elastic, containerized Spark clusters that can scale on-demand, while fully leveraging GPU acceleration - with minimal configuration effort.

## Requirements

This example was built with Cloudera On Cloud Public Cloud Runtime 7.3.1, CAI Workbench 2.0.53, Python 3.10, a single V100 Nvidia GPU, the Cloudera Spark 3.3.0 DEX CDE 1.24 Runtime Hotfix 1 and Spark Rapids v25.08.0.

You can reproduce this tutorial in your CAI environment with the following:

* A CAI Environment in Private or Public Cloud.
* A PBJ or Python based IDE Runtime and the Spark 3.3.0 DEX CDE 1.24 Hotfix 1 Spark Add On.
* A Spark Rapids compatible GPU such as V100, T4, A10, A100, L4, H100, and B100.

## Step by Step Instructions

#### 1. Create Project from Git Repository

Create a CAI Project by cloning the following git url: https://github.com/pdefusco/CAI_Rapids_Articles.git

When you create the project, make sure the Python 3.10 PBJ Runtime with Nvidia Drivers is present.

![alt text](../img/spark-rapids-etl-project.png)

#### 2. Launch CAI Session

Next, launch a Cloudera AI Session with the following dependencies:

```
Editor: PBJ Workbench
Kernel: Python 3.10
Edition: Nvidia GPU
Spark Runtime Addon: Spark 3.3.0 DEX CDE 1.24.1 Runtime Hotfix 1
Enable GPU: Select one gpu card
Resource Profile: 4 vCPU / 16 GiB Mem
```

You can ignore the warning about Spark and GPU compatibility.

![alt text](../img/spark-rapids-session-params.png)

#### 3. Run Spark Rapids Script

Once the Session becomes available, run the entire script. No code modifications are required.

Notice the ETL transformations end with an ```.explain()``` method call. Look at the output to confirm the Spark Physical Plan is leveraging GPU acceleration.

```
from pyspark.sql import functions as F
Define the path to your text file

file_path = "/home/cdsw/spark-rapids-ml/example.txt"  # Adjust if your file is in a different location
Read the text file into a DataFrame

df = spark.read.csv(file_path)
df.show()
+--------------------+
|                 _c0|
+--------------------+
|This is the first...|
|Here's another li...|
|And a third line ...|
+--------------------+

read_df = spark.read.table(“DataLakeTable”) read_df.show()

from pyspark.sql import functions as F
Define the path to your text file

file_path = "/home/cdsw/spark-rapids-ml/example.txt"  # Adjust if your file is in a different location
Read the text file into a DataFrame

df = spark.read.text(file_path)
— Step 1: Filter to keep only lines containing “line” (case-insensitive) — The output is a new filtered DataFrame ‘df_step1’

df_step1 = df.filter(df["value"].rlike("(?i)line"))
print("--- Step 1: Filtered for 'line' ---")
--- Step 1: Filtered for 'line' ---
df_step1.show(truncate=False)
+------------------------------------+
|value                               |
+------------------------------------+
|This is the first line of text.     |
|Here's another line with some words.|
|And a third line for good measure.  |
+------------------------------------+

read_df = spark.read.table(“DataLakeTable”) read_df.show()

from pyspark.sql import functions as F
Define the path to your text file

file_path = "/home/cdsw/spark-rapids-ml/example.txt"  # Adjust if your file is in a different location
Read the text file into a DataFrame

df = spark.read.text(file_path)
— Step 1: Filter to keep only lines containing “line” (case-insensitive) — The output is a new filtered DataFrame ‘df_step1’

df_step1 = df.filter(df["value"].rlike("(?i)line"))
print("--- Step 1: Filtered for 'line' ---")
--- Step 1: Filtered for 'line' ---
df_step1.show(truncate=False)
+------------------------------------+
|value                               |
+------------------------------------+
|This is the first line of text.     |
|Here's another line with some words.|
|And a third line for good measure.  |
+------------------------------------+

— Step 2: From the results of Step 1, filter out lines that contain “apache” — The input is ‘df_step1’, the output is ‘df_step2’

df_step2 = df_step1.filter(~df_step1["value"].contains("apache"))
print("--- Step 2: Filtered OUT 'apache' ---")
--- Step 2: Filtered OUT 'apache' ---
df_step2.show(truncate=False)
+------------------------------------+
|value                               |
+------------------------------------+
|This is the first line of text.     |
|Here's another line with some words.|
|And a third line for good measure.  |
+------------------------------------+

— Step 3: From the results of Step 2, transform the data by splitting the lines into an array of words — The input is ‘df_step2’, the output is ‘df_step3’

df_step3 = df_step2.select(F.split(df_step2["value"], " ").alias("words_array"))
print("--- Step 3: Transformed into array of words ---")
--- Step 3: Transformed into array of words ---
df_step3.show(truncate=False)
+-------------------------------------------+
|words_array                                |
+-------------------------------------------+
|[This, is, the, first, line, of, text.]    |
|[Here's, another, line, with, some, words.]|
|[And, a, third, line, for, good, measure.] |
+-------------------------------------------+

— Step 4: From the results of Step 3, filter rows where the word array has more than 2 elements — The input is ‘df_step3’, the output is ‘final_df’

final_df = df_step3.filter(F.size(df_step3["words_array"]) > 2)
print("--- Step 4: Filtered for arrays with size > 2 (Final Result) ---")
--- Step 4: Filtered for arrays with size > 2 (Final Result) ---
final_df.show(truncate=False)
+-------------------------------------------+
|words_array                                |
+-------------------------------------------+
|[This, is, the, first, line, of, text.]    |
|[Here's, another, line, with, some, words.]|
|[And, a, third, line, for, good, measure.] |
+-------------------------------------------+

final_df.explain()
== Physical Plan ==
GpuColumnarToRow false
+- GpuProject [split(value#226,  , -1,  , false) AS words_array#241], true
   +- GpuRowToColumnar targetsize(1073741824)
      +- *(1) Filter (((isnotnull(value#226) AND RLIKE(value#226, (?i)line)) AND NOT Contains(value#226, apache)) AND (size(split(value#226,  , -1), true) > 2))
         +- FileScan text [value#226] Batched: false, DataFilters: [isnotnull(value#226), RLIKE(value#226, (?i)line), NOT Contains(value#226, apache), (size(split(v..., Format: Text, Location: InMemoryFileIndex(1 paths)[file:/home/cdsw/spark-rapids-ml/example.txt], PartitionFilters: [], PushedFilters: [IsNotNull(value), Not(StringContains(value,apache))], ReadSchema: struct<value:string>
```

#### 4. Validate GPU utilization in the Spark UI

You can also explore the Spark UI and validate GPU utilization from there.

![alt text](../img/spark-rapids-ui-1.png)

![alt text](../img/spark-rapids-ui-2.png)


## Summary & Next Steps

In this project you learned about GPU accelerated ETL with Spark Rapids in Cloudera AI. You are encouraged to fork and reuse the code above in your environment.

**Additional Resources & Tutorials**
Explore these helpful tutorials and blogs to learn more about Spark Rapids and Cloudera AI:

* **Cloudera – “Cloudera Supercharges the Enterprise Data Cloud with NVIDIA”** — describes how Cloudera’s platform leverages the RAPIDS Accelerator to speed up ETL, Spark SQL, and analytics. ([blog.cloudera.com][1])
* **Cloudera – “NVIDIA RAPIDS in Cloudera Machine Learning”** — technical blog on how RAPIDS (cuDF, cuML, etc.) is integrated in Cloudera ML for data engineering + feature engineering. ([Cloudera][2])
* **NVIDIA – “RAPIDS Accelerator for Apache Spark v21.06 Release”** — details the 21.06 release, including ETL‑relevant operator support and profiling. ([NVIDIA Developer][3])
* **NVIDIA – “GPUs for ETL? Run Faster, Less Costly Workloads with NVIDIA RAPIDS Accelerator for Apache Spark …”** — shows a real-world ETL job on Spark + RAPIDS, cost-savings + speed. ([NVIDIA Developer][4])
* **NVIDIA – “RAPIDS Accelerator for Apache Spark Release v21.10”** — outlines major performance improvements, nested data types, and I/O support. ([NVIDIA Developer][5])
* **NVIDIA – “Saving Green: Accelerated Analytics Cuts Costs and Carbon”** — energy-efficiency and cost-reduction benefits for Spark ETL using RAPIDS. ([NVIDIA Blogs][6])
* **Cloudera / NVIDIA Joint (Whitepaper)** – *Turbocharge your ETL pipelines with NVIDIA GPUs and Cloudera Data Platform*. ([Cloudera][7])

[1]: https://blog.cloudera.com/cloudera-supercharges-the-enterprise-data-cloud-with-nvidia/?utm_source=chatgpt.com "Cloudera Supercharges the Enterprise Data Cloud with NVIDIA | Cloudera Blog"
[2]: https://www.cloudera.com/blog/technical/nvidia-rapids-in-cloudera-machine-learning.html?utm_source=chatgpt.com "NVIDIA RAPIDS in Cloudera Machine Learning | Blog | Cloudera"
[3]: https://developer.nvidia.com/blog/rapids-accelerator-for-apache-spark-v21-06-release/?utm_source=chatgpt.com "RAPIDS Accelerator for Apache Spark v21.06 Release | NVIDIA Technical Blog"
[4]: https://developer.nvidia.com/blog/gpus-for-etl-run-faster-less-costly-workloads-with-nvidia-rapids-accelerator-for-apache-spark-and-databricks/?utm_source=chatgpt.com "GPUs for ETL? Run Faster, Less Costly Workloads with NVIDIA RAPIDS Accelerator for Apache Spark and Databricks | NVIDIA Technical Blog"
[5]: https://developer.nvidia.com/blog/rapids-accelerator-for-apache-spark-release-v21-10/?utm_source=chatgpt.com "RAPIDS Accelerator for Apache Spark Release v21.10 | NVIDIA Technical Blog"
[6]: https://blogs.nvidia.com/blog/spark-rapids-energy-efficiency/?utm_source=chatgpt.com "Saving Green: Accelerated Analytics Cuts Costs and Carbon | NVIDIA Blog"
[7]: https://www.cloudera.com/content/dam/www/marketing/resources/partners/whitepapers/turbocharge-your-etl-pipelines-with-nvidia-gpus-and-cloudera-data-platform.pdf?daqp=true&utm_source=chatgpt.com "WHITE PAPER"
