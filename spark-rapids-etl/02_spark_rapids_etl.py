#****************************************************************************
# (C) Cloudera, Inc. 2020-2025
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
# #  Author(s): Paul de Fusco, James Horsch
#***************************************************************************/

import os, warnings, sys, logging
import pandas as pd
import numpy as np
from datetime import date
from pyspark.sql import SparkSession

spark = SparkSession\
  .builder\
  .appName("Spark-Rapids-Ml")\
  .config('spark.dynamicAllocation.enabled', 'false')\
  .config('spark.executor.cores', 1)\
  .config('spark.executor.resource.gpu.amount', 1)\
  .config('spark.executor.instances', 1)\
  .config('spark.executor.memory', '16g')\
  .config('spark.rapids.memory.pinnedPool.size', '2G')\
  .config('spark.plugins', 'com.nvidia.spark.SQLPlugin')\
  .config("spark.jars.packages", "com.nvidia:rapids-4-spark_2.12:25.08.0")\
  .config("spark.executor.resource.gpu.discoveryScript","/home/cdsw/getGpusResources.sh")\
  .config("spark.executor.resource.gpu.vendor", "nvidia.com")\
  .config("spark.rapids.shims-provider-override", "com.nvidia.spark.rapids.shims.spark330.SparkShimServiceProvider")\
  .config("spark.driver.memory","10g") \
  .config("spark.eventLog.enabled","true") \
  .config("spark.rapids.sql.concurrentGpuTasks", "2") \
  .config("spark.sql.files.maxPartitionBytes", "256m") \
  .config("spark.locality.wait", "0") \
  .config("spark.sql.adaptive.enabled", "true") \
  .config("spark.rapids.memory.pinnedPool.size", "2g") \
  .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "1g") \
  .config("spark.executor.memoryOverhead", "3g") \
  .config("spark.kryo.registrator", "com.nvidia.spark.rapids.GpuKryoRegistrator") \
  .config("spark.rapids.sql.enabled", "true") \
  .config("spark.rapids.sql.incompatibleOps.enabled", "true") \
  .config("spark.rapids.sql.udfCompiler.enabled", "true") \
  .config("spark.rapids.sql.format.csv.read.enabled", "true") \
  .config("spark.rapids.sql.format.csv.enabled", "true") \
  .config("spark.rapids.sql.variableFloatAgg.enabled", "true") \
  .config("spark.rapids.sql.explain", "ALL") \
  .config("spark.sql.hive.convertMetastoreParquet", "true") \
  .config("spark.rapids.sql.castFloatToString.enabled", "true") \
  .config("spark.rapids.sql.csv.read.float.enabled", "true") \
  .config("spark.rapids.sql.castStringToFloat.enabled", "true") \
  .config("spark.hadoop.fs.s3a.custom.signers", "RazS3SignerPlugin:org.apache.ranger.raz.hook.s3.RazS3SignerPlugin:org.apache.ranger.raz.hook.s3.RazS3SignerPluginInitializer") \
  .config("spark.hadoop.fs.s3a.s3.signing-algorithm", "RazS3SignerPlugin") \
  .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.ranger.raz.hook.s3.RazCredentialProvider") \
  .config("spark.kubernetes.executor.podTemplateFile", "/tmp/spark-executor.json") \
  .config("spark.kerberos.access.hadoopFileSystems", "s3a://pdf-oct-buk-a163bf71/data/") \
  .getOrCreate()

# Read from Data Lake Table
read_df = spark.read.table("DataLakeTable")
read_df.show()

# Read from Local Project File
from pyspark.sql import functions as F
# Define the path to your text file
file_path = "/home/cdsw/spark-rapids-etl/example.txt"  # Adjust if your file is in a different location

# Read the text file into a DataFrame
df = spark.read.text(file_path)

# --- Step 1: Filter to keep only lines containing "line" (case-insensitive) ---
# The output is a new filtered DataFrame 'df_step1'
df_step1 = df.filter(df["value"].rlike("(?i)line"))

print("--- Step 1: Filtered for 'line' ---")
df_step1.show(truncate=False)

# --- Step 2: From the results of Step 1, filter out lines that contain "apache" ---
# The input is 'df_step1', the output is 'df_step2'
df_step2 = df_step1.filter(~df_step1["value"].contains("apache"))

print("--- Step 2: Filtered OUT 'apache' ---")
df_step2.show(truncate=False)

# --- Step 3: From the results of Step 2, transform the data by splitting the lines into an array of words ---
# The input is 'df_step2', the output is 'df_step3'
df_step3 = df_step2.select(F.split(df_step2["value"], " ").alias("words_array"))

print("--- Step 3: Transformed into array of words ---")
df_step3.show(truncate=False)

# --- Step 4: From the results of Step 3, filter rows where the word array has more than 2 elements ---
# The input is 'df_step3', the output is 'final_df'
final_df = df_step3.filter(F.size(df_step3["words_array"]) > 2)

print("--- Step 4: Filtered for arrays with size > 2 (Final Result) ---")
final_df.show(truncate=False)

final_df.explain()
