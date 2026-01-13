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
# #  Author(s): Paul de Fusco, James Horsch
#***************************************************************************/

import os, warnings, sys, logging
import pandas as pd
import numpy as np
from datetime import date
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


STORAGE = "s3a://pdf-jan-26-buk-7c0e831f/data/"

spark = SparkSession\
  .builder\
  .appName("Spark-Rapids-ETL")\
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
  .config("spark.kerberos.access.hadoopFileSystems", STORAGE) \
  .getOrCreate()

# Read from Data Lake Table
df = spark.read.table("BenchmarkTableV2")

df_features = (
    df
    .withColumn(
        "total_assets",
        F.col("bank_account_balance") +
        F.col("sec_bank_account_balance") +
        F.col("savings_account_balance") +
        F.col("sec_savings_account_balance")
    )
    .withColumn(
        "total_liabilities",
        F.col("credit_card_balance") +
        F.col("mortgage_balance") +
        F.col("primary_loan_balance") +
        F.col("secondary_loan_balance") +
        F.col("uni_loan_balance")
    )
    .withColumn(
        "net_worth",
        F.col("total_assets") - F.col("total_liabilities")
    )
)

age_dim = (
    df_features
    .withColumn(
        "age_bucket",
        F.when(F.col("age") < 30, "18-29")
         .when(F.col("age") < 45, "30-44")
         .when(F.col("age") < 60, "45-59")
         .otherwise("60+")
    )
    .groupBy("age", "age_bucket")
    .count()     # forces shuffle
    .drop("count")
)

df_joined = (
    df_features
    .join(age_dim, on="age", how="inner")
)

fraud_metrics = (
    df_joined
    .groupBy("age_bucket", "fraud_trx")
    .agg(
        F.count("*").alias("txn_count"),
        F.sum("transaction_amount").alias("total_trx_amount"),
        F.avg("transaction_amount").alias("avg_trx_amount"),
        F.avg("net_worth").alias("avg_net_worth"),
        F.stddev("transaction_amount").alias("trx_stddev")
    )
)

final_summary = (
    fraud_metrics
    .groupBy("fraud_trx")
    .agg(
        F.sum("txn_count").alias("total_txns"),
        F.sum("total_trx_amount").alias("total_amount"),
        F.avg("avg_net_worth").alias("overall_avg_net_worth")
    )
)

final_summary.show()
