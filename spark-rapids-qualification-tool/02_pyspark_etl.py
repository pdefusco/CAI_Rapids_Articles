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
# #  Author(s): Paul de Fusco
#***************************************************************************/

import os, warnings, sys, logging
import pandas as pd
import numpy as np
from datetime import date
from pyspark.sql import SparkSession

STORAGE  = ""

spark = SparkSession\
  .builder\
  .appName("Spark-ETL")\
  .config('spark.driver.cores', 4)\
  .config('spark.driver.memory', '4g')\
  .config('spark.dynamicAllocation.enabled', 'true')\
  .config('spark.executor.cores', 4)\
  .config('spark.executor.memory', '8g')\
  .config("spark.kerberos.access.hadoopFileSystems", STORAGE) \
  .getOrCreate()

# Read from Data Lake Table
df = spark.read.table("DataLakeTable")

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
