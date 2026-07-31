import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

STORAGE = "s3a://pdf0714-buk-d7392db3/data"

spark = (
    SparkSession.builder
        .appName("Spark-ETL-v1")
        .config("spark.driver.cores", 4)
        .config("spark.driver.memory", "4g")
        .config("spark.dynamicAllocation.enabled", "true")
        .config("spark.executor.cores", 4)
        .config("spark.executor.memory", "8g")
        .config("spark.sql.shuffle.partitions", "800")
        .config("spark.kerberos.access.hadoopFileSystems", STORAGE)
        .getOrCreate()
)

###########################################################
# Read Data
###########################################################

df = (
    spark.read.table("BenchmarkTableV2")
    .repartition(800, "customer_id")
)

###########################################################
# Feature Engineering
###########################################################

df_features = (
    df
    .withColumn(
        "total_assets",
        F.col("bank_account_balance")
        + F.col("sec_bank_account_balance")
        + F.col("savings_account_balance")
        + F.col("sec_savings_account_balance")
    )
    .withColumn(
        "total_liabilities",
        F.col("credit_card_balance")
        + F.col("mortgage_balance")
        + F.col("primary_loan_balance")
        + F.col("secondary_loan_balance")
        + F.col("uni_loan_balance")
    )
    .withColumn(
        "net_worth",
        F.col("total_assets") - F.col("total_liabilities")
    )
    .withColumn(
        "credit_utilization",
        F.col("credit_card_balance") /
        (F.col("credit_card_balance") + F.lit(5000))
    )
    .withColumn(
        "debt_ratio",
        F.col("total_liabilities") /
        (F.col("total_assets") + F.lit(1))
    )
    .withColumn(
        "transaction_fee",
        F.col("transaction_amount") * F.lit(0.023)
    )
)

###########################################################
# Dimension Tables
###########################################################

age_dim = (
    df_features
    .select("age")
    .distinct()
    .withColumn(
        "age_bucket",
        F.when(F.col("age") < 30, "18-29")
         .when(F.col("age") < 45, "30-44")
         .when(F.col("age") < 60, "45-59")
         .otherwise("60+")
    )
)

state_dim = (
    df_features
    .select("state")
    .distinct()
    .withColumn(
        "state_region",
        F.when(F.col("state").isin("CA", "WA", "OR"), "West")
         .when(F.col("state").isin("NY", "NJ", "MA"), "East")
         .when(F.col("state").isin("TX", "OK"), "South")
         .otherwise("Other")
    )
)

merchant_dim = (
    df_features
    .select("merchant_category")
    .distinct()
    .withColumn(
        "merchant_risk",
        F.when(
            F.col("merchant_category").isin(
                "Gaming",
                "Crypto",
                "Electronics"
            ),
            "High"
        ).otherwise("Normal")
    )
)

###########################################################
# Multiple Joins
###########################################################

joined = (
    df_features
        .join(age_dim, "age")
        .join(state_dim, "state")
        .join(merchant_dim, "merchant_category")
)

###########################################################
# Large Aggregation
###########################################################

agg1 = (
    joined
    .groupBy(
        "state_region",
        "merchant_category",
        "merchant_risk",
        "age_bucket",
        "fraud_trx"
    )
    .agg(
        F.count("*").alias("txn_count"),
        F.sum("transaction_amount").alias("total_amount"),
        F.avg("transaction_amount").alias("avg_amount"),
        F.max("transaction_amount").alias("max_amount"),
        F.min("transaction_amount").alias("min_amount"),
        F.stddev("transaction_amount").alias("stddev_amount"),
        F.avg("net_worth").alias("avg_net_worth"),
        F.avg("credit_utilization").alias("avg_credit_utilization"),
        F.avg("debt_ratio").alias("avg_debt_ratio"),
        F.sum("transaction_fee").alias("total_fee")
    )
)

###########################################################
# Second Aggregation
###########################################################

agg2 = (
    agg1
    .groupBy(
        "state_region",
        "fraud_trx"
    )
    .agg(
        F.sum("txn_count").alias("transactions"),
        F.sum("total_amount").alias("total_amount"),
        F.avg("avg_net_worth").alias("avg_net_worth"),
        F.avg("avg_credit_utilization").alias("credit_utilization"),
        F.avg("avg_debt_ratio").alias("avg_debt_ratio"),
        F.sum("total_fee").alias("fees_collected")
    )
)

###########################################################
# Global Sort
###########################################################

final = (
    agg2
    .orderBy(
        F.desc("total_amount")
    )
)

final.show()

print("Completed Spark ETL v1")