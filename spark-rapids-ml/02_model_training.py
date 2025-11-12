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
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
import mlflow.sklearn
from xgboost import XGBClassifier
from datetime import date
import cml.data_v1 as cmldata
import pyspark.pandas as ps


USERNAME = os.environ["PROJECT_OWNER"]
DBNAME = os.environ["DBNAME_PREFIX"]+"_"+USERNAME
CONNECTION_NAME = os.environ["SPARK_CONNECTION_NAME"]
DATE = date.today()

import os
os.environ["PYSPARK_SUBMIT_ARGS"] = "--jars /home/cdsw/rapids-4-spark_2.12-25.08.0.jar pyspark-shell"

from pyspark.sql import SparkSession

spark = SparkSession\
  .builder\
  .appName("Spark-Rapids-Ml")\
  .config('spark.dynamicAllocation.enabled', 'false')\
  .config('spark.executor.cores', 1)\
  .config('spark.executor.resource.gpu.amount', 1)\
  .config('spark.executor.instances', 2)\
  .config('spark.executor.memory', '16g')\
  .config('spark.rapids.memory.pinnedPool.size', '2G')\
  .config('spark.plugins', 'com.nvidia.spark.SQLPlugin')\
  .config("spark.jars.packages", "com.nvidia:rapids-4-spark_2.12:25.08.0")\
  .config("spark.executor.resource.gpu.discoveryScript","/home/cdsw/getGpusResources.sh")\
  .config("spark.executor.resource.gpu.vendor", "nvidia.com")\
  .config("spark.rapids.shims-provider-override", "com.nvidia.spark.rapids.shims.spark332.SparkShimServiceProvider")\
  .config("spark.driver.memory","10g") \
  .config("spark.eventLog.enabled","true") \
  .getOrCreate()

df = spark.read("SELECT * FROM SPARK_CATALOG.{1}.{2};".format(DBNAME, USERNAME))
df.show()


                    .withColumn("age", "float", minValue=10, maxValue=100, random=True)
                    .withColumn("credit_card_balance", "float", minValue=100, maxValue=30000, random=True)
                    .withColumn("bank_account_balance", "float", minValue=0.01, maxValue=100000, random=True)
                    .withColumn("mortgage_balance", "float", minValue=0.01, maxValue=1000000, random=True)
                    .withColumn("sec_bank_account_balance", "float", minValue=0.01, maxValue=100000, random=True)
                    .withColumn("savings_account_balance", "float", minValue=0.01, maxValue=500000, random=True)
                    .withColumn("sec_savings_account_balance", "float", minValue=0.01, maxValue=500000, random=True)
                    .withColumn("total_est_nworth", "float", minValue=10000, maxValue=500000, random=True)
                    .withColumn("primary_loan_balance", "float", minValue=0.01, maxValue=5000, random=True)
                    .withColumn("secondary_loan_balance", "float", minValue=0.01, maxValue=500000, random=True)
                    .withColumn("uni_loan_balance", "float", minValue=0.01, maxValue=10000, random=True)
                    .withColumn("longitude", "float", minValue=-180, maxValue=180, random=True)
                    .withColumn("latitude", "float", minValue=-90, maxValue=90, random=True)
                    .withColumn("transaction_amount", "float", minValue=0.01, maxValue=30000, random=True)
                    .withColumn("fraud_trx", "string", values=["0", "1"], weights=[9, 1], random=True)

# 2. Transform data into a single vector column (required by Spark ML API)
feature_cols = ["age", "credit_card_balance", "bank_account_balance", "mortgage_balance", "sec_bank_account_balance", "savings_account_balance",
                    "sec_savings_account_balance", "total_est_nworth", "primary_loan_balance", "secondary_loan_balance", "uni_loan_balance",
                    "longitude", "latitude", "transaction_amount"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_assembled = assembler.transform(df)

# Split data into training and test sets
(training_data, test_data) = df_assembled.randomSplit([0.8, 0.2], seed=1234)

# 3. Use spark_rapids_ml.classification.RandomForestClassifier
# This automatically uses the GPU acceleration if the environment is set up
rf_classifier = RandomForestClassifier(labelCol="fraud_trx", featuresCol="features", numTrees=10)

# Train the model
print("Training RandomForest model...")
rf_model = rf_classifier.fit(training_data)
print("Model training complete.")

# 4. Make predictions on the test set
predictions = rf_model.transform(test_data)
predictions.select("prediction", "fraud_trx", "features").show(5)

# 5. Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="fraud_trx", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy: {accuracy}")

import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType, Int64TensorType
import onnx

# Define the input schema/types: name + shape + dtype
# Suppose your model expects features vector of length N
N = 14  # number of features in your Spark feature vector
initial_types = [("features", FloatTensorType([None, N]))]

# Convert to ONNX
onnx_model = onnxmltools.convert_sparkml(
    spark_model,
    name="SparkRapidsModel",
    initial_types=initial_types,
    target_opset=None  # you can specify a target ONNX opset version if desired
)

# Save model to file
onnxmltools.utils.save_model(onnx_model, "spark_model.onnx")

print("ONNX opset version:", onnx_model.opset_import[0].version)
