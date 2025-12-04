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

!pip install spark-rapids-ml

import os, warnings, sys, logging
import mlflow
import pandas as pd
import numpy as np
from datetime import date
from pyspark.ml.feature import VectorAssembler
from spark_rapids_ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

USERNAME = os.environ["PROJECT_OWNER"]
DBNAME = "DEMO_"+USERNAME
STORAGE = "s3a://pdf-oct-buk-a163bf71/data/"
DATE = date.today()

#os.environ["PYSPARK_SUBMIT_ARGS"] = "--jars /home/cdsw/rapids-4-spark_2.12-25.08.0.jar pyspark-shell"

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
  .config("spark.jars.packages", "com.nvidia:rapids-4-spark_2.12:25.10.0")\
  .config("spark.executor.resource.gpu.discoveryScript","/home/cdsw/getGpusResources.sh")\
  .config("spark.executor.resource.gpu.vendor", "nvidia.com")\
  .config("spark.rapids.shims-provider-override", "com.nvidia.spark.rapids.shims.spark351.SparkShimServiceProvider")\
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
  .config("spark.kerberos.access.hadoopFileSystems", "s3a://pdf-aw-buk-aec7c095/data/") \
  .getOrCreate()

from pyspark.sql import functions as F

# 1. Read from Data Lake Table
df = spark.read.table("DataLakeTable")
df.show()

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

#model_signature = infer_signature(X_train, y_pred)

# Convert to ONNX
onnx_model = onnxmltools.convert_sparkml(
    spark_model,
    name="SparkRapidsModel",
    initial_types=initial_types,
    target_opset=None  # you can specify a target ONNX opset version if desired
)

# Save model to Registry
#onnxmltools.utils.save_model(onnx_model, "fraud_classifier.onnx")
mlflow.onnx.log_model(onnx_model, "fraud-clf-onnx-spark-rapids-ml",
                      registered_model_name="fraud-clf-onnx-spark-rapids-ml"
                     )

#signature=model_signature)
