import os, warnings, sys, logging
import mlflow
import pandas as pd
import numpy as np
from datetime import date
from pyspark.ml.feature import VectorAssembler
# use the GPU-native implementation
from spark_rapids_ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
###from spark_rapids_ml.metrics.MulticlassMetrics import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

import pprint

USERNAME = os.environ["PROJECT_OWNER"]
DBNAME = "DEMO_"+USERNAME
CONNECTION_NAME = "go01-aw-dl"
STORAGE =  os.environ["DATA_STORAGE"] 
DATE = date.today()

RAPIDS_JAR = "/home/cdsw/rapids-4-spark_2.12-25.10.0.jar"


LOCAL_PACKAGES = "/home/cdsw/.local/lib/python3.10/site-packages"
# This is where the specific CUDA 12 NVRTC library lives
NVRTC_LIB_PATH = f"{LOCAL_PACKAGES}/nvidia/cuda_nvrtc/lib"
WRITABLE_CACHE_DIR = "/tmp/cupy_cache"

# Force-clear any hanging Py4J connections
from py4j.java_gateway import java_import

try:
    spark.stop()
except:
    pass


spark = SparkSession.builder \
    .appName("Spark-Rapids-32GB-Final") \
    .config("spark.executor.resource.gpu.vendor", "nvidia.com") \
    .config("spark.executorEnv.LD_LIBRARY_PATH", f"{NVRTC_LIB_PATH}:{os.environ.get('LD_LIBRARY_PATH', '')}") \
    .config("spark.executorEnv.PYTHONPATH", LOCAL_PACKAGES) \
    .config("spark.driver.memory", "12g") \
    .config("spark.driver.maxResultSize", "4g") \
    .config("spark.dynamicAllocation.enabled", "false") \
    .config("spark.executor.cores", 3) \
    .config("spark.executor.instances", 1) \
    .config("spark.executor.memory", "10g") \
    .config("spark.executor.memoryOverhead", "10g") \
    .config("spark.sql.autoBroadcastJoinThreshold", -1) \
    .config("spark.network.timeout", "800s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .config("spark.sql.broadcastTimeout", "1200") \
    .config("spark.rapids.memory.pinnedPool.size", "4g") \
    .config("spark.executor.resource.gpu.amount", 1) \
    .config("spark.task.resource.gpu.amount", 0.33) \
    .config("spark.jars", RAPIDS_JAR) \
    .config("spark.kerberos.access.hadoopFileSystems", "s3a://go01-demo/user/jprosser/spark-rapids-ml/") \
    .config("spark.plugins", "com.nvidia.spark.SQLPlugin") \
    .config("spark.rapids.sql.enabled", "true") \
    .config("spark.driver.extraJavaOptions", f"-Djava.library.path={NVRTC_LIB_PATH}") \
    .config("spark.sql.cache.serializer", "com.nvidia.spark.ParquetCachedBatchSerializer") \
    .config("spark.shuffle.service.enabled", "false") \
    .config('spark.sql.shuffle.partitions', '200') \
    .config('spark.shuffle.file.buffer', '64k') \
    .config('spark.shuffle.spill.compress', 'true') \
    .config("spark.driverEnv.CUPY_CACHE_DIR", WRITABLE_CACHE_DIR) \
    .config("spark.executorEnv.CUPY_CACHE_DIR", WRITABLE_CACHE_DIR) \
    .config("spark.hadoop.fs.defaultFS", "s3a://go01-demo/") \
    .config("spark.executor.resource.gpu.discoveryScript", "/home/cdsw/spark-rapids-ml/getGpusResources.sh") \
    .config("spark.rapids.shims-provider-override", "com.nvidia.spark.rapids.shims.spark351.SparkShimServiceProvider") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Enable CollectLimit so that large datasets are collected on the GPU.
# Not worth it for small datasets
spark.conf.set("spark.rapids.sql.exec.CollectLimitExec", "true")

# Enabled to let the GPU to handle the random sampling of rows for large datasets
spark.conf.set("spark.rapids.sql.exec.SampleExec", "true")

# Enabled to let allow more time for large broadcast joins
spark.conf.set("spark.sql.broadcastTimeout", "1200") # Increase to 20 mins
from pyspark.sql import functions as F

#spark.conf.set("spark.rapids.sql.explain", "ALL")
spark.conf.set("spark.rapids.sql.explain", "NOT_ON_GPU") # Only log when/why the GPU was not selected
spark.conf.set("spark.rapids.sql.variable.float.allow", "true") # Allow float math

# Allow the GPU to cast instead of pushing back to CPU just for cast
spark.conf.set("spark.rapids.sql.castFloatToDouble.enabled", "true") 
spark.conf.set("spark.rapids.sql.format.parquet.enabled", "true")

# Turning off Adaptive Query Execution (AQE) makes the entire SQL plan use the GPU
spark.conf.set("spark.sql.adaptive.enabled", "false")


## Do an initial test to confirm the the NVidia SQLPlugin
from pyspark.sql import functions as F

df = spark.read.table("DataLakeTable")
print(f"Columns: {len(df.columns)}")
print(f"Schema: {df.schema}")
df.limit(5).explain(mode="formatted")



feature_cols = ["age", "credit_card_balance", "bank_account_balance", "mortgage_balance", "sec_bank_account_balance", "savings_account_balance",
                    "sec_savings_account_balance", "total_est_nworth", "primary_loan_balance", "secondary_loan_balance", "uni_loan_balance",
                    "longitude", "latitude", "transaction_amount"]


(training_data, test_data) = df.randomSplit([0.8, 0.2], seed=1234)




# By using 'featuresCols' (list of strings), we avoid VectorAssembler and VectorUDT data types
# This keeps the data in GPU-friendly columnar format.
rf_classifier = RandomForestClassifier(
    labelCol="fraud_trx", 
    featuresCols=feature_cols, 
    numTrees=20
)
# [ NOTE: setting numTrees != 20 will result in a numTrees mismatch error when we do the onnx conversion}


# Run the training logic in C++ on the GPU via cuML
print("Training Spark RAPIDS ML model...")
rf_model = rf_classifier.fit(training_data)
print("Model training complete.")

# We drop 'probability' and 'rawPrediction' because they are VectorUDT types
# that Spark SQL would otherwise force back to the CPU for formatting.
predictions = rf_model.transform(test_data).drop("probability", "rawPrediction")

predictions.select("prediction", "fraud_trx").show(5)

# Show the plan that fully utilize the GPU at all stages
predictions.explain(mode="formatted")

evaluator = MulticlassClassificationEvaluator(
    labelCol="fraud_trx", 
    predictionCol="prediction", 
    metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy: {accuracy}")

import numpy as np
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

original_np_array = np.array

# The impurities list gets corrupted during model creation
def truncate_to_binary(obj, *args, **kwargs):
    # If this is the impurityStats list
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], (list, np.ndarray)):
        # Chop [0.0, 0.0, 0.0] -> [0.0, 0.0]
        return original_np_array([list(row)[:2] for row in obj], *args, **kwargs)
    return original_np_array(obj, *args, **kwargs)

# Inject the fix
np.array = truncate_to_binary

print("Moving cleaned model back to CPU...")
rf_model_cpu = rf_model.cpu()
spark.conf.set("spark.rapids.sql.explain", "NONE")
onnx_model = onnxmltools.convert_sparkml(
        rf_model_cpu, 
        initial_types=[("features", FloatTensorType([None, 14]))],
        spark_session=spark 
    )
onnxmltools.utils.save_model(onnx_model, "fraud_model_final.onnx")
print("✨ SUCCESS!")

# Save model to Registry
#onnxmltools.utils.save_model(onnx_model, "fraud_classifier.onnx")
mlflow.onnx.log_model(onnx_model, "fraud-clf-onnx-spark-rapids-ml",
                      registered_model_name="fraud-clf-onnx-spark-rapids-ml"
                     )
