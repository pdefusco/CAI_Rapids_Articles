#****************************************************************************
# (C) Cloudera, Inc. 2020-2023
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

import os
import numpy as np
from datetime import datetime
import dbldatagen as dg
from dbldatagen import DataGenerator
from pyspark.sql import SparkSession
import sys, random, os, json, random, configparser
from utils import *
from pyspark.sql.types import LongType, FloatType, IntegerType, StringType, \
                              DoubleType, BooleanType, ShortType, \
                              TimestampType, DateType, DecimalType, \
                              ByteType, BinaryType, ArrayType, MapType, \
                              StructType, StructField

def main():
    ## CDE PROPERTIES
    print("PARSING JOB ARGUMENTS...")
    storageLocation = sys.argv[1]
    print("DATA LOCATION IN CLOUD STORAGE:")
    print(storageLocation)

    try:
        spark = SparkSession \
            .builder \
            .appName("DATA GENERATION") \
            .getOrCreate()
    except Exception as e:
        print("LAUNCHING SPARK SESSION UNSUCCESSFUL")
        print('\n')
        print(f'caught {type(e)}: e')
        print(e)

    try:
        myDG = DataGen(spark)
        print("DATAGEN CLASS INSTANTIATED SUCCESSFULLY")
        sparkDf = myDG.dataGen()
        print("DATA GENERATED SUCCESSFULLY")
        #print("SHOW GENERATED DATA:\n")
        #sparkDf.show()
    except Exception as e:
        print("INSTANTIATING DATAGEN UNSUCCESSFUL")
        print('\n')
        print(f'caught {type(e)}: e')
        print(e)

    try:
        sparkDf.write.format("parquet").mode("overwrite").save(storageLocation)
        print("DATA SAVED TO CLOUD STORAGE SUCCESSFULLY")
    except Exception as e:
        print("SAVING DATAGEN DF UNSUCCESSFUL")
        print('\n')
        print(f'caught {type(e)}: e')
        print(e)

    #transactionsDf.write.format("json").mode("overwrite").save("/home/cdsw/jsonData2.json")

if __name__ == "__main__":
    main()
