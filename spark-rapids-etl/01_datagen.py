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
# #  Author(s): Paul de Fusco, James Horsch
#***************************************************************************/

import os
import numpy as np
import pandas as pd
from datetime import datetime
from pyspark.sql.types import LongType, IntegerType, StringType
from pyspark.sql import SparkSession
import dbldatagen as dg
import dbldatagen.distributions as dist
from dbldatagen import FakerTextFactory, DataGenerator, fakerText
from faker.providers import bank, credit_card, currency
import cml.data_v1 as cmldata


class BankDataGen:

    '''Class to Generate Banking Data'''

    def __init__(self, username, dbname, connectionName, storage):
        self.username = username
        self.dbname = dbname
        self.connectionName = connectionName
        self.storage = storage


    def dataGen(self, spark, shuffle_partitions_requested = 5, partitions_requested = 2, data_rows = 10000):
        """
        Method to create credit card transactions in Spark Df
        """

        # setup use of Faker
        FakerTextUS = FakerTextFactory(locale=['en_US'], providers=[bank])

        # partition parameters etc.
        spark.conf.set("spark.sql.shuffle.partitions", shuffle_partitions_requested)

        fakerDataspec = (DataGenerator(spark, rows=data_rows, partitions=partitions_requested)
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
                    )
        df = fakerDataspec.build()
        df = df.withColumn("fraud_trx", df["fraud_trx"].cast(IntegerType()))

        return df


    def createSparkConnection(self):
        """
        Method to create a Spark Connection using CML Data Connections
        """

        from pyspark import SparkContext
        SparkContext.setSystemProperty('spark.executor.cores', '2')
        SparkContext.setSystemProperty('spark.executor.memory', '4g')

        import cml.data_v1 as cmldata
        conn = cmldata.get_connection(self.connectionName)
        spark = conn.get_spark_session()

        return spark


    def saveFileToCloud(self, df):
        """
        Method to save credit card transactions df as csv in cloud storage
        """
        df.write.format("csv").mode('overwrite').option("header", "true").save(self.storage + "spark-rapids-ml-demo/" + self.username)


    def createDatabase(self, spark):
        """
        Method to create database before data generated is saved to new database and table
        """

        spark.sql("CREATE DATABASE IF NOT EXISTS {}".format(self.dbname))

        print("SHOW DATABASES LIKE '{}'".format(self.dbname))
        spark.sql("SHOW DATABASES LIKE '{}'".format(self.dbname)).show()


    def createOrReplace(self, df):
        """
        Method to create or append data to the BANKING TRANSACTIONS table
        The table is used to simulate batches of new data
        The table is meant to be updated periodically as part of a CML Job
        """

        try:
            df.writeTo("{0}.CC_TRX_{1}".format(self.dbname, self.username))\
              .using("iceberg").tableProperty("write.format.default", "parquet").append()

        except:
            df.writeTo("{0}.CC_TRX_{1}".format(self.dbname, self.username))\
                .using("iceberg").tableProperty("write.format.default", "parquet").createOrReplace()


    def createHiveTable(self, df, hive_table_name):
        """
        Method to create or append data to the BANKING TRANSACTIONS table
        The table is used to simulate batches of new data
        The table is meant to be updated periodically as part of a CML Job
        """

        df.write.mode("overwrite").saveAsTable(hive_table_name)


    def validateTable(self, spark):
        """
        Method to validate creation of table
        """

        print("SHOW TABLES FROM '{}'".format(self.dbname))
        spark.sql("SHOW TABLES FROM {}".format(self.dbname)).show()


def main():

    USERNAME = os.environ["PROJECT_OWNER"]
    DBNAME = "DEMO_"+USERNAME
    CONNECTION_NAME = "pdf-oct-aw-dl"
    STORAGE = "s3a://pdf-oct-buk-a163bf71/data/"

    # Instantiate BankDataGen class
    dg = BankDataGen(USERNAME, DBNAME, CONNECTION_NAME, STORAGE)

    # Create CML Spark Connection
    spark = dg.createSparkConnection()

    # Create Banking Transactions DF
    df = dg.dataGen(spark)

    # Create Spark Database
    #dg.createDatabase(spark)

    # Create Iceberg Table in Database
    #dg.createOrReplace(df)

    # Validate Iceberg Table in Database
    #dg.validateTable(spark)

    # Write Data to S3
    #dg.saveFileToCloud(df)

    # Save as Hive Table
    dg.createHiveTable(df, "DataLakeTable")


if __name__ == '__main__':
    main()
