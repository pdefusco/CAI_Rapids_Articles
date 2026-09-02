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

import os
from datetime import date

from pyspark.sql import functions as F
import cml.data_v1 as cmldata


class CalendarDimension:

    def __init__(self, connection_name, database):

        self.connection_name = connection_name
        self.database = database

    ############################################################

    def createSparkConnection(self):

        conn = cmldata.get_connection(self.connection_name)

        spark = conn.get_spark_session()

        from pyspark import SparkContext
        SparkContext.setSystemProperty('spark.executor.cores', '5')
        SparkContext.setSystemProperty('spark.executor.memory', '10g')

        return spark

    ############################################################

    def generateCalendar(
            self,
            spark,
            start="2024-01-01",
            end="2025-12-31"):

        days = spark.sql(f"""
            SELECT explode(
                sequence(
                    to_date('{start}'),
                    to_date('{end}'),
                    interval 1 day
                )
            ) AS calendar_date
        """)

        calendar = (

            days

            .withColumn(
                "date_key",
                F.date_format("calendar_date", "yyyyMMdd").cast("int")
            )

            .withColumn(
                "year",
                F.year("calendar_date")
            )

            .withColumn(
                "quarter",
                F.quarter("calendar_date")
            )

            .withColumn(
                "month",
                F.month("calendar_date")
            )

            .withColumn(
                "month_name",
                F.date_format("calendar_date", "MMMM")
            )

            .withColumn(
                "week_of_year",
                F.weekofyear("calendar_date")
            )

            .withColumn(
                "day_of_month",
                F.dayofmonth("calendar_date")
            )

            .withColumn(
                "day_of_week",
                F.dayofweek("calendar_date")
            )

            .withColumn(
                "day_name",
                F.date_format("calendar_date", "EEEE")
            )

            .withColumn(
                "is_weekend",
                F.col("day_of_week").isin([1, 7])
            )

            .withColumn(
                "is_month_end",
                F.last_day("calendar_date") == F.col("calendar_date")
            )

        )

        return calendar

    ############################################################

    def saveTable(self, df):

        df.write.mode("overwrite").saveAsTable(
            f"{self.database}.CALENDAR"
        )

        print(f"Calendar rows: {df.count()}")

        df.show(10, False)


############################################################

def main():

    USERNAME = os.environ["PROJECT_OWNER"]

    DATABASE = f"DEMO_{USERNAME}"

    CONNECTION_NAME = "se-aws-edl"

    generator = CalendarDimension(
        CONNECTION_NAME,
        DATABASE
    )

    spark = generator.createSparkConnection()

    calendar = generator.generateCalendar(spark)

    generator.saveTable(calendar)


############################################################

if __name__ == "__main__":

    main()
