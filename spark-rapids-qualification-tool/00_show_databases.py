import cml.data_v1 as cmldata

# Sample in-code customization of spark configurations
#from pyspark import SparkContext
#SparkContext.setSystemProperty('spark.executor.cores', '4')
#SparkContext.setSystemProperty('spark.executor.memory', '8g')

CONNECTION_NAME = "se-aws-edl"

conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

# Sample usage to run query through spark
EXAMPLE_SQL_QUERY = "show databases"
spark.sql(EXAMPLE_SQL_QUERY).show()

# --- Demo: show live data across tables ---
DATABASE = "demo_pauldefusco"
TABLES = ["TRX", "CUSTOMERS", "ACCOUNTS", "MERCHANTS", "BRANCHES", "CALENDAR"]

print(f"{'Table':<15} {'Rows':>12} {'Columns':>10}")
print("-" * 40)

table_dfs = {}
total_rows = 0
total_cols = 0
for t in TABLES:
    df = spark.table(f"{DATABASE}.{t}")
    count = df.count()
    num_cols = len(df.columns)
    table_dfs[t] = df
    total_rows += count
    total_cols += num_cols
    print(f"{t:<15} {count:>12,} {num_cols:>10}")

# --- Drill into schema + sample rows per table ---
for t, df in table_dfs.items():
    print(f"\n=== {t} ===")
    df.printSchema()
    df.show(5, truncate=False)

# --- Grand total summary ---
summary = f"""
{"=" * 40}
TOTAL DATA VOLUME ACROSS ALL TABLES
{"=" * 40}
Tables:  {len(TABLES)}
Rows:    {total_rows:,}
Columns: {total_cols}
{"=" * 40}
"""
print(summary)