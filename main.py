from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, when, year, month, trim, lower, regexp_replace, to_date, lit,
    avg, max, min, sum, regexp_extract, current_timestamp, input_file_name
)
from pyspark.sql.types import DoubleType
import re
from datetime import datetime
from pyspark.sql.functions import expr

# Initialize Spark
spark = SparkSession.builder.appName("bronze_silver_gold").getOrCreate()

RAW_DIR = "/app/data"
BRONZE_DIR = "/app/data/bronze"
SILVER_DIR = "/app/data/silver"
GOLD_DIR = "/app/data/gold"
# Clean existing bronze/silver/gold outputs
import shutil
import os

def clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"[CLEAN] Removed old directory: {path}")

for d in [BRONZE_DIR, SILVER_DIR, GOLD_DIR]:
    clean_dir(d)

# Bronze Layer
tables = {
    "feature_clickstream":"feature_clickstream.csv",
    "features_attributes":"features_attributes.csv",
    "features_financials":"features_financials.csv",
    "lms_loan_daily":"lms_loan_daily.csv",
}
def norm_colname(c):
    c = re.sub(r"\s+", "_", c.strip())
    c = re.sub(r"[^0-9a-zA-Z_]", "_", c)
    c = re.sub(r"_+", "_", c)
    return c.lower().strip("_")

for name, file in tables.items():
    path = f"{RAW_DIR}/{file}"
    df = spark.read.option("header", True).option("inferSchema", True).csv(path)
    for old in df.columns:
        df = df.withColumnRenamed(old, norm_colname(old))
    df = (
        df.withColumn("_ingest_time", current_timestamp())
          .withColumn("_ingest_date", lit(datetime.utcnow().strftime("%Y-%m-%d")))
          .withColumn("_source_name", lit(name))
          .withColumn("_source_file", input_file_name())
    )
    out = f"{BRONZE_DIR}/{name}"
    df.write.mode("overwrite").partitionBy("_ingest_date").parquet(out)
    print(f"[OK] {name} -> {out} ({df.count()} rows)")
print(f"[DEBUG] {name}: {df.count()} rows loaded from {file}")


# Silver Layer
tables = ["feature_clickstream","features_attributes","features_financials","lms_loan_daily"]
for t in tables:
    df = spark.read.parquet(f"{BRONZE_DIR}/{t}")
    cols = []
    seen = set()
    for c in df.columns:
        if c not in seen:
            cols.append(c)
            seen.add(c)
    df = df.select(cols)
    print("missing values:")
    df.select([count(when(col(c).isNull(),c)).alias(c) for c in df.columns]).show()
    if t =="feature_clickstream":
        if "snapshot_date" in df.columns:
            df = df.withColumn("year",year(col("snapshot_date")))
            df = df.withColumn("month",month(col("snapshot_date")))
        if "customer_id" in df.columns and "snapshot_date" in df.columns:
            df.dropDuplicates(["customer_id","snapshot_date"])
    elif t == "features_attributes":
        for c, dtype in df.dtypes:
            if dtype =="string":
                df = df.withColumn(c,trim(lower(col(c))))
        if "gender" in df.columns:
            df = df.withColumn(
                "gender",
                when(col("gender").isin("male", "m"), "M")
                .when(col("gender").isin("female", "f"), "F")
                .otherwise("U")
            )
        elif t == "features_financials":
            
            for c in df.columns:
                if "amount" in c or "balance" in c:
                    df = df.withColumn(c, regexp_replace(col(c), "[$,]", ""))
                    df = df.withColumn(c, expr(f"try_cast({c} as double)"))
    elif t == "lms_loan_daily":
        if "loan_date" in df.columns:
            df = df.withColumn("loan_date",to_date(col("loan_date"),"yyy-MM-dd"))
        if "due_date" in df.columns:
            df = df.withColumn("due_date",to_date(col("due_date"),"yyy-MM-dd"))
        if "due_date" in df.columns and "loan_date" in df.columns:
            df = df.withColumn("is_overdue",when(col("due_date")<col("loan_date"),lit(1))).otherwise(lit(0))
    out = f"{SILVER_DIR}/{t}"
    df.write.mode("overwrite").parquet(out)
    print(f"{t} saved to {out} ({df.count()} rows)")


# Gold Layer
clickstream = spark.read.parquet(f"{SILVER_DIR}/feature_clickstream")
attributes  = spark.read.parquet(f"{SILVER_DIR}/features_attributes")
financials  = spark.read.parquet(f"{SILVER_DIR}/features_financials")
loans       = spark.read.parquet(f"{SILVER_DIR}/lms_loan_daily")

click_agg = (
    clickstream.groupBy("customer_id")
    .agg(
        *[avg(c).alias(f"avg_{c}") for c in clickstream.columns if c.startswith("fe_")],
        *[max(c).alias(f"max_{c}") for c in clickstream.columns if c.startswith("fe_")],
        *[min(c).alias(f"min_{c}") for c in clickstream.columns if c.startswith("fe_")]
    )
)

attr_sel = (
    attributes
    .withColumn("age", regexp_extract(col("age"), r"(\d+)", 1).cast("int"))
    .drop("ssn")
    .select("customer_id", "name", "age", "occupation")
)

fin_sel = (
    financials
    .withColumn("annual_income", regexp_replace(col("annual_income"), "[^0-9.]", "").cast("double"))
    .withColumn("outstanding_debt", regexp_replace(col("outstanding_debt"), "[^0-9.]", "").cast("double"))
    .withColumn("credit_history_months",
                regexp_extract(col("credit_history_age"), r"(\d+)\s+Years", 1).cast("int") * 12 +
                regexp_extract(col("credit_history_age"), r"(\d+)\s+Months", 1).cast("int"))
    .drop("credit_history_age", "_source_name", "_source_file", "_ingest_time", "_ingest_date")
)

loan_agg = (
    loans.groupBy("customer_id")
    .agg(
        sum("paid_amt").alias("total_paid"),
        max("overdue_amt").alias("max_overdue"),
        avg("balance").alias("avg_balance"),
        max("tenure").alias("max_tenure")
    )
)

gold = (
    attr_sel
    .join(fin_sel, "customer_id", "left")
    .join(click_agg, "customer_id", "left")
    .join(loan_agg, "customer_id", "left")
)

out = f"{GOLD_DIR}/customer_features"
gold.write.mode("overwrite").parquet(out)
print(f"[OK] gold dataset saved to {out} ({gold.count()} rows, {len(gold.columns)} columns)")
