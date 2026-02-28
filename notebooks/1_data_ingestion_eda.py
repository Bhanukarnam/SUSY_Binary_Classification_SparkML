from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, stddev, min as smin, max as smax

PARQUET_PATH = "data/processed/susy_4m_parquet"

spark = (
    SparkSession.builder
    .appName("SUSY-EDA")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.driver.memory", "6g")
    .getOrCreate()
)

df = spark.read.parquet(PARQUET_PATH)

print("Rows:", df.count())
print("Columns:", len(df.columns))


print("Class distribution:")
df.groupBy("label").count().orderBy("label").show()


features = [f"f{i}" for i in range(1, 19)]
summary = df.select(
    *[mean(c).alias(f"{c}_mean") for c in features[:6]],
    *[stddev(c).alias(f"{c}_std") for c in features[:6]],
)
summary.show(truncate=False)

train, val, test = df.randomSplit([0.7, 0.15, 0.15], seed=42)

print("Split sizes:")
print("Train:", train.count())
print("Val  :", val.count())
print("Test :", test.count())

train.write.mode("overwrite").parquet("data/processed/susy_train")
val.write.mode("overwrite").parquet("data/processed/susy_val")
test.write.mode("overwrite").parquet("data/processed/susy_test")

print("Saved train/val/test parquet splits.")
spark.stop()
