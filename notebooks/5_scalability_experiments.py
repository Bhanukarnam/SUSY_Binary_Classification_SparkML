import time
import csv
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

TRAIN_PATH = "data/processed/susy_train"
TEST_PATH  = "data/processed/susy_test"
OUT_CSV    = "data/processed/scalability_results.csv"

feature_cols = [f"f{i}" for i in range(1, 19)]

def run_once(shuffle_partitions: int, train_fraction: float):
    spark = (
        SparkSession.builder
        .appName(f"SUSY-Scalability-sp{shuffle_partitions}-frac{train_fraction}")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.driver.memory", "6g")
        .getOrCreate()
    )


    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)

    gbt = GBTClassifier(labelCol="label", featuresCol="features", seed=42, maxIter=20, maxDepth=5)

    evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

    train = spark.read.parquet(TRAIN_PATH)
    test  = spark.read.parquet(TEST_PATH).cache()


    train_sample = train.sample(withReplacement=False, fraction=train_fraction, seed=42).cache()
    n_train = train_sample.count()

    pipeline = Pipeline(stages=[assembler, scaler, gbt])

    t0 = time.time()
    model = pipeline.fit(train_sample)
    train_time_s = time.time() - t0


    t1 = time.time()
    pred = model.transform(test)
    auc = evaluator.evaluate(pred)
    eval_time_s = time.time() - t1

    spark.stop()
    return n_train, train_time_s, eval_time_s, auc



weak_fracs = [0.05, 0.10, 0.15, 0.20]
strong_parts = [50, 100, 200, 400]

rows = []

fixed_parts = 200
for frac in weak_fracs:
    n_train, t_train, t_eval, auc = run_once(fixed_parts, frac)
    rows.append({
        "experiment": "weak_scaling",
        "shuffle_partitions": fixed_parts,
        "train_fraction": frac,
        "train_rows": n_train,
        "train_time_s": round(t_train, 2),
        "eval_time_s": round(t_eval, 2),
        "test_auc": round(auc, 4),
    })


fixed_frac = 0.10
for sp in strong_parts:
    n_train, t_train, t_eval, auc = run_once(sp, fixed_frac)
    rows.append({
        "experiment": "strong_scaling_proxy",
        "shuffle_partitions": sp,
        "train_fraction": fixed_frac,
        "train_rows": n_train,
        "train_time_s": round(t_train, 2),
        "eval_time_s": round(t_eval, 2),
        "test_auc": round(auc, 4),
    })


with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved: {OUT_CSV}")
for r in rows:
    print(r)
