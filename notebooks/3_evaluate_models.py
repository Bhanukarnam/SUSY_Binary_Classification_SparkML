import time
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, sum as ssum
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

TRAIN_PATH = "data/processed/susy_train"
TEST_PATH  = "data/processed/susy_test"

spark = (
    SparkSession.builder
    .appName("SUSY-Evaluate-Models")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.driver.memory", "6g")
    .getOrCreate()
)

train = spark.read.parquet(TRAIN_PATH)
test  = spark.read.parquet(TEST_PATH).cache()

feature_cols = [f"f{i}" for i in range(1, 19)]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)

evaluator_auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

def confusion_and_scores(pred_df, threshold=0.5):

    scored = pred_df.withColumn("p1", vector_to_array(col("probability"))[1]) \
                    .withColumn("pred", when(col("p1") >= threshold, 1.0).otherwise(0.0))

    tp = scored.filter((col("label") == 1.0) & (col("pred") == 1.0)).count()
    tn = scored.filter((col("label") == 0.0) & (col("pred") == 0.0)).count()
    fp = scored.filter((col("label") == 0.0) & (col("pred") == 1.0)).count()
    fn = scored.filter((col("label") == 1.0) & (col("pred") == 0.0)).count()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy  = (tp + tn) / (tp + tn + fp + fn)

    return tp, fp, tn, fn, precision, recall, f1, accuracy

def fit_and_eval(model, name):
    pipeline = Pipeline(stages=[assembler, scaler, model])

    t0 = time.time()
    fitted = pipeline.fit(train)
    train_time = time.time() - t0

    pred = fitted.transform(test)

    auc = evaluator_auc.evaluate(pred)
    tp, fp, tn, fn, precision, recall, f1, acc = confusion_and_scores(pred, threshold=0.5)

    print(f"\n=== {name} ===")
    print(f"Train time (s): {train_time:.1f}")
    print(f"Test ROC-AUC   : {auc:.4f}")
    print("Confusion Matrix (threshold=0.5)")
    print(f"TP={tp}  FP={fp}")
    print(f"FN={fn}  TN={tn}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    return {
        "model": name,
        "train_time_s": round(train_time, 1),
        "test_auc": round(auc, 4),
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }


lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=30, regParam=0.01, elasticNetParam=0.0)
rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed=42, numTrees=50, maxDepth=8)
gbt = GBTClassifier(labelCol="label", featuresCol="features", seed=42, maxIter=20, maxDepth=5)

results = []
results.append(fit_and_eval(lr, "LogisticRegression"))
results.append(fit_and_eval(rf, "RandomForest"))
results.append(fit_and_eval(gbt, "GBTClassifier"))


out_df = spark.createDataFrame(results)
out_df.coalesce(1).write.mode("overwrite").option("header", True).csv("data/processed/model_metrics_csv")

print("\nSaved metrics CSV to data/processed/model_metrics_csv (single part file).")
spark.stop()
