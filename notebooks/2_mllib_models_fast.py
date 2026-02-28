from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder

TRAIN_PATH = "data/processed/susy_train"
TEST_PATH  = "data/processed/susy_test"

spark = (
    SparkSession.builder
    .appName("SUSY-MLlib-FAST")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.driver.memory", "6g")
    .getOrCreate()
)

train = spark.read.parquet(TRAIN_PATH)
test  = spark.read.parquet(TEST_PATH)


train_small = train.sample(withReplacement=False, fraction=0.15, seed=42).cache()
print("Train small rows:", train_small.count())

feature_cols = [f"f{i}" for i in range(1, 19)]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)

evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

def run_tvs(model, grid, name):
    pipeline = Pipeline(stages=[assembler, scaler, model])

    tvs = TrainValidationSplit(
        estimator=pipeline,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        trainRatio=0.8,
        parallelism=4,
        seed=42
    )

    print(f"\n=== {name} (TrainValidationSplit) ===")
    tvs_model = tvs.fit(train_small)

    test_pred = tvs_model.transform(test)
    test_auc = evaluator.evaluate(test_pred)

    print(f"{name} | Test AUC: {test_auc:.4f}")
    return test_auc

lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=30)
lr_grid = (ParamGridBuilder()
           .addGrid(lr.regParam, [0.0, 0.01])
           .addGrid(lr.elasticNetParam, [0.0, 1.0])
           .build())

rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed=42)
rf_grid = (ParamGridBuilder()
           .addGrid(rf.numTrees, [50])
           .addGrid(rf.maxDepth, [8])
           .build())

gbt = GBTClassifier(labelCol="label", featuresCol="features", seed=42)
gbt_grid = (ParamGridBuilder()
            .addGrid(gbt.maxDepth, [3, 5])
            .addGrid(gbt.maxIter, [20])
            .build())

aucs = []
aucs.append(("LogisticRegression", run_tvs(lr, lr_grid, "LogisticRegression")))
aucs.append(("RandomForest",      run_tvs(rf, rf_grid, "RandomForest")))
aucs.append(("GBTClassifier",     run_tvs(gbt, gbt_grid, "GBTClassifier")))

print("\n=== Summary (Test AUC) ===")
for n, a in aucs:
    print(f"{n:20s}  Test AUC={a:.4f}")

spark.stop()
