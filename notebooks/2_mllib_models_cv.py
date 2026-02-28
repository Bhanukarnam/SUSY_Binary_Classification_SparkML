from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

TRAIN_PATH = "data/processed/susy_train"
VAL_PATH   = "data/processed/susy_val"
TEST_PATH  = "data/processed/susy_test"

spark = (
    SparkSession.builder
    .appName("SUSY-MLlib-Models-CV")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.driver.memory", "6g")
    .getOrCreate()
)

train = spark.read.parquet(TRAIN_PATH).cache()
val   = spark.read.parquet(VAL_PATH).cache()
test  = spark.read.parquet(TEST_PATH).cache()

feature_cols = [f"f{i}" for i in range(1, 19)]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")

scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)

evaluator_auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

def run_cv(model, param_grid, model_name: str):
    pipeline = Pipeline(stages=[assembler, scaler, model])

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator_auc,
        numFolds=3,
        parallelism=4,   
        seed=42
    )

    print(f"\n=== Training {model_name} with CrossValidator ===")
    cv_model = cv.fit(train)


    val_pred = cv_model.transform(val)
    test_pred = cv_model.transform(test)

    val_auc = evaluator_auc.evaluate(val_pred)
    test_auc = evaluator_auc.evaluate(test_pred)

    print(f"{model_name} | Val AUC:  {val_auc:.4f}")
    print(f"{model_name} | Test AUC: {test_auc:.4f}")

    return cv_model, val_auc, test_auc


lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=50)
lr_grid = (ParamGridBuilder()
           .addGrid(lr.regParam, [0.0, 0.01, 0.1])
           .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
           .build())

rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed=42)
rf_grid = (ParamGridBuilder()
           .addGrid(rf.numTrees, [50, 100])
           .addGrid(rf.maxDepth, [5, 10])
           .build())

gbt = GBTClassifier(labelCol="label", featuresCol="features", seed=42)
gbt_grid = (ParamGridBuilder()
            .addGrid(gbt.maxDepth, [3, 5])
            .addGrid(gbt.maxIter, [20, 50])
            .build())

results = []

lr_model, lr_val, lr_test = run_cv(lr, lr_grid, "LogisticRegression")
results.append(("LogisticRegression", lr_val, lr_test))

rf_model, rf_val, rf_test = run_cv(rf, rf_grid, "RandomForest")
results.append(("RandomForest", rf_val, rf_test))

gbt_model, gbt_val, gbt_test = run_cv(gbt, gbt_grid, "GBTClassifier")
results.append(("GBTClassifier", gbt_val, gbt_test))

print("\n=== Summary (AUC) ===")
for name, v, t in results:
    print(f"{name:20s}  Val AUC={v:.4f}  Test AUC={t:.4f}")


best = max(results, key=lambda x: x[1])[0]
print(f"\nBest by validation AUC: {best}")


spark.stop()
