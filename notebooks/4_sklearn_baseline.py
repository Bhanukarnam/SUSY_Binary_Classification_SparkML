import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.functions import vector_to_array

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

TRAIN_PATH = "data/processed/susy_train"
TEST_PATH  = "data/processed/susy_test"

N_TRAIN = 200_000
N_TEST  = 100_000

spark = SparkSession.builder.appName("SUSY-sklearn-baseline").getOrCreate()

train = spark.read.parquet(TRAIN_PATH)
test  = spark.read.parquet(TEST_PATH)

train_pd = train.sample(withReplacement=False, fraction=0.10, seed=42).limit(N_TRAIN).toPandas()
test_pd  = test.sample(withReplacement=False, fraction=0.25, seed=42).limit(N_TEST).toPandas()
spark.stop()

X_train = train_pd.drop(columns=["label"]).values
y_train = train_pd["label"].values

X_test  = test_pd.drop(columns=["label"]).values
y_test  = test_pd["label"].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

t0 = time.time()
clf = LogisticRegression(max_iter=200, n_jobs=-1)
clf.fit(X_train, y_train)
train_time = time.time() - t0


y_prob = clf.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(float)

auc = roc_auc_score(y_test, y_prob)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("=== scikit-learn LogisticRegression baseline ===")
print(f"Train rows: {N_TRAIN:,} | Test rows: {N_TEST:,}")
print(f"Train time (s): {train_time:.1f}")
print(f"Test ROC-AUC   : {auc:.4f}")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print("Confusion matrix [[TN, FP],[FN, TP]]:")
print(cm)


out = pd.DataFrame([{
    "model": "sklearn_LogisticRegression",
    "train_rows": N_TRAIN,
    "test_rows": N_TEST,
    "train_time_s": round(train_time, 1),
    "test_auc": round(auc, 4),
    "accuracy": round(acc, 4),
    "precision": round(prec, 4),
    "recall": round(rec, 4),
    "f1": round(f1, 4),
    "tn": int(cm[0,0]), "fp": int(cm[0,1]),
    "fn": int(cm[1,0]), "tp": int(cm[1,1]),
}])

out.to_csv("data/processed/sklearn_baseline_metrics.csv", index=False)
print("Saved: data/processed/sklearn_baseline_metrics.csv")
