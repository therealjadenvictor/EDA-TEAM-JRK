import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "financial_fraud_detection_dataset.csv"

TARGET = "is_fraud"

# ==================================================
# STEP 1: LOAD AND CHRONOLOGICALLY SPLIT DATA
# ==================================================

print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="ISO8601"
)

df = df.sort_values("timestamp").reset_index(drop=True)

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()

print("Train:", train_df.shape)
print("Validation:", val_df.shape)

# ==================================================
# STEP 2: CREATE HISTORICAL ACCOUNT FEATURES
# ==================================================

print("\nCreating account history features...")

combined = pd.concat(
    [train_df, val_df],
    ignore_index=True
).sort_values("timestamp").copy()

sender_groups = combined.groupby("sender_account", sort=False)

combined["sender_txn_count"] = sender_groups.cumcount()

previous_amount_sum = (
    sender_groups["amount"].cumsum() - combined["amount"]
)

previous_count = sender_groups.cumcount()

combined["sender_avg_amount"] = (
    previous_amount_sum / previous_count.replace(0, np.nan)
)

combined["receiver_txn_count"] = (
    combined.groupby("receiver_account", sort=False).cumcount()
)

# Restore chronological partition boundaries.
# Train and validation are already chronological.
train_df = combined.iloc[:train_end].copy()
val_df = combined.iloc[train_end:val_end].copy()

# ==================================================
# STEP 3: TEMPORAL FEATURES
# ==================================================

for partition in [train_df, val_df]:

    partition["hour_of_day"] = partition["timestamp"].dt.hour
    partition["day_of_week"] = partition["timestamp"].dt.dayofweek
    partition["month"] = partition["timestamp"].dt.month
    partition["is_weekend"] = (
        partition["day_of_week"].isin([5, 6]).astype(int)
    )

    if "fraud_type" in partition.columns:
        partition.drop(columns=["fraud_type"], inplace=True)

# ==================================================
# STEP 4: SELECT FEATURES
# ==================================================

num_features = [
    "amount",
    "spending_deviation_score",
    "velocity_score",
    "geo_anomaly_score",
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "sender_txn_count",
    "sender_avg_amount",
    "receiver_txn_count"
]

cat_features = [
    "transaction_type",
    "merchant_category",
    "device_used",
    "location",
    "payment_channel"
]

features = num_features + cat_features

X_train = train_df[features].copy()
X_val = val_df[features].copy()

y_train = train_df[TARGET]
y_val = val_df[TARGET]

# CatBoost handles missing numerical values natively.
# Convert categorical columns to strings and fill missing values.

for col in cat_features:

    X_train[col] = X_train[col].fillna("Unknown").astype(str)
    X_val[col] = X_val[col].fillna("Unknown").astype(str)

cat_indices = [
    X_train.columns.get_loc(col)
    for col in cat_features
]

# ==================================================
# STEP 5: CLASS WEIGHTS
# ==================================================

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

class_weights = [
    1.0,
    negative_count / positive_count
]

print("\nClass weights:", class_weights)

# ==================================================
# STEP 6: TRAIN CATBOOST
# ==================================================

print("\nTraining CatBoost...")

model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.1,
    loss_function="Logloss",
    eval_metric="PRAUC",
    class_weights=class_weights,
    random_seed=42,
    verbose=100,
    early_stopping_rounds=40,
    thread_count=-1,
    allow_writing_files=False
)

model.fit(
    X_train,
    y_train,
    cat_features=cat_indices,
    eval_set=(X_val, y_val),
    use_best_model=True
)

# ==================================================
# STEP 7: VALIDATION EVALUATION
# ==================================================

print("\nEvaluating CatBoost...")

probabilities = model.predict_proba(X_val)[:, 1]

predictions = (probabilities >= 0.5).astype(int)

print("\n========== CATBOOST RESULTS ==========")

print(
    "PR-AUC:",
    average_precision_score(y_val, probabilities)
)

print(
    "ROC-AUC:",
    roc_auc_score(y_val, probabilities)
)

print(
    "Precision:",
    precision_score(y_val, predictions, zero_division=0)
)

print(
    "Recall:",
    recall_score(y_val, predictions, zero_division=0)
)

print(
    "F1:",
    f1_score(y_val, predictions, zero_division=0)
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, predictions))

print("\nBest iteration:", model.best_iteration_)

# ==================================================
# STEP 8: SAVE MODEL
# ==================================================

output_path = BASE_DIR / "catboost_experiment_model.cbm"

model.save_model(output_path)

print("\nModel saved to:", output_path)
print("Experiment completed.")

