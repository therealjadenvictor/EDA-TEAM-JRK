
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

# 1. Define memory-optimized dtypes for 5M rows
DATATYPES = {
    'transaction_id': 'string',
    'sender_account': 'string',
    'receiver_account': 'string',
    'amount': 'float32',
    'spending_deviation_score': 'float32',
    'velocity_score': 'float32',
    'geo_anomaly_score': 'float32',
    'transaction_type': 'category',
    'merchant_category': 'category',
    'device_used': 'category',
    'location': 'category',
    'is_fraud': 'int8',
    'time_since_last_transaction': 'float32',
    'fraud_type': 'category',
    'timestamp': 'string'
}

# STEP 1: MEMORY-AWARE DATA INGESTION & AUDIT
def load_and_audit_dataset(file_path: str) -> pd.DataFrame:
    print("==================================================")
    print("STEP 1: MEMORY-AWARE DATA INGESTION & AUDIT")
    print("==================================================")
    
    print("[+] Ingesting dataset into memory...")
    df = pd.read_csv(file_path, dtype=DATATYPES)
    
    print("[+] Parsing timestamps (ISO8601 format)...")
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    
    mem_usage_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"\n[+] Loaded Dataset Shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"[+] Memory footprint     : {mem_usage_mb:.2f} MB")
    
    # Class balance audit
    print("\n--- Class Distribution Audit ---")
    fraud_counts = df['is_fraud'].value_counts()
    fraud_pct = df['is_fraud'].value_counts(normalize=True) * 100
    print(f"Legitimate (0) : {fraud_counts.get(0, 0):,} ({fraud_pct.get(0, 0.0):.2f}%)")
    print(f"Fraudulent (1) : {fraud_counts.get(1, 0):,} ({fraud_pct.get(1, 0.0):.2f}%)")
    
    # Duplicate check
    print("\n--- Duplicate Check ---")
    duplicates = df.duplicated().sum()
    print(f"Exact duplicate rows detected: {duplicates:,}")
    if duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Duplicates removed. Cleaned Dataset Shape: {df.shape[0]:,} rows")
        
    print("==================================================\n")
    return df

# STEP 2: TIME-AWARE LEAKAGE-SAFE DATA SPLITTING
def split_time_aware(
    df: pd.DataFrame,
    time_col: str = 'timestamp',
    target_col: str = 'is_fraud',
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("==================================================")
    print("STEP 2: TIME-AWARE LEAKAGE-SAFE DATA SPLITTING")
    print("==================================================")
    
    print(f"[+] Sorting {len(df):,} records chronologically by '{time_col}'...")
    df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
    
    total_rows = len(df_sorted)
    train_end = int(total_rows * train_ratio)
    val_end = int(total_rows * (train_ratio + val_ratio))
    
    train_df = df_sorted.iloc[:train_end].copy()
    val_df   = df_sorted.iloc[train_end:val_end].copy()
    test_df  = df_sorted.iloc[val_end:].copy()
    
    print("\n--- Partition Breakdown ---")
    splits = [('Training (70%)', train_df), ('Validation (15%)', val_df), ('Test Lockbox (15%)', test_df)]
    for name, partition in splits:
        print(f"\n{name}:")
        print(f"  • Rows         : {len(partition):,}")
        print(f"  • Date Horizon : {partition[time_col].min()}  -->  {partition[time_col].max()}")
        print(f"  • Fraud Count  : {partition[target_col].sum():,} ({partition[target_col].mean()*100:.2f}% prevalence)")
        
    assert train_df[time_col].max() <= val_df[time_col].min(), "Error: Train/Val overlap detected!"
    assert val_df[time_col].max() <= test_df[time_col].min(), "Error: Val/Test overlap detected!"
    print("\n[✓] Sanity Check Passed: Zero temporal overlap across partitions.")
    print("==================================================\n")
    
    return train_df, val_df, test_df
# STEP 2.5: HISTORICAL ACCOUNT BEHAVIOUR FEATURES
def create_account_history_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
):
    print("\n==================================================")
    print("STEP 2.5: HISTORICAL ACCOUNT BEHAVIOUR FEATURES")
    print("==================================================")

    combined = pd.concat(
        [train_df, val_df, test_df],
        ignore_index=True
    )

    combined = combined.sort_values("timestamp").copy()

    # Previous sender transaction count
    combined["sender_txn_count"] = (
        combined.groupby("sender_account", sort=False).cumcount()
    )

    # Previous average sender transaction amount
    sender_groups = combined.groupby("sender_account", sort=False)

    previous_amount_sum = sender_groups["amount"].cumsum() - combined["amount"]

    previous_count = sender_groups.cumcount()

    combined["sender_avg_amount"] = (
    previous_amount_sum / previous_count.replace(0, np.nan)
)

    # Previous receiver transaction count
    combined["receiver_txn_count"] = (
        combined.groupby("receiver_account", sort=False).cumcount()
    )

    # Separate the original chronological partitions
    train_end = len(train_df)
    val_end = train_end + len(val_df)

    train_features = combined.iloc[:train_end].copy()
    val_features = combined.iloc[train_end:val_end].copy()
    test_features = combined.iloc[val_end:].copy()

    # Restore original indexes
    train_features.index = train_df.index
    val_features.index = val_df.index
    test_features.index = test_df.index

    print("[✓] Historical account features created.")
    print("==================================================")

    return train_features, val_features, test_features

# STEP 3: LEAKAGE-SAFE PREPROCESSING & FEATURE ENGINEERING
def engineer_and_preprocess_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = 'is_fraud'
):
    print("==================================================")
    print("STEP 3: FEATURE ENGINEERING & TRAIN-ONLY PREPROCESSING")
    print("==================================================")

    # Work on copies to avoid modifying the original partitions
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    partitions = [
        ('Train', train_df),
        ('Validation', val_df),
        ('Test', test_df)
    ]

    print("[+] Extracting temporal features...")

    for name, partition in partitions:
        partition['hour_of_day'] = partition['timestamp'].dt.hour.astype('int8')
        partition['day_of_week'] = partition['timestamp'].dt.dayofweek.astype('int8')
        partition['month'] = partition['timestamp'].dt.month.astype('int8')
        partition['is_weekend'] = partition['day_of_week'].isin([5, 6]).astype('int8')

        if 'fraud_type' in partition.columns:
            partition.drop(columns=['fraud_type'], inplace=True)

    # Unreliable time_since_last_transaction deliberately excluded
    num_features = [
        'amount',
    'spending_deviation_score',
    'velocity_score',
    'geo_anomaly_score',
    'hour_of_day',
    'day_of_week',
    'month',
    'is_weekend',
    'sender_txn_count',
    'sender_avg_amount',
    'receiver_txn_count'
    ]

    cat_features = [
        'transaction_type',
        'merchant_category',
        'device_used',
        'location',
        'payment_channel'
    ]

    X_train = train_df[num_features + cat_features]
    y_train = train_df[target_col]

    X_val = val_df[num_features + cat_features]
    y_val = val_df[target_col]

    X_test = test_df[num_features + cat_features]
    y_test = test_df[target_col]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ])

    print("[+] Fitting preprocessor STRICTLY on Training set...")

    X_train_proc = preprocessor.fit_transform(X_train)
    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(exist_ok=True)

    joblib.dump(
       preprocessor,
       models_dir / "preprocessor.pkl"
    )

    print("[✓] Preprocessor saved.")
    print("[+] Transforming Validation and Test sets...")

    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    print("\n[✓] Preprocessing Complete:")
    print(f"  • Processed Train Shape      : {X_train_proc.shape[0]:,} rows x {X_train_proc.shape[1]} features")
    print(f"  • Processed Validation Shape : {X_val_proc.shape[0]:,} rows x {X_val_proc.shape[1]} features")
    print(f"  • Processed Test Shape       : {X_test_proc.shape[0]:,} rows x {X_test_proc.shape[1]} features")

    print("==================================================\n")

    return (
        (X_train_proc, y_train),
        (X_val_proc, y_val),
        (X_test_proc, y_test)
    )


# STEP 4: MODEL TRAINING & IMBALANCE STRATEGY
def train_leakage_safe_models(X_train, y_train, X_val, y_val):
    print("==================================================")
    print("STEP 4: MODEL TRAINING & COST-SENSITIVE IMBALANCE HANDLING")
    print("==================================================")
    
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    print(f"[+] Training Set Class Balance: {neg_count:,} Legitimate | {pos_count:,} Fraud")
    print(f"[+] Calculated Imbalance Scale Weight: {scale_pos_weight:.2f}")
    
    models = {
        'Logistic Regression (Baseline)': LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        'LightGBM Classifier': LGBMClassifier(
            scale_pos_weight=scale_pos_weight,
            n_estimators=300,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1
        ),
        'Random Forest Classifier': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
    }
    
    trained_models = {}
    val_predictions_proba = {}
    
    for name, model in models.items():
        print(f"\n[+] Training {name}...")
        model.fit(X_train, y_train)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        trained_models[name] = model
        if name == 'Logistic Regression (Baseline)':

            joblib.dump(
                model,
                
                Path(__file__).resolve().parent / "models" / "logistic_regression.pkl"
            )

            print("[✓] Logistic Regression model saved.")
        val_predictions_proba[name] = y_val_proba
        print(f"  • {name} trained successfully.")
        
    print("\n[✓] Model Training Complete.")
    print("==================================================\n")
    
    return trained_models, val_predictions_proba

# STEP 5: VALIDATION THRESHOLD OPTIMIZATION
def optimize_decision_thresholds(trained_models, val_predictions_proba, y_val):
    print("==================================================")
    print("STEP 5: VALIDATION-SET THRESHOLD OPTIMIZATION")
    print("==================================================")
    
    optimal_thresholds = {}
    thresholds_grid = np.linspace(0.01, 0.99, 99)
    
    for name, y_val_proba in val_predictions_proba.items():
        print(f"\n[+] Scanning threshold space for {name}...")
        best_f1 = -1.0
        best_thresh = 0.5
        
        for thresh in thresholds_grid:
            y_pred_bin = (y_val_proba >= thresh).astype(int)
            f1 = f1_score(y_val, y_pred_bin, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                
        optimal_thresholds[name] = best_thresh
        print(f"  • Selected Optimal Validation Threshold: {best_thresh:.2f} (Val F1: {best_f1:.4f})")
        
    print("\n[✓] Threshold Optimization Complete.")
    print("==================================================\n")
    
    return optimal_thresholds

# STEP 6: FINAL EVALUATION ON UNTOUCHED TEST SET

def evaluate_on_locked_test_set(
    trained_models,
    optimal_thresholds,
    X_test,
    y_test,
    c_fn: float = 100.0,
    c_fp: float = 5.0
):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report

    print("==================================================")
    print("STEP 6: MODEL EVALUATION & VISUALIZATION")
    print("==================================================")

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)

    test_results = []
    model_predictions = {}

    for name, model in trained_models.items():

        print(f"\n[+] Evaluating {name}...")

        y_test_proba = model.predict_proba(X_test)[:, 1]

        threshold = optimal_thresholds[name]
        y_test_pred = (y_test_proba >= threshold).astype(int)

        model_predictions[name] = (y_test, y_test_pred)

        tn, fp, fn, tp = confusion_matrix(
            y_test, y_test_pred
        ).ravel()

        pr_auc = average_precision_score(y_test, y_test_proba)
        roc_auc = roc_auc_score(y_test, y_test_proba)
        precision = precision_score(y_test, y_test_pred, zero_division=0)
        recall = recall_score(y_test, y_test_pred, zero_division=0)
        f1 = f1_score(y_test, y_test_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_test_pred)

        total_cost = (c_fn * fn) + (c_fp * fp)

        print(f"  • Threshold : {threshold:.2f}")
        print(f"  • PR-AUC    : {pr_auc:.4f}")
        print(f"  • ROC-AUC   : {roc_auc:.4f}")
        print(f"  • Precision : {precision:.4f}")
        print(f"  • Recall    : {recall:.4f}")
        print(f"  • F1        : {f1:.4f}")
        print(f"  • MCC       : {mcc:.4f}")
        print(f"  • Cost      : ${total_cost:,.2f}")

        test_results.append({
            "Model": name,
            "Threshold": threshold,
            "PR-AUC": pr_auc,
            "ROC-AUC": roc_auc,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "MCC": mcc,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Total Cost ($)": total_cost
        })

        # Save classification report
        report = classification_report(
            y_test,
            y_test_pred,
            target_names=["Legitimate", "Fraudulent"],
            zero_division=0
        )

        safe_name = name.replace(" ", "_").replace("/", "_")

        with open(
            output_dir / f"{safe_name}_classification_report.txt",
            "w",
            encoding="utf-8"
        ) as file:
            file.write(report)

    results_df = pd.DataFrame(test_results)

    # Save results table
    results_df.to_csv(
        output_dir / "model_comparison.csv",
        index=False
    )

    print("\n[+] Saved model comparison CSV.")

    # 1. PR-AUC comparison
    plt.figure(figsize=(9, 5))
    plt.bar(results_df["Model"], results_df["PR-AUC"])
    plt.title("Model Comparison - Precision Recall AUC")
    plt.ylabel("PR-AUC")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_dir / "pr_auc_comparison.png", dpi=300)
    plt.close()

    # 2. ROC-AUC comparison
    plt.figure(figsize=(9, 5))
    plt.bar(results_df["Model"], results_df["ROC-AUC"])
    plt.title("Model Comparison - ROC AUC")
    plt.ylabel("ROC-AUC")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_auc_comparison.png", dpi=300)
    plt.close()

    # 3. Precision, Recall and F1 comparison
    metrics = ["Precision", "Recall", "F1-Score"]

    results_df.set_index("Model")[metrics].plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.title("Precision, Recall and F1 Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_dir / "classification_metrics.png", dpi=300)
    plt.close()

    # 4. Confusion matrices
    for name, (actual, predicted) in model_predictions.items():

        fig, ax = plt.subplots(figsize=(7, 6))

        ConfusionMatrixDisplay.from_predictions(
            actual,
            predicted,
            display_labels=["Legitimate", "Fraudulent"],
            cmap="Blues",
            values_format=",",
            ax=ax
        )

        ax.set_title(f"Confusion Matrix - {name}")

        safe_name = name.replace(" ", "_").replace("/", "_")

        plt.tight_layout()
        plt.savefig(
            output_dir / f"{safe_name}_confusion_matrix.png",
            dpi=300
        )
        plt.close()

    print("\n[✓] All visualizations saved.")

    print("\n==================================================")
    print("MODEL PERFORMANCE COMPARISON")
    print("==================================================")

    print(
        results_df[
            ["Model", "PR-AUC", "ROC-AUC", "Precision",
             "Recall", "F1-Score", "MCC", "Total Cost ($)"]
        ].to_string(index=False)
    )

    print("\n[✓] Results saved in:", output_dir)
    print("==================================================")

    return results_df

# STEP 7: MISSING VALUE DIAGNOSTIC
def analyze_missing_values(df: pd.DataFrame) -> None:
    print("\n" + "=" * 50)
    print("STEP 7: MISSING VALUE DIAGNOSTIC")
    print("=" * 50)

    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100

    missing_report = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing Percentage": missing_percent
    })

    missing_report = missing_report.sort_values(
        by="Missing Percentage",
        ascending=False
    )

    print("\n[+] Missing-value report:")
    print(missing_report.to_string())

    print("\n[+] Total missing values:",
          missing_count.sum())
    print("\n[+] Missing time_since_last_transaction by class:")

    history_missing = df.groupby("is_fraud")[
        "time_since_last_transaction"
    ].apply(lambda x: x.isnull().mean() * 100)

    print(history_missing.rename({
        0: "Legitimate",
        1: "Fraudulent"
    }).to_string())

    print("\n[+] Transaction counts by class and missing history:")

    history_table = pd.crosstab(
        df["is_fraud"],
        df["time_since_last_transaction"].isnull(),
        normalize="index"
    ) * 100

    history_table.index = ["Legitimate", "Fraudulent"]
    history_table.columns = ["History Available", "History Missing"]

    print(history_table.round(2).to_string())
    print("=" * 50)
# STEP 8: NUMERICAL FEATURE ANALYSIS
def analyze_numerical_features(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("STEP 8: NUMERICAL FEATURE ANALYSIS")
    print("=" * 60)

    numerical_features = [
        "amount",
        "spending_deviation_score",
        "velocity_score",
        "geo_anomaly_score",
        "time_since_last_transaction"
    ]

    for feature in numerical_features:
        print(f"\n[+] Analyzing: {feature}")

        summary = df.groupby("is_fraud")[feature].agg(
            ["count", "mean", "median", "std", "min", "max"]
        )

        summary.index = ["Legitimate", "Fraudulent"]

        print(summary.round(3).to_string())

    print("\n[✓] Numerical feature analysis complete.")
    print("=" * 60)
# STEP 9: TRANSACTION TIME ANOMALY ANALYSIS
def analyze_time_anomalies(df: pd.DataFrame) -> None:

    print("\n" + "=" * 60)
    print("STEP 9: TRANSACTION TIME ANOMALY ANALYSIS")
    print("=" * 60)

    feature = "time_since_last_transaction"

    print("\n[+] Overall time-value distribution:")

    categories = pd.cut(
        df[feature],
        bins=[-float("inf"), 0, 1e-9, float("inf")],
        labels=["Negative", "Zero", "Positive"]
    )

    print(categories.value_counts(dropna=False).to_string())

    print("\n[+] Time-value distribution by fraud class:")

    time_groups = pd.DataFrame({
        "is_fraud": df["is_fraud"],
        "time_category": categories
    })

    distribution = pd.crosstab(
        time_groups["is_fraud"],
        time_groups["time_category"],
        normalize="index"
    ) * 100

    distribution.index = ["Legitimate", "Fraudulent"]

    print(distribution.round(2).to_string())

    print("\n[+] Detailed negative-value statistics:")

    negative_values = df[df[feature] < 0]

    print("Negative transaction times:",
          f"{len(negative_values):,}")

    print("Negative values among legitimate:",
          f"{(negative_values['is_fraud'] == 0).sum():,}")

    print("Negative values among fraudulent:",
          f"{(negative_values['is_fraud'] == 1).sum():,}")

    print("\n[✓] Time anomaly analysis complete.")
    print("=" * 60)
# STEP 10: TIMESTAMP CONSISTENCY ANALYSIS
def analyze_timestamp_consistency(df: pd.DataFrame) -> None:

    print("\n" + "=" * 60)
    print("STEP 10: TIMESTAMP CONSISTENCY ANALYSIS")
    print("=" * 60)

    feature = "time_since_last_transaction"

    print("\n[+] Timestamp range:")
    print("Earliest:", df["timestamp"].min())
    print("Latest:  ", df["timestamp"].max())

    print("\n[+] Sample records:")

    sample = df[
        ["timestamp", "sender_account", feature, "is_fraud"]
    ].sample(15, random_state=42)

    print(sample.to_string(index=False))

    print("\n[+] Negative-value examples:")

    negative_sample = df[df[feature] < 0][
        ["timestamp", "sender_account", feature, "is_fraud"]
    ].sample(10, random_state=42)

    print(negative_sample.to_string(index=False))

    print("\n[+] Timestamp consistency analysis complete.")
    print("=" * 60)
# STEP 11: CATEGORICAL FEATURE ANALYSIS
def analyze_categorical_features(df: pd.DataFrame) -> None:

    print("\n" + "=" * 60)
    print("STEP 11: CATEGORICAL FEATURE ANALYSIS")
    print("=" * 60)

    categorical_features = [
        "merchant_category",
        "transaction_type",
        "device_used",
        "payment_channel",
        "location"
    ]

    for feature in categorical_features:

        print(f"\n[+] Analyzing: {feature}")

        analysis = (
            df.groupby(feature, observed=True)["is_fraud"]
            .agg(
                Transactions="count",
                Fraud_Count="sum",
                Fraud_Rate="mean"
            )
        )

        analysis["Fraud_Rate"] *= 100

        analysis = analysis.sort_values(
            "Fraud_Rate",
            ascending=False
        )

        print(analysis.head(15).round(3).to_string())

    print("\n[✓] Categorical feature analysis complete.")
    print("=" * 60)
# MAIN EXECUTION PIPELINE
if __name__ == "__main__":
    DIAGNOSTIC_ONLY = False
    if DIAGNOSTIC_ONLY:
        dataset_path = Path(__file__).resolve().parent / "financial_fraud_detection_dataset.csv"

        print("\n[+] Running diagnostic-only mode...")

        df = load_and_audit_dataset(dataset_path)
        
        analyze_missing_values(df)
        analyze_numerical_features(df)
        analyze_time_anomalies(df)
        analyze_timestamp_consistency(df)
        analyze_categorical_features(df)

    if not DIAGNOSTIC_ONLY:
        dataset_path = Path(__file__).resolve().parent / "financial_fraud_detection_dataset.csv"

        # Step 1: Data Ingestion & Audit
        df = load_and_audit_dataset(dataset_path)

        # Step 2: Time-Aware Partitioning
        train_df, val_df, test_df = split_time_aware(df)
        train_df, val_df, test_df = create_account_history_features(
            train_df, val_df, test_df
        )
        # Step 3: Feature Engineering & Preprocessing
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = engineer_and_preprocess_features(
            train_df, val_df, test_df
        )

        # Step 4: Model Training
        models, val_probas = train_leakage_safe_models(X_train, y_train, X_val, y_val)

        # Step 5: Threshold Optimization
        optimal_thresholds = optimize_decision_thresholds(models, val_probas, y_val)

        # Step 6: Final Test Set Evaluation
        final_results = evaluate_on_locked_test_set(
            models, optimal_thresholds, X_test, y_test)
