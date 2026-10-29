#imports
import pandas as pd
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
    
    partitions = [('Train', train_df), ('Validation', val_df), ('Test', test_df)]
    
    print("[+] Extracting temporal features and dropping leaking column ('fraud_type')...")
    for name, partition in partitions:
        partition['hour_of_day'] = partition['timestamp'].dt.hour.astype('int8')
        partition['day_of_week'] = partition['timestamp'].dt.dayofweek.astype('int8')
        partition['month']       = partition['timestamp'].dt.month.astype('int8')
        partition['is_weekend']  = partition['day_of_week'].isin([5, 6]).astype('int8')
        
        if 'fraud_type' in partition.columns:
            partition.drop(columns=['fraud_type'], inplace=True)
            
    num_features = [
        'amount', 'spending_deviation_score', 'velocity_score',
        'geo_anomaly_score', 'time_since_last_transaction',
        'hour_of_day', 'day_of_week', 'month', 'is_weekend'
    ]
    cat_features = ['transaction_type', 'merchant_category', 'device_used', 'location']
    
    X_train, y_train = train_df[num_features + cat_features], train_df[target_col]
    X_val,   y_val   = val_df[num_features + cat_features],   val_df[target_col]
    X_test,  y_test  = test_df[num_features + cat_features],  test_df[target_col]
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ])
    
    print("[+] Fitting preprocessor STRICTLY on Training set...")
    X_train_proc = preprocessor.fit_transform(X_train)
    
    print("[+] Transforming Validation and Test sets using fitted training parameters...")
    X_val_proc  = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)
    
    print(f"\n[✓] Preprocessing Complete:")
    print(f"  • Processed Train Shape      : {X_train_proc.shape[0]:,} rows x {X_train_proc.shape[1]} features")
    print(f"  • Processed Validation Shape : {X_val_proc.shape[0]:,} rows x {X_val_proc.shape[1]} features")
    print(f"  • Processed Test Shape       : {X_test_proc.shape[0]:,} rows x {X_test_proc.shape[1]} features")
    print("==================================================\n")
    
    return (X_train_proc, y_train), (X_val_proc, y_val), (X_test_proc, y_test)

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
    """
    Step 6: Passes the untouched test set through the trained models once, 
    applies the validation-selected decision thresholds, and reports rare-event metrics.
    """
    print("==================================================")
    print("STEP 6: FINAL EVALUATION ON UNTOUCHED TEST SET")
    print("==================================================")
    
    test_results = []
    
    for name, model in trained_models.items():
        print(f"\n[+] Evaluating {name} on Locked Test Set...")
        
        # 1. Predict fraud probability scores on test set
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        # 2. Apply validation-selected decision threshold
        thresh = optimal_thresholds[name]
        y_test_pred = (y_test_proba >= thresh).astype(int)
        
        # 3. Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        
        # 4. Calculate rare-event metrics
        pr_auc = average_precision_score(y_test, y_test_proba)
        roc_auc = roc_auc_score(y_test, y_test_proba)
        precision = precision_score(y_test, y_test_pred, zero_division=0)
        recall = recall_score(y_test, y_test_pred, zero_division=0)
        f1 = f1_score(y_test, y_test_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_test_pred)
        
        # 5. Calculate operational cost: Cost = c_FN * FN + c_FP * FP
        total_cost = (c_fn * fn) + (c_fp * fp)
        
        print(f"  • Applied Threshold : {thresh:.2f}")
        print(f"  • PR-AUC (AUPRC)    : {pr_auc:.4f}")
        print(f"  • ROC-AUC           : {roc_auc:.4f}")
        print(f"  • Precision         : {precision:.4f}")
        print(f"  • Recall            : {recall:.4f}")
        print(f"  • Fraud F1-Score    : {f1:.4f}")
        print(f"  • MCC               : {mcc:.4f}")
        print(f"  • Confusion Matrix  : TP={tp:,} | FP={fp:,} | FN={fn:,} | TN={tn:,}")
        print(f"  • Est. Error Cost   : ${total_cost:,.2f} (c_FN=${c_fn:.0f}, c_FP=${c_fp:.0f})")
        
        test_results.append({
            'Model': name,
            'Threshold': thresh,
            'PR-AUC': pr_auc,
            'ROC-AUC': roc_auc,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'MCC': mcc,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
            'Total Cost ($)': total_cost
        })
        
    results_df = pd.DataFrame(test_results)
    
    print("\n==================================================")
    print("FINAL TEST SET PERFORMANCE COMPARISON SUMMARY")
    print("==================================================")
    print(results_df[['Model', 'Threshold', 'PR-AUC', 'Precision', 'Recall', 'F1-Score', 'MCC', 'Total Cost ($)']].to_string(index=False))
    print("==================================================\n")
    
    return results_df

# MAIN EXECUTION PIPELINE
if __name__ == "__main__":
    dataset_path = r"C:\Users\messa\Downloads\DATASET\financial_fraud_detection_dataset.csv"
    
    # Step 1: Data Ingestion & Audit
    df = load_and_audit_dataset(dataset_path)
    
    # Step 2: Time-Aware Partitioning
    train_df, val_df, test_df = split_time_aware(df)
    
    # Step 3: Feature Engineering & Preprocessing
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = engineer_and_preprocess_features(
        train_df, val_df, test_df
    )
    
    # Step 4: Model Training
    models, val_probas = train_leakage_safe_models(X_train, y_train, X_val, y_val)
    
    # Step 5: Threshold Optimization
    optimal_thresholds = optimize_decision_thresholds(models, val_probas, y_val)
    
    # Step 6: Final Test Set Evaluation
    final_results = evaluate_on_locked_test_set(models, optimal_thresholds, X_test, y_test)
