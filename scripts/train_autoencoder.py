"""Training script for Transaction Autoencoder Anomaly Detector.

Trains on legitimate transactions only, computes reconstruction error,
calibrates threshold on legitimate validation transactions, and evaluates
on validation and untouched test sets.
"""

import hashlib
import json
import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import torch

# Ensure repo root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder import TransactionAutoencoder, train_autoencoder
from src.config import CLEAN_PATH, ARTIFACTS, SEED, TARGET_FALSE_POSITIVE_RATE, TARGET_FALSE_POSITIVE_RATES, ALERT_CAPACITIES


def main():
    print("=" * 60)
    print("Step 2: Training Autoencoder Anomaly Detector")
    print("=" * 60)

    # 1. Load Clean Dataset
    print(f"Loading cleaned dataset from: {CLEAN_PATH}")
    clean_bytes = CLEAN_PATH.read_bytes()
    clean_sha256 = hashlib.sha256(clean_bytes).hexdigest()
    print(f"Dataset SHA-256: {clean_sha256}")

    data = pd.read_csv(CLEAN_PATH).sort_values("Time").reset_index(drop=True)
    print(f"Loaded dataset shape: {data.shape}")

    # 2. Chronological Split (60/20/20)
    train_end = int(len(data) * 0.60)
    validation_end = int(len(data) * 0.80)
    train = data.iloc[:train_end].copy()
    val = data.iloc[train_end:validation_end].copy()
    test = data.iloc[validation_end:].copy()

    feature_names = [col for col in data.columns if col not in ("Time", "Class")]
    print(f"Features ({len(feature_names)}): {feature_names}")

    legitimate_train = train.loc[train["Class"] == 0, feature_names]
    legitimate_val = val.loc[val["Class"] == 0, feature_names]

    print(f"Train rows: {len(train):,} (Legitimate: {len(legitimate_train):,}, Fraud: {train['Class'].sum():,})")
    print(f"Validation rows: {len(val):,} (Legitimate: {len(legitimate_val):,}, Fraud: {val['Class'].sum():,})")
    print(f"Test rows: {len(test):,} (Legitimate: {(test['Class']==0).sum():,}, Fraud: {test['Class'].sum():,})")

    # 3. Fit Scaler ONLY on Legitimate Training Transactions
    print("\nFitting StandardScaler strictly on legitimate training transactions...")
    scaler = StandardScaler().fit(legitimate_train)

    X_train_scaled = scaler.transform(legitimate_train)
    X_val_legit_scaled = scaler.transform(legitimate_val)

    # 4. Train Autoencoder
    print("\nTraining Deep Autoencoder on legitimate transaction manifold...")
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    hidden_dims = (20, 12)
    latent_dim = 6
    epochs = 40
    batch_size = 256

    model, history = train_autoencoder(
        train_features=X_train_scaled,
        val_features=X_val_legit_scaled,
        input_dim=len(feature_names),
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=1e-3,
        seed=SEED,
        device=device,
    )
    print(f"Training completed. Final epochs trained: {len(history['train_loss'])}")
    print(f"Initial train loss: {history['train_loss'][0]:.5f} -> Final train loss: {history['train_loss'][-1]:.5f}")
    print(f"Initial val loss:   {history['val_loss'][0]:.5f} -> Final val loss:   {history['val_loss'][-1]:.5f}")

    # 5. Compute Reconstruction Errors
    model.eval()
    def get_errors(df: pd.DataFrame) -> np.ndarray:
        scaled = scaler.transform(df[feature_names])
        tensor = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad():
            errs = model.compute_reconstruction_error(tensor).numpy().astype(float)
        return errs

    val_errors = get_errors(val)
    val_legit_errors = val_errors[val["Class"].to_numpy() == 0]
    test_errors = get_errors(test)

    # 6. Calibrate Thresholds on Legitimate Validation Transactions
    print("\nCalibrating anomaly thresholds on legitimate validation transactions...")
    thresholds = {}
    for target_fpr in TARGET_FALSE_POSITIVE_RATES:
        th = float(np.quantile(val_legit_errors, 1.0 - target_fpr))
        thresholds[str(target_fpr)] = th
        print(f"  Target FPR {target_fpr * 100:.1f}% -> Threshold MSE = {th:.5f}")

    primary_threshold = thresholds[str(TARGET_FALSE_POSITIVE_RATE)]

    # 7. Validation Evaluation
    val_labels = val["Class"].to_numpy()
    val_ap = float(average_precision_score(val_labels, val_errors))
    val_auc = float(roc_auc_score(val_labels, val_errors))
    val_preds = (val_errors >= primary_threshold).astype(int)
    val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(val_labels, val_preds, average="binary", zero_division=0)
    val_fpr = float((val_preds[val_labels == 0] == 1).mean())

    print(f"\n--- Validation Performance (1% target FPR) ---")
    print(f"  Average Precision (PR-AUC): {val_ap:.4f}")
    print(f"  ROC-AUC:                   {val_auc:.4f}")
    print(f"  Recall:                    {val_rec * 100:.2f}% ({val_preds[val_labels == 1].sum()}/{val_labels.sum()} frauds detected)")
    print(f"  Precision:                 {val_prec * 100:.2f}%")
    print(f"  Observed FPR:              {val_fpr * 100:.2f}%")

    # 8. Test Evaluation (Untouched Holdout)
    test_labels = test["Class"].to_numpy()
    test_ap = float(average_precision_score(test_labels, test_errors))
    test_auc = float(roc_auc_score(test_labels, test_errors))
    test_preds = (test_errors >= primary_threshold).astype(int)
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(test_labels, test_preds, average="binary", zero_division=0)
    test_fpr = float((test_preds[test_labels == 0] == 1).mean())
    test_detected_frauds = int(test_preds[test_labels == 1].sum())
    test_false_alarms = int(test_preds[test_labels == 0].sum())

    print(f"\n--- Test Performance (Untouched 20% Chronological Holdout) ---")
    print(f"  Average Precision (PR-AUC): {test_ap:.4f}")
    print(f"  ROC-AUC:                   {test_auc:.4f}")
    print(f"  Recall:                    {test_rec * 100:.2f}% ({test_detected_frauds}/{test_labels.sum()} frauds detected)")
    print(f"  Precision:                 {test_prec * 100:.2f}%")
    print(f"  F1-Score:                  {test_f1:.4f}")
    print(f"  Observed Test FPR:         {test_fpr * 100:.2f}%")
    print(f"  False Alarms:              {test_false_alarms} legitimate transactions flagged")

    # Precision & Recall at fixed capacities
    capacity_results = {}
    test_sorted_indices = np.argsort(-test_errors)
    total_test_frauds = int(test_labels.sum())
    print("\n--- Fixed Review Capacity Analysis ---")
    for k in ALERT_CAPACITIES:
        top_k_indices = test_sorted_indices[:k]
        frauds_in_top_k = int(test_labels[top_k_indices].sum())
        prec_at_k = frauds_in_top_k / k
        rec_at_k = frauds_in_top_k / total_test_frauds
        capacity_results[f"top_{k}"] = {
            "capacity": k,
            "frauds_detected": frauds_in_top_k,
            "precision_at_k": float(prec_at_k),
            "recall_at_k": float(rec_at_k),
        }
        print(f"  Top {k:4d} alerts: {frauds_in_top_k:2d}/{total_test_frauds} frauds | Precision@{k}: {prec_at_k*100:.1f}% | Recall@{k}: {rec_at_k*100:.1f}%")

    # 9. Save Artifact Bundle & Report
    bundle = {
        "model_version": "1.0",
        "model_type": "Autoencoder",
        "feature_names": feature_names,
        "scaler": scaler,
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": len(feature_names),
            "hidden_dims": hidden_dims,
            "latent_dim": latent_dim,
        },
        "threshold": primary_threshold,
        "thresholds_by_target_fpr": thresholds,
        "target_false_positive_rate": TARGET_FALSE_POSITIVE_RATE,
        "cleaned_dataset_sha256": clean_sha256,
        "training_legitimate_rows": len(legitimate_train),
        "random_seed": SEED,
        "history": history,
    }

    bundle_path = ARTIFACTS / "autoencoder_bundle.joblib"
    joblib.dump(bundle, bundle_path)
    print(f"\nSaved model bundle to: {bundle_path}")

    report = {
        "report_version": "1.0",
        "model_type": "Deep Autoencoder Anomaly Detector",
        "dataset_sha256": clean_sha256,
        "architecture": {
            "input_dim": len(feature_names),
            "encoder": [len(feature_names), *hidden_dims, latent_dim],
            "decoder": [latent_dim, *reversed(hidden_dims), len(feature_names)],
            "loss_function": "Mean Squared Error (MSE)",
        },
        "training": {
            "legitimate_rows": len(legitimate_train),
            "epochs_run": len(history["train_loss"]),
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
        },
        "threshold_calibration": {
            "target_fpr": TARGET_FALSE_POSITIVE_RATE,
            "calibrated_threshold": primary_threshold,
            "thresholds": thresholds,
        },
        "validation_metrics": {
            "average_precision": val_ap,
            "roc_auc": val_auc,
            "precision": float(val_prec),
            "recall": float(val_rec),
            "f1": float(val_f1),
            "fpr": val_fpr,
        },
        "test_metrics": {
            "average_precision": test_ap,
            "roc_auc": test_auc,
            "precision": float(test_prec),
            "recall": float(test_rec),
            "f1": float(test_f1),
            "fpr": test_fpr,
            "frauds_detected": test_detected_frauds,
            "total_frauds": total_test_frauds,
            "false_alarms": test_false_alarms,
        },
        "capacity_metrics": capacity_results,
    }

    report_path = ARTIFACTS / "autoencoder_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved audit report to: {report_path}")
    print("=" * 60)
    print("Step 2 completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

