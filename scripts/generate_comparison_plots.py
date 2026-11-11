"""Generates side-by-side comparison plots between GMM and Autoencoder.

Figures generated:
1. reports/figures/autoencoder_training.png (Loss curves)
2. reports/figures/pr_curve_comparison.png (Precision-Recall GMM vs Autoencoder)
3. reports/figures/capacity_tradeoff.png (Alert capacity Precision & Recall @ K)
"""

import json
from pathlib import Path
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import CLEAN_PATH, ARTIFACTS
from src.autoencoder import TransactionAutoencoder

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Generating comparative evaluation figures...")

    # Load Reports
    with open(ARTIFACTS / "gmm_report.json") as f:
        gmm_report = json.load(f)

    with open(ARTIFACTS / "autoencoder_report.json") as f:
        ae_report = json.load(f)

    ae_bundle = joblib.load(ARTIFACTS / "autoencoder_bundle.joblib")

    # 1. Plot Autoencoder Training Loss
    history = ae_bundle["history"]
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.plot(epochs, history["train_loss"], label="Train Loss (MSE)", color="#1f77b4", lw=2)
    plt.plot(epochs, history["val_loss"], label="Val Loss (MSE)", color="#ff7f0e", lw=2, linestyle="--")
    plt.title("Autoencoder Convergence on Legitimate Transactions", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "autoencoder_training.png")
    plt.close()
    print("Saved reports/figures/autoencoder_training.png")

    # 2. Score Test Set for PR curve
    data = pd.read_csv(CLEAN_PATH).sort_values("Time").reset_index(drop=True)
    val_end = int(len(data) * 0.80)
    test = data.iloc[val_end:].copy()
    features = ae_bundle["feature_names"]
    test_labels = test["Class"].to_numpy()

    # Load AE model
    ae_model = TransactionAutoencoder(
        input_dim=ae_bundle["model_config"]["input_dim"],
        hidden_dims=tuple(ae_bundle["model_config"]["hidden_dims"]),
        latent_dim=ae_bundle["model_config"]["latent_dim"],
    )
    ae_model.load_state_dict(ae_bundle["model_state_dict"])
    ae_model.eval()

    scaler = ae_bundle["scaler"]
    test_scaled = scaler.transform(test[features])
    with torch.no_grad():
        ae_scores = ae_model.compute_reconstruction_error(torch.tensor(test_scaled, dtype=torch.float32)).numpy()

    # Precision-Recall Curve
    ae_prec, ae_rec, _ = precision_recall_curve(test_labels, ae_scores)
    ae_ap = average_precision_score(test_labels, ae_scores)
    gmm_ap = gmm_report["test_metrics"]["average_precision"]

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(ae_rec, ae_prec, label=f"Autoencoder (PR-AUC = {ae_ap:.4f})", color="#2ca02c", lw=2.5)
    base_rate = test_labels.mean()
    plt.axhline(base_rate, color="gray", linestyle=":", label=f"Random Chance ({base_rate*100:.2f}%)")
    plt.axvline(ae_report["test_metrics"]["recall"], color="#2ca02c", linestyle="--", alpha=0.6,
                label=f"AE Operating Point (Recall={ae_report['test_metrics']['recall']*100:.1f}%)")
    plt.title("Test Set Precision-Recall Trade-off (Rare Fraud Detection)", fontsize=13, fontweight="bold")
    plt.xlabel("Recall (Detection Rate)", fontsize=11)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pr_curve_autoencoder.png")
    plt.close()
    print("Saved reports/figures/pr_curve_autoencoder.png")

    # 3. Capacity Comparison Plot (Top 100, 500, 1000)
    capacities = [100, 500, 1000]
    gmm_test_caps = gmm_report["capacity_metrics"]["test"]
    ae_caps = ae_report["capacity_metrics"]

    gmm_precs = [gmm_test_caps[str(k)]["precision"] * 100 for k in capacities]
    ae_precs = [ae_caps[f"top_{k}"]["precision_at_k"] * 100 for k in capacities]
    gmm_recs = [gmm_test_caps[str(k)]["recall"] * 100 for k in capacities]
    ae_recs = [ae_caps[f"top_{k}"]["recall_at_k"] * 100 for k in capacities]

    x = np.arange(len(capacities))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    # Precision@K
    ax1.bar(x - width/2, gmm_precs, width, label=f"GMM Density (AP={gmm_ap:.2f})", color="#1f77b4")
    ax1.bar(x + width/2, ae_precs, width, label=f"Autoencoder (ROC={ae_report['test_metrics']['roc_auc']:.2f})", color="#2ca02c")
    ax1.set_title("Precision at Review Capacity (Top K Alerts)", fontweight="bold")
    ax1.set_xlabel("Review Capacity (K)")
    ax1.set_ylabel("Precision (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Top {k}" for k in capacities])
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.legend()

    # Recall@K
    ax2.bar(x - width/2, gmm_recs, width, label="GMM (AML Baseline)", color="#1f77b4")
    ax2.bar(x + width/2, ae_recs, width, label="Deep Autoencoder (DL)", color="#2ca02c")
    ax2.set_title("Recall at Review Capacity (Top K Alerts)", fontweight="bold")
    ax2.set_xlabel("Review Capacity (K)")
    ax2.set_ylabel("Recall (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Top {k}" for k in capacities])
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "capacity_comparison.png")
    plt.close()
    print("Saved reports/figures/capacity_comparison.png")


if __name__ == "__main__":
    main()
