# Presentation Guide & Slide Outline
## Project 26: Card Fraud Triage via Gaussian Mixture Density Scoring and Autoencoder Anomaly Detection

**Target Presentation Length:** 5–7 Minutes  
**Audience:** Deep Learning & Machine Learning Reviewers / Evaluators

---

### Slide 1: Title & Problem Context
* **Slide Title:** Card Fraud Triage: The 'Hybrid' Logic from Density Scoring to Learned Nonlinear Reconstruction
* **Key Talking Points:**
  1. **The Fraud Dilemma:** In financial fraud detection, fraud is exceedingly rare (here, **0.1667%** — only 1 fraud per 600 transactions).
  2. **Why Supervised Learning Struggles:** Extreme class imbalance causes severe overfitting, vulnerability to zero-day fraud patterns, and reliance on labels that suffer from 30–90 day chargeback delays.
  3. **The Solution:** Unsupervised Anomaly Detection — train models *strictly on legitimate customer behavior*, score deviations, and prioritize high-risk transactions for human review queues.

---

### Slide 2: The 'Hybrid' Logic & Identifying the Gap
* **Slide Title:** Moving from Gaussian Density Estimation to Nonlinear Manifold Reconstruction
* **Visuals:** Diagram showing GMM Gaussian Ellipsoid vs Autoencoder Manifold Bottleneck.
* **Key Talking Points:**
  1. **AML Baseline (GMM):** Fits a mixture of Gaussians $p(x) = \sum \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$ using the EM algorithm. Anomaly Score $= -\log p(x)$.
  2. **The Gap in GMM:**
     * **Rigid Parametric Assumption:** Assumes normal behavior forms convex hyper-ellipsoids.
     * **Curse of Dimensionality:** Full covariance estimation in $D=29$ requires 435 parameters per component, risking ill-conditioned or singular matrices.
     * **Nonlinear Geometry:** Complex transaction relationships are non-Gaussian and cannot be tightly bounded by ellipsoids.
  3. **The Deep Learning Proposal (Autoencoder):**
     * Discards parametric density assumptions.
     * Maps inputs through an undercomplete bottleneck ($29 \to 20 \to 12 \to 6 \to 12 \to 20 \to 29$) to learn the intrinsic lower-dimensional nonlinear manifold.
     * **Anomaly Score:** Mean Squared Error (MSE) reconstruction loss $\|x - \hat{x}\|_2^2$.

---

### Slide 3: Rigorous Experimental Design (No Data Leakage)
* **Slide Title:** Methodology: Fair Benchmark & Strict Temporal Hygiene
* **Visuals:** Timeline diagram showing the 60/20/20 chronological split.
* **Key Talking Points:**
  1. **Chronological Splitting:**
     * Avoided random k-fold splitting (which causes future-to-past time leakage).
     * Chronologically partitioned into Train (60%), Validation (20%), and Untouched Test (20%).
  2. **Legitimate-Only Training:**
     * Scaler and Autoencoder/GMM were trained **exclusively on 169,893 legitimate transactions**.
     * Fraud labels and test rows were completely hidden during training.
  3. **Validation-Only Threshold Calibration:**
     * Alert threshold set to the 99th percentile of legitimate validation reconstruction errors (targeting a **1.0% False Positive Rate budget**).

---

### Slide 4: Deep Learning Architecture & Training
* **Slide Title:** Autoencoder Architecture & Optimization
* **Visuals:** [reports/figures/autoencoder_training.png](file:///Users/swagatob/Desktop/card-fraud-triage/reports/figures/autoencoder_training.png)
* **Key Talking Points:**
  1. **Architecture:** Symmetrical undercomplete architecture with LeakyReLU activations, BatchNorm1d, and 5% dropout for regularization.
  2. **Training Dynamics:**
     * Accelerated on Apple Silicon GPU (`mps`).
     * Trained with AdamW and `ReduceLROnPlateau` scheduler.
     * Loss smoothly converged from 0.796 down to 0.513 train MSE (0.678 val MSE).
  3. **Inference:** Fast, vectorized matrix operations producing per-sample reconstruction errors in milliseconds.

---

### Slide 5: Head-to-Head Results & Capacity Analysis
* **Slide Title:** Empirical Benchmark: GMM vs Deep Autoencoder
* **Visuals:** [reports/figures/capacity_comparison.png](file:///Users/swagatob/Desktop/card-fraud-triage/reports/figures/capacity_comparison.png) and [reports/figures/pr_curve_autoencoder.png](file:///Users/swagatob/Desktop/card-fraud-triage/reports/figures/pr_curve_autoencoder.png)
* **Comparison Table:**
  | Metric | GMM Baseline | Deep Autoencoder | Takeaway |
  |---|:---:|:---:|---|
  | **Test ROC-AUC** | 0.9353 | **0.9480** | **Autoencoder +0.0127**: Superior global separation |
  | **Test Recall @ 1% FPR** | **78.38%** (58/74) | 70.27% (52/74) | Both capture $>70\%$ unseen fraud |
  | **False Alarms** | 329 alerts | **302 alerts** | **Autoencoder saves 27 manual reviews** |
  | **Top 1,000 Capacity** | 60 frauds (81.1%) | 58 frauds (78.4%) | Both capture ~80% of frauds in top 1.76% queue |
* **Key Talking Points:**
  * Autoencoder achieves a higher ROC-AUC and reduces the false alarm review burden by 8.2%.
  * Both models successfully surface 4 out of every 5 frauds inside a manageable daily review capacity of 1,000 alerts.

---

### Slide 6: Conclusions & Operational Recommendations
* **Slide Title:** Synthesis & Production Fraud Triage Takeaways
* **Key Talking Points:**
  1. **Complementary Strengths:**
     * Autoencoders learn smooth nonlinear manifolds, providing cleaner normal boundaries and fewer false positives.
     * GMM identifies extreme density outliers in multimodal clusters.
  2. **Production Deployment Recommendation:**
     * An ensemble scoring mechanism: flag for immediate block if both models agree, or route to human review if either model exceeds its 1% FPR threshold.
  3. **Deliverables Completed:** Fully audited pipeline, trained PyTorch model bundle, test-verified scoring API, and interactive evaluation notebook.

---

### Quick Q&A Cheat Sheet (Anticipated Reviewer Questions)

* **Q: Why not use accuracy as a metric?**
  * *Answer:* With 0.1667% fraud prevalence, a naive model that predicts "legitimate" for every transaction achieves 99.83% accuracy while catching 0% of frauds. We evaluate on **ROC-AUC, Precision, Recall, and Precision@Capacity**.

* **Q: Why exclude Time from the model features?**
  * *Answer:* `Time` is elapsed seconds from the start of the recording. Using it as a feature would cause the model to memorize temporal timestamps rather than behavioral transaction characteristics. It is used strictly for chronological train/val/test splitting.

* **Q: Why is PR-AUC lower for Autoencoder than GMM even though ROC-AUC is higher?**
  * *Answer:* PR-AUC gives heavy weight to the precision of the very highest scores. GMM's density calculation penalizes extreme coordinate outliers exponentially, putting a cluster of frauds into ultra-high anomaly scores (yielding higher precision at tiny recall). Meanwhile, the Autoencoder's reconstruction MSE distributes errors more smoothly across all fraud dimensions, resulting in better global ranking (higher ROC-AUC) and fewer false alarms overall.

