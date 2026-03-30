# Fraud Detection 

A three-stage fraud detection analysis built on a highly imbalanced real-world transaction dataset (357 fraud cases out of 3.1 million transactions — a 1:8,806 ratio).

---

## Project Structure

```
Nasdaq/
├── train_fraud.parquet                   # Training data (labelled)
├── test_fraud_external.parquet           # Test data (no labels)
├── test_fraud_external_predicted.parquet # Output: test data + is_fraud_predicted
├── columns.txt                           # Column definitions
├── fraud_analysis2.py                    # Main analysis script (Colab)
└── plots/                                # All generated visualisations
    ├── 01_class_distribution.png
    ├── 02_transaction_type.png
    ├── 03_amount_distribution.png
    ├── 04_time_analysis.png
    ├── 05_balance_analysis.png
    ├── 06_flag_vs_actual.png
    
```

---

## How to Run

This project has the option to run on Python notebook.



**Dependencies**
```
pip install lightgbm xgboost scikit-learn pandas numpy matplotlib seaborn pyarrow
```

---

## Analysis Overview

### Stage 1 — Exploratory Data Analysis (EDA)

Understanding the data before building any model.

| Plot | What it shows |
|---|---|
| 01 — Class Distribution | 1:8,806 fraud imbalance — accuracy is a useless metric |
| 02 — Transaction Type | Fraud only exists in CASH_OUT and TRANSFER |
| 03 — Amount Distribution | Fraud amounts skew significantly higher (mean $1.5M vs $217K legit) |
| 04 — Time Analysis | 36% of fraud occurs 10pm–6am; all fraud in the last 108 hours |
| 05 — Balance Analysis | 97.8% of fraud drains the account completely vs 0.001% of legit |
| 06 — Existing Flag System | Current rule-based system catches only 2 of 357 frauds (0.6% recall) |

**Key finding:** The existing `is_flagged_fraud` system detects a narrow accounting anomaly
(`balance_before == balance_after` for TRANSFER type) — not actual fraud behaviour.
It has perfect precision (1.0) but catastrophically low recall (0.006).

---

### Stage 2 — Feature Engineering

28 features engineered from raw columns to capture fraud signals the model cannot derive on its own.

**Top signal features:**

| Feature | What it captures |
|---|---|
| `drains_account` | Transaction amount ≈ full account balance (complete drain) |
| `amount_to_init_ratio` | Transaction as proportion of initiator's balance |
| `init_became_zero` | Initiator balance reaches exactly $0 after transaction |
| `recip_zero_before` | Recipient account was empty before receiving (shell account signal) |
| `log_amount` | Log-transformed amount — prevents large values dominating |
| `is_night` | Hour 22:00–06:00 — disproportionate fraud rate overnight |
| `c2m` | Customer-to-Merchant transfers — structural fraud pattern |

---

### Stage 2 — From Classification to Cost-Sensitive Learning

**The problem with treating all fraud equally:**
Missing a $119 fraud and missing a $10M fraud were penalized identically in the original model.
This is misaligned with real business risk.

**Two weighting versions were implemented:**

| Version | Formula | Philosophy |
|---|---|---|
| V1 — No Floor | `base_w × (amount / mean_amount)` | Penalises large frauds more; small frauds get less attention |
| V2 — With Floor | `base_w × (1 + amount / mean_amount)` | Every fraud always weighted ≥ 8,806× more than any legit transaction |

Where `base_w = legit_count / fraud_count ≈ 8,806` (the class imbalance ratio).

V2 is recommended: the `+1` floor guarantees no fraud case is ever considered less important
than a legitimate transaction, regardless of its dollar amount.

**Business cost threshold optimization:**

Rather than maximizing F1, we find the threshold that minimizes total business cost:

```
Total Cost = Missed Fraud Value ($) + False Alarms × Cost per Alert ($)
```

Three cost-per-alert scenarios tested: $50 / $200 / $500 per investigation.

---

### Stage 3 — Model Comparison

All three models were trained with Version 2 (with floor) corrected weights and 5-fold Stratified CV.

| Metric | LightGBM ⭐ | XGBoost | Random Forest |
|---|---|---|---|
| **Recall** | **100%** | **100%** | 100%* |
| Precision | 99.4% | 99.4% | 100%* |
| Missed Fraud (FN) | **0** | **0** | 0* |
| False Alarms (FP) | 2 | 2 | 0* |
| Average Precision | 1.0000 | 1.0000 | 1.0000* |
| Training Speed | Fastest | Medium | Slowest |

*Random Forest results indicate overfitting — see note below.

**Why LightGBM is recommended:**
- Fastest training at 3.1M rows — critical for daily retraining in production
- Regularisation parameters (`lambda_l1`, `lambda_l2`, `min_child_samples`) prevent memorisation
- Sequential boosting adjusts gradient magnitudes — more robust to extreme class weights than RF
- Results validated independently by XGBoost agreement

**Random Forest overfitting warning:**
When passed extreme sample weights (`base_w ≈ 8,806`), Random Forest memorises the 357 fraud
cases rather than generalising. Bootstrap sampling + extreme weights causes individual trees to
isolate fraud cases in single-sample leaf nodes. Use `class_weight='balanced'` instead of
`sample_weight` for RF, or constrain with `min_samples_leaf=20`.

---

## Key Results

| | Value |
|---|---|
| Training set | 3,143,740 legit + 357 fraud |
| Fraud rate | 0.0114% (1 in 8,806) |
| OOF ROC-AUC | 1.0000 |
| OOF Average Precision | 1.0000 |
| Recall (optimal threshold) | 100% |
| Precision (optimal threshold) | 99.4% |
| False alarms / month | 2 |
| Missed frauds | 0 |
| Test predictions | 861 fraud / 2,247,484 total (0.038%) |

---

## Important Caveats

1. **Near-perfect performance is a warning sign.** The dominant feature `drains_account` is
   near-deterministic (97.8% of fraud drains fully vs 0.001% of legit). Performance is
   feature-driven, not algorithm-driven.

2. **Single-month data risk.** All 357 frauds share the same full-drain behaviour. If future
   fraudsters use partial drains (80% instead of 100%), the model may miss them entirely.

3. **Time-based validation needed.** Cross-validation within the same month does not test
   generalisation across time periods. A true hold-out should be a separate month of data.

4. **`day_of_month` pattern is a dataset artifact.** All fraud falls in the last 5 days of the
   month — this should not be relied upon in production without confirming it holds across
   multiple months.

---

## Methodology Notes

- **StratifiedKFold (5-fold):** Preserves the 0.011% fraud rate in each fold — prevents folds
  with zero fraud cases, which would cause AUC to be undefined
- **OOF predictions:** Each prediction made on data the model never saw during training —
  honest evaluation without data leakage
- **No early stopping:** With only ~71 fraud cases per validation fold, early stopping triggers
  on noise. Fixed `n_estimators=300` with L1/L2 regularisation used instead
- **`balance_after` features allowed:** Per the problem brief (`columns.txt`), these can be
  treated as "what the balance would be if the transaction completed"
