"""
Fraud Detection — Amount-Weighted Model (Version 1)
=====================================================
Weight formula:
    weight(legit)  = 1.0
    weight(fraud)  = base_w × (amount / mean_fraud_amount)

This is the ORIGINAL version before the floor correction.
The idea is to scale each fraud's importance by its dollar value relative
to the average fraud amount:
  - A fraud worth 2× the average gets 2× the class-imbalance weight
  - A fraud worth 0.5× the average gets 0.5× the class-imbalance weight

Known limitation of this formula:
  - Small-value fraud (< mean_amount) gets weight < base_w
  - Tiny frauds (e.g. $119) can get weight < 1.0 — LESS than a legit transaction
  - See fraud_model_weighted.py (v2) for the corrected formula with a floor

NOTE: This file is kept for reference / comparison against v2.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             classification_report, confusion_matrix)
import lightgbm as lgb

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path("/Users/mengyao/Documents/_MyDrive/Interviews/Nasdaq")
TRAIN    = BASE / "train_fraud.parquet"
TEST     = BASE / "test_fraud_external.parquet"
OUT      = BASE / "test_fraud_external_predicted_v1.parquet"
PLOT_DIR = BASE / "plots"
PLOT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({'figure.dpi': 130, 'font.size': 10})
FRAUD_COLOR = '#e74c3c'
OPS_COLOR   = '#2980b9'
TOTAL_COLOR = '#2c3e50'
OPT_COLOR   = '#f39c12'

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("1. LOADING DATA")
print("="*60)

train = pd.read_parquet(TRAIN)
test  = pd.read_parquet(TEST)
print(f"Train: {train.shape}  |  Test: {test.shape}")
print(f"Fraud rate: {train['is_fraud'].mean()*100:.4f}%  ({train['is_fraud'].sum()} cases)")

TYPE_MAP = {t: i for i, t in enumerate(sorted(
    pd.concat([train['transaction_type'], test['transaction_type']]).unique()
))}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("2. FEATURE ENGINEERING")
print("="*60)

def engineer_features(df):
    df = df.copy()

    # Log-transformed amounts and balances
    df['log_amount']            = np.log1p(df['transaction_amount'])
    df['log_init_bal_before']   = np.log1p(df['initiater_balance_before'].clip(lower=0))
    df['log_recip_bal_before']  = np.log1p(df['recipient_balance_before'].clip(lower=0))
    df['log_init_bal_after']    = np.log1p(df['initiater_balance_after'].clip(lower=0))
    df['log_recip_bal_after']   = np.log1p(df['recipient_balance_after'].clip(lower=0))

    # Balance changes
    df['init_balance_change']   = df['initiater_balance_before'] - df['initiater_balance_after']
    df['recip_balance_change']  = df['recipient_balance_after']  - df['recipient_balance_before']
    df['log_init_bal_change']   = np.log1p(df['init_balance_change'].clip(lower=0))
    df['log_recip_bal_change']  = np.log1p(df['recip_balance_change'].clip(lower=0))

    # Balance anomalies
    df['init_amount_mismatch']  = (np.abs(df['init_balance_change'] - df['transaction_amount']) > 1.0).astype(int)
    df['recip_amount_mismatch'] = (np.abs(df['recip_balance_change'] - df['transaction_amount']) > 1.0).astype(int)

    # Account drain signals
    df['drains_account']        = (
        (df['initiater_balance_before'] > 0) &
        (np.abs(df['transaction_amount'] - df['initiater_balance_before']) < 1.0)
    ).astype(int)
    df['init_became_zero']      = (
        (df['initiater_balance_before'] > 0) & (df['initiater_balance_after'] == 0)
    ).astype(int)
    df['recip_became_zero']     = (
        (df['recipient_balance_before'] > 0) & (df['recipient_balance_after'] == 0)
    ).astype(int)

    # Zero balance flags
    df['init_zero_before']      = (df['initiater_balance_before'] == 0).astype(int)
    df['recip_zero_before']     = (df['recipient_balance_before'] == 0).astype(int)

    # Amount ratios
    df['amount_to_init_ratio']  = df['transaction_amount'] / (df['initiater_balance_before'] + 1)
    df['amount_to_recip_ratio'] = df['transaction_amount'] / (df['recipient_balance_before'] + 1)

    # Entity type
    df['initiator_is_customer'] = (df['initiating_customer'].str[0] == 'C').astype(int)
    df['recipient_is_customer'] = (df['recipient_customer'].str[0]  == 'C').astype(int)
    df['c2c'] = ((df['initiator_is_customer']==1) & (df['recipient_is_customer']==1)).astype(int)
    df['c2m'] = ((df['initiator_is_customer']==1) & (df['recipient_is_customer']==0)).astype(int)

    # Time features
    df['hour_of_day']  = df['hours_elapsed'] % 24
    df['day_of_month'] = df['hours_elapsed'] // 24
    df['is_night']     = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] < 6)).astype(int)

    # Transaction type
    df['txn_type_enc'] = df['transaction_type'].map(TYPE_MAP).fillna(-1).astype(int)
    df['is_cash_out']  = (df['transaction_type'] == 'CASH_OUT').astype(int)
    df['is_transfer']  = (df['transaction_type'] == 'TRANSFER').astype(int)

    return df

FEATURES = [
    'log_amount', 'amount_to_init_ratio', 'amount_to_recip_ratio',
    'log_init_bal_before', 'log_recip_bal_before',
    'log_init_bal_after', 'log_recip_bal_after',
    'log_init_bal_change', 'log_recip_bal_change',
    'init_amount_mismatch', 'recip_amount_mismatch',
    'drains_account', 'init_became_zero', 'recip_became_zero',
    'init_zero_before', 'recip_zero_before',
    'initiator_is_customer', 'recipient_is_customer', 'c2c', 'c2m',
    'hour_of_day', 'day_of_month', 'is_night',
    'txn_type_enc', 'is_cash_out', 'is_transfer',
    'is_flagged_fraud', 'hours_elapsed',
]

train_fe = engineer_features(train)
test_fe  = engineer_features(test)
X        = train_fe[FEATURES].values
y        = train_fe['is_fraud'].values
amounts  = train['transaction_amount'].values

print(f"Features: {len(FEATURES)}  |  X_train: {X.shape}  |  X_test: {test_fe[FEATURES].shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ORIGINAL AMOUNT-WEIGHTED SAMPLE WEIGHTS  (no floor)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("3. SAMPLE WEIGHT CONSTRUCTION  (v1 — proportional, no floor)")
print("="*60)

legit_n           = int((y == 0).sum())
fraud_n           = int((y == 1).sum())
base_w            = legit_n / fraud_n          # class imbalance ratio ~8,806
fraud_amounts     = amounts[y == 1]
mean_fraud_amount = fraud_amounts.mean()

# ── ORIGINAL FORMULA ──────────────────────────────────────────────────────────
#
#   weight(legit)  = 1.0
#   weight(fraud)  = base_w × (amount / mean_fraud_amount)
#
# Rationale:
#   - base_w corrects for class imbalance (357 fraud vs 3.1M legit)
#   - (amount / mean) scales each fraud case by its relative dollar importance:
#       • A $3M fraud (2× mean ~$1.5M) gets weight = base_w × 2 = 17,612
#       • A $1.5M fraud (1× mean)      gets weight = base_w × 1 = 8,806
#       • A $750K fraud (0.5× mean)    gets weight = base_w × 0.5 = 4,403
#       • A $119 fraud (0.00008× mean) gets weight = base_w × 0.00008 ≈ 0.69  ← BUG
#
# The bug: small frauds get weight < 1, meaning the model treats them as
# LESS important than an ordinary legitimate transaction.
# See fraud_model_weighted.py (v2) for the fix.
# ─────────────────────────────────────────────────────────────────────────────
sample_weights          = np.ones(len(y))
sample_weights[y == 1]  = base_w * (fraud_amounts / mean_fraud_amount)

w_fraud = sample_weights[y == 1]
print(f"Legit weight             : 1.0  (all identical)")
print(f"Class imbalance ratio    : {base_w:.0f}  (base_w = legit_n / fraud_n)")
print(f"Mean fraud amount        : ${mean_fraud_amount:,.0f}")
print(f"Fraud weight min         : {w_fraud.min():.4f}  ← may be < 1.0!")
print(f"Fraud weight mean        : {w_fraud.mean():.0f}")
print(f"Fraud weight max         : {w_fraud.max():.0f}")
print(f"Fraud cases with weight < 1.0 : {(w_fraud < 1.0).sum()}  "
      f"({'⚠️  some fraud treated as less important than legit' if (w_fraud < 1.0).any() else '✅ none'})")
print(f"\nWeight examples:")
print(f"  $119 fraud   → weight {base_w*(119/mean_fraud_amount):>10.2f}"
      f"  {'⚠️  < 1.0 (less than legit!)' if base_w*(119/mean_fraud_amount) < 1 else ''}")
print(f"  $1.5M fraud  → weight {base_w*(1.5e6/mean_fraud_amount):>10.0f}")
print(f"  $10M fraud   → weight {base_w*(10e6/mean_fraud_amount):>10.0f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. MODEL TRAINING — LightGBM with amount-weighted loss
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("4. MODEL TRAINING")
print("="*60)

lgb_params = dict(
    objective         = 'binary',
    metric            = 'auc',
    boosting_type     = 'gbdt',
    n_estimators      = 300,
    learning_rate     = 0.05,
    num_leaves        = 63,
    max_depth         = 6,
    min_child_samples = 2,
    feature_fraction  = 0.8,
    bagging_fraction  = 0.8,
    bagging_freq      = 5,
    lambda_l1         = 1.0,
    lambda_l2         = 1.0,
    # Note: NOT using is_unbalance=True — sample_weight handles class imbalance
    random_state      = 42,
    verbose           = -1,
    n_jobs            = -1,
)

cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = np.zeros(len(X))
fold_aucs = []
fold_aps  = []

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X[tr_idx], y[tr_idx],
        sample_weight=sample_weights[tr_idx]
    )
    proba = model.predict_proba(X[va_idx])[:, 1]
    oof_proba[va_idx] = proba

    auc = roc_auc_score(y[va_idx], proba)
    ap  = average_precision_score(y[va_idx], proba)
    fold_aucs.append(auc)
    fold_aps.append(ap)
    print(f"  Fold {fold+1}: AUC={auc:.4f}  AP={ap:.4f}")

oof_auc = roc_auc_score(y, oof_proba)
oof_ap  = average_precision_score(y, oof_proba)
print(f"\n  OOF ROC-AUC         : {oof_auc:.4f}")
print(f"  OOF Avg Precision   : {oof_ap:.4f}")

# Train final model on all training data
print("\n  Training final model on full dataset...")
final_model = lgb.LGBMClassifier(**lgb_params)
final_model.fit(X, y, sample_weight=sample_weights)
print("  Done.")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. THRESHOLD — F1 maximisation with recall >= 0.5, precision >= 0.05
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("5. THRESHOLD OPTIMISATION")
print("="*60)

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y, oof_proba)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)

valid_mask = (recalls[:-1] >= 0.5) & (precisions[:-1] >= 0.05)
if valid_mask.any():
    best_idx       = np.where(valid_mask)[0][np.argmax(f1_scores[:-1][valid_mask])]
    best_threshold = float(thresholds[best_idx])
    best_precision = float(precisions[best_idx])
    best_recall    = float(recalls[best_idx])
    best_f1        = float(f1_scores[best_idx])
else:
    best_idx       = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])
    best_precision = float(precisions[best_idx])
    best_recall    = float(recalls[best_idx])
    best_f1        = float(f1_scores[best_idx])

oof_preds = (oof_proba >= best_threshold).astype(int)
cm        = confusion_matrix(y, oof_preds)
tn, fp_val, fn_val, tp_val = cm.ravel()

print(f"  Best threshold : {best_threshold:.6f}")
print(f"  Precision      : {best_precision*100:.1f}%")
print(f"  Recall         : {best_recall*100:.1f}%")
print(f"  F1             : {best_f1:.4f}")
print(f"  TP={tp_val}  FP={fp_val}  FN={fn_val}  TN={tn}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. TEST SET PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("6. TEST SET PREDICTIONS")
print("="*60)

test_proba = final_model.predict_proba(test_fe[FEATURES].values)[:, 1]
test_preds = (test_proba >= best_threshold).astype(int)

test_out = test.copy()
test_out['is_fraud_predicted'] = test_preds
test_out.to_parquet(OUT, index=False)

print(f"  Threshold (F1-opt)   : {best_threshold:.6f}")
print(f"  Predicted fraud      : {test_preds.sum():,} / {len(test_preds):,} ({test_preds.mean()*100:.3f}%)")
print(f"  Saved to             : {OUT}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. PLOTS — Weight distribution + PR curve
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("7. GENERATING PLOTS")
print("="*60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Amount-Weighted Fraud Model — v1 (Proportional, No Floor)',
             fontsize=13, fontweight='bold')

# ── Plot A: Weight formula visualised ─────────────────────────────────────────
ax = axes[0]
sort_idx = np.argsort(fraud_amounts)
ax.plot(fraud_amounts[sort_idx] / 1e6,
        base_w * (fraud_amounts[sort_idx] / mean_fraud_amount),
        color=FRAUD_COLOR, linewidth=2.5,
        label='v1 formula: base_w × (amount / mean)')
ax.axhline(1.0,   color='black',  linewidth=1.2, linestyle=':',  label='Legit weight = 1.0')
ax.axhline(base_w, color='navy',  linewidth=1.2, linestyle='--', label=f'base_w = {base_w:.0f}')
ax.fill_between(fraud_amounts[sort_idx] / 1e6, 0, 1,
                color='red', alpha=0.12, label='⚠️  DANGER ZONE: fraud < legit')

broken_mask = (base_w * (fraud_amounts / mean_fraud_amount)) < 1.0
if broken_mask.any():
    ax.scatter(fraud_amounts[broken_mask] / 1e6,
               base_w * (fraud_amounts[broken_mask] / mean_fraud_amount),
               color='red', s=80, zorder=6,
               label=f'{broken_mask.sum()} fraud cases with weight < 1')

ax.set_title('v1 Weight Formula\nbase_w × (amount / mean)', fontweight='bold')
ax.set_xlabel('Fraud Transaction Amount ($M)')
ax.set_ylabel('Sample Weight')
ax.legend(fontsize=8, loc='upper left')
ax.grid(alpha=0.3)

# ── Plot B: Precision-Recall curve ────────────────────────────────────────────
ax = axes[1]
ax.plot(recalls, precisions, color=FRAUD_COLOR, linewidth=2,
        label=f'PR curve  (AP = {oof_ap:.4f})')
ax.scatter([best_recall], [best_precision], s=200, color=OPT_COLOR,
           zorder=5, edgecolor='black', linewidth=1.5,
           label=f'Optimal point\n(thr={best_threshold:.4f})')
ax.annotate(
    f"P={best_precision*100:.1f}%\nR={best_recall*100:.1f}%\nF1={best_f1:.3f}",
    xy=(best_recall, best_precision),
    xytext=(best_recall - 0.25, best_precision - 0.2),
    fontsize=9, arrowprops=dict(arrowstyle='->', color='black'),
    bbox=dict(boxstyle='round', facecolor='#fff9e6', edgecolor=OPT_COLOR, alpha=0.9)
)
ax.set_title('Precision-Recall Curve\n(F1-optimal threshold)', fontweight='bold')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.05)

# ── Plot C: Confusion matrix ───────────────────────────────────────────────────
ax = axes[2]
cm_display = np.array([[tn, fp_val], [fn_val, tp_val]])
im = ax.imshow(cm_display, cmap='Blues')
plt.colorbar(im, ax=ax)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm_display[i, j]:,}', ha='center', va='center',
                fontsize=14, fontweight='bold',
                color='white' if cm_display[i, j] > cm_display.max() / 2 else 'black')
ax.set_xticks([0, 1]); ax.set_xticklabels(['Pred: Legit', 'Pred: Fraud'])
ax.set_yticks([0, 1]); ax.set_yticklabels(['Actual: Legit', 'Actual: Fraud'])
ax.set_title(f'OOF Confusion Matrix\n(threshold = {best_threshold:.4f})', fontweight='bold')

plt.tight_layout()
plt.savefig(PLOT_DIR / '12b_weighted_v1.png', bbox_inches='tight')
plt.close()
print("  Saved: plots/12b_weighted_v1.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FINAL SUMMARY  (v1 — proportional weights, no floor)")
print("="*60)
print(f"\nWeight formula    : base_w × (amount / mean_amount)")
print(f"Base_w            : {base_w:.0f}")
print(f"Mean fraud amount : ${mean_fraud_amount:,.0f}")
print(f"Weight range      : {w_fraud.min():.2f} – {w_fraud.max():,.0f}")
print(f"Fraud < legit wt  : {(w_fraud < 1.0).sum()} cases  ({'⚠️  known limitation' if (w_fraud < 1.0).any() else '✅ none'})")
print(f"\nOOF Performance:")
print(f"  ROC-AUC         : {oof_auc:.4f}")
print(f"  Avg Precision   : {oof_ap:.4f}")
print(f"  Threshold       : {best_threshold:.6f}")
print(f"  Recall          : {best_recall*100:.1f}%")
print(f"  Precision       : {best_precision*100:.1f}%")
print(f"  F1              : {best_f1:.4f}")
print(f"  TP={tp_val}  FP={fp_val}  FN={fn_val}")
print(f"\nTest predictions  : {test_preds.sum():,} fraud / {len(test_preds):,} total")
print(f"Output saved to   : {OUT}")
print(f"Plot saved to     : {PLOT_DIR}/12b_weighted_v1.png")
print("\n" + "="*60)
print("DONE")
print("="*60)
print("\n⚠️  NOTE: For the corrected formula (floor fix), see fraud_model_weighted.py")
