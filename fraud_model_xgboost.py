"""
Fraud Detection — XGBoost Cost-Sensitive Weighted Model
========================================================
Same methodology as fraud_model_weighted.py but using XGBoost instead of LightGBM.

LightGBM → XGBoost parameter mapping:
  num_leaves=63        → (no direct equiv; max_depth=6 gives ~63 leaves)
  min_child_samples=2  → min_child_weight=1
  feature_fraction=0.8 → colsample_bytree=0.8
  bagging_fraction=0.8 → subsample=0.8
  bagging_freq=5       → (XGBoost subsamples every tree by default)
  lambda_l1=1.0        → reg_alpha=1.0
  lambda_l2=1.0        → reg_lambda=1.0
  verbose=-1           → verbosity=0
  metric='auc'         → eval_metric='auc'
  objective='binary'   → objective='binary:logistic'
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
import xgboost as xgb

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path("/Users/mengyao/Documents/_MyDrive/Interviews/Nasdaq")
TRAIN    = BASE / "train_fraud.parquet"
TEST     = BASE / "test_fraud_external.parquet"
OUT      = BASE / "test_fraud_external_predicted_xgb.parquet"
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
# 3. AMOUNT-WEIGHTED SAMPLE WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("3. SAMPLE WEIGHT CONSTRUCTION")
print("="*60)

legit_n           = int((y == 0).sum())
fraud_n           = int((y == 1).sum())
base_w            = legit_n / fraud_n          # class imbalance ratio ~8,806
fraud_amounts     = amounts[y == 1]
mean_fraud_amount = fraud_amounts.mean()

# weight(legit)  = 1.0
# weight(fraud)  = base_w × (1 + amount / mean_fraud_amount)
# The "+1" guarantees every fraud weight >= base_w (floor = 8,806) >> legit
sample_weights          = np.ones(len(y))
sample_weights[y == 1]  = base_w * (1 + fraud_amounts / mean_fraud_amount)

w_fraud = sample_weights[y == 1]
print(f"Legit weight          : 1.0")
print(f"Fraud weight floor    : {base_w * 1:.0f}  (any fraud, even $0)")
print(f"Fraud weight mean     : {w_fraud.mean():.0f}")
print(f"Fraud weight max      : {w_fraud.max():.0f}  ($10M fraud)")
print(f"Any fraud weight < 1? : {(w_fraud < 1.0).any()}  ✅")
print(f"Any fraud weight < base_w? : {(w_fraud < base_w).any()}  ✅")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. MODEL TRAINING — XGBoost with 5-fold Stratified CV
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("4. MODEL TRAINING — XGBoost with 5-fold Stratified CV")
print("="*60)

xgb_params = dict(
    objective         = 'binary:logistic',   # LightGBM: 'binary'
    eval_metric       = 'auc',               # LightGBM: metric='auc'
    n_estimators      = 300,
    learning_rate     = 0.05,
    max_depth         = 6,
    min_child_weight  = 1,                   # LightGBM: min_child_samples=2
    colsample_bytree  = 0.8,                 # LightGBM: feature_fraction=0.8
    subsample         = 0.8,                 # LightGBM: bagging_fraction=0.8
    reg_alpha         = 1.0,                 # LightGBM: lambda_l1=1.0
    reg_lambda        = 1.0,                 # LightGBM: lambda_l2=1.0
    # Note: no scale_pos_weight — handled via sample_weight
    random_state      = 42,
    verbosity         = 0,                   # LightGBM: verbose=-1
    n_jobs            = -1,
)

cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = np.zeros(len(X))
fold_aucs = []
fold_aps  = []

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
    model = xgb.XGBClassifier(**xgb_params)
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
print(f"\n  OOF ROC-AUC         : {oof_auc:.4f}  (folds: {[f'{x:.4f}' for x in fold_aucs]})")
print(f"  OOF Avg Precision   : {oof_ap:.4f}  (folds: {[f'{x:.4f}' for x in fold_aps]})")

# Train final model on all training data
print("\n  Training final model on full dataset...")
final_model = xgb.XGBClassifier(**xgb_params)
final_model.fit(X, y, sample_weight=sample_weights)
print("  Done.")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. COST-SENSITIVE THRESHOLD OPTIMISATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("5. COST-SENSITIVE THRESHOLD OPTIMISATION")
print("="*60)

thresholds_sweep = np.linspace(0.0001, 0.9999, 3000)
cost_scenarios   = [50, 200, 500]

results = {}
for cpa in cost_scenarios:
    fc, oc, tc, rec_l, prec_l, fp_l, fn_l = [], [], [], [], [], [], []

    for thr in thresholds_sweep:
        preds = (oof_proba >= thr).astype(int)
        tp = int(((preds==1) & (y==1)).sum())
        fp = int(((preds==1) & (y==0)).sum())
        fn = int(((preds==0) & (y==1)).sum())

        missed_val = fraud_amounts[(preds[y==1] == 0)].sum()

        fc.append(missed_val)
        oc.append(fp * cpa)
        tc.append(missed_val + fp * cpa)
        fp_l.append(fp)
        fn_l.append(fn)
        rec_l.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        prec_l.append(tp / (tp + fp) if (tp + fp) > 0 else 0)

    results[cpa] = dict(
        fc=np.array(fc), oc=np.array(oc), tc=np.array(tc),
        rec=np.array(rec_l), prec=np.array(prec_l),
        fp=np.array(fp_l), fn=np.array(fn_l)
    )

    opt     = int(np.argmin(tc))
    opt_thr = thresholds_sweep[opt]
    print(f"\n  Cost per alert = ${cpa}")
    print(f"    Optimal threshold : {opt_thr:.6f}")
    print(f"    Recall            : {rec_l[opt]*100:.1f}%")
    print(f"    Precision         : {prec_l[opt]*100:.1f}%")
    print(f"    False alarms/month: {fp_l[opt]}")
    print(f"    Fraud missed      : {fn_l[opt]}")
    print(f"    Fraud cost        : ${fc[opt]:,.0f}")
    print(f"    Ops cost          : ${oc[opt]:,.0f}")
    print(f"    Total cost        : ${tc[opt]:,.0f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. GENERATE TEST PREDICTIONS (using $200/alert optimal threshold)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("6. TEST SET PREDICTIONS")
print("="*60)

opt_thr    = thresholds_sweep[int(np.argmin(results[200]['tc']))]
test_proba = final_model.predict_proba(test_fe[FEATURES].values)[:, 1]
test_preds = (test_proba >= opt_thr).astype(int)

test_out = test.copy()
test_out['is_fraud_predicted'] = test_preds
test_out.to_parquet(OUT, index=False)

print(f"  Optimal threshold (${200}/alert): {opt_thr:.6f}")
print(f"  Predicted fraud  : {test_preds.sum():,} / {len(test_preds):,} ({test_preds.mean()*100:.3f}%)")
print(f"  Saved to         : {OUT}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("7. GENERATING PLOTS")
print("="*60)

fig = plt.figure(figsize=(20, 20))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot A: Weight formula comparison ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
old_w = base_w * (fraud_amounts / mean_fraud_amount)
new_w = base_w * (1 + fraud_amounts / mean_fraud_amount)
sort_idx = np.argsort(fraud_amounts)

ax1.plot(fraud_amounts[sort_idx]/1e6, old_w[sort_idx],
         color='gray', linewidth=2, linestyle='--',
         label='Old formula (BROKEN)\nbase_w × (amount/mean)\n→ some weights < 1.0')
ax1.plot(fraud_amounts[sort_idx]/1e6, new_w[sort_idx],
         color=FRAUD_COLOR, linewidth=2.5,
         label='Corrected formula\nbase_w × (1 + amount/mean)\n→ all weights ≥ 8,806')
ax1.axhline(1.0, color='black', linewidth=1.2, linestyle=':',
            label='Legit weight = 1.0')
ax1.axhline(base_w, color='navy', linewidth=1.2, linestyle=':',
            label=f'Base imbalance weight = {base_w:.0f}')
ax1.fill_between(fraud_amounts[sort_idx]/1e6, 0, 1,
                 color='red', alpha=0.1, label='DANGER ZONE: fraud < legit')

broken_mask = old_w < 1.0
if broken_mask.any():
    ax1.scatter(fraud_amounts[broken_mask]/1e6, old_w[broken_mask],
                color='red', s=80, zorder=6,
                label=f'{broken_mask.sum()} fraud cases with weight < 1 (❌)')

ax1.set_title('Weight Formula: Old vs Corrected', fontweight='bold')
ax1.set_xlabel('Fraud Transaction Amount ($M)')
ax1.set_ylabel('Sample Weight')
ax1.legend(fontsize=7.5, loc='upper left')
ax1.grid(alpha=0.3)

# ── Plot B: Weight philosophy bar chart ───────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
categories = ['Legit\ntransaction', '$0 fraud\n(min floor)',
              '$119 fraud\n(smallest)', '$1.5M fraud\n(average)', '$10M fraud\n(max)']
w_values   = [
    1,
    base_w * (1 + 0),
    base_w * (1 + 119 / mean_fraud_amount),
    base_w * (1 + 1),
    base_w * (1 + 10e6 / mean_fraud_amount),
]
bar_colors = ['#2980b9', '#e74c3c', '#e74c3c', '#c0392b', '#922b21']
bars = ax2.bar(categories, [min(v, 75000) for v in w_values],
               color=bar_colors, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, w_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
             f'{val:,.0f}', ha='center', fontsize=8.5, fontweight='bold')
ax2.axhline(1.0, color='black', linewidth=1.5, linestyle='--', alpha=0.4,
            label='Legit baseline = 1')
ax2.set_title('Corrected Weight Philosophy\n"Every fraud always >> any legit"',
              fontweight='bold')
ax2.set_ylabel('Sample Weight')
ax2.tick_params(axis='x', labelsize=8.5)
ax2.legend(fontsize=9)
ax2.text(0.5, 0.92,
    'The floor (base_w=8,806) guarantees\neven a $0 fraud matters 8,806× more\nthan any legitimate transaction',
    transform=ax2.transAxes, ha='center', va='top', fontsize=8.5,
    bbox=dict(boxstyle='round', facecolor='#fff9e6', edgecolor=OPT_COLOR, alpha=0.9))

# ── Plot C: Main cost curve (full width) ──────────────────────────────────────
ax3   = fig.add_subplot(gs[1, :])
cpa   = 200
res   = results[cpa]
opt   = int(np.argmin(res['tc']))
opt_t = thresholds_sweep[opt]

ax3.plot(thresholds_sweep, res['fc']/1e6,  color=FRAUD_COLOR,  linewidth=2,
         label='Fraud cost  =  total $ value of missed fraud transactions')
ax3.plot(thresholds_sweep, res['oc']/1e6,  color=OPS_COLOR,    linewidth=2,
         label=f'Ops cost  =  false alarms × ${cpa} per investigation')
ax3.plot(thresholds_sweep, res['tc']/1e6,  color=TOTAL_COLOR,  linewidth=3,
         label='Total cost  =  Fraud cost + Ops cost', zorder=5)
ax3.axvline(opt_t, color=OPT_COLOR, linestyle='--', linewidth=2.5, zorder=6,
            label=f'Balanced point  (threshold = {opt_t:.4f})')
ax3.scatter([opt_t], [res['tc'][opt]/1e6], s=220,
            color=OPT_COLOR, zorder=7, edgecolor='black', linewidth=1.5)

ax3.annotate(
    f"  BALANCED POINT\n"
    f"  Threshold  = {opt_t:.4f}\n"
    f"  Recall     = {res['rec'][opt]*100:.1f}%\n"
    f"  Precision  = {res['prec'][opt]*100:.1f}%\n"
    f"  False alarms / month = {res['fp'][opt]}\n"
    f"  Fraud missed         = {res['fn'][opt]}\n"
    f"  Fraud cost  = ${res['fc'][opt]/1e6:.3f}M\n"
    f"  Ops cost    = ${res['oc'][opt]:,.0f}\n"
    f"  Total cost  = ${res['tc'][opt]/1e6:.3f}M",
    xy=(opt_t, res['tc'][opt]/1e6),
    xytext=(min(opt_t + 0.15, 0.75), res['tc'][opt]/1e6 + 60),
    fontsize=9,
    arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff9e6',
              edgecolor=OPT_COLOR, linewidth=1.5, alpha=0.97)
)
ax3.axvspan(0,     opt_t, alpha=0.04, color=FRAUD_COLOR,
            label='← Fraud cost dominates (threshold too low)')
ax3.axvspan(opt_t, 1,     alpha=0.04, color=OPS_COLOR,
            label='Ops cost dominates (threshold too high) →')

ax3.set_title(
    f'Business Cost Curve — Finding the Balanced Point  [XGBoost]\n'
    f'(Cost per alert = ${cpa}  |  Corrected weights: every fraud ≥ {base_w:.0f}× legit)',
    fontweight='bold', fontsize=12
)
ax3.set_xlabel('Classification Threshold', fontsize=11)
ax3.set_ylabel('Total Cost ($M)', fontsize=11)
ax3.legend(fontsize=8.5, loc='upper right')
ax3.grid(alpha=0.3)
ax3.set_xlim(0, 1)

# ── Plot D: All 3 cost scenarios ──────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
cols_s = ['#27ae60', '#e67e22', '#e74c3c']
for cpa_s, col in zip(cost_scenarios, cols_s):
    res_s = results[cpa_s]
    opt_s = int(np.argmin(res_s['tc']))
    ax4.plot(thresholds_sweep, res_s['tc']/1e6, color=col, linewidth=2,
             label=f'${cpa_s}/alert → thr={thresholds_sweep[opt_s]:.4f}  '
                   f'Recall={res_s["rec"][opt_s]*100:.0f}%  FP={res_s["fp"][opt_s]}')
    ax4.scatter([thresholds_sweep[opt_s]], [res_s['tc'][opt_s]/1e6],
                s=120, color=col, edgecolor='black', zorder=5)
ax4.set_title('Optimal Threshold Sensitivity\nAcross Operational Cost Assumptions',
              fontweight='bold')
ax4.set_xlabel('Classification Threshold')
ax4.set_ylabel('Total Cost ($M)')
ax4.legend(fontsize=8, loc='upper right')
ax4.grid(alpha=0.3)
ax4.set_xlim(0, 1)

# ── Plot E: False alarms vs missed fraud trade-off space ──────────────────────
ax5 = fig.add_subplot(gs[2, 1])
res200 = results[200]
opt200 = int(np.argmin(res200['tc']))
sc = ax5.scatter(res200['fp'], res200['fc']/1e6,
                 c=thresholds_sweep, cmap='RdYlGn_r', s=8, alpha=0.6)
ax5.scatter([res200['fp'][opt200]], [res200['fc'][opt200]/1e6],
            s=250, color=OPT_COLOR, edgecolor='black', linewidth=2,
            zorder=5, label=f'Balanced point  (thr={thresholds_sweep[opt200]:.4f})')
ax5.annotate(
    f"FP={res200['fp'][opt200]}\nMissed=${res200['fc'][opt200]/1e6:.3f}M",
    xy=(res200['fp'][opt200], res200['fc'][opt200]/1e6),
    xytext=(res200['fp'][opt200] + 15, res200['fc'][opt200]/1e6 + 25),
    fontsize=8.5,
    arrowprops=dict(arrowstyle='->', color='black', lw=1),
    bbox=dict(boxstyle='round', facecolor='#fff9e6', edgecolor=OPT_COLOR, alpha=0.9)
)
plt.colorbar(sc, ax=ax5, label='Threshold value')
ax5.set_title('Operational Trade-off Space\nFalse Alarms vs Missed Fraud ($M)',
              fontweight='bold')
ax5.set_xlabel('False Alarms per Month')
ax5.set_ylabel('Missed Fraud Value ($M)')
ax5.legend(fontsize=8.5)
ax5.grid(alpha=0.3)

plt.suptitle('Cost-Sensitive Fraud Detection — XGBoost with Corrected Amount Weighting',
             fontsize=14, fontweight='bold', y=1.01)
plt.savefig(PLOT_DIR / '13_cost_optimisation_xgb.png', bbox_inches='tight')
plt.close()
print("  Saved: 13_cost_optimisation_xgb.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
opt_res = results[200]
opt_idx = int(np.argmin(opt_res['tc']))

print("\n" + "="*60)
print("FINAL SUMMARY  [XGBoost]")
print("="*60)
print(f"\nModel             : XGBoost (binary:logistic)")
print(f"Weight formula    : base_w × (1 + amount/mean_amount)")
print(f"Weight floor      : {base_w:.0f}  (every fraud always >> legit)")
print(f"Threshold (opt)   : {thresholds_sweep[opt_idx]:.6f}  ($200/alert scenario)")
print(f"\nOOF Performance:")
print(f"  ROC-AUC         : {oof_auc:.4f}")
print(f"  Avg Precision   : {oof_ap:.4f}")
print(f"  Recall          : {opt_res['rec'][opt_idx]*100:.1f}%")
print(f"  Precision       : {opt_res['prec'][opt_idx]*100:.1f}%")
print(f"  False alarms/mo : {opt_res['fp'][opt_idx]}")
print(f"  Fraud missed    : {opt_res['fn'][opt_idx]}")
print(f"  Total cost/mo   : ${opt_res['tc'][opt_idx]:,.0f}")
print(f"\nTest predictions  : {test_preds.sum():,} fraud / {len(test_preds):,} total")
print(f"Output saved to   : {OUT}")
print(f"Plot saved to     : {PLOT_DIR}/13_cost_optimisation_xgb.png")
print("\n" + "="*60)
print("DONE")
print("="*60)
