"""
Fraud Detection Analysis
========================
Steps:
1. EDA + Feature Engineering
2. Model Training (train_fraud.parquet only)
3. Predict on test_fraud_external.parquet → append 'is_fraud_predicted'
4. Evaluate model performance
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
#using StratifiedKFold for cross validations. (n_splits= 5)

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, ConfusionMatrixDisplay
)
import lightgbm as lgb

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path("/Users/mengyao/Documents/_MyDrive/Interviews/Nasdaq")
TRAIN_PATH = BASE / "train_fraud.parquet"
TEST_PATH  = BASE / "test_fraud_external.parquet"
OUT_PATH   = BASE / "test_fraud_external_predicted.parquet"
PLOT_DIR   = BASE / "plots"
PLOT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})
PALETTE = {'fraud': '#e74c3c', 'legit': '#2980b9'}

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("1. LOADING DATA")
print("="*60)

train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)

print(f"Train shape : {train.shape}")
print(f"Test  shape : {test.shape}")
print(f"Train columns: {train.columns.tolist()}")
print(f"\nTrain dtypes:\n{train.dtypes}")
print(f"\nMissing values (train):\n{train.isnull().sum()}")
print(f"\nFraud rate : {train['is_fraud'].mean()*100:.4f}%")
print(f"Fraud count: {train['is_fraud'].sum():,} / {len(train):,}")
print(f"\nTransaction types:\n{train['transaction_type'].value_counts()}")
print(f"\nBasic stats:\n{train[['transaction_amount','initiater_balance_before','recipient_balance_before']].describe()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. EDA PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("2. EXPLORATORY DATA ANALYSIS")
print("="*60)

# ── Plot 1: Target distribution ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

counts = train['is_fraud'].value_counts()
axes[0].bar(['Legitimate', 'Fraud'], counts.values,
            color=[PALETTE['legit'], PALETTE['fraud']], edgecolor='black', linewidth=0.5)
axes[0].set_title('Transaction Class Distribution', fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 100, f'{v:,}\n({v/len(train)*100:.4f}%)', ha='center', fontsize=9)

axes[1].pie(counts.values, labels=['Legitimate', 'Fraud'],
            autopct='%1.4f%%', colors=[PALETTE['legit'], PALETTE['fraud']],
            startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
axes[1].set_title('Class Balance (extremely imbalanced)', fontweight='bold')

plt.tight_layout()
plt.savefig(PLOT_DIR / "01_class_distribution.png", bbox_inches='tight')
plt.close()
print("  Saved: 01_class_distribution.png")

# ── Plot 2: Transaction type vs fraud ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

type_counts = train.groupby(['transaction_type', 'is_fraud']).size().unstack(fill_value=0)
type_counts.columns = ['Legitimate', 'Fraud']
type_counts.plot(kind='bar', ax=axes[0],
                 color=[PALETTE['legit'], PALETTE['fraud']],
                 edgecolor='black', linewidth=0.5)
axes[0].set_title('Transaction Type: Legit vs Fraud (Count)', fontweight='bold')
axes[0].set_xlabel('')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=30)
axes[0].legend()

fraud_rate_by_type = train.groupby('transaction_type')['is_fraud'].mean() * 100
fraud_rate_by_type.sort_values(ascending=False).plot(kind='bar', ax=axes[1],
    color=PALETTE['fraud'], edgecolor='black', linewidth=0.5)
axes[1].set_title('Fraud Rate (%) by Transaction Type', fontweight='bold')
axes[1].set_xlabel('')
axes[1].set_ylabel('Fraud Rate (%)')
axes[1].tick_params(axis='x', rotation=30)
for i, v in enumerate(fraud_rate_by_type.sort_values(ascending=False)):
    axes[1].text(i, v + 0.001, f'{v:.4f}%', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(PLOT_DIR / "02_transaction_type.png", bbox_inches='tight')
plt.close()
print("  Saved: 02_transaction_type.png")
print(f"  Fraud by type:\n{train.groupby('transaction_type')['is_fraud'].agg(['sum','mean']).round(4)}")

# ── Plot 3: Amount distribution ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

legit_amounts = train[train['is_fraud']==0]['transaction_amount']
fraud_amounts = train[train['is_fraud']==1]['transaction_amount']

axes[0].hist(np.log1p(legit_amounts), bins=60, color=PALETTE['legit'], alpha=0.7,
             label='Legitimate', density=True)
axes[0].hist(np.log1p(fraud_amounts), bins=60, color=PALETTE['fraud'], alpha=0.8,
             label='Fraud', density=True)
axes[0].set_title('log(n+1) Distribution by Class', fontweight='bold')
axes[0].set_xlabel('log(n + 1)')
axes[0].set_ylabel('Density')
axes[0].legend()

# Amount stats by class
amount_stats = train.groupby('is_fraud')['transaction_amount'].describe()
print(f"\n  Amount stats by class:\n{amount_stats}")

# Amount by type for fraud
axes[1].boxplot(
    [np.log1p(train[train['transaction_type']==t]['transaction_amount'])
     for t in sorted(train['transaction_type'].unique())],
    labels=sorted(train['transaction_type'].unique()),
    patch_artist=True,
    medianprops=dict(color='red', linewidth=2)
)
axes[1].set_title('log(n+1) by Transaction Type', fontweight='bold')
axes[1].set_ylabel('log(n + 1)')
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(PLOT_DIR / "03_amount_distribution.png", bbox_inches='tight')
plt.close()
print("  Saved: 03_amount_distribution.png")

# ── Plot 4: Time analysis ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

fraud_by_hour = train[train['is_fraud']==1].groupby('hours_elapsed').size()
legit_by_hour = train[train['is_fraud']==0].groupby('hours_elapsed').size()

# Normalize to rate per transaction
fraud_rate_by_hour = train.groupby('hours_elapsed')['is_fraud'].mean() * 100
txn_vol_by_hour   = train.groupby('hours_elapsed').size()

ax2 = axes[0].twinx()
axes[0].fill_between(txn_vol_by_hour.index, txn_vol_by_hour.values,
                     alpha=0.3, color=PALETTE['legit'], label='Transaction volume')
ax2.plot(fraud_rate_by_hour.index, fraud_rate_by_hour.values,
         color=PALETTE['fraud'], linewidth=1, alpha=0.6, label='Fraud rate')
rolling = fraud_rate_by_hour.rolling(12, center=True).mean()
ax2.plot(rolling.index, rolling.values, color='darkred', linewidth=2.5, label='12h rolling avg')
axes[0].set_title('Transaction Volume and Fraud Rate Over 744 Hours', fontweight='bold')
axes[0].set_ylabel('Transaction Count', color=PALETTE['legit'])
ax2.set_ylabel('Fraud Rate (%)', color=PALETTE['fraud'])
axes[0].legend(loc='upper left')
ax2.legend(loc='upper right')

# Hourly pattern
hour_fraud_rate = train.copy()
hour_fraud_rate['hour_of_day'] = hour_fraud_rate['hours_elapsed'] % 24
hourly = hour_fraud_rate.groupby('hour_of_day')['is_fraud'].mean() * 100
hourly_vol = hour_fraud_rate.groupby('hour_of_day').size()
axes[1].bar(hourly.index, hourly.values, color=PALETTE['fraud'], alpha=0.8, label='Fraud rate (%)')
axes[1].set_title('Fraud Rate by Hour of Day', fontweight='bold')
axes[1].set_xlabel('Hour of Day')
axes[1].set_ylabel('Fraud Rate (%)')
axes[1].set_xticks(range(24))

plt.tight_layout()
plt.savefig(PLOT_DIR / "04_time_analysis.png", bbox_inches='tight')
plt.close()
print("  Saved: 04_time_analysis.png")

# ── Plot 5: Balance analysis ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# ── Top-left: Account Drain — dual y-axis to show both classes ────────────────
 
# left axis = legit counts, right axis = fraud counts.
train_copy = train.copy()
train_copy['drains_account'] = (
    (train_copy['initiater_balance_before'] > 0) &
    (np.abs(train_copy['transaction_amount'] - train_copy['initiater_balance_before']) < 1.0)
).astype(int)

drain_counts = train_copy.groupby(['drains_account', 'is_fraud']).size().unstack(fill_value=0)
drain_counts.index = ['Does NOT\ndrain', 'Drains\naccount']
drain_counts.columns = ['Legitimate', 'Fraud']

x = np.arange(len(drain_counts))
width = 0.35
ax_r = axes[0][0].twinx()   # right axis for fraud

bars_l = axes[0][0].bar(x - width/2, drain_counts['Legitimate'], width,
                         color=PALETTE['legit'], edgecolor='black', linewidth=0.5,
                         label='Legitimate (left axis)')
bars_r = ax_r.bar(x + width/2, drain_counts['Fraud'], width,
                   color=PALETTE['fraud'], edgecolor='black', linewidth=0.5,
                   label='Fraud (right axis)')

axes[0][0].set_title('Account Drain: Legit vs Fraud\n(dual axis — scales differ)', fontweight='bold')
axes[0][0].set_ylabel('Legitimate Count', color=PALETTE['legit'])
ax_r.set_ylabel('Fraud Count', color=PALETTE['fraud'])
axes[0][0].set_xticks(x)
axes[0][0].set_xticklabels(drain_counts.index)
axes[0][0].tick_params(axis='y', colors=PALETTE['legit'])
ax_r.tick_params(axis='y', colors=PALETTE['fraud'])

# Label bars with counts
for bar in bars_l:
    axes[0][0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=8,
                    color=PALETTE['legit'])
for bar in bars_r:
    ax_r.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
              f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=8,
              color=PALETTE['fraud'])

lines_l, labels_l = axes[0][0].get_legend_handles_labels()
lines_r, labels_r = ax_r.get_legend_handles_labels()
axes[0][0].legend(lines_l + lines_r, labels_l + labels_r, fontsize=7, loc='upper center')

# Finding annotation
drain_fraud_pct = drain_counts.loc['Drains\naccount', 'Fraud'] / drain_counts['Fraud'].sum() * 100
drain_legit_pct = drain_counts.loc['Drains\naccount', 'Legitimate'] / drain_counts['Legitimate'].sum() * 100
axes[0][0].text(0.5, -0.18,
    f'Finding: {drain_fraud_pct:.0f}% of fraud drains the account vs only {drain_legit_pct:.1f}% of legit',
    transform=axes[0][0].transAxes, ha='center', fontsize=8,
    color='darkred', style='italic',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe5e5', edgecolor='#e74c3c', alpha=0.8))

# ── Top-right: Initiator Balance Before ───────────────────────────────────────
# What to look for: do fraudsters start with more or less money than legit senders?
legit_init = np.log1p(train[train['is_fraud']==0]['initiater_balance_before'].clip(lower=0))
fraud_init = np.log1p(train[train['is_fraud']==1]['initiater_balance_before'].clip(lower=0))
axes[0][1].hist(legit_init, bins=50, color=PALETTE['legit'], alpha=0.7, label='Legitimate', density=True)
axes[0][1].hist(fraud_init, bins=50, color=PALETTE['fraud'], alpha=0.8, label='Fraud', density=True)
axes[0][1].set_title('log(Initiator Balance Before + 1) by Class', fontweight='bold')
axes[0][1].set_xlabel('log(Balance + 1)')
axes[0][1].set_ylabel('Density')
axes[0][1].legend()

legit_zero_pct = (train[train['is_fraud']==0]['initiater_balance_before'] == 0).mean() * 100
fraud_zero_pct = (train[train['is_fraud']==1]['initiater_balance_before'] == 0).mean() * 100
axes[0][1].text(0.5, -0.18,
    f'Finding: {fraud_zero_pct:.0f}% of fraud starts at $0 vs {legit_zero_pct:.0f}% of legit — fraudsters have real money before striking',
    transform=axes[0][1].transAxes, ha='center', fontsize=8,
    color='darkred', style='italic',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe5e5', edgecolor='#e74c3c', alpha=0.8))

# ── Bottom-left: Recipient Balance Before ─────────────────────────────────────
# What to look for: are fraud recipients "mule" accounts that start empty?
legit_recip = np.log1p(train[train['is_fraud']==0]['recipient_balance_before'].clip(lower=0))
fraud_recip = np.log1p(train[train['is_fraud']==1]['recipient_balance_before'].clip(lower=0))
axes[1][0].hist(legit_recip, bins=50, color=PALETTE['legit'], alpha=0.7, label='Legitimate', density=True)
axes[1][0].hist(fraud_recip, bins=50, color=PALETTE['fraud'], alpha=0.8, label='Fraud', density=True)
axes[1][0].set_title('log(Recipient Balance Before + 1) by Class', fontweight='bold')
axes[1][0].set_xlabel('log(Balance + 1)')
axes[1][0].set_ylabel('Density')
axes[1][0].legend()

fraud_recip_zero = (train[train['is_fraud']==1]['recipient_balance_before'] == 0).mean() * 100
legit_recip_zero = (train[train['is_fraud']==0]['recipient_balance_before'] == 0).mean() * 100
axes[1][0].text(0.5, -0.18,
    f'Finding: {fraud_recip_zero:.0f}% of fraud recipients start at $0 vs {legit_recip_zero:.0f}% of legit — recipient accounts are often empty shells',
    transform=axes[1][0].transAxes, ha='center', fontsize=8,
    color='darkred', style='italic',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe5e5', edgecolor='#e74c3c', alpha=0.8))

# ── Bottom-right: Initiator Balance Decrease ──────────────────────────────────

train_copy['init_balance_change'] = (train_copy['initiater_balance_before']
                                     - train_copy['initiater_balance_after'])
legit_change = np.log1p(train_copy[train_copy['is_fraud']==0]['init_balance_change'].clip(lower=0))
fraud_change = np.log1p(train_copy[train_copy['is_fraud']==1]['init_balance_change'].clip(lower=0))
axes[1][1].hist(legit_change, bins=50, color=PALETTE['legit'], alpha=0.7, label='Legitimate', density=True)
axes[1][1].hist(fraud_change, bins=50, color=PALETTE['fraud'], alpha=0.8, label='Fraud', density=True)
axes[1][1].set_title('log(Initiator Balance Decrease + 1) by Class', fontweight='bold')
axes[1][1].set_xlabel('log(Balance Decrease + 1)')
axes[1][1].set_ylabel('Density')
axes[1][1].legend()

fraud_med_decrease = train_copy[train_copy['is_fraud']==1]['init_balance_change'].median()
legit_med_decrease = train_copy[train_copy['is_fraud']==0]['init_balance_change'].clip(lower=0).median()
axes[1][1].text(0.5, -0.18,
    f'Finding: median balance decrease — Fraud ${fraud_med_decrease:,.0f} vs Legit ${legit_med_decrease:,.0f} — fraud drains far more per transaction',
    transform=axes[1][1].transAxes, ha='center', fontsize=8,
    color='darkred', style='italic',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe5e5', edgecolor='#e74c3c', alpha=0.8))

plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig(PLOT_DIR / "05_balance_analysis.png", bbox_inches='tight')
plt.close()
print("  Saved: 05_balance_analysis.png")

# ── Plot 6: Existing fraud flag analysis ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cross = pd.crosstab(train['is_flagged_fraud'], train['is_fraud'],
                    rownames=['System Flagged'], colnames=['Actual Fraud'])
sns.heatmap(cross, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            linewidths=1)
axes[0].set_title('System Flag vs Actual Fraud', fontweight='bold')

flag_fraud_overlap = ((train['is_flagged_fraud']==1) & (train['is_fraud']==1)).sum()
actual_fraud = train['is_fraud'].sum()
flagged_total = train['is_flagged_fraud'].sum()

from sklearn.metrics import classification_report as cr
if flagged_total > 0:
    flag_report = cr(train['is_fraud'], train['is_flagged_fraud'],
                     target_names=['Legit', 'Fraud'], output_dict=True)
    metrics_df = pd.DataFrame(flag_report).T.loc[['Legit', 'Fraud'],
                                                  ['precision', 'recall', 'f1-score']]
    metrics_df.plot(kind='bar', ax=axes[1], edgecolor='black', linewidth=0.5)
    axes[1].set_title("Existing Flag System Performance", fontweight='bold')
    axes[1].set_ylabel('Score')
    axes[1].set_xticklabels(['Legitimate', 'Fraud'], rotation=0)
    axes[1].legend(loc='upper right')
    axes[1].set_ylim(0, 1.15)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%.3f', fontsize=8, padding=2)

plt.tight_layout()
plt.savefig(PLOT_DIR / "06_flag_vs_actual.png", bbox_inches='tight')
plt.close()
print("  Saved: 06_flag_vs_actual.png")
print(f"\n  System flag stats:")
print(f"  - Total flagged: {flagged_total:,} ({flagged_total/len(train)*100:.4f}%)")
print(f"  - True fraud: {actual_fraud:,} ({actual_fraud/len(train)*100:.4f}%)")
print(f"  - Flag captures {flag_fraud_overlap}/{actual_fraud} frauds (recall = {flag_fraud_overlap/actual_fraud*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("3. FEATURE ENGINEERING")
print("="*60)

TYPE_MAP = {t: i for i, t in enumerate(sorted(
    pd.concat([train['transaction_type'], test['transaction_type']]).unique()
))}

def engineer_features(df):
    df = df.copy()

    # ── Log-transformed amounts / balances ────────────────────────────────
    df['log_amount']           = np.log1p(df['transaction_amount'])
    df['log_init_bal_before']  = np.log1p(df['initiater_balance_before'].clip(lower=0))
    df['log_recip_bal_before'] = np.log1p(df['recipient_balance_before'].clip(lower=0))
    df['log_init_bal_after']   = np.log1p(df['initiater_balance_after'].clip(lower=0))
    df['log_recip_bal_after']  = np.log1p(df['recipient_balance_after'].clip(lower=0))

    # ── Balance changes (post-transaction) ────────────────────────────────
    df['init_balance_change']  = df['initiater_balance_before'] - df['initiater_balance_after']
    df['recip_balance_change'] = df['recipient_balance_after']  - df['recipient_balance_before']
    df['log_init_bal_change']  = np.log1p(df['init_balance_change'].clip(lower=0))
    df['log_recip_bal_change'] = np.log1p(df['recip_balance_change'].clip(lower=0))

    # ── Balance anomaly: amount doesn't match balance change ──────────────
    # For a legitimate transaction, amount ≈ initiator's balance decrease
    df['init_change_vs_amount'] = np.abs(df['init_balance_change'] - df['transaction_amount'])
    df['recip_change_vs_amount']= np.abs(df['recip_balance_change'] - df['transaction_amount'])
    df['init_amount_mismatch']  = (df['init_change_vs_amount'] > 1.0).astype(int)
    df['recip_amount_mismatch'] = (df['recip_change_vs_amount'] > 1.0).astype(int)

    # ── "Drain account" — transaction amount equals balance before ─────────
    df['drains_account']        = (
        (df['initiater_balance_before'] > 0) &
        (np.abs(df['transaction_amount'] - df['initiater_balance_before']) < 1.0)
    ).astype(int)

    # ── Zero balance signals ───────────────────────────────────────────────
    df['init_zero_before']  = (df['initiater_balance_before']  == 0).astype(int)
    df['init_zero_after']   = (df['initiater_balance_after']   == 0).astype(int)
    df['recip_zero_before'] = (df['recipient_balance_before']  == 0).astype(int)
    df['recip_zero_after']  = (df['recipient_balance_after']   == 0).astype(int)

    # Became zero after (common in fraud: full drain)
    df['init_became_zero']  = ((df['initiater_balance_before']  > 0) & (df['initiater_balance_after']  == 0)).astype(int)
    df['recip_became_zero'] = ((df['recipient_balance_before']  > 0) & (df['recipient_balance_after']  == 0)).astype(int)

    # ── Amount relative to initiator balance ──────────────────────────────
    df['amount_to_init_ratio']  = df['transaction_amount'] / (df['initiater_balance_before'] + 1)
    df['amount_to_recip_ratio'] = df['transaction_amount'] / (df['recipient_balance_before'] + 1)

    # ── Entity type ────────────────────────────────────────────────────────
    df['initiator_is_customer'] = (df['initiating_customer'].str[0] == 'C').astype(int)
    df['recipient_is_customer']  = (df['recipient_customer'].str[0] == 'C').astype(int)
    df['c2c'] = ((df['initiator_is_customer'] == 1) & (df['recipient_is_customer'] == 1)).astype(int)
    df['c2m'] = ((df['initiator_is_customer'] == 1) & (df['recipient_is_customer'] == 0)).astype(int)

    # ── Time features ──────────────────────────────────────────────────────
    df['hour_of_day'] = df['hours_elapsed'] % 24
    df['day_of_month'] = df['hours_elapsed'] // 24
    df['is_night']    = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] < 6)).astype(int)

    # ── Transaction type ───────────────────────────────────────────────────
    df['txn_type_enc'] = df['transaction_type'].map(TYPE_MAP).fillna(-1).astype(int)
    # Binary flags for high-fraud types
    df['is_cash_out']  = (df['transaction_type'] == 'CASH_OUT').astype(int)
    df['is_transfer']  = (df['transaction_type'] == 'TRANSFER').astype(int)

    return df

train_fe = engineer_features(train)
test_fe  = engineer_features(test)

FEATURES = [
    # Amount
    'log_amount',
    'amount_to_init_ratio',
    'amount_to_recip_ratio',
    # Balances before
    'log_init_bal_before',
    'log_recip_bal_before',
    # Balances after (as-if-completed signals)
    'log_init_bal_after',
    'log_recip_bal_after',
    # Balance changes
    'log_init_bal_change',
    'log_recip_bal_change',
    # Anomaly features
    'init_amount_mismatch',
    'recip_amount_mismatch',
    'drains_account',
    'init_became_zero',
    'recip_became_zero',
    'init_zero_before',
    'recip_zero_before',
    # Entity type
    'initiator_is_customer',
    'recipient_is_customer',
    'c2c', 'c2m',
    # Time
    'hour_of_day',
    'day_of_month',
    'is_night',
    # Transaction type
    'txn_type_enc',
    'is_cash_out',
    'is_transfer',
    # Existing system flag
    'is_flagged_fraud',
    # Raw time
    'hours_elapsed',
]

TARGET = 'is_fraud'
X_train = train_fe[FEATURES]
y_train = train_fe[TARGET]
X_test  = test_fe[FEATURES]

print(f"  Features ({len(FEATURES)}): {FEATURES}")
print(f"  X_train: {X_train.shape},  X_test: {X_test.shape}")
print(f"  Class balance: {y_train.value_counts().to_dict()}")

# Feature analysis: mean per class
print("\n  Feature means by class:")
means = train_fe.groupby('is_fraud')[FEATURES[:10]].mean().round(4)
print(means)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("4. MODEL TRAINING — LightGBM with 5-fold Stratified CV")
print("="*60)

fraud_n  = int(y_train.sum())
legit_n  = int(len(y_train) - fraud_n)
spw      = legit_n / fraud_n
print(f"  Fraud: {fraud_n:,}  Legit: {legit_n:,}  scale_pos_weight: {spw:.1f}")

lgb_params = dict(
    objective         = 'binary',
    metric            = 'auc',
    boosting_type     = 'gbdt',
    n_estimators      = 300,          # fixed; no early stopping (too few positives)
    learning_rate     = 0.05,
    num_leaves        = 63,
    max_depth         = 6,
    min_child_samples = 2,            # allow small leaf nodes for rare class
    feature_fraction  = 0.8,
    bagging_fraction  = 0.8,
    bagging_freq      = 5,
    lambda_l1         = 1.0,
    lambda_l2         = 1.0,
    is_unbalance      = True,         # auto class weight; skip manual spw
    random_state      = 42,
    verbose           = -1,
    n_jobs            = -1,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_proba   = np.zeros(len(X_train))
fold_aucs   = []
fold_aps    = []

X_train_arr = X_train.values
y_train_arr = y_train.values

for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train_arr, y_train_arr)):
    X_tr, y_tr = X_train_arr[tr_idx], y_train_arr[tr_idx]
    X_va, y_va = X_train_arr[va_idx], y_train_arr[va_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_va)[:, 1]
    oof_proba[va_idx] = proba

    auc = roc_auc_score(y_va, proba)
    ap  = average_precision_score(y_va, proba)
    fold_aucs.append(auc)
    fold_aps.append(ap)
    print(f"  Fold {fold+1}: AUC={auc:.4f}  AP={ap:.4f}")

oof_auc = roc_auc_score(y_train_arr, oof_proba)
oof_ap  = average_precision_score(y_train_arr, oof_proba)
print(f"\n  OOF ROC-AUC          : {oof_auc:.4f}  (folds: {[f'{x:.4f}' for x in fold_aucs]})")
print(f"  OOF Average Precision: {oof_ap:.4f}  (folds: {[f'{x:.4f}' for x in fold_aps]})")

# ── Threshold optimisation: maximize F1 on OOF ───────────────────────────────
precisions, recalls, thresholds = precision_recall_curve(y_train_arr, oof_proba)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)

# Only consider thresholds where we have non-trivial recall (>0.5) and precision (>0.1)
valid_mask = (recalls[:-1] >= 0.5) & (precisions[:-1] >= 0.05)
if valid_mask.any():
    best_idx = np.where(valid_mask)[0][np.argmax(f1_scores[:-1][valid_mask])]
else:
    best_idx = np.argmax(f1_scores[:-1])

best_threshold = float(thresholds[best_idx])
print(f"\n  Best threshold: {best_threshold:.6f}")
print(f"  At threshold → Precision={precisions[best_idx]:.4f}  Recall={recalls[best_idx]:.4f}  F1={f1_scores[best_idx]:.4f}")

oof_preds = (oof_proba >= best_threshold).astype(int)
print("\n  OOF Classification Report:")
print(classification_report(y_train_arr, oof_preds, target_names=['Legitimate', 'Fraud']))

# ── Train final model on ALL training data ────────────────────────────────────
print("  Training final model on full training data…")
final_model = lgb.LGBMClassifier(**lgb_params)
final_model.fit(X_train_arr, y_train_arr)
print("  Done.")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. GENERATE TEST PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("5. PREDICTING ON TEST SET")
print("="*60)

test_proba = final_model.predict_proba(X_test.values)[:, 1]
test_preds = (test_proba >= best_threshold).astype(int)

test_out = test.copy()
test_out['is_fraud_predicted'] = test_preds
test_out.to_parquet(OUT_PATH, index=False)

print(f"  Predicted fraud: {test_preds.sum():,} / {len(test_preds):,} ({test_preds.mean()*100:.3f}%)")
print(f"  Saved to: {OUT_PATH}")
print(f"  Columns: {test_out.columns.tolist()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. EVALUATION PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("6. EVALUATION PLOTS")
print("="*60)

# ── Plot 7: ROC + PR curves ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

fpr, tpr, _ = roc_curve(y_train_arr, oof_proba)
axes[0].plot(fpr, tpr, color=PALETTE['fraud'], linewidth=2,
             label=f'LightGBM (AUC = {oof_auc:.4f})')
axes[0].plot([0,1],[0,1], 'k--', linewidth=1, label='Random classifier')
axes[0].fill_between(fpr, tpr, alpha=0.12, color=PALETTE['fraud'])
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve (5-fold OOF)', fontweight='bold')
axes[0].legend(loc='lower right')
axes[0].grid(alpha=0.3)

axes[1].plot(recalls, precisions, color=PALETTE['fraud'], linewidth=2,
             label=f'LightGBM (AP = {oof_ap:.4f})')
axes[1].axvline(recalls[best_idx], color='gray', linestyle='--', linewidth=1)
axes[1].axhline(precisions[best_idx], color='gray', linestyle=':', linewidth=1)
axes[1].scatter(recalls[best_idx], precisions[best_idx], s=130, color='black', zorder=5,
                label=f'Optimal: P={precisions[best_idx]:.3f}, R={recalls[best_idx]:.3f}, F1={f1_scores[best_idx]:.3f}')
baseline_pr = y_train_arr.mean()
axes[1].axhline(baseline_pr, color='navy', linestyle='--', linewidth=1, alpha=0.5,
                label=f'Random baseline = {baseline_pr:.5f}')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve (5-fold OOF)', fontweight='bold')
axes[1].legend(loc='upper right', fontsize=8)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "07_roc_pr_curves.png", bbox_inches='tight')
plt.close()
print("  Saved: 07_roc_pr_curves.png")

# ── Plot 8: Confusion matrix ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm = confusion_matrix(y_train_arr, oof_preds)
ConfusionMatrixDisplay(cm, display_labels=['Legitimate', 'Fraud']).plot(
    ax=axes[0], cmap='Blues', colorbar=True)
axes[0].set_title('Confusion Matrix — Raw Counts (OOF)', fontweight='bold')

cm_norm = confusion_matrix(y_train_arr, oof_preds, normalize='true')
ConfusionMatrixDisplay(cm_norm, display_labels=['Legitimate', 'Fraud']).plot(
    ax=axes[1], cmap='Blues', colorbar=True, values_format='.2%')
axes[1].set_title('Confusion Matrix — Normalized by True Class (OOF)', fontweight='bold')

plt.tight_layout()
plt.savefig(PLOT_DIR / "08_confusion_matrix.png", bbox_inches='tight')
plt.close()
print("  Saved: 08_confusion_matrix.png")

# ── Plot 9: Feature importance ────────────────────────────────────────────────
importance = pd.Series(final_model.feature_importances_, index=FEATURES).sort_values(ascending=True)
top_n = importance.tail(20)

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#e74c3c' if i >= len(top_n)-5 else '#2980b9'
          for i in range(len(top_n))]
top_n.plot(kind='barh', ax=ax, color=colors, edgecolor='black', linewidth=0.4)
ax.set_title('Top 20 Feature Importances (LightGBM)', fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig(PLOT_DIR / "09_feature_importance.png", bbox_inches='tight')
plt.close()
print("  Saved: 09_feature_importance.png")

# ── Plot 10: Score distribution ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

fraud_scores = oof_proba[y_train_arr == 1]
legit_scores = oof_proba[y_train_arr == 0]

axes[0].hist(legit_scores, bins=100, color=PALETTE['legit'], alpha=0.7,
             label='Legitimate', density=True)
axes[0].hist(fraud_scores, bins=30, color=PALETTE['fraud'], alpha=0.8,
             label='Fraud', density=True)
axes[0].axvline(best_threshold, color='black', linestyle='--', linewidth=2,
                label=f'Threshold = {best_threshold:.4f}')
axes[0].set_title('OOF Predicted Probability by True Class', fontweight='bold')
axes[0].set_xlabel('Predicted Fraud Probability')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].set_yscale('log')

# ── Plot 11: F1 / Precision / Recall vs threshold ────────────────────────────
axes[1].plot(thresholds, f1_scores[:-1], color=PALETTE['fraud'], linewidth=2, label='F1')
axes[1].plot(thresholds, precisions[:-1], color='green', linewidth=1.5, linestyle='--', label='Precision')
axes[1].plot(thresholds, recalls[:-1], color='orange', linewidth=1.5, linestyle='--', label='Recall')
axes[1].axvline(best_threshold, color='black', linestyle=':', linewidth=2,
                label=f'Optimal = {best_threshold:.4f}')
axes[1].set_xlabel('Classification Threshold')
axes[1].set_ylabel('Score')
axes[1].set_title('Precision, Recall, F1 vs Threshold', fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "10_score_distribution_threshold.png", bbox_inches='tight')
plt.close()
print("  Saved: 10_score_distribution_threshold.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
cr_dict = classification_report(y_train_arr, oof_preds, target_names=['Legitimate', 'Fraud'],
                                 output_dict=True)
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"\nDataset")
print(f"  Training records : {len(train):>12,}")
print(f"  Test records     : {len(test):>12,}")
print(f"  Fraud in train   : {fraud_n:>12,}  ({fraud_n/len(train)*100:.4f}%)")

print(f"\nModel: LightGBM (gradient boosted trees)")
print(f"  Validation: 5-fold Stratified Cross-Validation (OOF)")

print(f"\nPerformance (OOF)")
print(f"  ROC-AUC            : {oof_auc:.4f}")
print(f"  Average Precision  : {oof_ap:.4f}")
print(f"  Threshold chosen   : {best_threshold:.6f}")

print(f"\n  Fraud class metrics @ chosen threshold:")
print(f"    Precision  : {cr_dict['Fraud']['precision']:.4f}")
print(f"    Recall     : {cr_dict['Fraud']['recall']:.4f}")
print(f"    F1-score   : {cr_dict['Fraud']['f1-score']:.4f}")
print(f"    Support    : {int(cr_dict['Fraud']['support'])}")

print(f"\n  Legit class metrics:")
print(f"    Precision  : {cr_dict['Legitimate']['precision']:.4f}")
print(f"    Recall     : {cr_dict['Legitimate']['recall']:.4f}")
print(f"    Specificity: {cm[0,0]/(cm[0,0]+cm[0,1]):.4f}")

print(f"\n  Confusion matrix:")
print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
print(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")

print(f"\nTest Predictions")
print(f"  Predicted fraud: {test_preds.sum():,} / {len(test_preds):,} ({test_preds.mean()*100:.3f}%)")
print(f"  Output file    : {OUT_PATH}")
print(f"  Plot directory : {PLOT_DIR}/")

print("\n" + "="*60)
print("DONE")
print("="*60)
