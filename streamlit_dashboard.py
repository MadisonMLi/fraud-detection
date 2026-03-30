"""
Fraud Detection — Interactive Streamlit Dashboard
"""
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

BASE     = Path(__file__).parent
PLOT_DIR = BASE / "plots"

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Fraud Detection Dashboard")
st.caption("Nasdaq Interview — Fraud Analysis · Train: 3.1M transactions · Test: 2.2M transactions")

# ── Load data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    train = pd.read_parquet(BASE / "train_fraud.parquet")
    test  = pd.read_parquet(BASE / "test_fraud_external_predicted.parquet")
    return train, test

train, test = load_data()
fraud = train[train['is_fraud'] == 1]
legit = train[train['is_fraud'] == 0]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to", [
    "📊 Dataset Overview",
    "🔎 EDA Plots",
    "🤖 Model Performance",
    "🔮 Test Predictions",
])

# ═══════════════════════════════════════════════════════════════════════════════
if section == "📊 Dataset Overview":
    st.header("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training Records",   f"{len(train):,}")
    col2.metric("Fraud Cases",        f"{train['is_fraud'].sum():,}")
    col3.metric("Fraud Rate",         f"{train['is_fraud'].mean()*100:.4f}%")
    col4.metric("Existing Flag Recall", "0.56%")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Transaction Type Distribution")
        type_fraud = train.groupby('transaction_type')['is_fraud'].agg(['sum','count','mean'])
        type_fraud.columns = ['Fraud Count', 'Total', 'Fraud Rate']
        type_fraud['Fraud Rate %'] = (type_fraud['Fraud Rate'] * 100).round(4)
        st.dataframe(type_fraud[['Total','Fraud Count','Fraud Rate %']].sort_values('Fraud Count', ascending=False))

    with col2:
        st.subheader("Amount Statistics by Class")
        amt_stats = train.groupby('is_fraud')['transaction_amount'].describe()
        amt_stats.index = ['Legitimate', 'Fraud']
        st.dataframe(amt_stats[['mean','50%','75%','max']].rename(columns={'50%':'median'}).style.format("${:,.0f}"))

    st.divider()
    st.subheader("Raw Training Data Sample")
    st.dataframe(train.sample(200, random_state=42), height=300)

# ═══════════════════════════════════════════════════════════════════════════════
elif section == "🔎 EDA Plots":
    st.header("🔎 Exploratory Data Analysis")

    plots = {
        "01 — Class Distribution":        "01_class_distribution.png",
        "02 — Transaction Type":          "02_transaction_type.png",
        "03 — Amount Distribution":       "03_amount_distribution.png",
        "04 — Time Analysis":             "04_time_analysis.png",
        "05 — Balance Analysis":          "05_balance_analysis.png",
        "06 — Flag System Effectiveness": "06_flag_vs_actual.png",
    }

    selected = st.selectbox("Select plot", list(plots.keys()))
    img_path = PLOT_DIR / plots[selected]

    if img_path.exists():
        img = Image.open(img_path)
        st.image(img, use_container_width=True)
    else:
        st.warning(f"Plot not found: {img_path}. Run fraud_analysis.py first.")

    st.divider()
    st.subheader("Key EDA Findings")
    findings = {
        "Account Drain":      "97.8% of fraud transactions empty the account completely (amount ≈ balance before). Only 0.001% of legit do this.",
        "Transaction Types":  "100% of fraud is CASH_OUT or TRANSFER. Zero fraud in PAYMENT, CASH_IN, or DEBIT.",
        "Amount Size":        "Fraud amounts are 7× larger on average ($1.52M vs $217K).",
        "Entity Type":        "100% of fraud is Customer → Customer. No merchants ever involved.",
        "Night-time":         "36% of fraud at night (10pm–6am) vs only 6.3% of legit.",
        "Existing Flag":      "Current system catches only 2 of 357 frauds (0.56% recall). Essentially useless.",
        "Repeat Offenders":   "Every fraud has a unique sender and receiver — no repeat accounts.",
    }
    for title, finding in findings.items():
        st.info(f"**{title}:** {finding}")

# ═══════════════════════════════════════════════════════════════════════════════
elif section == "🤖 Model Performance":
    st.header("🤖 Model Performance — LightGBM (5-fold OOF)")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("ROC-AUC",          "1.0000")
    col2.metric("Avg Precision",    "0.6837")
    col3.metric("Fraud Recall",     "99.72%")
    col4.metric("Fraud Precision",  "72.95%")
    col5.metric("F1-Score",         "0.843")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm_data = pd.DataFrame({
            "Predicted Legit": [3143608, 1],
            "Predicted Fraud": [132, 356]
        }, index=["Actual Legit", "Actual Fraud"])
        st.dataframe(cm_data.style.highlight_max(axis=None, color='#d4edda')
                                   .format("{:,}"))
        st.caption("TN=3,143,608 | FP=132 | FN=1 | TP=356")

    with col2:
        st.subheader("Fold-level Performance")
        fold_df = pd.DataFrame({
            "Fold": [1, 2, 3, 4, 5],
            "AUC":  [1.0000, 1.0000, 1.0000, 1.0000, 0.9999],
            "AP":   [0.9998, 1.0000, 1.0000, 0.9316, 0.3522],
        })
        st.dataframe(fold_df.style.background_gradient(subset=['AUC','AP'], cmap='RdYlGn'))
        st.caption("Fold 5 AP=0.35 caused by 127 legit transactions scoring identically to fraud")

    st.divider()

    plots = {
        "ROC + PR Curves":     "07_roc_pr_curves.png",
        "Confusion Matrix":    "08_confusion_matrix.png",
        "Feature Importance":  "09_feature_importance.png",
        "Score Distribution":  "10_score_distribution_threshold.png",
    }
    selected = st.selectbox("Select evaluation plot", list(plots.keys()))
    img_path = PLOT_DIR / plots[selected]
    if img_path.exists():
        st.image(Image.open(img_path), use_container_width=True)
    else:
        st.warning(f"Plot not found. Run fraud_analysis.py first.")

# ═══════════════════════════════════════════════════════════════════════════════
elif section == "🔮 Test Predictions":
    st.header("🔮 Test Set Predictions")

    col1, col2, col3 = st.columns(3)
    col1.metric("Test Records",      f"{len(test):,}")
    col2.metric("Predicted Fraud",   f"{test['is_fraud_predicted'].sum():,}")
    col3.metric("Predicted Rate",    f"{test['is_fraud_predicted'].mean()*100:.3f}%")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Predicted Fraud by Transaction Type")
        type_pred = test.groupby('transaction_type')['is_fraud_predicted'].agg(['sum','count','mean'])
        type_pred.columns = ['Predicted Fraud', 'Total', 'Rate']
        type_pred['Rate %'] = (type_pred['Rate'] * 100).round(3)
        st.dataframe(type_pred[['Total','Predicted Fraud','Rate %']].sort_values('Predicted Fraud', ascending=False))

    with col2:
        st.subheader("Predicted Fraud Amount Stats")
        pred_fraud = test[test['is_fraud_predicted'] == 1]
        pred_legit = test[test['is_fraud_predicted'] == 0]
        amt_comp = pd.DataFrame({
            'Predicted Fraud': pred_fraud['transaction_amount'].describe(),
            'Predicted Legit': pred_legit['transaction_amount'].describe(),
        })
        st.dataframe(amt_comp.style.format("${:,.2f}"))

    st.divider()
    st.subheader("Flagged Transactions (Predicted Fraud)")
    flagged = test[test['is_fraud_predicted'] == 1].sort_values('transaction_amount', ascending=False)
    st.dataframe(flagged.head(500), height=400)

    st.download_button(
        label="⬇️ Download Full Predictions as CSV",
        data=test.to_csv(index=False),
        file_name="test_fraud_external_predicted.csv",
        mime="text/csv",
    )
