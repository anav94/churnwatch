import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="ChurnWatch – Risk Cohorts & What-If", layout="wide")

# Fixed path → looks for model.joblib inside the app folder
MODEL_PATH = Path(__file__).parent / "model.joblib"
model = joblib.load(MODEL_PATH)

LEAK_COLS = ["Churn","Churn Label","Churn Value","Churn Score","Churn Reason"]
NUM_CANON = ["TotalCharges","Total Charges","MonthlyCharges","Monthly Charges","tenure","Tenure Months"]

def _read_table(file):
    name = file.name.lower() if hasattr(file, "name") else str(file).lower()
    if name.endswith((".xlsx",".xls")):
        return pd.read_excel(file)
    return pd.read_csv(file)

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in LEAK_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()
        df.loc[df[c]=="", c] = np.nan
    for name in NUM_CANON:
        if name in df.columns:
            df[name] = pd.to_numeric(df[name], errors="coerce")
    for c in df.select_dtypes(include=[np.number]).columns:
        med = df[c].median()
        df[c] = df[c].fillna(0 if pd.isna(med) else med)
    for c in df.select_dtypes(exclude=[np.number]).columns:
        if df[c].isna().any():
            mode = df[c].mode(dropna=True)
            df[c] = df[c].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    pre = model.named_steps["preprocessor"]
    exp_cat = list(pre.transformers_[0][2])
    exp_num = list(pre.transformers_[1][2])
    for col in exp_cat:
        if col not in df.columns:
            df[col] = "Unknown"
    for col in exp_num:
        if col not in df.columns:
            df[col] = 0
    return df

st.title("ChurnWatch – Risk Cohorts & What-If")
uploaded = st.file_uploader("Upload a CSV or Excel (Telco Customer Churn)", type=["csv","xlsx","xls"])

if uploaded is not None:
    raw = _read_table(uploaded)
    feats = _clean(raw)
    probs = model.predict_proba(feats)[:,1]
    view = feats.copy()
    view["churn_probability"] = probs
    st.subheader("Top risk cohort")
    st.dataframe(view.sort_values("churn_probability", ascending=False).head(20))

    st.subheader("What-if analysis")
    idx = st.number_input("Row index", min_value=0, max_value=len(feats)-1, value=0, step=1)
    row = feats.iloc[[idx]].copy()

    if "Tenure Months" in row.columns:
        val = float(row.iloc[0]["Tenure Months"])
        row.iloc[0, row.columns.get_loc("Tenure Months")] = st.slider("Tenure Months", 0.0, max(float(val*2), 200.0), val, 1.0)
    if "Monthly Charges" in row.columns:
        val = float(row.iloc[0]["Monthly Charges"])
        row.iloc[0, row.columns.get_loc("Monthly Charges")] = st.slider("Monthly Charges", 0.0, max(float(val*2), 200.0), val, 1.0)
    if "Total Charges" in row.columns:
        val = float(row.iloc[0]["Total Charges"])
        row.iloc[0, row.columns.get_loc("Total Charges")] = st.slider("Total Charges", 0.0, max(float(val*2), 200.0), val, 1.0)

    for col, choices in {
        "Contract": ["Month-to-month","One year","Two year"],
        "Internet Service": ["DSL","Fiber optic","No"],
        "Tech Support": ["No","Yes","No internet service"],
        "Online Security": ["No","Yes","No internet service"],
        "Online Backup": ["No","Yes","No internet service"],
        "Device Protection": ["No","Yes","No internet service"],
        "Paperless Billing": ["No","Yes"],
        "Phone Service": ["No","Yes"]
    }.items():
        if col in row.columns:
            cur = str(row.iloc[0][col]) if pd.notna(row.iloc[0][col]) else choices[0]
            row.iloc[0, row.columns.get_loc(col)] = st.selectbox(col, options=choices, index=choices.index(cur) if cur in choices else 0)

    new_prob = float(model.predict_proba(row)[:,1][0])
    st.metric("New churn probability", f"{new_prob:.3f}")
else:
    st.info("Upload the Telco churn file to see cohorts and what-if analysis.")
