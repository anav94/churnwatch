from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "Churn"
LEAK_COLS = ["Churn Label","Churn Value","Churn Score","Churn Reason"]
CAT_HINT = ["gender","SeniorCitizen","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"]
NUM_HINT = ["tenure","MonthlyCharges","TotalCharges","Tenure Months","Monthly Charges","Total Charges"]

def _read_table(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)

def load_telco(path: str) -> pd.DataFrame:
    df = _read_table(path)
    if TARGET_COL not in df.columns:
        low = {c: c.strip().lower() for c in df.columns}
        alts = [c for c in df.columns if low[c] in ("churn","churn value","churn label","exited")]
        if alts:
            df = df.rename(columns={alts[0]: TARGET_COL})
    if TARGET_COL in df.columns:
        if df[TARGET_COL].dtype == object:
            s = df[TARGET_COL].astype(str).str.strip().str.title()
            df[TARGET_COL] = s.map({"Yes": 1, "No": 0})
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
        df = df.dropna(subset=[TARGET_COL])
        df[TARGET_COL] = df[TARGET_COL].astype("Int64")
    for name in ["TotalCharges","Total Charges"]:
        if name in df.columns:
            df[name] = pd.to_numeric(df[name], errors="coerce")
    for idcol in ["customerID","CustomerID"]:
        if idcol in df.columns:
            df = df.drop(columns=[idcol])
    for c in df.select_dtypes(include=[np.number]).columns:
        med = df[c].median()
        df[c] = df[c].fillna(0 if pd.isna(med) else med)
    for c in df.select_dtypes(exclude=[np.number]).columns:
        if not df[c].empty:
            df[c] = df[c].fillna(df[c].mode().iloc[0])
    return df

def split_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    drop_cols = [TARGET_COL] + [c for c in LEAK_COLS if c in df.columns]
    y = df[TARGET_COL]
    X = df.drop(columns=drop_cols)
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or c in CAT_HINT]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return X, y, cat_cols, num_cols

def build_preprocessor(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                              ("num", StandardScaler(), num_cols)])

def train_test(df: pd.DataFrame, test_size=0.2, random_state=42):
    X, y, cat_cols, num_cols = split_features(df)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y), cat_cols, num_cols
