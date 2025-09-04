import argparse, os
import pandas as pd
import joblib
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPerformancePreset

DEFAULT_OUT = "monitoring/reports/report.html"

def _read_table(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)

def _ensure_churn(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Churn" not in df.columns:
        if "Churn Value" in df.columns:
            df["Churn"] = pd.to_numeric(df["Churn Value"], errors="coerce")
        elif "Churn Label" in df.columns:
            df["Churn"] = (
                df["Churn Label"].astype(str).str.strip().str.title().map({"Yes":1,"No":0})
            )
    df["Churn"] = pd.to_numeric(df["Churn"], errors="coerce")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_csv", required=True)
    ap.add_argument("--current_csv", required=True)
    ap.add_argument("--model_path", default="artifacts/model.joblib")
    ap.add_argument("--out_html", default=DEFAULT_OUT)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_html), exist_ok=True)

    ref = _read_table(args.reference_csv)
    cur = _read_table(args.current_csv)

    ref = _ensure_churn(ref)
    cur = _ensure_churn(cur)

    common = [c for c in ref.columns if c in cur.columns]
    ref = ref[common].copy()
    cur = cur[common].copy()

    try:
        model = joblib.load(args.model_path)
        feat_cols = [c for c in common if c != "Churn"]
        if feat_cols:
            ref["prediction"] = model.predict_proba(ref[feat_cols])[:, 1]
            cur["prediction"] = model.predict_proba(cur[feat_cols])[:, 1]
    except Exception:
        pass

    report = Report(metrics=[
        DataDriftPreset(),
        ClassificationPerformancePreset(target_column="Churn", prediction_column="prediction"),
    ])
    report.run(reference_data=ref, current_data=cur)
    report.save_html(args.out_html)
    print(args.out_html)

if __name__ == "__main__":
    main()
