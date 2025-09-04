from __future__ import annotations
import os
import argparse
import joblib
import json
import numpy as np
import pandas as pd
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
from .data_prep import load_telco, train_test, build_preprocessor
from .utils import action_playbook

def make_model(name: str):
    name = name.lower()
    if name in ["logreg", "logistic", "lr"]:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    elif name in ["xgb", "xgboost"]:
        clf = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=0,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    return clf

def log_metrics(y_true, y_proba):
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
    }

def save_confusion_plot(y_true, y_proba, out_path: str):
    y_pred = (y_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha='center', va='center')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def shap_summary(pipeline: Pipeline, X_sample: pd.DataFrame, out_png: str, max_display=12):
    pre = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    Xtx = pre.transform(X_sample)
    cat = pre.named_transformers_["cat"].get_feature_names_out()
    num = pre.transformers_[1][2]
    feat_names = list(cat) + list(num)
    explainer = shap.Explainer(model, Xtx)
    vals = explainer(Xtx)
    shap.summary_plot(vals.values, feature_names=feat_names, show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def write_playbook(top_features: pd.Series, out_csv: str):
    rows = []
    for fname, score in top_features.items():
        suggestion = action_playbook.get(fname.split("=")[0], "Retention call with tailored offer.")
        rows.append({"feature": fname, "importance": float(score), "suggested_action": suggestion})
    pd.DataFrame(rows).to_csv(out_csv, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--experiment', default='ChurnWatch')
    ap.add_argument('--model', default='xgb', choices=['xgb','logreg'])
    ap.add_argument('--test_size', type=float, default=0.2)
    ap.add_argument('--random_state', type=int, default=42)
    ap.add_argument('--mlflow_uri', default='./mlruns')
    args = ap.parse_args()

    os.makedirs('artifacts', exist_ok=True)

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    df = load_telco(args.data_path)
    (X_train, X_test, y_train, y_test), cat_cols, num_cols = train_test(df, args.test_size, args.random_state)

    pre = build_preprocessor(cat_cols, num_cols)
    model = make_model(args.model)

    pipe = Pipeline([
        ('preprocessor', pre),
        ('model', model)
    ])

    with mlflow.start_run() as run:
        pipe.fit(X_train, y_train)
        proba_val = pipe.predict_proba(X_test)[:,1]
        metrics = log_metrics(y_test, proba_val)
        for k,v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.log_params({
            'model': args.model,
            'test_size': args.test_size,
            'random_state': args.random_state
        })
        # SHAP summary
        sample = X_test.sample(min(500, len(X_test)), random_state=args.random_state)
        shap_png = 'artifacts/shap_summary.png'
        shap_summary(pipe, sample, shap_png)
        mlflow.log_artifact(shap_png)
        # Top features by mean |SHAP|
        pre_tx = pipe.named_steps['preprocessor']
        feat_names = list(pre_tx.named_transformers_['cat'].get_feature_names_out()) + list(pre_tx.transformers_[1][2])
        expl = shap.Explainer(pipe.named_steps['model'], pre_tx.transform(sample))
        shap_vals = expl(pre_tx.transform(sample)).values
        import numpy as np
        mean_abs = np.abs(shap_vals).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:12]
        top_series = pd.Series(mean_abs[top_idx], index=[feat_names[i] for i in top_idx])
        play_csv = 'artifacts/eligibility_playbook.csv'
        write_playbook(top_series, play_csv)
        mlflow.log_artifact(play_csv)
        # Confusion matrix plot
        cm_png = 'artifacts/confusion_matrix.png'
        save_confusion_plot(y_test.values, proba_val, cm_png)
        mlflow.log_artifact(cm_png)
        # Persist pipeline for API/Streamlit
        joblib.dump(pipe, 'artifacts/model.joblib')
        mlflow.sklearn.log_model(pipe, artifact_path='model')
        print(json.dumps({"run_id": run.info.run_id, **metrics}, indent=2))

if __name__ == '__main__':
    main()
