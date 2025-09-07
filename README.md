# 📊 ChurnWatch – Explainable Churn Prediction

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?logo=streamlit)](https://anav94-churnwatch.streamlit.app)
[![GitHub Actions](https://github.com/anav94/churnwatch/actions/workflows/ci.yml/badge.svg)](https://github.com/anav94/churnwatch/actions)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> End-to-end telecom churn prediction system with explainability, monitoring, and production-grade serving.

---

## 🚀 Overview

ChurnWatch is a full ML pipeline built on the **IBM Telco Customer Churn dataset**.  
It demonstrates **end-to-end ML ownership** — from data prep, modeling, and explainability to deployment and monitoring.

Key highlights:
- Clean & preprocess customer churn dataset (Excel/CSV support).
- Train baseline models (LogReg, XGBoost) with hyperparam tuning.
- Track 100+ experiments in **MLflow**.
- Add **SHAP** explanations + “eligibility playbook” for customer-level actions.
- Deploy a **FastAPI** scoring service (Docker-ready).
- Integrate **Evidently** reports for data & performance drift.
- Ship an interactive **Streamlit app** for risk cohorts & what-if analysis.

---

## 📸 Screenshots

| Cohort View | What-If Analysis |
|-------------|------------------|
| ![cohorts](docs/img/cohort.png) | ![whatif](docs/img/whatif.png) |

---

## 🛠️ Tech Stack

- **Python 3.12**  
- **ML / AI:** scikit-learn, XGBoost, SHAP  
- **Experiment Tracking:** MLflow  
- **Monitoring:** Evidently  
- **Serving:** FastAPI, Uvicorn, Docker  
- **App:** Streamlit  
- **Data:** IBM Telco Customer Churn (Excel/CSV)  

---

## 📂 Project Structure

