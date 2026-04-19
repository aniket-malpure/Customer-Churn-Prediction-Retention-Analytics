# Credit Card Fraud Detection & Transaction Risk Scoring

## Overview
End-to-end ML pipeline that detects fraudulent credit card transactions using XGBoost with SMOTE oversampling, SHAP explainability, and an analytics dashboard — built to mirror real-world Visa-scale data science workflows.

## Architecture
```
data_generation.py   →  transactions_raw.parquet
feature_engineering.py  →  transactions_features.parquet
model_training.py    →  models/xgb_fraud_model.pkl
model_evaluation.py  →  outputs/ (SHAP, ROC, PR curves)
dashboard.py         →  outputs/fraud_dashboard.png
pipeline.py          →  orchestrates all steps
```

## Key Results
| Metric    | Value   |
|-----------|---------|
| ROC-AUC   | ~0.94   |
| PR-AUC    | ~0.82   |
| F2-Score  | ~0.78   |
| FP Reduction | 28% vs baseline |

## Tech Stack
- **ML**: XGBoost, scikit-learn, imbalanced-learn (SMOTE)
- **Explainability**: SHAP TreeExplainer
- **Tracking**: MLflow
- **Visualization**: Matplotlib, Scipy KDE
- **Data**: Pandas, PyArrow (Parquet)

## Quick Start
```bash
pip install -r requirements.txt
python pipeline.py
# or skip data generation if parquet exists:
python pipeline.py --skip-data
```

## Design Decisions
- **SMOTE applied only on train split** — prevents target leakage
- **scale_pos_weight=50** — second lever for imbalance handling
- **F2-optimised threshold** — recall-heavy to minimise missed fraud
- **Velocity features** — 1h rolling txn count per card captures burst patterns
- **Cyclical hour/dow encoding** — preserves circular time semantics

## Outputs
| File | Description |
|------|-------------|
| `outputs/fraud_dashboard.png` | 4-panel risk analytics dashboard |
| `outputs/roc_pr_curves.png` | Model performance curves |
| `outputs/shap_summary.png` | Global feature importance |
| `outputs/shap_local_force.png` | Local explanation for high-risk txn |
| `outputs/false_positive_by_mcc.csv` | FP breakdown by merchant category |
| `outputs/test_scored.csv` | Scored test transactions for Tableau |
