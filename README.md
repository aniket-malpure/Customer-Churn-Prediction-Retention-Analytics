# Customer Churn Prediction & Retention Analytics

## Overview
End-to-end ML system predicting credit card customer churn using gradient boosting,
K-Means customer segmentation, and a causal inference A/B testing framework with
CUPED variance reduction — built to reflect real-world bank/merchant analytics at Visa.

## Architecture
```
data_generation.py     →  customers_raw.parquet       (200k synthetic customers)
feature_engineering.py →  customers_features.parquet  (27 engineered features)
model_training.py      →  models/churn_model.pkl      (best of 6 models)
segmentation.py        →  customers_segmented.parquet (5 K-Means cohorts)
ab_testing.py          →  outputs/ab_test_*.csv/png   (causal inference)
dashboard.py           →  outputs/churn_dashboard.png
pipeline.py            →  orchestrates all steps
```

## Key Results
| Metric           | Value     |
|------------------|-----------|
| Best Model       | LightGBM  |
| ROC-AUC          | ~0.91     |
| F1-Score         | ~0.74     |
| Churn Reduction  | ~15% (A/B)|
| Customer Segments| 5         |

## Tech Stack
- **ML**: LightGBM, XGBoost, scikit-learn (6 models compared)
- **Clustering**: K-Means + PCA + Silhouette selection
- **Causal Inference**: A/B testing, two-proportion z-test, bootstrap CI, CUPED
- **Tracking**: MLflow
- **Visualization**: Matplotlib, Scipy KDE

## Quick Start
```bash
pip install -r requirements.txt
python pipeline.py
# or to skip data re-generation:
python pipeline.py --skip-data
```

## Module Details

### `feature_engineering.py`
- **RFM scoring** — Recency, Frequency, Monetary proxies
- **Spend CoV** — coefficient of variation as engagement decay signal
- **Utilisation bands** — ordinal bucketing of credit utilisation
- **Interaction features** — high_util_no_rewards, inactive_mobile, etc.

### `segmentation.py`
| Segment | Label | Churn Risk |
|---------|-------|------------|
| 0 | High-Value Active | Low |
| 1 | Dormant Low-Spend | High |
| 2 | Young Digital-First | Medium |
| 3 | Revolving Debt-Prone | High |
| 4 | Mature Occasional | Medium |

### `ab_testing.py`
- **RCT simulation** — stratified 50/50 treatment/control split
- **Balance check** — Standardised Mean Difference (SMD < 0.1 = balanced)
- **ATE estimation** — difference-in-means + 2000-iteration bootstrap CI
- **CUPED** — pre-experiment covariate adjustment for variance reduction
- **HTE** — heterogeneous treatment effects by customer segment

## Outputs
| File | Description |
|------|-------------|
| `outputs/churn_dashboard.png` | 5-panel retention analytics dashboard |
| `outputs/customer_segments.png` | PCA cluster scatter plot |
| `outputs/segment_profiles.csv` | Segment-level churn + spend profiles |
| `outputs/ab_test_results.png` | Campaign lift visualisation |
| `outputs/ab_test_summary.csv` | Raw vs CUPED ATE comparison |
| `outputs/hte_by_segment.csv` | Heterogeneous treatment effects |
| `outputs/model_comparison.csv` | 6-model AUC/PR-AUC comparison |
| `outputs/balance_check.csv` | Covariate balance table |
