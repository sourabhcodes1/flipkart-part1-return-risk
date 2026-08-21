# Flipkart Return-Risk Scoring Pipeline (Part 1)

This is Part 1 of the Flipkart Order Intelligence & Support Assistant project:
a return-risk model trained on a deterministic, seeded synthetic order history.
Its saved artifact (`models/return_risk_model.pkl`) is the fixed input consumed
by Part 3's `check_return_risk` tool.

## Contents

- `generate_orders.py` — deterministic dataset generator (`np.random.default_rng(42)`,
  fixed category/payment lists). Do not modify.
- `orders_dataset.csv` — the generated dataset (6,000 rows, 13 columns).
- `train_return_risk.py` — the full pipeline: data verification, leakage-free
  preprocessing, Dummy baseline, Logistic Regression + threshold sweep,
  Random Forest + `GridSearchCV`, feature importance + permutation importance,
  subgroup analysis, and final artifact save.
- `report.md` — full captured console output from the last run (all metrics,
  tables, and written analysis paragraphs required by the brief).
- `models/return_risk_model.pkl` — the final, tuned Random Forest
  `Pipeline` (preprocessing + model together), saved with `joblib.dump`.
- `models/risk_thresholds.json` — `t*_rf` (the F1-maximising threshold on the
  saved Random Forest's own `predict_proba`) and the risk-bucket cut points
  derived from it, for Part 3 to load without recomputing.

## How to reproduce

```bash
pip install scikit-learn pandas numpy joblib

# 1. Regenerate the exact seeded dataset
python3 generate_orders.py

# 2. Run the full training/evaluation pipeline
python3 train_return_risk.py
```

Running `train_return_risk.py` will:
1. Print/verify dataset stats (row count, return rate, missingness) and
   classify the `rating_given` missingness mechanism.
2. Build a `ColumnTransformer` + `Pipeline` (median/mode impute → one-hot →
   scale), fit only on the training split.
3. Train and report a `DummyClassifier` baseline.
4. Train a `class_weight="balanced"` Logistic Regression, report default-threshold
   metrics, then sweep thresholds 0.10–0.90 (step 0.02) and report the
   F1-maximising threshold and its trade-offs.
5. Run `GridSearchCV` (5-fold `StratifiedKFold`, scored on `roc_auc`) over a
   Random Forest, report the best params/CV AUC/test AUC.
6. Report top-5 `.feature_importances_`, then compute
   `permutation_importance` on the held-out test split for the same features
   and compare rankings.
7. Break out recall/precision by `product_category` and `payment_method`,
   flag the weakest subgroup, and propose a concrete fix.
8. Re-run the threshold sweep on the **Random Forest's own** `predict_proba`
   (not the Logistic Regression's) to compute `t*_rf`, then save the final
   pipeline to `models/return_risk_model.pkl` and the risk-bucket config to
   `models/risk_thresholds.json`. A sanity check reloads the saved model via
   `joblib.load` and confirms its `predict_proba` output on the test split is
   bit-identical to the in-memory model.

All of the above output — every table, metric, and written paragraph the
brief requires — is captured verbatim in `report.md`.

## Headline results (from the last run — see `report.md` for full detail)

- Dataset: 6,000 rows, 13 columns, return rate ≈ 22.75%, `rating_given`
  missing on ≈ 13.05% of rows (MAR, conditional on `payment_method`).
- Dummy baseline: accuracy ≈ 77%, F1 (class 1) = **0.0** — the "high
  accuracy, zero recall" trap.
- Logistic Regression (default threshold): ROC-AUC ≈ 0.625, F1 ≈ 0.392.
- Best Random Forest (`GridSearchCV`): `max_depth=6, n_estimators=100`,
  CV ROC-AUC ≈ 0.618, test ROC-AUC ≈ 0.614 (gap ≈ 0.004 — no severe overfitting).
- Top-5 impurity features: `payment_method_COD`, `price_inr`,
  `customer_tenure_days`, `delivery_distance_km`, `discount_pct`.
  Permutation importance shows `customer_tenure_days`, `delivery_distance_km`,
  and `discount_pct` collapse to ~zero/negative real signal on held-out data —
  `payment_method` and (to a smaller degree) `price_inr` are the features that
  actually generalize.
- Weakest subgroup: `Electronics` (recall well below the overall average) —
  proposed fix is a category-specific interaction feature plus a
  category-specific decision threshold.
- `t*_rf ≈ 0.46`; risk buckets: **Low** if `proba < 0.46`, **High** if
  `proba ≥ 0.61`, else **Medium**.

## Git workflow

Work was done on `feature/return-risk-pipeline` (created off `main`) with
commits for the generator, the dataset, the pipeline, the report, and the
saved artifacts, then merged back into `main` — visible via
`git log --graph --all`.

## Deployment

See the "How to deploy this" section at the end of the project write-up /
chat response for options (containerized REST API, batch scoring job, or as
a tool inside the Part 3 LangGraph agent).
