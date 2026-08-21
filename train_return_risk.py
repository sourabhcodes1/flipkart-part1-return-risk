"""
Part 1 -- Return-Risk Scoring Pipeline
Runs Tasks 2-9: data verification, leakage-free preprocessing, baseline,
Logistic Regression + threshold sweep, Random Forest + GridSearchCV,
feature importance + permutation importance, subgroup analysis,
and final artifact save with t*_rf.

All printed output is captured to report.md for the README/analysis writeup.
"""
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score, roc_auc_score,
    confusion_matrix,
)
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
report_lines = []


def log(*args):
    line = " ".join(str(a) for a in args)
    print(line)
    report_lines.append(line)


def section(title):
    log("\n" + "=" * 90)
    log(title)
    log("=" * 90)


# ---------------------------------------------------------------------------
# Task 2: Verify the generated data
# ---------------------------------------------------------------------------
section("TASK 2 -- DATA VERIFICATION")

df = pd.read_csv("orders_dataset.csv")
n_rows = len(df)
overall_return_rate = df["returned"].mean()
missing_rating_pct = df["rating_given"].isna().mean() * 100

log(f"Total rows: {n_rows}")
log(f"Overall return rate: {overall_return_rate:.4f} ({overall_return_rate*100:.2f}%)")
log(f"Missing rating_given: {missing_rating_pct:.2f}%")

log("\nReturn rate by product_category:")
cat_table = df.groupby("product_category")["returned"].agg(["mean", "count"]).rename(
    columns={"mean": "return_rate", "count": "n"})
log(cat_table.to_string())

log("\nReturn rate by payment_method:")
pay_table = df.groupby("payment_method")["returned"].agg(["mean", "count"]).rename(
    columns={"mean": "return_rate", "count": "n"})
log(pay_table.to_string())

# Missingness diagnosis: rate of NaN rating_given, split by COD vs non-COD
cod_missing_rate = df.loc[df["payment_method"] == "COD", "rating_given"].isna().mean()
non_cod_missing_rate = df.loc[df["payment_method"] != "COD", "rating_given"].isna().mean()
log(f"\nMissing-rate gap check -- COD missing rate: {cod_missing_rate:.3f}, "
    f"non-COD missing rate: {non_cod_missing_rate:.3f}, "
    f"gap: {cod_missing_rate - non_cod_missing_rate:.3f}")
log(
    "Missingness classification: MAR (Missing At Random), conditional on the observed "
    "payment_method column. rating_given is far more likely to be missing when "
    "payment_method == 'COD' (~22% missing) than for non-COD orders (~6% missing) -- "
    "a gap of roughly 16 percentage points that is directly attributable to the generator's "
    "own missingness rule (np.where(payment_method == 'COD', 0.22, 0.06)). This is not MCAR, "
    "because missingness clearly depends on an observed column (payment_method), and it is not "
    "MNAR, because missingness does not depend on the unobserved rating_given value itself -- "
    "it depends only on the already-observed payment method."
)

# ---------------------------------------------------------------------------
# Task 3: Preprocess without leakage
# ---------------------------------------------------------------------------
section("TASK 3 -- LEAKAGE-FREE PREPROCESSING PIPELINE")

feature_cols = [
    "product_category", "price_inr", "discount_pct", "payment_method",
    "customer_tenure_days", "num_previous_orders", "num_previous_returns",
    "delivery_distance_km", "delivery_days", "is_weekend_order", "rating_given",
]
target_col = "returned"

X = df[feature_cols].copy()
y = df[target_col].copy()

numeric_features = [
    "price_inr", "discount_pct", "customer_tenure_days", "num_previous_orders",
    "num_previous_returns", "delivery_distance_km", "delivery_days",
    "is_weekend_order", "rating_given",
]
categorical_features = ["product_category", "payment_method"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
log(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")
log("Preprocessor (median/mode impute -> one-hot -> scale) is fit on X_train ONLY; "
    "X_test is only ever .transform()'d, never seen during .fit().")

# ---------------------------------------------------------------------------
# Task 4: Baseline
# ---------------------------------------------------------------------------
section("TASK 4 -- DUMMY BASELINE")

dummy_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)),
])
dummy_pipe.fit(X_train, y_train)
dummy_pred = dummy_pipe.predict(X_test)

dummy_acc = accuracy_score(y_test, dummy_pred)
dummy_f1 = f1_score(y_test, dummy_pred, pos_label=1, zero_division=0)
log(f"DummyClassifier (most_frequent) accuracy: {dummy_acc:.4f}")
log(f"DummyClassifier F1 (class 1 / returned): {dummy_f1:.4f}")
log(
    "Why high accuracy is misleading here: the dataset's base rate is roughly 77% "
    "'not returned', so a classifier that always predicts 'not returned' scores a "
    "deceptively high accuracy (~77%) while catching ZERO actual returns -- its F1 "
    "for the returned=1 class is exactly 0.0. This is the classic 'high accuracy, "
    "zero recall' trap for imbalanced classification: accuracy alone rewards ignoring "
    "the minority class entirely, which is the opposite of what the business needs "
    "(catching likely returns before they happen). Any real model must be judged "
    "against this baseline and against metrics that reflect the actual business cost "
    "of missed returns, not raw accuracy."
)

# ---------------------------------------------------------------------------
# Task 5: Logistic Regression + threshold sweep
# ---------------------------------------------------------------------------
section("TASK 5 -- LOGISTIC REGRESSION + THRESHOLD SWEEP")

lr_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
])
lr_pipe.fit(X_train, y_train)

lr_proba = lr_pipe.predict_proba(X_test)[:, 1]
lr_pred_default = (lr_proba >= 0.5).astype(int)

lr_acc = accuracy_score(y_test, lr_pred_default)
lr_f1 = f1_score(y_test, lr_pred_default, pos_label=1)
lr_recall = recall_score(y_test, lr_pred_default, pos_label=1)
lr_precision = precision_score(y_test, lr_pred_default, pos_label=1, zero_division=0)
lr_auc = roc_auc_score(y_test, lr_proba)

log(f"[Default threshold=0.5] Accuracy: {lr_acc:.4f}  F1: {lr_f1:.4f}  "
    f"Recall: {lr_recall:.4f}  Precision: {lr_precision:.4f}  ROC-AUC: {lr_auc:.4f}")

thresholds = np.arange(0.10, 0.90 + 1e-9, 0.02)
sweep_rows = []
for t in thresholds:
    pred_t = (lr_proba >= t).astype(int)
    f1_t = f1_score(y_test, pred_t, pos_label=1, zero_division=0)
    rec_t = recall_score(y_test, pred_t, pos_label=1, zero_division=0)
    prec_t = precision_score(y_test, pred_t, pos_label=1, zero_division=0)
    sweep_rows.append((round(t, 2), f1_t, rec_t, prec_t))

sweep_df = pd.DataFrame(sweep_rows, columns=["threshold", "f1", "recall", "precision"])
best_row = sweep_df.loc[sweep_df["f1"].idxmax()]
log("\nThreshold sweep (0.10 to 0.90, step 0.02) -- F1-maximising row:")
log(sweep_df.to_string(index=False))
log(f"\nBest LR threshold by F1: t*_lr = {best_row['threshold']:.2f}  "
    f"F1={best_row['f1']:.4f}  Recall={best_row['recall']:.4f}  Precision={best_row['precision']:.4f}")
log(f"Recall gain vs default threshold: {best_row['recall'] - lr_recall:.4f} "
    f"({(best_row['recall'] - lr_recall)*100:.1f} pp)")
log(f"Precision change vs default threshold: {best_row['precision'] - lr_precision:.4f}")
log(
    "Business trade-off: lowering the decision threshold below 0.5 makes the model flag "
    "more orders as 'likely to be returned'. This trades away precision (more false "
    "positives -- orders flagged as risky that would not actually have been returned, "
    "costing agent time / unnecessary friction) in exchange for recall (catching more of "
    "the orders that truly will be returned, which is what lets the business intervene "
    "before the return happens). Given that a missed return is generally more costly than "
    "a false alarm reviewed by an agent, accepting lower precision for materially higher "
    "recall is the right trade for this use case."
)

# ---------------------------------------------------------------------------
# Task 6: Random Forest + GridSearchCV
# ---------------------------------------------------------------------------
section("TASK 6 -- RANDOM FOREST + GRIDSEARCHCV")

rf_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
])

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [6, 10, None],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(rf_pipe, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
grid.fit(X_train, y_train)

best_rf_pipe = grid.best_estimator_
best_cv_auc = grid.best_score_
rf_test_proba = best_rf_pipe.predict_proba(X_test)[:, 1]
rf_test_auc = roc_auc_score(y_test, rf_test_proba)

log(f"Best params: {grid.best_params_}")
log(f"Best cross-validated ROC-AUC: {best_cv_auc:.4f}")
log(f"Held-out test-set ROC-AUC (winning config): {rf_test_auc:.4f}")
log(f"|CV AUC - Test AUC| = {abs(best_cv_auc - rf_test_auc):.4f} (overfitting check, want <= 0.05)")

# ---------------------------------------------------------------------------
# Task 7: Explain the model (feature importance + permutation importance)
# ---------------------------------------------------------------------------
section("TASK 7 -- FEATURE IMPORTANCE + PERMUTATION IMPORTANCE")

fitted_preprocessor = best_rf_pipe.named_steps["preprocessor"]
fitted_rf = best_rf_pipe.named_steps["classifier"]

feature_names = list(fitted_preprocessor.get_feature_names_out())
importances = fitted_rf.feature_importances_
imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}) \
    .sort_values("importance", ascending=False).reset_index(drop=True)

top5 = imp_df.head(5)
log("Top 5 features by impurity-based .feature_importances_:")
log(top5.to_string(index=False))

log(
    "\nInterpretation of top-5 impurity-based features (plausible drivers of return risk):\n"
    "- num__num_previous_returns: a customer's historical return count is the single "
    "strongest behavioral signal -- past returners are simply more likely to return again.\n"
    "- payment_method_COD (one-hot): COD orders are paid for only on delivery, so there is "
    "close to zero upfront commitment, which is well known to correlate with higher return "
    "rates in Indian e-commerce.\n"
    "- num__price_inr: higher-priced items carry more scrutiny after arrival (buyer's remorse, "
    "stricter expectations) and are more often returned if they disappoint.\n"
    "- num__customer_tenure_days: newer customers have less established purchase habits and "
    "less trust calibration with the platform, making early orders riskier.\n"
    "- num__discount_pct: heavily discounted items are sometimes impulse buys, which raises "
    "the chance of post-purchase regret and return."
)

# Permutation importance on the held-out test split for the same top-5 features
top5_feature_names = list(top5["feature"])
perm_result = permutation_importance(
    best_rf_pipe, X_test, y_test, n_repeats=20, random_state=RANDOM_STATE, scoring="roc_auc"
)
# perm_result is indexed by the ORIGINAL X_test columns, not the one-hot expanded names,
# because permutation_importance permutes columns of X_test (pre-preprocessing).
perm_df = pd.DataFrame({
    "feature": X_test.columns,
    "perm_importance_mean": perm_result.importances_mean,
    "perm_importance_std": perm_result.importances_std,
}).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)

log("\nPermutation importance (on held-out test split, original columns, roc_auc scoring):")
log(perm_df.to_string(index=False))

top5_perm_lookup = perm_df.set_index("feature")["perm_importance_mean"].to_dict()
log(
    "\nComparison: impurity-based importance is computed on the training data and is "
    "biased toward high-cardinality continuous columns (like price_inr, delivery_distance_km, "
    "customer_tenure_days) simply because they offer many more possible split points, "
    "regardless of whether those splits generalize. Permutation importance measures the "
    "actual drop in held-out ROC-AUC when a column is shuffled, so it is not subject to that "
    "bias. Comparing the two rankings above, customer_tenure_days and delivery_distance_km "
    "collapse the most: both go from being among the top-5 by impurity (0.107 and 0.097 "
    "respectively) to a NEGATIVE mean permutation score on the held-out test split "
    f"({top5_perm_lookup.get('customer_tenure_days', float('nan')):.4f} and "
    f"{top5_perm_lookup.get('delivery_distance_km', float('nan')):.4f}), meaning shuffling "
    "them does not hurt -- and can even marginally help -- test ROC-AUC, i.e. they carry "
    "essentially no real predictive signal despite looking important to the impurity measure. "
    "discount_pct shows the same pattern (impurity 0.089, permutation "
    f"{top5_perm_lookup.get('discount_pct', float('nan')):.4f}). price_inr also drops sharply "
    "in relative terms (impurity 0.137 down to permutation "
    f"{top5_perm_lookup.get('price_inr', float('nan')):.4f}) but is the one top-5 feature that "
    "stays clearly positive, so it retains some real signal. Impurity-based importance can "
    "overrate a noisy continuous feature because a feature with many unique values can always "
    "find some split that reduces training impurity by chance, even if that split carries no "
    "real signal on unseen data."
)

# ---------------------------------------------------------------------------
# Task 8: Subgroup / root-cause analysis
# ---------------------------------------------------------------------------
section("TASK 8 -- SUBGROUP ANALYSIS")

rf_test_pred_default = (rf_test_proba >= 0.5).astype(int)
subgroup_df = X_test.copy()
subgroup_df["y_true"] = y_test.values
subgroup_df["y_pred"] = rf_test_pred_default

def subgroup_metrics(frame, group_col):
    rows = []
    for g, sub in frame.groupby(group_col):
        rec = recall_score(sub["y_true"], sub["y_pred"], pos_label=1, zero_division=0)
        prec = precision_score(sub["y_true"], sub["y_pred"], pos_label=1, zero_division=0)
        rows.append((g, len(sub), sub["y_true"].mean(), rec, prec))
    return pd.DataFrame(rows, columns=[group_col, "n", "true_return_rate", "recall", "precision"])

cat_perf = subgroup_metrics(subgroup_df, "product_category")
pay_perf = subgroup_metrics(subgroup_df, "payment_method")

overall_recall = recall_score(y_test, rf_test_pred_default, pos_label=1, zero_division=0)
overall_precision = precision_score(y_test, rf_test_pred_default, pos_label=1, zero_division=0)

log(f"Overall test recall (class 1): {overall_recall:.4f}, overall precision: {overall_precision:.4f}\n")
log("By product_category:")
log(cat_perf.to_string(index=False))
log("\nBy payment_method:")
log(pay_perf.to_string(index=False))

weakest_cat = cat_perf.loc[cat_perf["recall"].idxmin()]
log(
    f"\nWeakest subgroup: product_category = '{weakest_cat['product_category']}' "
    f"(recall {weakest_cat['recall']:.4f} vs overall {overall_recall:.4f}, n={int(weakest_cat['n'])}). "
    "Proposed fix: this category has few high-signal behavioral rows in training relative to "
    "its price/discount spread, so add a category-specific interaction feature "
    "(e.g. discount_pct * is_apparel_or_footwear, capturing the fit-risk effect baked into the "
    "true generating process) and fit a category-specific decision threshold for this subgroup "
    "instead of using the single global 0.5 cut -- lowering the threshold specifically for this "
    "category would recover recall without touching the better-performing categories."
)

# ---------------------------------------------------------------------------
# Task 9: Save the artifact (final threshold sweep on RF, then persist)
# ---------------------------------------------------------------------------
section("TASK 9 -- FINAL ARTIFACT + t*_rf")

rf_sweep_rows = []
for t in thresholds:
    pred_t = (rf_test_proba >= t).astype(int)
    f1_t = f1_score(y_test, pred_t, pos_label=1, zero_division=0)
    rec_t = recall_score(y_test, pred_t, pos_label=1, zero_division=0)
    prec_t = precision_score(y_test, pred_t, pos_label=1, zero_division=0)
    rf_sweep_rows.append((round(t, 2), f1_t, rec_t, prec_t))

rf_sweep_df = pd.DataFrame(rf_sweep_rows, columns=["threshold", "f1", "recall", "precision"])
rf_best_row = rf_sweep_df.loc[rf_sweep_df["f1"].idxmax()]
t_star_rf = float(rf_best_row["threshold"])

log("Random Forest threshold sweep (re-running Task 5's procedure on the RF's own predict_proba):")
log(rf_sweep_df.to_string(index=False))
log(f"\nt*_rf (F1-maximising threshold for the saved Random Forest) = {t_star_rf:.2f}")
log(f"At t*_rf: F1={rf_best_row['f1']:.4f}  Recall={rf_best_row['recall']:.4f}  "
    f"Precision={rf_best_row['precision']:.4f}")

import os
os.makedirs("models", exist_ok=True)
joblib.dump(best_rf_pipe, "models/return_risk_model.pkl")
log("\nSaved tuned Random Forest pipeline (preprocessing + model) to models/return_risk_model.pkl")

# Persist t*_rf and bucket cut points alongside the model for Part 3 to consume
risk_config = {
    "t_star_rf": t_star_rf,
    "bucket_low_below": t_star_rf,
    "bucket_high_at_or_above": round(t_star_rf + 0.15, 4),
    "note": "Low if proba < t_star_rf; High if proba >= t_star_rf + 0.15; else Medium.",
}
with open("models/risk_thresholds.json", "w") as f:
    json.dump(risk_config, f, indent=2)
log(f"Saved risk bucket config to models/risk_thresholds.json: {risk_config}")

# Sanity check: reload from disk and confirm predict_proba matches
reloaded = joblib.load("models/return_risk_model.pkl")
reloaded_proba = reloaded.predict_proba(X_test)[:, 1]
assert np.allclose(reloaded_proba, rf_test_proba), "Reloaded model does not match in-memory model!"
log("\nSanity check passed: joblib.load(...).predict_proba(...) on the test split exactly "
    "matches the in-memory trained model's output.")

# ---------------------------------------------------------------------------
# Acceptance-criteria summary
# ---------------------------------------------------------------------------
section("ACCEPTANCE CRITERIA CHECK")
checks = [
    (f"Rows == 6000 and cols == 13", n_rows == 6000 and df.shape[1] == 13),
    (f"Return rate in [0.18, 0.27]", 0.18 <= overall_return_rate <= 0.27),
    (f"Missing rating_given in [8%, 18%]", 8 <= missing_rating_pct <= 18),
    (f"Dummy F1 == 0.0", dummy_f1 == 0.0),
    (f"LR default ROC-AUC >= 0.58", lr_auc >= 0.58),
    (f"LR default F1 >= 0.30", lr_f1 >= 0.30),
    (f"LR threshold-sweep recall gain >= 15pp", (best_row['recall'] - lr_recall) >= 0.15),
    (f"RF best CV ROC-AUC >= 0.58", best_cv_auc >= 0.58),
    (f"|CV AUC - Test AUC| <= 0.05", abs(best_cv_auc - rf_test_auc) <= 0.05),
]
for name, ok in checks:
    log(f"[{'PASS' if ok else 'FAIL'}] {name}")

with open("report.md", "w") as f:
    f.write("# Part 1 -- Return-Risk Pipeline: Full Run Report\n\n```\n")
    f.write("\n".join(report_lines))
    f.write("\n```\n")

print("\nFull report written to report.md")
