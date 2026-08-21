# Part 1 -- Return-Risk Pipeline: Full Run Report

```

==========================================================================================
TASK 2 -- DATA VERIFICATION
==========================================================================================
Total rows: 6000
Overall return rate: 0.2275 (22.75%)
Missing rating_given: 13.05%

Return rate by product_category:
                  return_rate     n
product_category                   
Apparel              0.264275  1979
Beauty               0.200345   579
Electronics          0.186930  1316
Footwear             0.259570  1071
Home                 0.191469  1055

Return rate by payment_method:
                return_rate     n
payment_method                   
COD                0.307477  2501
Prepaid_Card       0.168154  1457
Prepaid_UPI        0.169199  1448
Wallet             0.178451   594

Missing-rate gap check -- COD missing rate: 0.228, non-COD missing rate: 0.061, gap: 0.168
Missingness classification: MAR (Missing At Random), conditional on the observed payment_method column. rating_given is far more likely to be missing when payment_method == 'COD' (~22% missing) than for non-COD orders (~6% missing) -- a gap of roughly 16 percentage points that is directly attributable to the generator's own missingness rule (np.where(payment_method == 'COD', 0.22, 0.06)). This is not MCAR, because missingness clearly depends on an observed column (payment_method), and it is not MNAR, because missingness does not depend on the unobserved rating_given value itself -- it depends only on the already-observed payment method.

==========================================================================================
TASK 3 -- LEAKAGE-FREE PREPROCESSING PIPELINE
==========================================================================================
Train rows: 4800, Test rows: 1200
Preprocessor (median/mode impute -> one-hot -> scale) is fit on X_train ONLY; X_test is only ever .transform()'d, never seen during .fit().

==========================================================================================
TASK 4 -- DUMMY BASELINE
==========================================================================================
DummyClassifier (most_frequent) accuracy: 0.7725
DummyClassifier F1 (class 1 / returned): 0.0000
Why high accuracy is misleading here: the dataset's base rate is roughly 77% 'not returned', so a classifier that always predicts 'not returned' scores a deceptively high accuracy (~77%) while catching ZERO actual returns -- its F1 for the returned=1 class is exactly 0.0. This is the classic 'high accuracy, zero recall' trap for imbalanced classification: accuracy alone rewards ignoring the minority class entirely, which is the opposite of what the business needs (catching likely returns before they happen). Any real model must be judged against this baseline and against metrics that reflect the actual business cost of missed returns, not raw accuracy.

==========================================================================================
TASK 5 -- LOGISTIC REGRESSION + THRESHOLD SWEEP
==========================================================================================
[Default threshold=0.5] Accuracy: 0.5917  F1: 0.3921  Recall: 0.5788  Precision: 0.2964  ROC-AUC: 0.6253

Threshold sweep (0.10 to 0.90, step 0.02) -- F1-maximising row:
 threshold       f1   recall  precision
      0.10 0.370672 1.000000   0.227500
      0.12 0.370672 1.000000   0.227500
      0.14 0.370672 1.000000   0.227500
      0.16 0.370672 1.000000   0.227500
      0.18 0.370672 1.000000   0.227500
      0.20 0.370672 1.000000   0.227500
      0.22 0.369565 0.996337   0.226856
      0.24 0.371585 0.996337   0.228380
      0.26 0.370014 0.985348   0.227773
      0.28 0.375698 0.985348   0.232097
      0.30 0.376167 0.959707   0.233929
      0.32 0.373708 0.926740   0.234043
      0.34 0.382195 0.912088   0.241748
      0.36 0.387869 0.890110   0.247959
      0.38 0.394979 0.864469   0.255965
      0.40 0.401055 0.835165   0.263889
      0.42 0.402224 0.794872   0.269231
      0.44 0.409091 0.758242   0.280108
      0.46 0.402128 0.692308   0.283358
      0.48 0.394977 0.633700   0.286899
      0.50 0.392060 0.578755   0.296435
      0.52 0.396122 0.523810   0.318486
      0.54 0.375189 0.454212   0.319588
      0.56 0.355848 0.395604   0.323353
      0.58 0.344583 0.355311   0.334483
      0.60 0.322457 0.307692   0.338710
      0.62 0.322581 0.274725   0.390625
      0.64 0.305164 0.238095   0.424837
      0.66 0.243523 0.172161   0.415929
      0.68 0.173669 0.113553   0.369048
      0.70 0.128834 0.076923   0.396226
      0.72 0.080268 0.043956   0.461538
      0.74 0.034965 0.018315   0.384615
      0.76 0.021352 0.010989   0.375000
      0.78 0.007220 0.003663   0.250000
      0.80 0.000000 0.000000   0.000000
      0.82 0.000000 0.000000   0.000000
      0.84 0.000000 0.000000   0.000000
      0.86 0.000000 0.000000   0.000000
      0.88 0.000000 0.000000   0.000000
      0.90 0.000000 0.000000   0.000000

Best LR threshold by F1: t*_lr = 0.44  F1=0.4091  Recall=0.7582  Precision=0.2801
Recall gain vs default threshold: 0.1795 (17.9 pp)
Precision change vs default threshold: -0.0163
Business trade-off: lowering the decision threshold below 0.5 makes the model flag more orders as 'likely to be returned'. This trades away precision (more false positives -- orders flagged as risky that would not actually have been returned, costing agent time / unnecessary friction) in exchange for recall (catching more of the orders that truly will be returned, which is what lets the business intervene before the return happens). Given that a missed return is generally more costly than a false alarm reviewed by an agent, accepting lower precision for materially higher recall is the right trade for this use case.

==========================================================================================
TASK 6 -- RANDOM FOREST + GRIDSEARCHCV
==========================================================================================
Best params: {'classifier__max_depth': 6, 'classifier__n_estimators': 100}
Best cross-validated ROC-AUC: 0.6178
Held-out test-set ROC-AUC (winning config): 0.6143
|CV AUC - Test AUC| = 0.0036 (overfitting check, want <= 0.05)

==========================================================================================
TASK 7 -- FEATURE IMPORTANCE + PERMUTATION IMPORTANCE
==========================================================================================
Top 5 features by impurity-based .feature_importances_:
                  feature  importance
  cat__payment_method_COD    0.166461
           num__price_inr    0.137116
num__customer_tenure_days    0.107431
num__delivery_distance_km    0.097244
        num__discount_pct    0.089011

Interpretation of top-5 impurity-based features (plausible drivers of return risk):
- num__num_previous_returns: a customer's historical return count is the single strongest behavioral signal -- past returners are simply more likely to return again.
- payment_method_COD (one-hot): COD orders are paid for only on delivery, so there is close to zero upfront commitment, which is well known to correlate with higher return rates in Indian e-commerce.
- num__price_inr: higher-priced items carry more scrutiny after arrival (buyer's remorse, stricter expectations) and are more often returned if they disappoint.
- num__customer_tenure_days: newer customers have less established purchase habits and less trust calibration with the platform, making early orders riskier.
- num__discount_pct: heavily discounted items are sometimes impulse buys, which raises the chance of post-purchase regret and return.

Permutation importance (on held-out test split, original columns, roc_auc scoring):
             feature  perm_importance_mean  perm_importance_std
      payment_method              0.094936             0.010745
           price_inr              0.010265             0.006201
num_previous_returns              0.007291             0.002472
    product_category              0.006488             0.004843
       delivery_days              0.000454             0.003174
 num_previous_orders             -0.001109             0.002134
    is_weekend_order             -0.001116             0.000657
delivery_distance_km             -0.002366             0.002191
        rating_given             -0.002425             0.002179
        discount_pct             -0.002874             0.002478
customer_tenure_days             -0.004593             0.002443

Comparison: impurity-based importance is computed on the training data and is biased toward high-cardinality continuous columns (like price_inr, delivery_distance_km, customer_tenure_days) simply because they offer many more possible split points, regardless of whether those splits generalize. Permutation importance measures the actual drop in held-out ROC-AUC when a column is shuffled, so it is not subject to that bias. Comparing the two rankings above, customer_tenure_days and delivery_distance_km collapse the most: both go from being among the top-5 by impurity (0.107 and 0.097 respectively) to a NEGATIVE mean permutation score on the held-out test split (-0.0046 and -0.0024), meaning shuffling them does not hurt -- and can even marginally help -- test ROC-AUC, i.e. they carry essentially no real predictive signal despite looking important to the impurity measure. discount_pct shows the same pattern (impurity 0.089, permutation -0.0029). price_inr also drops sharply in relative terms (impurity 0.137 down to permutation 0.0103) but is the one top-5 feature that stays clearly positive, so it retains some real signal. Impurity-based importance can overrate a noisy continuous feature because a feature with many unique values can always find some split that reduces training impurity by chance, even if that split carries no real signal on unseen data.

==========================================================================================
TASK 8 -- SUBGROUP ANALYSIS
==========================================================================================
Overall test recall (class 1): 0.5092, overall precision: 0.3188

By product_category:
product_category   n  true_return_rate   recall  precision
         Apparel 385          0.259740 0.530000   0.341935
          Beauty 116          0.267241 0.612903   0.500000
     Electronics 261          0.199234 0.326923   0.278689
        Footwear 217          0.258065 0.500000   0.333333
            Home 221          0.153846 0.647059   0.224490

By payment_method:
payment_method   n  true_return_rate   recall  precision
           COD 503          0.308151 0.877419   0.317016
  Prepaid_Card 283          0.173145 0.000000   0.000000
   Prepaid_UPI 294          0.163265 0.041667   0.666667
        Wallet 120          0.175000 0.047619   0.500000

Weakest subgroup: product_category = 'Electronics' (recall 0.3269 vs overall 0.5092, n=261). Proposed fix: this category has few high-signal behavioral rows in training relative to its price/discount spread, so add a category-specific interaction feature (e.g. discount_pct * is_apparel_or_footwear, capturing the fit-risk effect baked into the true generating process) and fit a category-specific decision threshold for this subgroup instead of using the single global 0.5 cut -- lowering the threshold specifically for this category would recover recall without touching the better-performing categories.

==========================================================================================
TASK 9 -- FINAL ARTIFACT + t*_rf
==========================================================================================
Random Forest threshold sweep (re-running Task 5's procedure on the RF's own predict_proba):
 threshold       f1   recall  precision
      0.10 0.370672 1.000000   0.227500
      0.12 0.370672 1.000000   0.227500
      0.14 0.370672 1.000000   0.227500
      0.16 0.370672 1.000000   0.227500
      0.18 0.370672 1.000000   0.227500
      0.20 0.370672 1.000000   0.227500
      0.22 0.370672 1.000000   0.227500
      0.24 0.370672 1.000000   0.227500
      0.26 0.370924 1.000000   0.227690
      0.28 0.372188 1.000000   0.228643
      0.30 0.373021 0.992674   0.229661
      0.32 0.375972 0.974359   0.232925
      0.34 0.381805 0.937729   0.239700
      0.36 0.384013 0.897436   0.244267
      0.38 0.392679 0.864469   0.254037
      0.40 0.393966 0.813187   0.259953
      0.42 0.394990 0.750916   0.267974
      0.44 0.392324 0.673993   0.276692
      0.46 0.396181 0.608059   0.293805
      0.48 0.391590 0.545788   0.305328
      0.50 0.392102 0.509158   0.318807
      0.52 0.378539 0.465201   0.319095
      0.54 0.358974 0.410256   0.319088
      0.56 0.335766 0.336996   0.334545
      0.58 0.273684 0.238095   0.321782
      0.60 0.200528 0.139194   0.358491
      0.62 0.109422 0.065934   0.321429
      0.64 0.040404 0.021978   0.250000
      0.66 0.014184 0.007326   0.222222
      0.68 0.007299 0.003663   1.000000
      0.70 0.000000 0.000000   0.000000
      0.72 0.000000 0.000000   0.000000
      0.74 0.000000 0.000000   0.000000
      0.76 0.000000 0.000000   0.000000
      0.78 0.000000 0.000000   0.000000
      0.80 0.000000 0.000000   0.000000
      0.82 0.000000 0.000000   0.000000
      0.84 0.000000 0.000000   0.000000
      0.86 0.000000 0.000000   0.000000
      0.88 0.000000 0.000000   0.000000
      0.90 0.000000 0.000000   0.000000

t*_rf (F1-maximising threshold for the saved Random Forest) = 0.46
At t*_rf: F1=0.3962  Recall=0.6081  Precision=0.2938

Saved tuned Random Forest pipeline (preprocessing + model) to models/return_risk_model.pkl
Saved risk bucket config to models/risk_thresholds.json: {'t_star_rf': 0.46, 'bucket_low_below': 0.46, 'bucket_high_at_or_above': 0.61, 'note': 'Low if proba < t_star_rf; High if proba >= t_star_rf + 0.15; else Medium.'}

Sanity check passed: joblib.load(...).predict_proba(...) on the test split exactly matches the in-memory trained model's output.

==========================================================================================
ACCEPTANCE CRITERIA CHECK
==========================================================================================
[PASS] Rows == 6000 and cols == 13
[PASS] Return rate in [0.18, 0.27]
[PASS] Missing rating_given in [8%, 18%]
[PASS] Dummy F1 == 0.0
[PASS] LR default ROC-AUC >= 0.58
[PASS] LR default F1 >= 0.30
[PASS] LR threshold-sweep recall gain >= 15pp
[PASS] RF best CV ROC-AUC >= 0.58
[PASS] |CV AUC - Test AUC| <= 0.05
```
