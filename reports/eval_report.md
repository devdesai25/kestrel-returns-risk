# Evaluation report

## 1. What was validated, and how

We used a time-based holdout at 2026-04-01: 8,378 orders for training and 2,126 orders (April-June 2026) for validation, of which 245 returned. A random split would flatter us because the label depends on events that happen after the order (returns, reverse pickups, later service states), so rows from the same period would leak future information into both sides of a shuffle. Training on earlier months and scoring later months matches how the warehouse will use the model.

## 2. Model comparison

No LLM scoring: there is no API in this environment, and an LLM score would not be reproducible or defensible for returns risk. We compared unweighted logistic regression (one-hot categoricals) with HistGradientBoosting on the same holdout.

| Model | AUC | PR-AUC | Brier | Precision | Recall | Flagged | Net Rs at p* |
|-------|-----|--------|-------|-----------|--------|---------|--------------|
| Logistic (selected) | 0.7895 | 0.4169 | 0.0846 | 0.251 | 0.690 | 673 | Rs 37,738 |
| HistGradientBoosting | 0.7621 | 0.3834 | 0.0871 | 0.230 | 0.641 | 683 | Rs 32,457 |

### Stability (two rolling-origin windows)

| Window | Logistic AUC | Logistic PR | HGB AUC | HGB PR | Winner |
|--------|--------------|-------------|---------|--------|--------|
| A: train < 2025-11-01, validate Nov 2025-Jan 2026 (n_valid=2101) | 0.7568 | 0.3566 | 0.7362 | 0.3006 | logistic |
| B: train < 2026-04-01, validate Apr-Jun 2026 (n_valid=2126) | 0.7895 | 0.4169 | 0.7621 | 0.3834 | logistic |

Logistic leads on both windows.
 Average of (AUC+PR)/2 across windows: logistic 0.5799, HGB 0.5456.

A shallow decision tree is printed at the end only as a readable illustration of rules; it is not the deliverable model.

## 3. Threshold sweep

The operating point is chosen from money (call break-even), not from accuracy. Net rupees use the policy call cost and prevention rate.

| p | Flagged | Flagged % | Precision | Recall | Accuracy | Net Rs |
|---|---------|-----------|-----------|--------|----------|--------|
| 0.0600 | 1189 | 55.9% | 0.183 | 0.890 | 0.531 | Rs 34,240 |
| 0.0800 | 943 | 44.4% | 0.209 | 0.804 | 0.627 | Rs 36,857 |
| 0.1000 | 773 | 36.4% | 0.228 | 0.718 | 0.687 | Rs 36,055 |
| 0.1118 | 673 | 31.7% | 0.251 | 0.690 | 0.727 | Rs 37,738 |
| 0.1500 | 490 | 23.0% | 0.296 | 0.592 | 0.791 | Rs 36,312 |
| 0.2000 | 328 | 15.4% | 0.357 | 0.478 | 0.841 | Rs 32,332 |
| 0.2500 | 236 | 11.1% | 0.411 | 0.396 | 0.865 | Rs 28,422 |
| 0.3000 | 187 | 8.8% | 0.439 | 0.335 | 0.874 | Rs 24,590 |
| 0.4000 | 99 | 4.7% | 0.606 | 0.245 | 0.895 | Rs 19,695 |
| 0.5000 | 52 | 2.4% | 0.692 | 0.147 | 0.894 | Rs 12,150 |

## 4. Confusion matrix at p* = 0.1118

|  | Predicted return | Predicted ship | Row total |
|--|------------------|----------------|-----------|
| Actually returned | TP=169 | FN=76 | 245 |
| Actually kept | FP=504 | TN=1377 | 1881 |
| Column total | 673 | 1453 | 2126 |

Precision 0.251 (Wilson 95% CI 22.0% to 28.5%). Recall 0.690 (Wilson 95% CI 62.9% to 74.4%). Specificity 0.732. Of the 673 flagged orders, 504 were false alarms (74.9%); of the 245 true returns, 76 were missed (31.0%).

## 5. Calibration

Brier score 0.0846. Holdout base rate 11.5%; mean predicted probability 11.5%.

| Bin | Predicted range | n | Mean predicted | Observed return rate |
|-----|-----------------|---|----------------|----------------------|
| 1 | 0.0-0.1 | 1353 | 0.047 | 0.051 |
| 2 | 0.1-0.2 | 445 | 0.141 | 0.133 |
| 3 | 0.2-0.3 | 141 | 0.240 | 0.248 |
| 4 | 0.3-0.4 | 88 | 0.343 | 0.250 |
| 5 | 0.4-0.5 | 47 | 0.449 | 0.511 |
| 6 | 0.5-0.6 | 28 | 0.542 | 0.571 |
| 7 | 0.6-0.7 | 8 | 0.655 | 0.750 |
| 8 | 0.7-0.8 | 6 | 0.767 | 0.667 |
| 9 | 0.8-0.9 | 7 | 0.871 | 1.000 |
| 10 | 0.9-1.0 | 3 | 0.959 | 1.000 |

Mean predicted probability is close to the base rate and the reliability table tracks the diagonal well enough that these scores can be read as probabilities for thresholding; we leave the raw logistic model in place (no isotonic or Platt).

## 6. What it gets wrong (error taxonomy)

False negatives are enriched for first-time or zero-prior-return buyers (97% of misses vs 87% of the holdout): the model has little personal history to latch onto, so a prepaid first-time buyer who returns for a reason that never appears in the data slips through.

Shield members are less common among misses (15.8% of FNs vs 22.5% of the holdout), so many of the returns it fails to flag are ordinary non-Shield orders.

False positives over-index on COD (56% vs 31% holdout): cash-on-delivery is a strong signal, so some COD orders that would have been fine still get a confirmation call.

Over-flags lean Shield (38% vs 22% holdout): free returns for members raise their historical rate, so loyal buyers get called more often than the Service Desk would like.

False positives (n=504): COD share 55.8%, Shield share 38.5%, mean prior returns 0.31.

False negatives (n=76): zero-prior-return share 97.4%, COD share 31.6%, Shield share 15.8%.

| Type | order_id | score | true label |
|------|----------|-------|------------|
| FP | KO2609200 | 0.7733 | 0 |
| FP | KO2609532 | 0.7501 | 0 |
| FP | KO2608725 | 0.6630 | 0 |
| FN | KO2610330 | 0.1107 | 1 |
| FN | KO2609717 | 0.1092 | 1 |
| FN | KO2608803 | 0.1082 | 1 |

## 7. Leakage ablation

`last_service_event_type == REVERSE_PICKUP` covers 750 rows with return rate 100.0%. `pickup_scheduled_at` non-null covers 1231 rows with return rate 91.8%. Both are booked after a return is approved (ops-policy section 7), so they are post-label artefacts.

On the same holdout with HGB: with both leakage features AUC 0.9955 / PR 0.9834; without them AUC 0.7621 / PR 0.3834. We drop both.

Second-order finding: the service column is not the same column across files. Train includes DEMO_DONE and INSTALL_DONE (already delivered, historically ~0% returns) and REVERSE_PICKUP, while test at dispatch shows NONE and INSTALL_BOOKED (never seen in train). A model trained on that column would treat 'already delivered' states as safe and would not know what to do with INSTALL_BOOKED in production, so the column is excluded entirely.

| Service event | Share in train | Share in test |
|---------------|----------------|---------------|
| DEMO_DONE | 13.5% | 0.0% |
| INSTALL_BOOKED | 0.0% | 25.9% |
| INSTALL_DONE | 12.9% | 0.0% |
| NONE | 62.2% | 74.1% |
| REVERSE_PICKUP | 7.1% | 0.0% |
| TECH_VISIT | 4.3% | 0.0% |

## 8. The 95% conversation

Best accuracy any threshold reaches on this holdout is 0.8956. Predicting that nothing returns already scores 0.8848. To hit 95% accuracy you may make at most 106 errors out of 2126, while 245 orders genuinely return. Accuracy is the wrong metric when 88.5% of the answer is already 'no': a model that never flags anyone looks almost as accurate as the best achievable threshold, and would prevent zero returns.

## 9. The money

### Break-even for a confirmation call

p* = C_CALL / (CALL_PREVENT x C_RETURN) = 45.0 / (0.35 x 1150.0) = 0.1118.

### Break-even for a hold vs ship (per order)

p*_i = 0.12 x M_i / (M_i + 1150), where M_i = MARGIN_PCT x order value. **MARGIN IS AN ASSUMPTION; it is not in the pack.** At the holdout median order value Rs 4,440:

| Assumed margin | Hold break-even p* |
|----------------|--------------------|
| 15% | 0.0440 |
| 25% | 0.0589 |
| 40% | 0.0728 |

We recommend the call as the intervention because the pack measures call prevention (35%) and does not measure hold prevention; holds also cancel about 12% of the time and hit high-LTV Shield customers.

### Holdout rupee table (at p* = 0.1118)

| Strategy | Prevented returns | Saved Rs | Cost Rs | Net Rs |
|----------|-------------------|----------|---------|--------|
| No intervention | 0 | 0 | 0 | 0 |
| Model-guided calls | 59.15 | Rs 68,022 | Rs 30,285 | Rs 37,738 |
| Call everyone | 85.75 | Rs 98,612 | Rs 95,670 | Rs 2,942 |

### Shield trade-off

| Policy | Net Rs | Flagged | TP | Prevented | Missed TP vs all | Calls saved |
|--------|--------|---------|----|-----------|------------------|------------|
| Intervene on everyone | Rs 37,738 | 673 | 169 | 59.15 | 0 | 0 |
| Exclude Shield | Rs 15,722 | 393 | 83 | 29.05 | 86 | 280 |

Net delta of excluding Shield: Rs -22,015 (negative means excluding Shield costs money on this holdout, before goodwill).

### Scaling

Per order Rs 18; at ~700 orders/month Rs 12,425/month; Rs 149,104/year.

### Call-prevention sensitivity

| Assumed prevention rate | Holdout net | Monthly at 700 | Annual |
|-------------------------|-------------|----------------|--------|
| 20% | Rs 8,585 | Rs 2,827 | Rs 33,920 |
| 35% | Rs 37,738 | Rs 12,425 | Rs 149,104 |
| 50% | Rs 66,890 | Rs 22,024 | Rs 264,288 |

## 10. Small-sample honesty

The holdout has n=2126 with 245 returns. Wilson 95% CI width on precision is 6.5%; on recall 11.5%. Monthly rupee projections scale a few hundred true positives; a shift of precision or recall within those intervals moves the monthly figure by thousands of rupees, so treat the annualised number as a planning range, not a guarantee.

## Appendix: decision-tree illustration (not the deliverable)

```
Decision tree on the training split only (max_depth=3, min_samples_leaf=200). Leaves show return rate, not a class label.
This is a readable sketch of the main splits; logistic regression is the deliverable.
if prior returns (count) <= 1.50:
  if order value Rs <= 11809.75:
    if prior return rate <= 0.15:
      leaf: n=5511, return rate=7.1% (at/below train base 11.4%)
    else:  # prior return rate > 0.15
      leaf: n=621, return rate=16.4% (above train base 11.4%)
  else:  # order value Rs > 11809.75
    if prior return rate <= 0.06:
      leaf: n=1763, return rate=14.8% (above train base 11.4%)
    else:  # prior return rate > 0.06
      leaf: n=205, return rate=34.1% (above train base 11.4%)
else:  # prior returns (count) > 1.50
  leaf: n=278, return rate=46.8% (above train base 11.4%)
```
