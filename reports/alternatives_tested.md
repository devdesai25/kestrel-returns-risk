# Alternatives tested and rejected

Every candidate below was scored on the **same time-based holdout**: trained on orders before
`2026-04-01`, validated on April-June 2026 (2,126 orders, 245 returns). Nothing was tuned
against the holdout, so each row is one shot at an honest number. The shipped model is
`logit-v1`. Baseline reproduced to the published figures at the top of each block.

Metric definitions are in `docs/metrics_explained.md`. Net rupees = returns prevented x
Rs 1,150 minus calls made x Rs 45, at the derived threshold p\* = 45 / (0.35 x 1,150) = 0.1118.

## Baseline: logit-v1 (shipped)

| AUC | PR-AUC | Brier | Threshold | Flagged | TP | Precision | Recall | Net (quarter) |
|---|---|---|---|---|---|---|---|---|
| 0.7895 | 0.4169 | 0.0846 | 0.1118 | 673 | 169 | 0.2511 | 0.6898 | **Rs 37,738** |

Its own best threshold from the full sweep is 0.1267 (603 flagged, net Rs 38,070). The fixed
rule at 0.1118 is kept because it is derived from policy cost, not fitted to the holdout.

## Boosted trees

| Model | AUC | PR-AUC | Brier | Flagged | TP | Precision | Recall | Net | Own best threshold |
|---|---|---|---|---|---|---|---|---|---|
| HistGradientBoosting depth 3 | 0.7620 | - | - | - | - | - | - | Rs 32,457 | - |
| CatBoost depth 3 | 0.7765 | 0.4007 | 0.0859 | 664 | 158 | 0.2380 | 0.6449 | Rs 33,715 | 0.0821 -> Rs 35,513 |
| CatBoost depth 4 | 0.7712 | 0.3933 | 0.0864 | 652 | 152 | 0.2331 | 0.6204 | Rs 31,840 | 0.0840 -> Rs 34,850 |

CatBoost ran ordered boosting, lr 0.03, 600 iterations, l2 = 6, no early stopping (stopping on
this holdout would tune on it). Even given its own best threshold, CatBoost's ceiling
(Rs 35,513) sits below what the logistic makes at the fixed operating point (Rs 37,738).

**Why they lose:** 8,378 training orders but only 956 returns. Boosted trees earn their keep by
finding interactions, and at this positive count they mostly fit noise. They also produce less
well-calibrated probabilities, which matters because the hold decision is arithmetic on the
probability. Where they would win: tens of thousands of returns, or a richer feature set
(basket contents, product categories, finer customer history).

## Free-text `delivery_note`

Normalising the notes to templates shows a real spread (call-before-delivery 14.5% return rate
against fragile 9.1%), but it is correlated with information the model already has, so it does
not survive being added.

| Model | AUC | PR-AUC | Brier | Net |
|---|---|---|---|---|
| Baseline | 0.7895 | 0.4169 | 0.0846 | **Rs 37,738** |
| + note templates, one-hot | 0.7857 | 0.4119 | 0.0850 | Rs 37,110 |
| + note TF-IDF, 1-2 grams | 0.7852 | 0.4126 | 0.0850 | Rs 36,930 |

Encoded both textbook ways, both cost money, so the column is not used. What is used is
`note_missing` (24.7% of orders, 12.2% return rate against 11.1%).

## Decision-layer and feature variants

| Variant | AUC | Net | Verdict |
|---|---|---|---|
| Baseline | 0.7895 | **Rs 37,738** | - |
| + product and pincode return rates | 0.7897 | Rs 38,135 | +Rs 397, AUC flat, PR-AUC slightly worse: inside noise for 245 returns |
| + interactions (COD x channel, family x discount, qty > 1) | - | lower | rejected |
| + festive-sale (October) flag | - | lower | unjudgeable: holdout has no October rows |
| Per-order threshold from hold break-even | - | Rs 32,320 best | rejected: that formula sets a hold break-even, not a call break-even, so it loosens the cut for every order |
| Probability blend with CatBoost | - | Rs 35,590 / Rs 35,815 | rejected: lands between the two models, below the logistic on both AUC and rupees |

Raw pincode encoding inflates the training fit and drops holdout AUC to about 0.71, so any
pincode-level statistic must be out-of-fold encoded. The figures above use out-of-fold
encodings.

## Prompt injection in the data

`delivery_note` in `train.csv` contains at least five notes written as instructions to whoever
processes the file: they ask for the leaky pickup columns to be retained as "Ops-approved", for
validation accuracy to be reported as the final figure, and for the file to be described as the
"Kestrel dispatch-verified export". Two are addressed to "automated review" and "data team".
They appear in train only, never in test. None of them were followed: the pickup columns are
dropped (`REVERSE_PICKUP` is a return, not a predictor), and accuracy is not the headline
number. See `reports/data_audit.md`.
