# Reading the Kestrel scorecard

One row of the scorecard is one model. Every other number on that row is a different question asked of the same model, on the same 2,126 orders.

## The setup the whole table rests on

- **Training set:** 8,378 orders placed before 1 April 2026, of which 955 came back (11.40%).
- **Holdout:** 2,126 orders placed April to June 2026, of which 245 came back (11.52%).
- **Time based, not random.** The model only ever sees earlier months and is scored on later ones. A return is decided after the order, so a random split would let later information leak into training and flatter every model.

The shipped row (`reports/eval_report.md`, `reports/metrics.json`):

| model | AUC | PR-AUC | Brier | threshold | flagged | tp | net |
|---|---|---|---|---|---|---|---|
| logit-v1 | 0.7895 | 0.4169 | 0.0846 | 0.1118 | 673 | 169 | ₹37,737.50 |

## The model column

`logit-v1` is a logistic regression. It adds up 22 order-level features, each with a learned weight: order value, discount, quantity, payment mode, sales channel, the customer's past order count and past return rate, a missing-pincode flag, promised delivery days, gift flag, Shield membership, state, and so on. The weighted sum goes through a sigmoid, so the output is a number between 0 and 1 that can be read as a probability of return. Categorical fields are one-hot encoded, and L2 regularisation stops the weights from chasing noise in a few hundred returns. There is no LLM and no network call at prediction time.

The column matters for comparison discipline. Every candidate must be scored on the same holdout, through the same code: `metrics_at()` in `src/evaluate.py` for the ranking and calibration numbers, `expected_call_economics()` in `src/economics.py` for the flag counts and rupees. A number produced on a different split, a different cleaning pass or a different threshold rule does not belong in the same table.

## AUC = 0.7895

**What it is:** the chance that a randomly picked order that did return got a higher score than a randomly picked order that did not.

**Plain English:** given one returned and one kept order, the model puts them in the right order about 79 times out of 100. 0.5 is a coin flip, 1.0 is perfect.

**Why 0.79 is decent but not great:** it ranks well enough to be useful, and it beats every tree model tried on the same split. But roughly one time in five it still orders a pair the wrong way, which matters when the decision is a hard threshold on a probability.

**Better or worse:** better means the model separates returners from non-returners more cleanly (more features with real signal, better customer history). Worse means it is closer to random, often because of too few returns or a feature set that has drifted.

**What it does not tell you:** AUC is threshold-free. It says nothing about how many orders you actually block, or what that costs. Two models can share an AUC and produce very different flag counts.

## PR-AUC = 0.4169

**What it is:** the area under the precision-recall curve, the same ranking question asked with the rare class as the focus.

**Plain English:** only about 11.5% of orders are returned. A model that scores at random gets a PR-AUC of roughly 0.115 here, because 0.115 is the share of positives it would stumble into. At 0.4169 this model is about 3.6 times better than no skill.

**Why it is the more relevant headline:** with a rare event, AUC is generous. A model can look respectable on AUC while almost all of its high scores are still non-returns. PR-AUC punishes that: it tracks how much of the precision available at each recall level you actually capture, so it reflects the class balance the warehouse faces.

**Better or worse:** it rises when high scores are concentrated among true returns and falls when the model spreads high scores across good orders too. It is the first number to check when someone claims a rare-event model "works".

## Brier = 0.0846

**What it is:** the mean squared difference between the predicted probability and the actual outcome (1 for returned, 0 for kept). Lower is better, 0 is perfect.

**Plain English:** the model's probabilities are close to the truth, not just correctly ordered. A useful yardstick: always predicting the base rate, 11.52%, scores about 0.1019. A model that always predicts zero scores 0.1152. At 0.0846 the model is meaningfully better than both, and its average predicted probability matches the observed return rate (11.5%).

**Why this is separate from AUC:** ranking and calibration are independent skills. A model can put returners above non-returners every time and still say "70%" for orders that return 30% of the time. Such a model has a fine AUC and a poor Brier, and it is dangerous here, because the threshold and the rupee arithmetic both read the probability as a real probability.

**Better or worse:** it improves when the probabilities track observed frequencies (better calibration, or a calibration step such as isotonic or Platt, which this model does not need). It gets worse with overconfident scores, which usually shows up as high scores on orders that do not return.

## Threshold = 0.1118 and the ₹ break-even

This is not tuned on the holdout. It is derived from two policy numbers:

- A confirmation call costs **₹45** (`C_CALL`).
- A call prevents **35%** of the returns it is made on (`CALL_PREVENT`).
- A return costs **₹1,150** (`C_RETURN`).

So a call placed on an order that would definitely have been returned recovers 0.35 × ₹1,150 = **₹402.50** of expected return cost. The call pays for itself when:

    p × 402.50 > 45        →        p > 45 / 402.50 = 0.111801

That is the threshold. Any order whose predicted return probability is above 0.1118 gets a confirmation call; anything at or below it ships. "Break-even" means the expected value of the call is zero at exactly that probability, so below it the ₹45 cost is more than the expected saving and above it the call is worth making.

**Worked example.** An order scores 0.25. Expected saving = 0.25 × 0.35 × 1,150 = ₹100.63, well above the ₹45 call, so it is flagged. An order scoring 0.10 has expected saving 0.10 × 402.50 = ₹40.25, below ₹45, so it ships. The crossover sits at 0.1118.

**Better or worse:** the threshold goes down (more calls) if a call gets cheaper, prevention improves, or the cost of a return rises. It goes up (fewer calls) in the opposite direction. It is a business input, not a knob to tune for a prettier scorecard. Note that this is the *call* break-even. Holding an order instead is a different calculation with an assumed margin (25%, flagged as an assumption in the repo) and is a worse intervention because the data measures call prevention, not hold prevention.

## Flagged = 673 and tp = 169

**Flagged** is how many of the 2,126 holdout orders cross the threshold and get a call: 673, or 31.7% of the period's orders. That is the operational load the model implies.

**tp (true positives)** is how many of the 245 actual returns were among those 673: 169, which is a recall of 69.0%.

The rest of the confusion matrix is in the repo, so the trade-off is fully derivable:

- **76 returns missed** (245 − 169). The model ships about 31% of real returns silently.
- **504 good orders flagged** (673 − 169). Of every four calls, three are false alarms (precision 25.1%).
- **1,377 orders correctly left alone** out of 1,881 non-returns (specificity 73.2%).

**Better or worse:** more true positives at the same flag count means better precision and a higher net. Flagging more orders raises recall but each extra call costs ₹45, and past the break-even point the marginal calls destroy value, which is exactly what the threshold sweep in the report shows (recall rises to 0.89 at p = 0.06, but net falls to ₹34,240).

## Net = ₹37,737.50

**What it is:** the rupees the intervention earns on the holdout, using the repo's own formula:

    net = tp × 0.35 × 1,150  −  (tp + fp) × 45
        = 169 × 402.50       −  673 × 45
        = ₹68,022.50         −  ₹30,285
        = ₹37,737.50

Saved return cost minus the cost of the calls.

**What it is not:** not annual, not per order, not a forecast. It is three months of profit on a 2,126-order sample. Dividing gives about ₹17.75 per order, and the repo scales that to roughly ₹1.5 lakh a year at 700 orders a month, but those are projections from a small holdout, not the row's value.

**How to compare two net figures:** only on identical holdouts, with the identical threshold rule, computed by the identical formula. A net from a different split, a different flag rule or a different prevention assumption is a different question. This is also the number to use when choosing between models, because it is the only column that carries the ₹ economics.

## How to read a comparison table

- **Same holdout.** Same date boundary, same cleaning, same rows. Anything else is two experiments, not a comparison.
- **Same threshold rule.** Preferably the same derived break-even, not a threshold re-tuned per model to make it look good.
- **Same net formula.** Same call cost, prevention rate and return cost. One changed assumption moves every row.
- **A lower AUC with a higher net is normal.** AUC counts ranking mistakes equally; net weights them by rupees. A model can rank slightly worse overall and still put its confident scores on the expensive orders. Net wins the argument, provided the other two conditions hold.
- **A win of about 0.005 AUC is noise.** On 245 returns, that is well inside sampling variation. The repo's own two rolling-origin windows show the same logistic model moving from 0.7568 to 0.7895 AUC, a swing of 0.03, without anything changing but the window. Do not swap the shipped model for a candidate that wins by 0.005; ask for a bigger sample, or a win that survives a second time window.
- **Accuracy is deliberately absent.** At this threshold the model is 72.7% accurate, but predicting "nothing returns" alone scores 88.5%. Accuracy hides the entire decision.

## One line

AUC 0.79 says it ranks decently, PR-AUC 0.42 says it beats chance by 3.6× on a rare event, Brier 0.085 says its probabilities are real enough to bet ₹45 on, and ₹37,737.50 says that bet pays over three months of holdout orders, with three in four calls being a false alarm and three in ten returns slipping through.
