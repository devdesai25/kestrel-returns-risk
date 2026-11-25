# Submission answers

Form answers in the order the form asks them. Every number here is reproducible from
`reports/metrics.json`; the code that produced it is in `src/`. Two items are deliberately left
blank for you to fill in honestly (Q8 and Q10) and are marked as such.

Holdout design throughout: train on orders before 2026-04-01, validate on 2026-04-01 to
2026-06-30. 8,378 training orders, 2,126 validation orders, 245 validation returns, base rate
11.5%. Never a random split.

---

## What model is actually shipped, and why that one

**A regularised logistic regression ("logit-v1"), plus rule-based reason strings generated from
the model's own inputs.** Not a random forest, not XGBoost, not an LLM.

The reasoning, in the order it mattered:

- **No LLM in the scoring path.** It costs money per call, is not reproducible run-to-run, and
  cannot be defended to a Finance Controller. The reasons here are generated in about a
  millisecond, offline, for free, and are the same every time you ask.
- **The sample is small: 8,378 training orders with 956 returns.** At that size a linear model
  with sensible features is competitive, and it gives calibrated probabilities, which is what the
  rupee arithmetic depends on.
- **It won the comparison, it was not assumed.** I fitted gradient-boosted trees (HistGradientBoosting:
  many small decision trees, each one fitted to the errors the previous ones made) on the same
  time-based split and it lost on both held-out windows: 0.762 AUC against 0.790, and 0.383
  PR-AUC against 0.417. It also produced worse-calibrated probabilities and less money at the
  operating point (₹32,457 against ₹37,738). So trees were tried and not needed, rather than
  ignored.
- **A single decision tree is in the report as an illustration only**, because it is readable,
  but it is a bad deliverable: it cannot express a smooth probability and it overfits the few
  hundred returns quickly.

What gradient boosting would buy if Kestrel had three years and 50k orders: automatic interactions
(COD only being dangerous above a certain value, say). It could not buy them here.

---

## 1. What did you build, and what business outcome does it move? State the number and the money.

A pre-dispatch returns-risk scorer for Kestrel Home. Every order gets a probability of being
returned, a recommended action, and four plain-English reasons a warehouse clerk or a service-desk
agent can act on. It ships as a single HTTP endpoint plus one screen, and it runs on a laptop with
no API key and no network.

The business outcome is fewer reverse pickups per hundred orders dispatched. On the held-out
quarter: it flags 31.7% of orders (673 of 2,126), catches 69.0% of returns (169 of 245), and 25.1%
of what it flags actually returns. At the policy call cost of ₹45 and a prevented return worth
₹1,150, that is ₹68,022 saved against ₹30,285 spent, **₹37,738 net for the quarter**. At 700
orders a month that is **₹12,425 a month, around ₹1.5 lakh a year**.

The same quarter dispatched with no model returns 245 orders and saves nothing. Calling every
single order costs ₹95,670 to prevent 86 returns and nets ₹2,942, which is roughly break-even. The
value is in the targeting, not the calling.

## 2. What score do you expect predictions.csv to get on the hidden outcomes, on which metric, and why that metric? How did you estimate it?

**PR-AUC (average precision), expected 0.40, and I would put a range of 0.36 to 0.44 on it.**
ROC-AUC alongside it at 0.78 to 0.80.

PR-AUC is the right metric because only 11.5% of orders return. ROC-AUC flatters a model on data
this imbalanced: a model can score 0.79 AUC while being useless at the top of the list, which is
the only part of the list anyone acts on. Average precision measures exactly the thing the
warehouse experiences, which is how many of the orders we pull out are genuinely bad. Accuracy is
worse than useless here, for reasons in Q4.

How I estimated it: a time-based holdout (0.4169), then repeated the exercise on a second, earlier
window to see whether that number was luck. The model scored 0.3566 on November 2025 to January
2026 and 0.4169 on April to June 2026, so 0.40 is the middle of two independent estimates rather
than one optimistic draw. predictions.csv comes from a refit on all 10,504 cleaned rows, which
should be marginally better than the 8,378-row model that produced the holdout figure, so if
anything 0.40 is slightly conservative. The honest risk on the downside is that the hidden period
drifts (new SKUs, festive demand, a gateway change), which is the same risk that took the first
window down to 0.357.

If the grader scores accuracy instead: expect about 0.88 at a 0.5 cut, because 88.5% of orders are
kept. That number is available to anyone who predicts "nothing returns" and it is why I do not
optimise for it.

## 3. How do you know it works? Sample size, how you checked, error rate, and the kind of case it gets wrong.

Sample: 2,126 held-out orders with 245 returns, trained on 8,378 earlier orders. Two windows
checked, not one. I also checked that the model beats the trivial alternatives, not just that it
has a nice AUC: at the operating threshold it beats "flag nobody" (₹0) and "call everybody"
(₹2,942) by a wide margin on held-out money.

Error rate at the operating threshold (0.1118): 504 false alarms and 76 misses out of 2,126
orders. Precision 0.251 (95% CI 0.220 to 0.285), recall 0.690 (0.629 to 0.744), specificity 0.732.
So 74.9% of the calls it asks for are wasted, and 31.0% of returns get through. That is the actual
performance, not the accuracy number.

The cases it gets wrong, from profiling the errors rather than guessing:

- **Misses (76).** 97.4% of the returns it missed came from customers with no prior returns,
  against 87% of the holdout. It has almost nothing to work with on a first-time buyer, and those
  are the returns that cost the most to predict. A prepaid first-time buyer who returns for a
  reason that never appears in the data (damaged on arrival, expectation mismatch, changed mind)
  is invisible to this model and will stay invisible until something other than order data is
  joined to it.
- **False alarms (504).** They over-index on cash on delivery (55.8% of false alarms against
  31.1% of the holdout) and on Shield members (38.5% against 22.2%). In other words it calls the
  people the Service Desk already warned about, and it calls a lot of COD orders that would have
  been fine. That is a cost of ₹45 each, not a disaster, but it is the reason the recommendation
  is a call and not a hold.

Calibration is honest enough to threshold on: Brier 0.0846, mean predicted 11.5% against a base
rate of 11.5%, and the reliability bins track the diagonal. I left the raw logistic model in place
rather than applying Platt or isotonic.

## 4. Did you change, narrow, or push back on the client's ask? What, when, and why.

Four changes, all before writing any code, all from reading the policy and Finance's reply rather
than the brief.

**A. 95% accuracy is not achievable, and I said so.** 88.5% of orders are kept, so "flag nothing"
is already 88.5% accurate, and the best any threshold achieves on the holdout is 89.6%. Reaching
95% would allow 106 errors across 2,126 orders while 245 orders genuinely returned. I put that
number in the memo rather than quietly shipping a model that scores 72% accuracy and calling it a
success.

**B. ₹600 became ₹1,150.** Ritu's figure is a recollection. Finance's is the policy figure and the
one that gets reported, so every rupee in the memo uses ₹1,150. At ₹600 the whole intervention is
barely worth running.

**C. Flag-and-hold became flag-and-call.** A hold is not free: the policy says 12% of orders held
over 24 hours get cancelled, and every cancellation costs the margin. The pack measures call
prevention (35%) and does not measure hold prevention at all, so a hold cannot even be priced
honestly from the data. The call is the only intervention in the pack with a measured effect. I
implemented the hold arithmetic anyway, so the client can see what a hold would have to be worth to
justify itself, and I recommend a hold only for the flagged orders a call cannot reach.

**D. Shield, narrowed rather than deferred.** Ritu said "we'll deal with Shield later". The data
says Shield members are 22.2% of orders and 36.1% of returns. Excluding them from intervention
gives up ₹22,015 a quarter of modelled value, most of it in returns we would no longer prevent.
Holding them is worse than excluding them, because holds cancel 12% of the time and these are the
highest-value customers in the file. So the model keeps Shield as a feature and the recommendation
is explicitly: call Shield members, never hold them. That is a decision she can take today without
a policy change.

## 5. What is wrong with what you are handing us?

In rough order of how much it would cost you.

1. **The 35% call-prevention figure is the client's own estimate from an uncontrolled pilot, and my
   entire rupee case scales linearly with it.** At 20% the monthly net is ₹2,827; at 50% it is
   ₹22,024. Nothing in the pack lets me verify it. Until someone runs a controlled count, the
   annual figure is a planning range, not a forecast.
2. **The margin assumption (25%) is invented.** It is used only to price a hold, never to set the
   operating threshold, but it is still an assumption wearing a number's clothes. Policy gives no
   margin, and a hold cannot be priced without it.
3. **The model is strictly additive.** A logistic regression cannot express "COD is only dangerous
   above ₹15,000". I tried gradient boosting and it lost on both windows (0.762 against 0.790 AUC),
   which I read as the signal being close to additive with a lot of noise on top. It could equally
   mean the trees needed more data than 8.4k rows. If someone engineers interactions by hand it
   might beat this, and I would not be surprised.
4. **The reasons are associations, not causes.** They are produced by replacing each field with a
   typical value and re-scoring. With correlated fields (state, customer city, product family) the
   credit is split arbitrarily between them, and the wording is necessarily more confident than the
   evidence. "Robot vacuums come back about 1 in 5 times" is true; "your robot vacuum will come
   back" is not what it means.
5. **The de-duplication assumes the partner feed is a pure re-import.** In this export it is: 651
   pairs, identical on every field including the label. If that feed ever carries a later
   correction (an address fix, a cancellation), dropping it silently discards the correction.
6. **The October 2025 ÷100 repair is keyed to the month, not to the gateway.** It is verified, not
   guessed: all 700 October rows are inflated, and after dividing, the value-to-list-price
   distribution matches every other month almost exactly. But a repeat incident would need a new
   rule, and nothing in the pipeline would notice it happening.
7. **The service is a demo, not production.** Single process, stdlib server, no authentication, no
   logging, no persistence, no audit trail of which orders were scored, no rate limiting, no
   concurrency guarantees. It will hold up for a review and a pilot. It should not be pointed at
   the live order feed as it stands.
8. **One global threshold.** A ₹45 call is cheap enough that per-order thresholds barely change the
   answer, so I did not build them for calls (I did for holds, where the order value matters). At a
   higher call cost this becomes wrong quickly.
9. **No monitoring, no drift alarm.** Trained on data ending 30 June 2026. If returns shift because
   of a new SKU, the gateway, or festive demand, calibration drifts and nothing in what I handed
   over will tell you.
10. **The model extrapolates badly on an implausible order value.** `order_value_inr` cannot be
    trusted (see the October incident), and a linear model given a value far outside its training
    range will answer confidently and wrongly: a ₹21,999 order on a ₹4,048 SKU scored 0.0002
    ("ship") before I caught it, for a COD customer with three prior returns. The service now
    clamps the value-to-list-price ratio to the range real orders occupy and returns a warning
    field when it does. The scoring function itself is still unbounded, so anything calling the
    model directly rather than the service is exposed to this.
11. **predictions.csv flags 32.5% of test orders against 31.7% on the holdout.** Expected, since
    the file comes from a refit, and consistent, but I cannot verify it without the labels.

## 6. What did you deliberately leave out, and why that rather than something else?

- **No LLM in the scoring path.** It would not be reproducible, would cost money per order, and
  could not be defended to Finance. The reasons are generated from the model itself, in
  milliseconds, for free.
- **The two service columns, despite them being worth 0.99 AUC.** `last_service_event_type` and
  `pickup_scheduled_at` are written when a return is already in progress. Keeping them would have
  made the model look excellent and useless, because at dispatch time neither value exists. The
  ablation is in the report: 0.9955 with, 0.7621 without. This was the single biggest decision in
  the task and it is the one I would most want to be judged on.
- **Deep learning or per-customer sequence models.** Customer history exists here as two integers,
  prior orders and prior returns. There is nothing to sequence.
- **`delivery_note` free text.** I measured it: every keyword bucket lands between 0.09 and 0.13
  against a base rate of 0.114, and it added 0.007 AUC. It is a note field, not a signal, and
  tokenising it would have been the kind of complexity that reads as sophistication and buys
  nothing. Only "note missing" survives.
- **Pincode as a feature.** 3,479 distinct values over 10.5k rows is memorisation waiting to
  happen. City and state carry the geography instead.
- **Per-segment thresholds and a hold optimiser.** Both depend on the unmeasured hold-prevention
  rate, so building them would have meant inventing the number that makes them look good.
- **A web framework.** The brief asks for something that starts from the README on a clean machine.
  Two dependencies and no build step beats FastAPI plus Docker plus a database for a pilot.

## 7. Anything you built or found that nobody asked for?

- **The second leak, in the same column everyone would catch the first one in.** Train has
  DEMO_DONE and INSTALL_DONE, which return 0% of the time because they mean "already delivered and
  demoed". Test has INSTALL_BOOKED, which never appears in train. So the column is as-of-export-day
  in one file and at-dispatch in the other. Dropping REVERSE_PICKUP is the obvious catch; noticing
  that the remaining levels are still post-hoc, and that the test file speaks a different dialect,
  is the one that matters.
- **Proof that the accuracy target is arithmetically impossible**, rather than an opinion: best
  threshold accuracy 0.8956, majority baseline 0.8848.
- **A threshold sweep that agrees with the algebra.** The analytically derived break-even (0.1118)
  is also the empirical maximum of net rupees on the holdout (₹37,738; the neighbours at 0.10 and
  0.15 give ₹36,055 and ₹36,312). Two independent routes to the same operating point, which is the
  strongest evidence I have that the threshold is not a tuned knob.
- **The call-everybody comparison.** Net ₹2,942 against ₹37,738 for the targeted version. It
  answers "why not just call everyone" with a number instead of an argument.
- **The Shield number.** Excluding Shield costs ₹22,015 a quarter in modelled value, which turns
  "we'll deal with Shield later" into a decision with a price on it.
- **The README's pincode explanation is wrong.** It says 000000 means walk-in partner orders. In
  train the 848 zero-pincode rows are mostly app, marketplace and web orders (300/238/218) with
  only 92 partner_outlet. Nothing depends on this, since the flag is treated as missing either
  way, but the documentation does not match the file.
- **A note for whoever builds the next feature.** Policy §9 says resolution events before
  1 October 2025 came from the legacy Zoho log in UTC and were never converted. Any pickup-latency
  or time-of-day feature built on pre-October-2025 service timestamps is 5 hours 30 minutes off.
  It does not affect this model, because those columns are dropped, and it will quietly corrupt
  the first feature anyone adds back.
- **A service that starts without the data pack.** Verified: `/health`, `/sample`, `/score` and the
  screen all work with the data directory absent, because everything the model needs is in the
  committed bundle.

## 8. What did you use AI for?

> **[CANDIDATE TO COMPLETE. This must be your own honest account. The review is looking for
> specifics, not modesty: which tools and models, where they helped, where they wasted your time,
> and what you threw away. Suggested things to cover, only if true for you: which assistant(s) you
> used for the EDA and the leakage check; whether you had it draft the service and how much you
> rewrote; where it was wrong; what you discarded. Do not claim work you did not do.]**

## 9. Someone picks this up on Monday and you are unreachable. The three things they need to know.

1. **The two service columns are a trap, not treasure.** If you re-add `last_service_event_type` or
   `pickup_scheduled_at` the AUC jumps to 0.99 and the model becomes worthless, because both are
   written after a return is approved and neither exists at dispatch. The ablation in
   `reports/eval_report.md` shows both numbers. Anyone who "improves" this model by restoring them
   has broken it.
2. **The threshold is a money decision, not a tuning parameter.** p\* = call cost / (prevention rate
   × return cost) = 45 / (0.35 × 1150) = 0.1118. If Finance changes the cost of a return, or the
   pilot shows a different prevention rate, the threshold moves and it should. Do not retune it by
   eye against accuracy.
3. **Never hold a Shield order.** Holds cancel 12% of the time and Shield members are the highest
   value customers in the file, which is the one way this project can lose money in a way the model
   cannot see. Call them. Also, the 35% prevention figure is unvalidated and the whole rupee case
   scales with it, so run the four-week counted pilot before anyone quotes the annual number.

## 10. Honest hours spent.

> **[CANDIDATE TO COMPLETE: total hours, and roughly how they split across reading the pack, EDA
> and the leakage check, modelling, the service, and the write-up.]**

## 11. What does one prediction cost, and what would a month cost at Kestrel's volume (about 700 orders/month)? Show the arithmetic.

**One prediction: ₹0 in marginal compute.** Scoring is one logistic regression: a dot product over
about 40 encoded inputs, roughly 0.1 ms on one CPU core. The four reasons cost five more dot
products. There is no GPU, no network call, no third-party API and no per-token billing. Electricity
for 700 scores a month is under a hundredth of a kilowatt-hour, which rounds to zero.

**A month at 700 orders:**

| Item | Arithmetic | Cost |
|---|---|---|
| Model compute | 700 × ₹0 | ₹0 |
| Calls placed | 700 × 31.7% flagged = 222 calls × ₹45 | ₹9,972 |
| Server, if it needs its own | any small existing box; nothing new required | ₹0 |
| **Total running cost** | | **₹9,972/month** |

Against that, prevented returns: 222 calls at a 25.1% hit rate = 56 true flags, times the 35%
prevention rate = 19.5 returns prevented, times ₹1,150 = ₹22,400. Net **₹12,425/month**, which is
the same figure as the per-order scaling of the holdout (₹37,738 ÷ 2,126 × 700). The only real
recurring cost in the system is the phone call, and that is a decision Kestrel already makes, not a
cost of the software.

---

## The screen recording (yours to record)

The recording is yours to make, not something that can be generated. Suggested 3-minute shape, no
slides:

- **0:00 to 0:40, the problem and the trap.** 11.5% of orders return. Two columns in the file
  predict the label perfectly (0.99 AUC) and are unusable, because they are written after the
  return starts. Show the ablation table, one sentence on why you caught it.
- **0:40 to 1:20, the number.** Time-based holdout, what the model catches and misses, and say out
  loud that 95% accuracy was never on the table and why (88.5% is "flag nothing").
- **1:20 to 2:30, the thing running.** Paste an order into the screen, get a score, a band and four
  reasons a clerk would understand. Then the same for a clean order, to show it is not just a
  flag-everything model.
- **2:30 to 3:00, what you threw away and what you would do next.** The LLM scoring idea, the
  leakage columns, the deep model that lost to a straight line, and the four-week pilot that would
  settle the 35%.

## What still has to be done by the candidate

- Record the 3-minute screen recording.
- Fill in Q8 (AI usage) and Q10 (hours honestly spent).
- Paste these answers into the tracker form and submit, with the repo URL.
