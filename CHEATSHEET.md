# Notebook cheat sheet: what every part does and why (Group AIR 2)

Read your own part closely and everyone else's once. In Q&A **anyone can be asked about any part**.
Then write the comments in your part **in your own words**.

---

## The big picture (everyone must know this)

- **Goal:** predict true CO (`co_gt`, mg/m³) from 5 cheap gas sensors + weather + time of day. This is a "virtual sensor".
- **C1 (accuracy):** ≥ 90% of predictions within ±0.5 mg/m³. **No model meets it** (best ≈ 80%).
- **C2 (speed):** ≤ 1 ms per prediction. **All models meet it.**
- **Recommended model:** HistGradientBoosting (HGB). Test MAE 0.350, R² 0.82, 79.9% within ±0.5.
- **Key insight:** every model drops to 57–65% in the Dec–Jan fold, so the problem is **drift over time** (season and sensor ageing), not one model overfitting. The fix is **recalibration** (retraining on recent data).

---

## Part A: M1 (Ashish): data and shared protocol

| Cell | What it does | Why |
|---|---|---|
| A.1 | Imports, `RANDOM_STATE = 42` | Same results every run |
| A.2 | Loads the CSV, sorts by time | The split depends on time order |
| A.3 | Replaces `-200` with NaN, counts missing values | The dataset uses -200 to mean "missing" |
| A.4 | Drops `nmhc_gt` (90% missing) and rows with no `co_gt` → 7,674 rows | Can't train or score without a target |
| A.5 | Summary statistics | CO is skewed: median 1.8, max 11.9 |
| A.6 | Histogram + mean CO by hour | Rush-hour peaks → time features are useful |
| A.7 | Correlation heatmap | S2 (NMHC sensor) r = 0.92 is the strongest input; weather barely correlates directly |
| A.8 | Daily true CO vs CO sensor | Shows **drift** → we must split by time |
| A.9–A.10 | hour, weekday, weekend; hour as sin/cos | sin/cos makes 23:00 and 00:00 neighbours |
| A.11 | 12 features (5 sensors + 3 weather + 4 time) | Every model uses the same list |
| A.12 | First 80% = train, last 20% = test | Test = the future (Jan–Apr 2005); used only once |
| A.13 | `TimeSeriesSplit(5)` | Each fold trains on the past and validates on the next block, so no future leakage |
| A.14 | C1, C2, `pct_within`, scorers | MAE is the main metric (same unit as C1) |
| A.15 | `make_preprocessor`: median impute (+ scale) | It sits inside the pipeline, so it is fitted on training folds only → no leakage |
| A.16 | `run_search()`: GridSearchCV with train scores | Every model is tuned the same way, so the comparison is fair |
| A.17–A.18 | Linear and Ridge (alpha 316) | The reference the brief requires. Ridge val MAE 0.430, 72.6% |

**Likely questions:** Why not a random split? (Drift: neighbouring hours would leak between train and test.) Why median imputation inside the pipeline? (No leakage.) Why MAE? (Same unit as C1, and less sensitive to spikes than RMSE.)

---

## Part B: M2 (Shamyuktha): Random Forest

| Cell | What it does | Why |
|---|---|---|
| Regularised search | 8 configs, 200 trees: depth 10/20, leaf 5/10, max_features 0.7/sqrt | Limit overfitting |
| rf_final + test cell | A forest scored on the test set (**exploration only**, marked in the notebook) | Not used for selection |
| Feature importance | S2 sensor ≈ 0.86 of the importance | One sensor carries most of the signal |
| Error diagnosis by CO level | Error grows with CO; above 3.4 mg/m³ only ~40% within | High CO is the weak spot |
| Improved grid | 200/300 trees, depth None/20, leaf 1/2 | Best RF: 300 trees, depth 20, leaf 2 → val MAE 0.369, 79.4% |
| Log target | `TransformedTargetRegressor` with log1p | Tried to handle the skew; no gain (0.371) |
| Sample weights | Extra weight on high-CO rows (1.5–8) | Tried to fix high-CO error; no gain (≈0.369–0.372) |

**Likely questions:** Why is the RF gap large (0.28)? (Deep trees memorise the training data.) Why not pick RF? (Same accuracy as HGB, but 11 s to train and 67 MB vs 0.19 s and 191 KB.) **Why is RF slow?** Hundreds of deep trees, each checking all 12 features at every split. We cut it from 600 to 300 trees with the same accuracy, which halved its cost and the notebook's run time.

---

## Part C: M3 (Pooja): HistGradientBoosting (the recommended model)

| Cell | What it does | Why |
|---|---|---|
| C.3 | Pipeline with just the model, no imputer | HGB handles NaN natively and doesn't need scaling |
| C.4–C.5 | 12 configs: lr 0.05, iter 100/200, leaves 7/15/31, L2 0/0.5 | Best: 100 iterations, 15 leaves, L2 0.5 |
| C.6 | Results: val MAE 0.360, 80.8% | Best of all models |
| C.7 | Out-of-fold predictions on the same folds | Gives honest predictions for every validation hour |
| C.8–C.9 | Actual vs predicted, residuals | Under-predicts high CO; error spread grows with CO |
| C.10–C.12 | Permutation importance on the last fold | S2 sensor 0.68, then hour_sin, weekday |
| C.13 | Overall OOF metrics | MAE 0.360, R² 0.82 |

**Likely questions:** Why HGB over RF? (Same or better accuracy, 60× faster to train, 350× smaller, fit time grows only slowly with more data → cheap to retrain for drift.) What does L2 / max_leaf_nodes do? (Both limit complexity → smaller overfitting gap.)

---

## Part D: M4 (Mithil): SVR and KNN

| Cell | What it does | Why |
|---|---|---|
| Part 1–6 (original) | Random split, KFold(shuffle), 8 features, RandomizedSearch for SVR, Grid for KNN | First exploration; scores look better (MAE 0.289) **only because a random split leaks neighbouring hours** |
| D.2 | SVR re-tuned under the shared protocol | Best C=1, gamma=0.03, eps=0.05 → val MAE 0.366, 79.8% |
| D.3 | KNN re-tuned under the shared protocol | Best k=15, distance, p=1 → 0.433, 71.8%, train MAE 0.002 (memorises) |

**Likely questions:** Why were the first SVR/KNN numbers better? (Random split = leakage, and no time features.) Why does SVR need scaling? (The RBF kernel uses distances.) Why does KNN overfit? (Distance weighting makes training predictions exact.)

---

## Part E: M1 (Ashish): overfitting analysis

| Cell | What it does | Why |
|---|---|---|
| E.1 | Ridge learning curve | Is the model data-limited or overfitting? |
| E.2 | Ridge % within per fold | Folds 4–5 (winter) drop |
| E.3 | Table of the best tuned model per family | One comparable table |
| E.4 | Train vs validation MAE bars | KNN gap 0.43, RF 0.28, HGB 0.20, Ridge 0.15 |
| E.5 | % within per fold, all models | **All models drop to 57–65% in Dec–Jan → drift** |
| E.6 | Learning curves, all models | The validation error stays above the training error for every model |

**Likely question:** How do you know it's drift and not overfitting? (If it were overfitting, only the flexible models would fail. All six fail in the same fold, including the linear ones.)

---

## Parts F–H: M5 (Siddharth): cost, selection and test

| Cell | What it does | Why |
|---|---|---|
| F–G | Initial benchmark with default models on one hold-out (**exploration**, marked) | First look at timing |
| H.1 | Single-threaded timing: fit, batch ms/sample, single-row ms, size | A fair comparison (same thread count for every model) |
| H.2 | Fit time at 25/50/75/100% of the data, exponent b | SVR b≈1.9 (quadratic), RF ≈1.1, HGB ≈0.2 |
| H.3 | Main table + accuracy-vs-cost plot | The report's main table |
| H.4 | Selection rule (validation only) | Keep the models that meet C2 → none meets C1 → closest to C1 = HGB |
| H.5–H.6 | Test set used **once** | MAE 0.350, 79.9% within: matches validation (80.8%) → no hidden overfitting |
| H.7 | Error by CO level and month | Below 2 mg/m³ ≈ 87–93%; above 5 mg/m³ only 37% |

**Likely questions:** Why single-threaded? (Fair comparison.) Single-row latency is 1.6–2 ms, so does C2 fail? (That's scikit-learn's per-call overhead; batch prediction is 0.008 ms. In production you'd batch or export the model.) What would change the recommendation? (Recalibration allowed, no retraining possible → SVR, a relative tolerance at high CO, strict single-call latency.)

---

## Honest notes for Q&A

- The test set was also scored inside some members' exploration cells (M2, M4, M5). These are marked in the notebook and **not used for selection**. The official test score is H.6.
- AI use (Claude) is declared in the report's statement of contribution. Be ready to say what you used it for.
