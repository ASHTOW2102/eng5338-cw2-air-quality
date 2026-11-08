# ENG5338 CW2 — Virtual CO Sensor (UCI Air Quality)

Group coursework: predict true CO concentration (`co_gt`, mg/m³) from five low-cost metal-oxide sensors, weather and time of day, and recommend a scikit-learn pipeline a deployed "virtual sensor" could use.

## Engineering question and constraints (draft — team to confirm)

**Question:** Can a model running on cheap sensor readings replace the reference CO analyser between calibrations?

| ID | Constraint | Value |
|----|-----------|-------|
| C1 | Accuracy | ≥ 90 % of predictions within ±0.5 mg/m³ of the true CO |
| C2 | Latency  | ≤ 1 ms prediction time per sample |

> ⚠️ **Open decision:** with untuned models no model reaches 90 % on the winter fold, where it drops to about 60 %. Before tuning, decide whether to keep ±0.5 or use a relative/absolute hybrid tolerance justified from the application.

## Shared protocol (frozen — owner M1)

- **Data:** `data/air-quality.csv`. Replace `-200` with NaN, drop `nmhc_gt` (90 % missing), drop rows with no `co_gt` → 7,674 rows.
- **Features (12):** 5 sensors + temperature, RH, AH + `hour_sin`, `hour_cos`, `weekday`, `is_weekend`.
- **Split:** chronological 80/20. Train is 2004-03-10 → 2005-01-29; test is 2005-01-29 → 2005-04-04.
- **CV:** `TimeSeriesSplit(n_splits=5)` on train only.
- **Metrics:** MAE (primary, used for refit), RMSE, R², % within tolerance; train scores reported beside validation scores.
- **Preprocessing:** median imputation (+ StandardScaler where the model needs it), fitted inside the pipeline so nothing leaks between folds.
- **Reference:** Linear Regression and Ridge. Ridge gets validation MAE 0.430 and 72.6 % within tolerance.

All of this lives in [`src/cw2_common.py`](src/cw2_common.py). Import it and **do not redefine the split, folds or metrics**.

## Repository layout

```
data/air-quality.csv                       UCI Air Quality dataset (CC BY 4.0)
src/cw2_common.py                          shared protocol: data, features, split, CV, scorers, run_search()
notebooks/ENG5338_CW2_notebook.ipynb       THE group notebook (submitted). M1 merges everyone's cells here.
notebooks/members/m2_random_forest.ipynb   M2 working notebook
notebooks/members/m3_hist_gradient_boosting.ipynb   M3
notebooks/members/m4_svr_knn.ipynb         M4
notebooks/members/m5_cost_and_test.ipynb   M5
results/figures/                           exported figures for the report/slides
report/template_cw2_ENG5.tex               report template (max 8 pages + 1 appendix)
```

## Roles

| Member | Role | Notebook part |
|--------|------|---------------|
| M1 — Ashish Chaturvedi | Data, question & constraints, shared protocol, linear reference, overfitting analysis, notebook integration | Parts A, E |
| M2 | Random Forest | Part B |
| M3 | HistGradientBoosting | Part C |
| M4 | SVR (RBF) / KNN | Part D |
| M5 | Cost measurement (timing, latency vs C2, scalability) and the single final test score | Parts F, G |

Writing: M1 → report §1–2, M2–M4 → their model in §3, M5 → cost in §3 and §4.

## How to work

**Setup (local):**
```bash
git clone https://github.com/ASHTOW2102/eng5338-cw2-air-quality.git
cd eng5338-cw2-air-quality
pip install -r requirements.txt
jupyter notebook
```

**Setup (Google Colab):** in the first cell run
```python
!git clone https://github.com/ASHTOW2102/eng5338-cw2-air-quality.git
import sys; sys.path.insert(0, "/content/eng5338-cw2-air-quality/src")
from cw2_common import *
```

**Rules:**
1. Work **only in your own notebook** in `notebooks/members/`. Two people editing the same `.ipynb` causes merge conflicts that are painful to resolve.
2. Always `git pull` before you start, then commit and push small changes often.
3. Use `run_search(name, pipe, grid)` from `cw2_common` so every model has the same folds, metrics and train/val columns.
4. **Never use `X_test`.** Only M5 scores it, once, on the agreed pipeline.
5. When your model is done, tell M1 your best params, train/val MAE, % within tolerance and fit time. M1 merges your cells into the group notebook.
6. Only M1 edits `src/cw2_common.py` and `notebooks/ENG5338_CW2_notebook.ipynb`.

## Submission (one per group)

- `ENG5338_CW2_report_<groupID>.pdf`
- `ENG5338_CW2_presentation_<groupID>.pdf` (10 min + 10 min Q&A, where any member can be asked about any part of the notebook)
- `ENG5338_CW2_notebook_<groupID>.ipynb`, which must reproduce every result in the report. Put this repo's link on the report's first page.
