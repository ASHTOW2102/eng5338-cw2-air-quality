# ENG5338 CW2 — Virtual CO Sensor (UCI Air Quality)

Group notebook: `ENG5338_CW2_notebook.ipynb`

| Part | Owner | Content |
|------|-------|---------|
| A | M1 | Data cleaning, EDA, question & constraints, shared protocol (chronological 80/20 split, `TimeSeriesSplit(5)`, scorers, `run_search`), Linear/Ridge reference |
| B | M2 | Random Forest |
| C | M3 | HistGradientBoosting |
| D | M4 | SVR / KNN |
| E | M1 | Overfitting analysis |
| F | M5 | Computational cost |
| G | M5 | Final recommended pipeline, scored once on the test set |

**To run:** put `air-quality.csv` in the same folder as the notebook, then run all cells. It needs numpy, pandas, scikit-learn, matplotlib and seaborn.

## M5 — Testing and Cost Lead

The M5 module, `testing.py`, benchmarks the team's finalized model pipelines and supports selection of the recommended model. The Python file is maintained separately from the shared notebook.

### M5 responsibilities

- Compare Linear Regression/Ridge, Random Forest, HistGradientBoosting and SVR or KNN using M1's fixed data protocol.
- Measure single-threaded training time, validation prediction latency (ms per sample) and, where implemented, memory/model size.
- Compare validation MAE, R² and the percentage of predictions within ±0.5 mg/m³.
- Check the agreed constraints: at least 90% within ±0.5 mg/m³ and prediction latency no greater than 1 ms per sample.
- Recommend the most suitable pipeline **using validation results only**.
- Evaluate the selected model once on the untouched final test set.

### Running the script

When `testing.py` has been committed to this branch and the required teammate pipelines are available:

```powershell
python testing.py
```

The script must use the same chronological 80/20 split, five-fold `TimeSeriesSplit`, feature set and preprocessing defined by M1 in `ENG5338_CW2_notebook.ipynb`. Do not create independent random splits or use the test set for model selection.

> The `Test_and_Cost` branch documents M5's work. `testing.py` must be added from the local coursework directory before issuing the `v1.0` tag.
