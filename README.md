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
