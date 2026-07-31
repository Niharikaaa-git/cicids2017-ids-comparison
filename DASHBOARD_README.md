# IDS Comparison Dashboard

A Streamlit app that turns your `results/` and `plots/` folders into an
interactive dashboard — good for showing live during your internship
presentation instead of flipping through static images.

## Setup

1. Put `app.py` in the **same folder** as your notebooks (`00_Preprocessing.ipynb`
   through `07_Comparison.ipynb`) — it needs to sit next to your `results/`
   and `plots/` directories.
2. Install the extra packages it needs (you'll already have most of these):
   ```bash
   pip install streamlit plotly pillow
   ```
3. Make sure you've already run `00` through `07` at least once, so
   `results/comparison_table.csv` and the `plots/*.png` files exist.

## Run it

```bash
streamlit run app.py
```

This opens a browser tab automatically (usually `http://localhost:8501`).
Leave the terminal window open while you're presenting — closing it shuts
the app down, same as the Jupyter server.

## What's in it

Five tabs:
- **Comparison Table** — the full 9-metric results table, sortable, with a download button
- **Metric Charts** — pick any metric from a dropdown, see it as an interactive bar chart across all 6 models, plus an ML-family vs DL-family comparison
- **Confusion Matrices** — pick a model from a dropdown to view its confusion matrix, or expand to see all 6 side-by-side
- **Explainability (SHAP)** — the summary and waterfall plots for your best ML model and best DL model
- **Data Exploration (EDA)** — your class distribution, imbalance, correlation, and feature distribution plots from preprocessing

Everything at the top (Best Model, Best F1-Score, Best Accuracy, Best ROC-AUC)
updates automatically based on whatever is in `results/comparison_table.csv`
— so once you re-run your notebooks with the real full dataset, just restart
the app and it picks up the new numbers with no code changes needed.

## For your presentation

This is meant to run live during your demo: switch tabs to walk through
EDA → model comparison → best model → explainability, instead of a static
slide deck. You can still keep your PPTX for the written parts (Introduction,
Literature Survey, Methodology) and switch over to this dashboard specifically
for the Results and Discussion section.
