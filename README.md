# CIC-IDS2017 — ML vs DL Comparison (Split Notebooks)

Same pipeline as the single-notebook version, but split into one
notebook per model so you can run, inspect, and take outputs from each
model independently. All 6 models still train and evaluate on the
**exact same preprocessed data**, so the comparison stays fair.

## Files

| File | What it does |
|---|---|
| `utils.py` | Shared code used by every notebook: the `evaluate()` metrics function and result save/load helpers. Must sit in the same folder as the notebooks. |
| `00_Preprocessing.ipynb` | Load all CIC-IDS2017 CSVs → clean → EDA → label encode → split → scale → SMOTE → **saves everything to `processed_data/`** |
| `01_RandomForest.ipynb` | Loads `processed_data/`, trains RF, evaluates, saves its own results/plot/model |
| `02_XGBoost.ipynb` | Same, for XGBoost |
| `03_LightGBM.ipynb` | Same, for LightGBM |
| `04_MLP.ipynb` | Same, for the MLP |
| `05_CNN1D.ipynb` | Same, for the 1D CNN |
| `06_AutoencoderMLP.ipynb` | Same, for Autoencoder+MLP |
| `07_Comparison.ipynb` | Collects all 6 models' saved results → comparison table + bar charts + confusion-matrix grid → SHAP on the best ML model and best DL model |

## Run order

1. **`00_Preprocessing.ipynb`** — run once, first. Set `DATA_DIR` to your
   CIC-IDS2017 CSV folder in the "Load Dataset" cell.
2. **`01`–`06`**, in any order (each is independent, they only depend on
   step 1). Run each fully; every notebook saves its own outputs before
   finishing.
3. **`07_Comparison.ipynb`** — run last, after all 6 model notebooks
   have been run at least once.

## Where outputs land

- `processed_data/` — arrays + encoders from step 1 (shared inputs)
- `results/` — one `*_results.json` per model (metrics + confusion
  matrix + DL training history), plus `comparison_table.csv` from `07`
- `models/` — one saved model file per model (`.joblib` for ML, `.keras`
  for DL)
- `plots/` — one confusion-matrix PNG per model, training-curve PNGs
  for the 3 DL models, EDA plots from `00`, and the final comparison
  bar charts / SHAP plots from `07`

Because each model's results are a separate JSON/PNG/model file, you
can directly compare, share, or drop any single model's output into
your paper without needing to touch the others.

## Install

```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn tensorflow shap matplotlib seaborn joblib jupyter
```

## Design notes (same as the combined version)

- Scaler fit on train only (no leakage); SMOTE applied to training data
  only, with `k_neighbors` auto-capped for rare classes.
- All 6 models are trained on the identical SMOTE-resampled training
  set and scored with the identical `evaluate()` function on the
  identical untouched test set — that's what `utils.py` enforces.
- Autoencoder is trained unsupervised on pre-SMOTE data; its encoder
  then transforms the resampled training data for the MLP head.
- SHAP: `TreeExplainer` for the best ML model, `KernelExplainer` for
  the best DL model (broader TF-version compatibility than
  `DeepExplainer`), both on a capped sample size for tractable runtime.
- `SEED = 42` throughout for reproducibility.

## If your dataset is very large

Add `.sample(n=..., random_state=SEED)` right after the merge step in
`00_Preprocessing.ipynb`, and/or reduce `nsamples` in the
`KernelExplainer.shap_values(...)` call in `07_Comparison.ipynb`.

## What you DO need to redo each time you come back:

Open Git Bash
cd into your project folder:
bash
   cd ~/Desktop/[folder name]
Start the Jupyter server again:
bash
   python.exe -m notebook
Open the browser link it gives you
   Open whichever notebook you want to continue with

## Claude's Plan was REAL

