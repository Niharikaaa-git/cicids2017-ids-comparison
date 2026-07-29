"""
utils.py — shared helpers used by every notebook in this project, so
that evaluation is computed identically for every model and file
naming stays consistent between notebooks.
"""
import os
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix,
)

PROCESSED_DIR = "processed_data"
RESULTS_DIR = "results"
MODELS_DIR = "models"
PLOTS_DIR = "plots"

for d in (PROCESSED_DIR, RESULTS_DIR, MODELS_DIR, PLOTS_DIR):
    os.makedirs(d, exist_ok=True)


def evaluate(name, y_test, y_pred, y_proba, train_time, infer_time, n_classes):
    """Computes the full 9-metric evaluation suite. Same function used
    by every model notebook so scores are directly comparable."""
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec,
        "F1-Score": f1, "ROC-AUC": auc, "MCC": mcc, "Cohen's Kappa": kappa,
        "Train Time (s)": train_time, "Prediction Time (s)": infer_time,
        "Confusion Matrix": cm.tolist(),
    }


def save_results(name, metrics_dict, history=None):
    """Saves one model's metrics (+ optional Keras training history) as
    JSON under results/, so 07_Comparison.ipynb can later collect all
    of them without needing the original model objects in memory."""
    payload = dict(metrics_dict)
    if history is not None:
        payload["history"] = history
    path = os.path.join(RESULTS_DIR, f"{name.replace(' ', '_').replace('+', '')}_results.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results to {path}")


def load_all_results():
    """Loads every *_results.json in results/ into a list of dicts."""
    out = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.endswith("_results.json"):
            with open(os.path.join(RESULTS_DIR, fname)) as f:
                out.append(json.load(f))
    return out
