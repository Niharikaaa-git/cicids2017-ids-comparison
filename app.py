"""
IDS Model Comparison Dashboard
------------------------------
A Streamlit app that reads the results/, plots/, and processed_data/
folders produced by 00_Preprocessing.ipynb -> 07_Comparison.ipynb and
presents them as an interactive dashboard.

Run with:
    streamlit run app.py

Must be run from the same folder as your results/ and plots/ directories
(i.e. your project folder, alongside the 8 notebooks).
"""
import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

RESULTS_DIR = "results"
PLOTS_DIR = "plots"

st.set_page_config(page_title="IDS Model Comparison Dashboard", layout="wide")

# ---------------------------------------------------------------- helpers
@st.cache_data
def load_comparison_table():
    path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_all_results():
    results = []
    if not os.path.isdir(RESULTS_DIR):
        return results
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.endswith("_results.json"):
            with open(os.path.join(RESULTS_DIR, fname)) as f:
                results.append(json.load(f))
    return results


def show_image_if_exists(path, caption=None):
    if os.path.exists(path):
        st.image(Image.open(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Not found yet: {path}")


# ---------------------------------------------------------------- header
st.title("🛡️ Intrusion Detection: ML vs DL Model Comparison")
st.caption("CIC-IDS2017 — Random Forest, XGBoost, LightGBM, MLP, 1D CNN, Autoencoder+MLP")

df = load_comparison_table()
all_results = load_all_results()

if df is None or len(all_results) == 0:
    st.error(
        "No results found. Run `00_Preprocessing.ipynb` through `07_Comparison.ipynb` "
        "first, then launch this app from the same project folder."
    )
    st.stop()

# ---------------------------------------------------------------- top metrics
best_row = df.sort_values("F1-Score", ascending=False).iloc[0]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Best Model", best_row["Model"])
col2.metric("Best F1-Score", f"{best_row['F1-Score']:.3f}")
col3.metric("Best Accuracy", f"{best_row['Accuracy']:.3f}")
col4.metric("Best ROC-AUC", f"{best_row['ROC-AUC']:.3f}")

st.divider()

# ---------------------------------------------------------------- tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Comparison Table", "📈 Metric Charts", "🧩 Confusion Matrices",
    "🔍 Explainability (SHAP)", "🗂️ Data Exploration (EDA)"
])

# ---- Tab 1: Table
with tab1:
    st.subheader("Full Results Table")
    st.dataframe(
        df.style.background_gradient(
            subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], cmap="Greens"
        ),
        use_container_width=True,
    )
    st.download_button(
        "Download table as CSV",
        df.to_csv(index=False).encode("utf-8"),
        "comparison_table.csv",
        "text/csv",
    )

# ---- Tab 2: Interactive charts
with tab2:
    st.subheader("Metric Comparison Across Models")
    metric = st.selectbox(
        "Choose a metric",
        ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "MCC",
         "Cohen's Kappa", "Train Time (s)", "Prediction Time (s)"],
    )
    sorted_df = df.sort_values(metric, ascending=False)
    fig = px.bar(
        sorted_df, x="Model", y=metric, color="Model",
        text_auto=".3f", title=f"{metric} by Model",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("ML vs DL Family Comparison")
    ML_MODELS = {"Random Forest", "XGBoost", "LightGBM"}
    df_family = df.copy()
    df_family["Family"] = df_family["Model"].apply(lambda m: "ML" if m in ML_MODELS else "DL")
    fig2 = px.box(df_family, x="Family", y=metric, points="all", color="Family",
                  title=f"{metric} distribution: ML family vs DL family")
    st.plotly_chart(fig2, use_container_width=True)

# ---- Tab 3: Confusion matrices
with tab3:
    st.subheader("Per-Model Confusion Matrices")
    model_files = {
        "Random Forest": "RandomForest_confusion_matrix.png",
        "XGBoost": "XGBoost_confusion_matrix.png",
        "LightGBM": "LightGBM_confusion_matrix.png",
        "MLP": "MLP_confusion_matrix.png",
        "1D CNN": "CNN1D_confusion_matrix.png",
        "Autoencoder+MLP": "AutoencoderMLP_confusion_matrix.png",
    }
    chosen = st.selectbox("Choose a model", list(model_files.keys()))
    show_image_if_exists(os.path.join(PLOTS_DIR, model_files[chosen]), caption=chosen)

    with st.expander("Show all confusion matrices side-by-side"):
        show_image_if_exists(os.path.join(PLOTS_DIR, "all_confusion_matrices.png"))

# ---- Tab 4: SHAP
with tab4:
    st.subheader("SHAP Explainability — Best ML Model")
    c1, c2 = st.columns(2)
    with c1:
        show_image_if_exists(os.path.join(PLOTS_DIR, "shap_summary_best_ml.png"), "Summary Plot")
    with c2:
        show_image_if_exists(os.path.join(PLOTS_DIR, "shap_waterfall_best_ml.png"), "Waterfall Plot")

    st.subheader("SHAP Explainability — Best DL Model")
    c3, c4 = st.columns(2)
    with c3:
        show_image_if_exists(os.path.join(PLOTS_DIR, "shap_summary_best_dl.png"), "Summary Plot")
    with c4:
        show_image_if_exists(os.path.join(PLOTS_DIR, "shap_waterfall_best_dl.png"), "Waterfall Plot")

# ---- Tab 5: EDA
with tab5:
    st.subheader("Dataset Exploration")
    c1, c2 = st.columns(2)
    with c1:
        show_image_if_exists(os.path.join(PLOTS_DIR, "eda_class_distribution.png"), "Class Distribution")
        show_image_if_exists(os.path.join(PLOTS_DIR, "eda_correlation_heatmap.png"), "Correlation Heatmap")
    with c2:
        show_image_if_exists(os.path.join(PLOTS_DIR, "eda_class_imbalance_pie.png"), "Class Imbalance")
        show_image_if_exists(os.path.join(PLOTS_DIR, "eda_histograms.png"), "Feature Distributions")

st.divider()
st.caption("Built for the CIC-IDS2017 ML vs DL intrusion detection comparison project.")
