import pickle
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc
)

# =========================
# STREAMLIT PAGE SETTINGS
# =========================
st.set_page_config(page_title="Attrition Prediction Dashboard", layout="wide")
st.title("📊 Attrition Prediction Dashboard")
st.caption("Logistic Regression (Tuned + F2-optimized) — with safe/optimized LIME & SHAP")

# =========================
# HELPERS & CACHING
# =========================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    with open("results.pkl", "rb") as f:
        artifacts = pickle.load(f)

    # Format fleksibel
    if len(artifacts) == 3:
        X_test_scaled, y_test, y_proba_best = artifacts
        # coba ambil nama feature asli dari X_test_scaled kalau itu DataFrame
        if hasattr(X_test_scaled, "columns"):
            feature_names = list(X_test_scaled.columns)
            X_test_scaled = X_test_scaled.values  # pastikan jadi numpy
        else:
            feature_names = [f"Feature {i}" for i in range(X_test_scaled.shape[1])]
    elif len(artifacts) == 4:
        X_test_scaled, y_test, y_proba_best, feature_names = artifacts
        if hasattr(X_test_scaled, "values"):  # kalau masih DataFrame
            X_test_scaled = X_test_scaled.values
    else:
        raise ValueError("results.pkl tidak sesuai format, harus 3 atau 4 objek.")

    with open("best_log_reg.pkl", "rb") as f:
        best_log_reg = pickle.load(f)

    X_test_np = np.asarray(X_test_scaled)
    return X_test_np, y_test, y_proba_best, best_log_reg, feature_names

X_test_np, y_test, y_proba_best, model, feature_names = load_artifacts()

@st.cache_data(show_spinner=False)
def compute_curves(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)
    return fpr, tpr, precision, recall, roc_auc, pr_auc

# =========================
# CACHE LIME EXPLAINER
# =========================
@st.cache_resource(show_spinner=False)
def get_lime_explainer(X_train, feature_names):
    return LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=["Stay", "Attrition"],
        mode="classification"
    )

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("⚙️ Pengaturan")
threshold = st.sidebar.slider("Threshold Probabilitas", 0.0, 1.0, 0.35, 0.01)
y_pred = (y_proba_best >= threshold).astype(int)

# LIME controls
st.sidebar.markdown("---")
st.sidebar.subheader("LIME")
lime_idx = st.sidebar.number_input("Index sampel untuk LIME", 0, int(len(X_test_np)-1), 0)
lime_num_features = st.sidebar.slider("Top features (LIME)", 3, min(20, X_test_np.shape[1]), 10)
lime_num_samples = st.sidebar.slider("Num samples (LIME)", 200, 5000, 1000, 100)

# SHAP controls
st.sidebar.markdown("---")
st.sidebar.subheader("SHAP")
shap_sample_size = st.sidebar.slider("Jumlah sampel SHAP", 10, min(500, len(X_test_np)), 100, 10)
shap_plot_type = st.sidebar.selectbox("Tipe plot SHAP", ["bar", "beeswarm"], index=0)

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 Confusion Matrix", "📈 ROC Curve", "📊 Precision-Recall", "🟢 LIME", "🟣 SHAP"
])

# =========================
# TAB 1 — Confusion Matrix
# =========================
with tab1:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Stay", "Attrition"], yticklabels=["Stay", "Attrition"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.markdown("**Classification Report**")
    st.text(classification_report(y_test, y_pred, target_names=["Stay", "Attrition"]))

# =========================
# TAB 2 — ROC
# =========================
with tab2:
    st.subheader("ROC Curve")
    fpr, tpr, _, _, roc_auc, _ = compute_curves(y_test, y_proba_best)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

# =========================
# TAB 3 — PR
# =========================
with tab3:
    st.subheader("Precision-Recall Curve")
    _, _, precision, recall, _, pr_auc = compute_curves(y_test, y_proba_best)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    st.pyplot(fig)

# =========================
# TAB 4 — LIME
# =========================
with tab4:
    st.subheader("LIME Explanation (on-demand)")
    st.write("Klik tombol di bawah untuk menghitung LIME (cached explainer, jadi lebih cepat).")

    run_lime = st.button("▶️ Jalankan LIME untuk 1 sampel")
    if run_lime:
        with st.spinner("Menghitung LIME..."):
            try:
                explainer = get_lime_explainer(X_test_np, feature_names)
                instance = X_test_np[int(lime_idx)]
                exp = explainer.explain_instance(
                    instance,
                    lambda x: model.predict_proba(np.asarray(x).reshape(len(x), -1)),
                    num_features=int(lime_num_features),
                    num_samples=int(lime_num_samples)
                )
                st.components.v1.html(exp.as_html(), height=800)
            except Exception as e:
                st.error(f"Gagal menghitung LIME: {e}")

# =========================
# TAB 5 — SHAP
# =========================
with tab5:
    st.subheader("SHAP Summary (on-demand)")
    st.write("Klik tombol di bawah untuk menghitung SHAP pada subset data.")

    run_shap = st.button("▶️ Jalankan SHAP untuk subset")
    if run_shap:
        with st.spinner("Menghitung SHAP..."):
            try:
                import shap
                X_sub = X_test_np[:int(shap_sample_size)]
                explainer = shap.Explainer(model, X_sub, feature_names=feature_names)
                shap_values = explainer(X_sub)

                if shap_plot_type == "bar":
                    shap.summary_plot(shap_values, X_sub, feature_names=feature_names, plot_type="bar", show=False)
                else:
                    shap.summary_plot(shap_values, X_sub, feature_names=feature_names, show=False)

                fig = plt.gcf()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Gagal menghitung/menampilkan SHAP: {e}")

# =========================
# FOOTNOTE
# =========================
st.caption(
    "Tip: jika masih terasa lambat, kecilkan 'Num samples (LIME)' dan 'Jumlah sampel SHAP' di sidebar. "
    "LIME/SHAP dihitung hanya saat tombol ditekan."
)
