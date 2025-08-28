import pickle
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve,
    precision_recall_curve, auc, fbeta_score, accuracy_score, f1_score
)
from lime.lime_tabular import LimeTabularExplainer

# =========================
# STREAMLIT PAGE SETTINGS
# =========================
st.set_page_config(page_title="Attrition Prediction Dashboard", layout="wide")
st.title("📊 Attrition Prediction Dashboard")
st.caption("Logistic Regression (Tuned + F2-optimized) — with LIME & SHAP explanations")

# =========================
# HELPERS & CACHING
# =========================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    with open("results.pkl", "rb") as f:
        artifacts = pickle.load(f)

    if len(artifacts) == 3:
        X_test_scaled, y_test, y_proba_best = artifacts
        feature_names = list(X_test_scaled.columns) if hasattr(X_test_scaled, "columns") else [f"Feature {i}" for i in range(X_test_scaled.shape[1])]
        X_test_scaled = X_test_scaled.values if hasattr(X_test_scaled, "values") else X_test_scaled
    elif len(artifacts) == 4:
        X_test_scaled, y_test, y_proba_best, feature_names = artifacts
        X_test_scaled = X_test_scaled.values if hasattr(X_test_scaled, "values") else X_test_scaled
    else:
        raise ValueError("results.pkl harus berisi 3 atau 4 objek")

    with open("best_log_reg.pkl", "rb") as f:
        best_log_reg = pickle.load(f)

    return np.asarray(X_test_scaled), y_test, y_proba_best, best_log_reg, feature_names

X_test_np, y_test, y_proba_best, model, feature_names = load_artifacts()

@st.cache_data(show_spinner=False)
def compute_curves(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)
    return fpr, tpr, precision, recall, roc_auc, pr_auc

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
st.sidebar.header("⚙️ Pengaturan Threshold & Metrics")
threshold = st.sidebar.slider("Threshold Probabilitas", 0.0, 1.0, 0.35, 0.01)
y_pred = (y_proba_best >= threshold).astype(int)

# Metrics tambahan
f2 = fbeta_score(y_test, y_pred, beta=2)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
st.sidebar.markdown(f"**F2-score:** {f2:.3f}")
st.sidebar.markdown(f"**F1-score:** {f1:.3f}")
st.sidebar.markdown(f"**Accuracy:** {accuracy:.3f}")

# LIME controls
st.sidebar.markdown("---")
st.sidebar.subheader("LIME")
lime_num_features = st.sidebar.slider("Top features (LIME)", 3, min(20, X_test_np.shape[1]), 5)
lime_num_samples = st.sidebar.slider("Num samples (LIME)", 100, 2000, 500, 100)

# SHAP controls
st.sidebar.markdown("---")
st.sidebar.subheader("SHAP")
shap_sample_size = st.sidebar.slider("Jumlah sampel SHAP", 10, min(500, len(X_test_np)), 100, 10)
shap_plot_type = st.sidebar.selectbox("Tipe plot SHAP", ["bar", "beeswarm"], index=0)

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📌 Confusion Matrix", "📈 ROC Curve", "📊 Precision-Recall",
    "🟢 LIME", "🟣 SHAP", "📊 Probabilitas Prediksi", "📥 Upload CSV Baru"
])

# =========================
# TAB 1 — Confusion Matrix + Metrics
# =========================
with tab1:
    st.subheader("Confusion Matrix & Metrics")
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
# TAB 2 — ROC Curve
# =========================
with tab2:
    st.subheader("ROC Curve")
    fpr, tpr, _, _, roc_auc, _ = compute_curves(y_test, y_proba_best)
    st.markdown(f"**ROC AUC:** {roc_auc:.3f}")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

# =========================
# TAB 3 — Precision-Recall Curve
# =========================
with tab3:
    st.subheader("Precision-Recall Curve")
    _, _, precision, recall, _, pr_auc = compute_curves(y_test, y_proba_best)
    st.markdown(f"**PR AUC:** {pr_auc:.3f}")
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
    st.write("Klik tombol di bawah untuk menghitung LIME untuk 1 sampel dari dataset test.")
    
    lime_idx = st.number_input("Index sampel untuk LIME", 0, int(len(X_test_np)-1), 0)
    run_lime = st.button("▶️ Jalankan LIME untuk 1 sampel test")
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
    st.write("Klik tombol di bawah untuk menghitung SHAP pada subset data test.")
    
    run_shap = st.button("▶️ Jalankan SHAP untuk subset test")
    if run_shap:
        with st.spinner("Menghitung SHAP..."):
            try:
                import shap
                X_sub = X_test_np[:int(shap_sample_size)]
                explainer = shap.LinearExplainer(model, X_sub, feature_names=feature_names)
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
# TAB 6 — Histogram Probabilitas Prediksi
# =========================
with tab6:
    st.subheader("Distribusi Probabilitas Prediksi")
    fig, ax = plt.subplots()
    ax.hist(y_proba_best[y_test==0], bins=20, alpha=0.6, label="Stay")
    ax.hist(y_proba_best[y_test==1], bins=20, alpha=0.6, label="Attrition")
    ax.axvline(threshold, color="red", linestyle="--", label="Threshold")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)

# =========================
# TAB 7 — Upload CSV Baru
# =========================
with tab7:
    st.subheader("Upload CSV Baru untuk Prediksi Batch")
    uploaded_file = st.file_uploader("Upload CSV baru (kolom harus sama dengan fitur model)", type="csv")
    
    if uploaded_file is not None:
        try:
            new_data = pd.read_csv(uploaded_file)
            
            if list(new_data.columns) != feature_names:
                st.error("Kolom CSV tidak sesuai dengan fitur model!")
            else:
                new_X = new_data.values
                new_proba = model.predict_proba(new_X)[:, 1]
                new_pred = (new_proba >= threshold).astype(int)
                new_data["Prob_Attrition"] = new_proba
                new_data["Prediksi"] = np.where(new_pred==1, "Attrition", "Stay")
                
                st.dataframe(new_data)
                
                # Histogram probabilitas baru
                fig, ax = plt.subplots()
                ax.hist(new_proba, bins=20, alpha=0.7, color="orange")
                ax.axvline(threshold, color="red", linestyle="--", label="Threshold")
                ax.set_xlabel("Predicted Probability")
                ax.set_ylabel("Count")
                ax.legend()
                st.pyplot(fig)
                
                st.markdown("⚠️ Untuk LIME/SHAP, pilih beberapa sampel dari CSV baru secara manual.")
                
        except Exception as e:
            st.error(f"Gagal membaca CSV baru: {e}")

# =========================
# FOOTNOTE
# =========================
st.caption(
    "Tip: jika masih terasa lambat, kecilkan 'Num samples (LIME)' dan 'Jumlah sampel SHAP' di sidebar. "
    "LIME/SHAP dihitung hanya saat tombol ditekan."
)
