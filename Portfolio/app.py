"""
Streamlit app – BUY / HOLD / SELL signal prediction using the deployed
KernelPCA + Logistic Regression pipeline.

Run locally:
    streamlit run app.py

Requirements (same as requirements.txt):
    numpy==1.26.4
    scipy==1.15.1
    scikit-learn==1.3.2
    pandas==2.2.0
    imbalanced-learn==0.12.0
    shap==0.44.0
    streamlit
"""

import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
import shap
from imblearn.pipeline import Pipeline

# ── Load artefacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return load('./finalized_kpca_model.joblib')

@st.cache_resource
def load_explainer():
    with open('./explainer_kpca.shap', 'rb') as f:
        return shap.LinearExplainer.load(f)

model     = load_model()
explainer = load_explainer()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title('📈 Stock Signal Predictor – HW5')
st.markdown(
    'Upload your feature CSV (one row = one trading day) '
    'to get a **BUY / HOLD / SELL** signal.'
)

uploaded = st.file_uploader('Upload feature CSV', type='csv')

if uploaded:
    X_new = pd.read_csv(uploaded, index_col=0)

    preds     = model.predict(X_new)
    label_map = {1: '🟢 BUY', 0: '🟡 HOLD', -1: '🔴 SELL'}

    st.subheader('Predictions')
    out = pd.DataFrame({'Signal': preds}, index=X_new.index).replace(label_map)
    st.dataframe(out)

    # ── Probability breakdown ─────────────────────────────────────────────────
    lr_model = model.named_steps['model']
    preprocessing = Pipeline(steps=model.steps[:-1])
    X_t = preprocessing.transform(X_new)

    n_components = X_t.shape[1]
    comp_names   = [f'KPC_{i+1}' for i in range(n_components)]
    X_t_df       = pd.DataFrame(X_t, columns=comp_names)

    probas = lr_model.predict_proba(X_t_df)
    class_labels = [label_map.get(c, str(c)) for c in lr_model.classes_]
    proba_df = pd.DataFrame(probas, columns=class_labels, index=X_new.index)

    st.subheader('Prediction Probabilities')
    st.dataframe(proba_df.style.format("{:.2%}"))

    # ── SHAP waterfall for first row ──────────────────────────────────────────
    st.subheader('SHAP Explanation – First Row')

    sv = explainer(X_t_df)

    if len(sv.shape) == 3:
        # Multiclass: pick class with highest predicted probability for sample 0
        proba_0 = lr_model.predict_proba(X_t_df.iloc[:1])[0]
        cls_idx = int(np.argmax(proba_0))
        sv_plot = shap.Explanation(
            values        = sv.values[0, :, cls_idx],
            base_values   = sv.base_values[0, cls_idx],
            data          = sv.data[0],
            feature_names = comp_names,
        )
        st.caption(f"Explaining class: **{label_map.get(lr_model.classes_[cls_idx], lr_model.classes_[cls_idx])}**")
    else:
        sv_plot = sv[0]

    fig, ax = plt.subplots()
    shap.plots.waterfall(sv_plot, show=False)
    st.pyplot(fig)

    # ── SHAP bar summary (all rows) ───────────────────────────────────────────
    st.subheader('SHAP Feature Importance – All Rows')

    if len(sv.shape) == 3:
        # Average absolute SHAP across classes for summary
        mean_abs_shap = np.abs(sv.values).mean(axis=(0, 2))
        fig2, ax2 = plt.subplots(figsize=(8, max(4, n_components // 3)))
        sorted_idx = np.argsort(mean_abs_shap)
        ax2.barh(
            [comp_names[i] for i in sorted_idx],
            mean_abs_shap[sorted_idx],
            color='steelblue'
        )
        ax2.set_xlabel('Mean |SHAP value|')
        ax2.set_title('Average Feature Importance (all classes)')
        st.pyplot(fig2)
    else:
        fig2, ax2 = plt.subplots()
        shap.plots.bar(sv, show=False)
        st.pyplot(fig2)

else:
    st.info('👆 Upload a CSV file with the same feature columns used during training to get started.')
    st.markdown("""
    ### Expected columns
    The CSV should contain the **25+ technical indicators** generated in the notebook, e.g.:
    `return_1d`, `logret_1d`, `sma_5`, `ema_5`, `macd`, `bb_width`, `rsi_14`, etc.
    
    One row per trading day, with the date as the index.
    """)
