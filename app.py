"""
Parkinson's Disease Prediction App
-----------------------------------
Loads the trained SVM model (deploy_SVM.pkl) and lets a user either:
  1. Enter voice-measurement values manually to get a single prediction, or
  2. Upload a CSV of multiple patients to get batch predictions.

Run locally with:
    streamlit run app.py
"""

import pickle

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# PAGE CONFIG — must be the first Streamlit command in the script
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Parkinson's Disease Predictor",
    page_icon="🧠",
    layout="wide",
)

# ----------------------------------------------------------------------
# FEATURE DEFINITIONS
# The order below MUST match the column order the model was trained on
# (i.e. df.drop(['name', 'status'], axis=1) from the training notebook).
# Each entry stores (min, max, mean) taken from the original training
# dataset, so the input widgets have realistic, sensible ranges.
# ----------------------------------------------------------------------
FEATURES = {
    "MDVP:Fo(Hz)":      (88.333, 260.105, 154.229),
    "MDVP:Fhi(Hz)":     (102.145, 592.030, 197.105),
    "MDVP:Flo(Hz)":     (65.476, 239.170, 116.325),
    "MDVP:Jitter(%)":   (0.00168, 0.03316, 0.00622),
    "MDVP:Jitter(Abs)": (0.00001, 0.00026, 0.00004),
    "MDVP:RAP":         (0.00068, 0.02144, 0.00331),
    "MDVP:PPQ":         (0.00092, 0.01958, 0.00345),
    "Jitter:DDP":       (0.00204, 0.06433, 0.00992),
    "MDVP:Shimmer":     (0.00954, 0.11908, 0.02971),
    "MDVP:Shimmer(dB)": (0.08500, 1.30200, 0.28225),
    "Shimmer:APQ3":     (0.00455, 0.05647, 0.01566),
    "Shimmer:APQ5":     (0.00570, 0.07940, 0.01788),
    "MDVP:APQ":         (0.00719, 0.13778, 0.02408),
    "Shimmer:DDA":      (0.01364, 0.16942, 0.04699),
    "NHR":              (0.00065, 0.31482, 0.02485),
    "HNR":              (8.44100, 33.04700, 21.88597),
    "RPDE":             (0.25657, 0.68515, 0.49854),
    "DFA":              (0.57428, 0.82529, 0.71810),
    "spread1":          (-7.96498, -2.43403, -5.68440),
    "spread2":          (0.00627, 0.45049, 0.22651),
    "D2":               (1.42329, 3.67116, 2.38183),
    "PPE":              (0.04454, 0.52737, 0.20655),
}

FEATURE_GROUPS = {
    "Fundamental Frequency": ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)"],
    "Jitter (frequency variation)": [
        "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    ],
    "Shimmer (amplitude variation)": [
        "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3",
        "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
    ],
    "Noise-to-Tone Ratios": ["NHR", "HNR"],
    "Nonlinear Dynamical Measures": ["RPDE", "DFA", "spread1", "spread2", "D2", "PPE"],
}

MODEL_PATH = "deploy_SVM.pkl"


# ----------------------------------------------------------------------
# MODEL LOADING — cached so the pickle file is only read once per session
# ----------------------------------------------------------------------
@st.cache_resource
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


from typing import Optional, Tuple

def predict_single(model, values: dict) -> Tuple[int, Optional[float]]:
    """Run the model on one patient's values.

    Returns (prediction, confidence_score).
    confidence_score is None if the model has no decision_function/predict_proba.
    """
    ordered_values = [values[feat] for feat in FEATURES]
    X = pd.DataFrame([ordered_values], columns=list(FEATURES.keys()))
    prediction = int(model.predict(X)[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(X)[0][prediction])
    elif hasattr(model, "decision_function"):
        # SVC(kernel='linear') without probability=True only has this.
        # It's a raw distance from the decision boundary, not a probability,
        # so we just show it as a confidence/margin score.
        confidence = float(model.decision_function(X)[0])

    return prediction, confidence


# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 About")
    st.write(
        "This app predicts whether a patient's voice recording shows signs "
        "of **Parkinson's disease**, using a Support Vector Machine (SVM) "
        "trained on 22 biomedical voice-measurement features."
    )
    st.write(
        "**Status codes**\n"
        "- `1` → Parkinson's detected\n"
        "- `0` → Healthy"
    )
    st.caption(
        "Model file expected: `deploy_SVM.pkl` in the same folder as this app."
    )

# ----------------------------------------------------------------------
# LOAD MODEL (with a friendly error if the pickle file is missing)
# ----------------------------------------------------------------------
try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(
        f"Could not find `{MODEL_PATH}`. Place your trained pickle file in "
        "the same folder as `app.py`, then refresh this page."
    )
    st.stop()

st.title("Parkinson's Disease Prediction")
st.write(
    "Enter a patient's vocal measurements below, or upload a CSV for "
    "batch predictions."
)

tab_single, tab_batch = st.tabs(["🧍 Single Patient", "📄 Batch (CSV Upload)"])

# ----------------------------------------------------------------------
# TAB 1 — SINGLE PATIENT, MANUAL INPUT
# ----------------------------------------------------------------------
with tab_single:
    st.subheader("Enter Voice Measurements")

    input_values = {}
    for group_name, feature_list in FEATURE_GROUPS.items():
        with st.expander(group_name, expanded=(group_name == "Fundamental Frequency")):
            cols = st.columns(3)
            for i, feature in enumerate(feature_list):
                min_val, max_val, mean_val = FEATURES[feature]
                # step chosen relative to the range so the +/- buttons feel usable
                step = round((max_val - min_val) / 100, 6)
                input_values[feature] = cols[i % 3].number_input(
                    label=feature,
                    min_value=float(min_val),
                    max_value=float(max_val * 1.5),  # allow some headroom above training max
                    value=float(mean_val),
                    step=step,
                    format="%.5f",
                )

    st.divider()

    if st.button("🔍 Predict", type="primary", use_container_width=True):
        prediction, confidence = predict_single(model, input_values)

        if prediction == 1:
            st.error("### Result: Parkinson's Disease Detected 🔴")
        else:
            st.success("### Result: Healthy — No Parkinson's Detected 🟢")

        if confidence is not None:
            st.caption(f"Model confidence / decision score: `{confidence:.4f}`")

        st.warning(
            "⚠️ This tool is for educational purposes only and is **not** a "
            "medical diagnosis. Always consult a qualified doctor.",
            icon="⚠️",
        )

# ----------------------------------------------------------------------
# TAB 2 — BATCH PREDICTION VIA CSV UPLOAD
# ----------------------------------------------------------------------
with tab_batch:
    st.subheader("Upload a CSV File")
    st.write(
        "The CSV must contain these "
        f"**{len(FEATURES)} columns** (extra columns like `name` or "
        "`status` are ignored if present):"
    )
    st.code(", ".join(FEATURES.keys()), language="text")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        missing_cols = [c for c in FEATURES if c not in batch_df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            X_batch = batch_df[list(FEATURES.keys())]
            predictions = model.predict(X_batch)

            results_df = batch_df.copy()
            results_df["Prediction"] = np.where(
                predictions == 1, "Parkinson's Detected", "Healthy"
            )

            st.success(f"Predictions complete for {len(results_df)} patients.")
            st.dataframe(results_df, use_container_width=True)

            csv_output = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv_output,
                file_name="parkinsons_predictions.csv",
                mime="text/csv",
            )