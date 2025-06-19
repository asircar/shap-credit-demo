import streamlit as st
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------
# Streamlit Config
# --------------------------------------------
st.set_page_config(page_title="SHAP Credit Risk Explainer", layout="wide")
st.title("🔍 SHAP Explainability for Credit Risk Model")

# --------------------------------------------
# Load model, explainer, and test set
# --------------------------------------------
model = joblib.load("models/xgb_model.pkl")
explainer = joblib.load("models/shap_explainer.pkl")
X_test = joblib.load("models/X_test.pkl")

# --------------------------------------------
# Sidebar: Applicant Selection
# --------------------------------------------
st.sidebar.header("Select Applicant")
index = st.sidebar.slider("Choose an applicant index", min_value=0, max_value=len(X_test) - 1, value=0)

# Grab selected row
row = X_test.iloc[[index]]
prediction = model.predict(row)[0]
probability = model.predict_proba(row)[0][prediction]
shap_values = explainer(row)

# --------------------------------------------
# Show Prediction
# --------------------------------------------
label = "Good Credit" if prediction == 0 else "Bad Credit"
st.markdown(f"### 🧾 Prediction: **{label}** ({probability:.2%} confidence)")

# --------------------------------------------
# Local SHAP Waterfall Plot
# --------------------------------------------
st.markdown("### 🔍 SHAP Waterfall Plot (Why this prediction?)")
fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[0], max_display=12, show=False)
st.pyplot(fig)

# --------------------------------------------
# Optional: Feature Values Table
# --------------------------------------------
with st.expander("📊 Show Feature Values for This Applicant", expanded=False):
    st.dataframe(row.T.rename(columns={index: "Value"}))
