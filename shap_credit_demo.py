import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# ---------------------------------------------
# Step 1: Load and Prepare the Data
# ---------------------------------------------

# Load from local file (space-separated)
df = pd.read_csv("data/german_credit_data.csv", sep='\\s+', header=None)

# Add column names: 20 features + 1 target
df.columns = [
    "checking_account", "duration", "credit_history", "purpose", "credit_amount",
    "savings", "employment_since", "installment_rate", "personal_status", "guarantors",
    "residence_since", "property", "age", "installment_plans", "housing",
    "existing_credits", "job", "liable_people", "telephone", "foreign_worker",
    "target"
]

# Transform target: 1 = good → 0, 2 = bad → 1
df["target"] = df["target"].replace({1: 0, 2: 1})

# Split features and label
X = df.drop(columns=["target"])
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ---------------------------------------------
# Step 2: Train the Model
# ---------------------------------------------

model = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# ---------------------------------------------
# Step 3: Compute SHAP Values
# ---------------------------------------------

explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Optional: Visualize global importance (manual inspection)
shap.summary_plot(shap_values, X_test)

# Optional: Visualize one prediction (manual inspection)
shap.plots.waterfall(shap_values[0])

# ---------------------------------------------
# Step 4: Save Model, SHAP Explainer, and Test Data
# ---------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/xgb_model.pkl")
joblib.dump(explainer, "models/shap_explainer.pkl")
joblib.dump(X_test.reset_index(drop=True), "models/X_test.pkl")

print("✅ Model, explainer, and test data saved to 'models/' folder.")
