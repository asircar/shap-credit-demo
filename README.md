# 🔍 SHAP Explainability for Credit Risk Modeling

This project demonstrates how to use SHAP (SHapley Additive exPlanations) to interpret predictions from a credit risk classifier. It provides both **global explanations** (what drives model decisions overall) and **local explanations** (why a specific applicant is predicted as high or low risk).

---

## 🧠 Project Summary

- **Problem:** Binary classification of credit risk (good vs bad credit)
- **Model:** XGBoost classifier
- **Explainability:** SHAP for global and local interpretability
- **Interface:** Streamlit app for interactive exploration

---

## 📁 Folder Structure

```
shap-credit-demo/
│
├── data/                   # Contains german_credit_data.csv (space-delimited)
├── models/                 # Saved model, SHAP explainer, test set
│   ├── xgb_model.pkl
│   ├── shap_explainer.pkl
│   └── X_test.pkl
├── streamlit_app.py        # Interactive Streamlit dashboard
├── shap_credit_demo.py     # Training and SHAP generation script
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

---

## 🚀 How to Run

### 1. Clone the repo and install dependencies

```bash
git clone https://github.com/your-username/shap-credit-demo.git
cd shap-credit-demo
python -m venv shap-env
source shap-env/bin/activate      # or .\shap-env\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Run the training script (only once)

```bash
python shap_credit_demo.py
```

This will:
- Train an XGBoost model
- Generate SHAP values
- Save everything into the `models/` folder

### 3. Launch the Streamlit app

```bash
streamlit run streamlit_app.py
```

Then visit `http://localhost:8501` in your browser.

---

## 📊 Features of the App

- **Applicant selection:** Browse individual predictions
- **Waterfall plot:** Understand why the model predicted good or bad credit
- **Summary plot:** See which features matter most across all applicants
- **Feature values:** Inspect input data for any selected applicant

---

## 📦 Dependencies

- Python ≥ 3.8
- XGBoost
- SHAP
- pandas, matplotlib, scikit-learn
- Streamlit

Install via:

```bash
pip install -r requirements.txt
```

---

## 🔐 Optional Add-ons

- Add GPT or templated text for human-readable explanations
- Deploy the app using [Streamlit Cloud](https://streamlit.io/cloud) or Docker
- Extend with scenario simulation: "What if the credit usage was lower?"

---

## 📄 Data Source

Dataset: [German Credit Data (UCI)](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))  
- 1000 samples  
- 24 numerical features  
- Binary classification: good vs bad credit

---

## 🧠 Author

**Arunava Sircar**  
[LinkedIn](https://www.linkedin.com/in/arunava-sircar) • [GitHub](https://github.com/your-username) • [Website](#)

---

## 📜 License

MIT License. Use freely and responsibly.
