# 📊 Customer Churn Prediction

Predicts whether a bank customer is likely to churn based on demographic, financial, and account-related information, using and comparing six classification models. Includes a Jupyter notebook for the full analysis and a Streamlit app for real-time churn risk assessment.

## Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customerchurnprediction-lhmpewgqdn3dwj85jsgkkz.streamlit.app/))

🔗 [Try it live](https://customerchurnprediction-lhmpewgqdn3dwj85jsgkkz.streamlit.app/))

An interactive Streamlit dashboard where you enter a customer's details (credit score, age, balance, tenure, etc.) and get their churn probability, risk level, and a prediction back instantly.

[App screenshot](https://github.com/user-attachments/assets/7bade4d8-a3fa-4083-9ad0-b103a863e340)

## Project Structure

```
customer_churn_prediction/
│
├── data/
│   └── Churn_Modelling.csv
│
├── notebook/
│   └── customer_churn_prediction_25_09_2026.ipynb
│
├── streamlit_app/
│   └── app.py
│
├── README.md
├── churn_model.pkl              # Saved scaler, threshold, and feature names
├── churn_model_comparison.csv   # Metrics for all 6 models
├── requirements.txt
└── xgb_model.json               # Saved best model (XGBoost)
```

## Dataset

<!-- Fill in the exact row count from Churn_Modelling.csv -->
~10,000 bank customer records, with columns including:

| Column | Description |
|---|---|
| `CreditScore` | Customer's credit score |
| `Geography` | Customer's country (France, Germany, Spain) |
| `Gender` | Male / Female |
| `Age` | Customer's age |
| `Tenure` | Years as a customer |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products used |
| `HasCrCard` | Whether the customer has a credit card |
| `IsActiveMember` | Whether the customer is an active member |
| `EstimatedSalary` | Customer's estimated salary |
| `Exited` | Target variable — 1 if the customer churned, 0 otherwise |

<!-- Add the dataset source link if from Kaggle or elsewhere, e.g.: -->
<!-- Source: [Kaggle — Bank Customer Churn](https://www.kaggle.com/...) -->

## Approach

1. **EDA** — distributions, missing values, correlation analysis, churn rate by geography/gender/age group
2. **Data Cleaning** — handled missing values, removed non-predictive identifier columns (e.g. `RowNumber`, `CustomerId`, `Surname`)
3. **Feature Engineering** — encoded categorical variables (`Geography`, `Gender`), scaled numerical features
4. **Class Imbalance Handling** — addressed churn/no-churn class imbalance in the training data
5. **Modeling** — Logistic Regression, Decision Tree, Random Forest, XGBoost, Gaussian Naive Bayes, and a Deep Learning model
6. **Evaluation** — Precision, Recall, F1-Score, ROC-AUC across all models
7. **Model Saving** — best model (XGBoost) saved with its scaler, decision threshold, and feature list for use in the Streamlit app

## Results

| Model | Threshold | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **XGBoost** | **0.45** | **0.850** | **0.636** | **0.609** | **0.622** | **0.861** |
| Random Forest | 0.59 | 0.846 | 0.630 | 0.590 | 0.609 | 0.859 |
| Decision Tree | 0.63 | 0.834 | 0.585 | 0.624 | 0.604 | 0.833 |
| Gaussian Naive Bayes | 0.56 | 0.752 | 0.426 | 0.631 | 0.508 | 0.792 |
| Logistic Regression | 0.61 | 0.774 | 0.454 | 0.553 | 0.498 | 0.778 |

**Best model: XGBoost** — highest ROC-AUC (0.861) and the best F1-score, using a custom decision threshold of 0.45 (rather than the default 0.5) to better balance precision and recall for catching more actual churn cases.

> **Note:** a Deep Learning model was also explored during development but isn't included in `churn_model_comparison.csv` — only these five models have final logged metrics.

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/Addychauhan/customer_churn_prediction.git
cd customer_churn_prediction
pip install -r requirements.txt
```

### Run the notebook

[Open in Google Colab]https://colab.research.google.com/github/Addychauhan/customer_churn_prediction/blob/main/notebook/customer_churn_prediction_25_09_2026.ipynb)

### Run the Streamlit app

```bash
streamlit run streamlit_app/app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Requirements

```
streamlit
scikit-learn
xgboost
pandas
numpy
plotly
matplotlib
seaborn
```

## Notes

- The app uses a custom decision threshold (0.45, not the default 0.5) to classify churn, tuned to balance precision and recall for this use case.
- If you retrain the model with a different XGBoost or scikit-learn version than what's installed when running the app, you may see an `InconsistentVersionWarning`. Match your library versions to the ones used for training to avoid this.

## License

MIT — feel free to use, modify, and share.
