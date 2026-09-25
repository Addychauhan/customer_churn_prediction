
import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
import plotly.graph_objects as go


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 38px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 17px;
    margin-bottom: 30px;
}

.metric-card {
    background: white;
    padding: 18px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    text-align: center;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = xgb.XGBClassifier()
    model.load_model("xgb_model.json")

    return model


@st.cache_resource
def load_support():

    with open("churn_model.pkl", "rb") as f:
        support = pickle.load(f)

    return support


model = load_model()
support = load_support()

scaler = support["scaler"]
threshold = support["threshold"]
feature_names = support["feature_names"]


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📊 Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning based customer churn risk assessment'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# MODEL INFORMATION
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <h4>🤖 Model</h4>
            <h3>{support["model_name"]}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <h4>🎯 Decision Threshold</h4>
            <h3>{threshold:.2f}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <h4>🔢 Features</h4>
            <h3>{len(feature_names)}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">👤 Customer Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    credit_score = st.slider(
        "Credit Score",
        350,
        850,
        650
    )

    age = st.slider(
        "Age",
        18,
        92,
        38
    )

    tenure = st.slider(
        "Tenure (years)",
        0,
        10,
        5
    )


with col2:

    balance = st.number_input(
        "Balance",
        min_value=0.0,
        max_value=300000.0,
        value=60000.0,
        step=1000.0
    )

    num_products = st.selectbox(
        "Number of Products",
        [1, 2, 3, 4]
    )

    salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        max_value=300000.0,
        value=100000.0,
        step=1000.0
    )


with col3:

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    has_cr_card = st.selectbox(
        "Has Credit Card?",
        ["Yes", "No"]
    )

    is_active = st.selectbox(
        "Active Member?",
        ["Yes", "No"]
    )


# =========================================================
# SINGLE PREDICTION BUTTON
# =========================================================

st.markdown("---")

predict_button = st.button(
    "🔍 Predict Customer Churn",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # -----------------------------------------------------
    # Create input dataframe
    # -----------------------------------------------------

    input_data = pd.DataFrame([{

        "CreditScore": credit_score,

        "Age": age,

        "Tenure": tenure,

        "Balance": balance,

        "NumOfProducts": num_products,

        "HasCrCard":
            1 if has_cr_card == "Yes" else 0,

        "IsActiveMember":
            1 if is_active == "Yes" else 0,

        "EstimatedSalary": salary,

        "Geography_Germany":
            1 if geography == "Germany" else 0,

        "Geography_Spain":
            1 if geography == "Spain" else 0,

        "Gender_Male":
            1 if gender == "Male" else 0

    }])


    # -----------------------------------------------------
    # Arrange columns
    # -----------------------------------------------------

    input_data = input_data.reindex(
        columns=feature_names,
        fill_value=0
    )


    # -----------------------------------------------------
    # Scale input
    # -----------------------------------------------------

    scaled_input = scaler.transform(input_data)


    # -----------------------------------------------------
    # Churn probability
    # -----------------------------------------------------

    proba = float(
        model.predict_proba(scaled_input)[0, 1]
    )


    # -----------------------------------------------------
    # Prediction using saved threshold
    # -----------------------------------------------------

    prediction = (
        "Churn"
        if proba >= threshold
        else "No Churn"
    )


    # =====================================================
    # RESULT
    # =====================================================

    st.markdown("---")

    st.subheader("🎯 Churn Probability")


    # -----------------------------------------------------
    # Speedometer Gauge
    # -----------------------------------------------------

    fig = go.Figure(
        go.Indicator(

            mode="gauge+number",

            value=proba * 100,

            number={
                "suffix": "%",
                "font": {
                    "size": 32
                }
            },

            title={
                "text": "Customer Churn Probability"
            },

            gauge={

                "axis": {
                    "range": [0, 100]
                },

                "bar": {
                    "thickness": 0.25
                },

                "steps": [

                    {
                        "range": [0, 30],
                        "color": "#2ecc71"
                    },

                    {
                        "range": [30, 60],
                        "color": "#f1c40f"
                    },

                    {
                        "range": [60, 100],
                        "color": "#e74c3c"
                    }

                ],

                "threshold": {

                    "line": {
                        "color": "black",
                        "width": 4
                    },

                    "thickness": 0.75,

                    "value": threshold * 100

                }
            }
        )
    )


    fig.update_layout(
        height=350,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # =====================================================
    # PROBABILITY + THRESHOLD
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Churn Probability",
            f"{proba * 100:.1f}%"
        )

    with col2:

        st.metric(
            "Decision Threshold",
            f"{threshold * 100:.1f}%"
        )


    # =====================================================
    # RISK LEVEL
    # =====================================================

    if proba < 0.30:

        risk = "🟢 Low Risk"

    elif proba < 0.60:

        risk = "🟡 Medium Risk"

    else:

        risk = "🔴 High Risk"


    st.write(f"### Risk Level: {risk}")


    # =====================================================
    # FINAL PREDICTION
    # =====================================================

    if prediction == "Churn":

        st.error(
            "⚠️ Customer is predicted to **CHURN**"
        )

    else:

        st.success(
            "✅ Customer is predicted to have **NO CHURN**"
        )