import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

# -------------------
# PAGE CONFIG
# -------------------
st.set_page_config(
    page_title="IBM HR Attrition Predictor",
    page_icon="💼",
    layout="wide"
)

# -------------------
# BACKGROUND IMAGE
# -------------------
def add_bg_from_local(image_file):
    """Set background image from local file."""
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add IBM logo background
add_bg_from_local("ibm_logo2.png")

# -------------------
# CUSTOM CSS FOR SELECTBOXES & SLIDERS
# -------------------
st.markdown(
    """
    <style>
    /* Change background and text color for all selectboxes */
    div[data-baseweb="select"] > div {
        background-color: white !important;
        color: black !important; 
    }

    /* Change dropdown menu items */
    div[role="listbox"] > div {
        background-color: white !important;
        color: black !important;
        font-weight: bold !important;
        font-size: 18px !important;
    }

    /* Change slider label text color */
    .stSlider label {
        color: white !important;
        font-weight: bold !important;
        font-size: 18px !important;
    }

    /* Change slider number display color */
    .stSlider span {
        color: red !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------
# LOAD MODEL & ENCODERS
# -------------------
model = joblib.load("decision_tree_attrition.pkl")

# Load label encoders for each categorical column
le_businesstravel = joblib.load("label_encoder_BusinessTravel.pkl")
le_department = joblib.load("label_encoder_Department.pkl")
le_educationfield = joblib.load("label_encoder_EducationField.pkl")
le_gender = joblib.load("label_encoder_Gender.pkl")
le_jobrole = joblib.load("label_encoder_JobRole.pkl")
le_maritalstatus = joblib.load("label_encoder_MaritalStatus.pkl")
le_over18 = joblib.load("label_encoder_Over18.pkl")
le_overtime = joblib.load("label_encoder_OverTime.pkl")

# -------------------
# SIDEBAR STORY
# -------------------
st.sidebar.header("📚 My Tuning Journey")
st.sidebar.write("""
I experimented with **GridSearchCV** to tune hyperparameters like max depth, 
criterion, and min samples per leaf.  
However, the **baseline Decision Tree** outperformed tuned versions — 
achieving higher accuracy and more true positives for attrition.

**Lesson learned:** Sometimes simpler is better.  
It reminded me to always benchmark against a strong baseline before tuning.
""")
st.sidebar.info("Model: Baseline Decision Tree Classifier\nFocus: Accuracy + Interpretability")

# -------------------
# TITLE & DESCRIPTION
# -------------------
st.markdown(
    "<h1 style='text-align: center; color: #cc0808;'>💼 IBM HR Analytics - Employee Attrition Predictor</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align: center;'>Predict whether an employee will leave the company using HR profile data.</h3>",
    unsafe_allow_html=True
)

st.write("Fill in the details below to get an attrition prediction:")

# -------------------
# INPUT FIELDS FOR ALL 30 FEATURES
# -------------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("🎂 Age", 18, 60, 30)
    businesstravel = st.selectbox("✈️ Business Travel", le_businesstravel.classes_)
    daily_rate = st.slider("📊 Daily Rate", 100, 1500, 800, step=10)
    department = st.selectbox("🏢 Department", le_department.classes_)
    distance_from_home = st.slider("🚗 Distance From Home (km)", 1, 30, 10)
    education = st.selectbox("🎓 Education (1=Below College, 5=Doctor)", [1, 2, 3, 4, 5])
    education_field = st.selectbox("📚 Education Field", le_educationfield.classes_)
    environment_satisfaction = st.selectbox("🌿 Environment Satisfaction (1=Low, 4=High)", [1, 2, 3, 4])
    gender = st.selectbox("⚧ Gender", le_gender.classes_)
    hourly_rate = st.slider("💵 Hourly Rate", 10, 100, 50)

with col2:
    job_involvement = st.selectbox("🛠️ Job Involvement (1=Low, 4=High)", [1, 2, 3, 4])
    job_level = st.slider("📈 Job Level", 1, 5, 2)
    job_role = st.selectbox("🧑‍💼 Job Role", le_jobrole.classes_)
    job_satisfaction = st.selectbox("😊 Job Satisfaction (1=Low, 4=High)", [1, 2, 3, 4])
    marital_status = st.selectbox("💍 Marital Status", le_maritalstatus.classes_)
    monthly_income = st.slider("💰 Monthly Income", 1000, 20000, 5000, step=500)
    monthly_rate = st.slider("💳 Monthly Rate", 2000, 30000, 10000, step=500)
    num_companies_worked = st.slider("🏢 Num Companies Worked", 0, 10, 2)
    over18 = st.selectbox("🔞 Over 18", le_over18.classes_)
    overtime = st.selectbox("⏱️ OverTime", le_overtime.classes_)

# Second row for remaining features
col3, col4 = st.columns(2)

with col3:
    percent_salary_hike = st.slider("📈 Percent Salary Hike", 0, 30, 15)
    performance_rating = st.slider("⭐ Performance Rating", 1, 4, 3)
    relationship_satisfaction = st.selectbox("🤝 Relationship Satisfaction (1=Low, 4=High)", [1, 2, 3, 4])
    stock_option_level = st.slider("📦 Stock Option Level", 0, 3, 1)
    total_working_years = st.slider("⌛ Total Working Years", 0, 40, 10)

with col4:
    training_times_last_year = st.slider("📚 Training Times Last Year", 0, 10, 3)
    work_life_balance = st.selectbox("⚖️ Work Life Balance (1=Bad, 4=Best)", [1, 2, 3, 4])
    years_at_company = st.slider("🏢 Years at Company", 0, 40, 3)
    years_in_current_role = st.slider("📌 Years in Current Role", 0, 20, 2)
    years_since_last_promotion = st.slider("🎯 Years Since Last Promotion", 0, 15, 1)
    years_with_curr_manager = st.slider("👨‍💼 Years with Current Manager", 0, 20, 3)

# -------------------
# ENCODE CATEGORICAL FEATURES
# -------------------
businesstravel_enc = le_businesstravel.transform([businesstravel])[0]
department_enc = le_department.transform([department])[0]
education_field_enc = le_educationfield.transform([education_field])[0]
gender_enc = le_gender.transform([gender])[0]
job_role_enc = le_jobrole.transform([job_role])[0]
marital_status_enc = le_maritalstatus.transform([marital_status])[0]
over18_enc = le_over18.transform([over18])[0]
overtime_enc = le_overtime.transform([overtime])[0]

# -------------------
# BUILD FEATURE VECTOR IN EXACT ORDER
# -------------------
features = np.array([[
    age, businesstravel_enc, daily_rate, department_enc, distance_from_home,
    education, education_field_enc, environment_satisfaction, gender_enc,
    hourly_rate, job_involvement, job_level, job_role_enc, job_satisfaction,
    marital_status_enc, monthly_income, monthly_rate, num_companies_worked,
    overtime_enc, percent_salary_hike, performance_rating,
    relationship_satisfaction, stock_option_level, total_working_years,
    training_times_last_year, work_life_balance, years_at_company,
    years_in_current_role, years_since_last_promotion, years_with_curr_manager
]])

# -------------------
# PREDICTION
# -------------------
if st.button("🔍 Predict Attrition"):
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Likely to Leave — Probability: **{prob:.2f}**")
    else:
        st.success(f"✅ Likely to Stay — Probability: **{prob:.2f}**")

# -------------------
# FOOTER
# -------------------
st.markdown(
    "<hr><p style='text-align: center;'>Made with ❤️ by Ajay</p>",
    unsafe_allow_html=True
)
