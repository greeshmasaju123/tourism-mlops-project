
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download


MODEL_REPO = "Greeshmadudu/tourism-wellness-model"


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="tourism_model.joblib",
        repo_type="model",
    )
    return joblib.load(model_path)


st.set_page_config(page_title="Wellness Tourism Predictor", layout="centered")
st.title("Wellness Tourism Package Predictor")

with st.form("customer_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration = st.number_input("Duration of Pitch", min_value=0.0, value=12.0)
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    persons = st.number_input("Number of Persons Visiting", min_value=1, value=2)
    followups = st.number_input("Number of Followups", min_value=0.0, value=3.0)
    product = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    trips = st.number_input("Number of Trips", min_value=0.0, value=2.0)
    passport = st.selectbox("Passport", [0, 1])
    satisfaction = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    car = st.selectbox("Own Car", [0, 1])
    children = st.number_input("Number of Children Visiting", min_value=0.0, value=1.0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    income = st.number_input("Monthly Income", min_value=0.0, value=25000.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    model = load_model()
    row = pd.DataFrame(
        [{
            "Age": age,
            "TypeofContact": contact,
            "CityTier": city_tier,
            "DurationOfPitch": duration,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": persons,
            "NumberOfFollowups": followups,
            "ProductPitched": product,
            "PreferredPropertyStar": star,
            "MaritalStatus": marital,
            "NumberOfTrips": trips,
            "Passport": passport,
            "PitchSatisfactionScore": satisfaction,
            "OwnCar": car,
            "NumberOfChildrenVisiting": children,
            "Designation": designation,
            "MonthlyIncome": income,
        }]
    )
    probability = model.predict_proba(row)[0, 1]
    prediction = "Likely to purchase" if probability >= 0.5 else "Not likely to purchase"
    st.metric("Prediction", prediction)
    st.progress(float(probability))
    st.write(f"Purchase probability: {probability:.1%}")
