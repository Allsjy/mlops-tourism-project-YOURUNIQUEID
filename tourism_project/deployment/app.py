import os
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Get Hugging Face Model Repo ID from environment variables (Streamlit Secrets)
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO")

if not HF_MODEL_REPO:
    st.error("HF_MODEL_REPO environment variable not set. Please configure it in Streamlit Secrets.")
else:
    try:
        # Download the model from Hugging Face Model Hub
        model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="best_tourism_package_model.joblib", repo_type="model")
        model = joblib.load(model_path)
        st.success(f"Model loaded from Hugging Face Model Hub: {HF_MODEL_REPO}")

        # Streamlit UI for Tourism Package Prediction
        st.title("Tourism Package Prediction")
        st.write("Fill the customer details below to predict if they'll purchase a travel package")

        # Collect user input
        Age = st.slider("Age", 18, 70, 30)
        TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        CityTier = st.selectbox("City Tier", [1, 2, 3])
        DurationOfPitch = st.slider("Duration of Pitch (mins)", 0, 100, 15)
        Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        Gender = st.selectbox("Gender", ["Male", "Female", "Others"])
        NumberOfPersonVisiting = st.slider("Number of Persons Visiting", 1, 5, 2)
        NumberOfFollowups = st.slider("Number of Follow-ups", 1, 10, 3)
        ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
        MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
        NumberOfTrips = st.slider("NumberOfTrips", 1, 20, 3)
        Passport = st.selectbox("Has Passport?", ["Yes", "No"])
        PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
        OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
        NumberOfChildrenVisiting = st.slider("Number of Children Visited", 0, 5, 1)
        Designation = st.selectbox("Designation", ["Executive", "Manager", "AVP", "VP", "Sr. Manager"])
        MonthlyIncome = st.number_input("Monthly Income", min_value=1000.0, value=30000.0)

        # ----------------------------
        # Prepare input data
        # ----------------------------
        input_data = pd.DataFrame([{
            'Age': Age,
            'TypeofContact': TypeofContact,
            'CityTier': CityTier,
            'DurationOfPitch': DurationOfPitch,
            'Occupation': Occupation,
            'Gender': Gender,
            'NumberOfPersonVisiting': NumberOfPersonVisiting,
            'NumberOfFollowups': NumberOfFollowups,
            'ProductPitched': ProductPitched,
            'PreferredPropertyStar': PreferredPropertyStar,
            'MaritalStatus': MaritalStatus,
            'NumberOfTrips': NumberOfTrips,
            'Passport': 1 if Passport == "Yes" else 0,
            'PitchSatisfactionScore': PitchSatisfactionScore,
            'OwnCar': 1 if OwnCar == "Yes" else 0,
            'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
            'Designation': Designation,
            'MonthlyIncome': MonthlyIncome
        }])

        # Set the classification threshold
        classification_threshold = 0.5

        # Predict button
        if st.button("Predict"):
            prob = model.predict_proba(input_data)[:, 1]
            pred = int(prob >= classification_threshold)
            result = "will purchase the travel package" if pred == 1 else "is unlikely to purchase"
            st.write(f"Prediction: Customer {result}")

    except Exception as e:
        st.error(f"Error loading model from Hugging Face: {e}")
