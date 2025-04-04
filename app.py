# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 15:49:44 2025

@author: ankit
"""

import streamlit as st
import pandas as pd
import pickle
import os

# Load the trained model (ensure the model file exists)
@st.cache_resource
def load_model():
    model_path = "logistic_model.pkl"
    if not os.path.exists(model_path):
        st.error("Model file not found! Please upload 'logistic_model.pkl'.")
        st.stop()
    
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

# Main function
def main():
    st.title("🚢 Titanic Survival Prediction")
    st.write("Enter passenger details to predict survival.")

    # User input fields
    pclass = st.selectbox("Passenger Class (1 = First, 2 = Second, 3 = Third)", [1, 2, 3])
    sex = st.selectbox("Sex", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=8, value=0)
    parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=6, value=0)
    fare = st.number_input("Fare Paid ($)", min_value=0.0, max_value=500.0, value=30.0)
    embarked = st.selectbox("Embarked Port", ["C (Cherbourg)", "Q (Queenstown)", "S (Southampton)"])

    # Convert categorical inputs
    sex_encoded = 1 if sex == "Male" else 0
    embarked_mapping = {"C (Cherbourg)": 0, "Q (Queenstown)": 1, "S (Southampton)": 2}
    embarked_encoded = embarked_mapping[embarked]

    # Create input DataFrame
    input_data = pd.DataFrame([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]],
                              columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"])

    # Load model
    classifier = load_model()

    # Make prediction
    if st.button("🚀 Predict"):
        try:
            prediction = classifier.predict(input_data)
            survival_prob = classifier.predict_proba(input_data)[0][1]

            if prediction[0] == 1:
                st.success(f"✅ The passenger is predicted to **survive** with a probability of {survival_prob:.2f}.")
            else:
                st.error(f"❌ The passenger is predicted **not to survive** with a probability of {1 - survival_prob:.2f}.")
            
            # Display probability visually
            st.progress(int(survival_prob * 100))

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Run main function
if __name__ == "__main__":
    main()

