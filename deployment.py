# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 18:41:57 2025

@author: ankit
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression

# Load the trained model (ensure the model is saved as 'logistic_model.pkl')
@st.cache_data
def load_model():
    with open("logistic_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

classifier = load_model()

# Streamlit UI
st.title("Titanic Survival Prediction")
st.write("Enter passenger details to predict survival.")

# User input fields
pclass = st.selectbox("Passenger Class (1 = First, 2 = Second, 3 = Third)", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("SibSp", min_value=0, max_value=5, value=0)
parch = st.number_input("Parch", min_value=0, max_value=5, value=0)
fare = st.number_input("Fare Paid", min_value=0.0, max_value=500.0, value=30.0)
embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])

# Convert categorical inputs
sex_encoded = 1 if sex == "Male" else 0
embarked_mapping = {"C": 0, "Q": 1, "S": 2}
embarked_encoded = embarked_mapping[embarked]

# Create input DataFrame
input_data = pd.DataFrame([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]],
                          columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"])

# Make prediction
if st.button("Predict"):
    prediction = classifier.predict(input_data)
    survival_prob = classifier.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.success(f"The passenger is predicted to **survive** with a probability of {survival_prob:.2f}.")
    else:
        st.error(f"The passenger is predicted **not to survive** with a probability of {1 - survival_prob:.2f}.")
