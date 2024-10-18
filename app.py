import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load model and encoders
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app styling
st.set_page_config(page_title="Customer Churn Prediction App", layout="centered", initial_sidebar_state="collapsed")

# Apply custom CSS
st.markdown("""
    <style>
    .main {
   
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 15px rgba(0,0,0,0.1);
    }
    h1 {
        color: #4A90E2;
    }
    .stButton button {
        background-color: #4A90E2;
        color: white;
        border-radius: 10px;
    }
    .stButton button:hover {
        background-color: #357ABD;
    }
    </style>
    """, unsafe_allow_html=True)

# Header and description
st.title("Customer Churn Prediction App")
st.write("""
    **Welcome!** This app predicts the likelihood of a customer to churn based on their demographic and geographic information.
    Fill in the form below to get an instant prediction.
    """)

# Collect user input
st.subheader("Customer Information")

# Split into two columns
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0], help="Select the customer's geography.")
    gender = st.selectbox('Gender', label_encoder_gender.classes_, help="Select the customer's gender.")
    age = st.slider('Age', 18, 92, help="Set the customer's age.")

with col2:
    balance = st.number_input('Balance', min_value=0.0, step=100.0, help="Enter the customer's account balance.")
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, step=10, help="Enter the customer's credit score.")
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, step=1000.0, help="Enter the customer's estimated salary.")

# Collect remaining data in two more columns
col3, col4 = st.columns(2)

with col3:
    tenure = st.slider('Tenure', 0, 10, help="Select the number of years the customer has been with the bank.")
    num_of_products = st.slider('Number of Products', 1, 4, help="Select the number of products the customer has.")

with col4:
    has_cr_card = st.selectbox('Has Credit Card', [0, 1], help="Does the customer have a credit card? (0 = No, 1 = Yes)")
    is_active_member = st.selectbox('Is Active Member', [0, 1], help="Is the customer an active member? (0 = No, 1 = Yes)")

# Prepare input data for the model
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Predict button
if st.button('Submit'):
    with st.spinner('Calculating churn prediction...'):
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Predict churn
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]

        # Display the result
        st.success(f'Churn Probability: {prediction_proba:.2f}')
        if prediction_proba > 0.5:
            st.error('The customer is likely to churn.')
        else:
            st.success('The customer is not likely to churn.')

# Footer with links
st.markdown("""
    <br><hr>
    <p style='text-align: center;'>
        Developed by <a href="https://saikiranmandula.vercel.app/" target="_blank">Saikiran</a> | 
        <a href="https://github.com/saikiran888" target="_blank">GitHub Repo</a>
    </p>
    """, unsafe_allow_html=True)
