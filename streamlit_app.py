
import streamlit as st
import joblib
import pandas as pd

# Load the trained pipeline (scaler + model)
try:
    pipeline = joblib.load("delivery_delay_pipeline.sav")
except FileNotFoundError:
    st.error("Error: 'delivery_delay_pipeline.sav' not found. Please ensure the model is saved correctly and is in the same directory.")
    st.stop()

st.set_page_config(page_title="Delivery Delay Prediction")

st.title("Delivery Delay Prediction App")
st.write("Enter the features below to predict if a delivery will be delayed.")

# Define the feature columns - ensure this matches the training data order
feature_columns = [
    'Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
    'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
    'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
    'Warehouse_Processing_Time'
]

# Create input widgets for each feature
with st.form("prediction_form"):
    st.header("Delivery Details")
    
    delivery_distance = st.number_input("Delivery Distance (km)", min_value=0.0, value=20.0, step=0.1)
    traffic_congestion = st.selectbox("Traffic Congestion (1-5, 5 being highest)", options=[1, 2, 3, 4, 5], index=2)
    weather_condition = st.selectbox("Weather Condition (1-5, 5 being worst)", options=[1, 2, 3, 4, 5], index=1)
    delivery_slot = st.selectbox("Delivery Slot (1-3)", options=[1, 2, 3], index=1)
    driver_experience = st.number_input("Driver Experience (years)", min_value=0, value=5, step=1)
    num_stops = st.number_input("Number of Stops", min_value=1, value=5, step=1)
    vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, value=3, step=1)
    road_condition_score = st.selectbox("Road Condition Score (1-5, 5 being best)", options=[1, 2, 3, 4, 5], index=2)
    package_weight = st.number_input("Package Weight (kg)", min_value=0.1, value=5.0, step=0.1)
    fuel_efficiency = st.number_input("Fuel Efficiency (km/l)", min_value=1.0, value=10.0, step=0.1)
    warehouse_processing_time = st.number_input("Warehouse Processing Time (minutes)", min_value=0, value=30, step=1)
    
    submitted = st.form_submit_button("Predict Delivery Status")

if submitted:
    # Create a DataFrame from the input values
    input_data = pd.DataFrame([[ 
        delivery_distance, traffic_congestion, weather_condition, delivery_slot,
        driver_experience, num_stops, vehicle_age, road_condition_score,
        package_weight, fuel_efficiency, warehouse_processing_time
    ]], columns=feature_columns)
    
    try:
        # Make prediction using the pipeline
        prediction = pipeline.predict(input_data)[0]
        prediction_proba = pipeline.predict_proba(input_data)[0][1] # Probability of delay (class 1)
        
        st.subheader("Prediction Result:")
        if prediction == 1:
            st.error(f"The delivery is predicted to be **DELAYED** with a probability of {prediction_proba:.2f}.")
        else:
            st.success(f"The delivery is predicted to be **ON TIME** with a probability of {1 - prediction_proba:.2f}.")
            
        st.write(f"Probability of Delay: {prediction_proba:.2f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
