import streamlit as st
import requests
from backend import model_helper
import pandas as pd

st.set_page_config(page_title="CodeX Pricing Prediction", layout="wide")

API_URL = "http://localhost:8000/predict"

st.title("CodeX Energy Drink Pricing Prediction")
st.markdown("Predict the optimal price range for energy drinks based on consumer survey responses.")

st.sidebar.header("Consumer Profile")
st.sidebar.markdown("Provide details about the consumer to get a pricing prediction.")

# We will layout the form using columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    age = st.number_input("Age", min_value=18, max_value=100, value=25)
    gender = st.selectbox("Gender", model_helper.UI_OPTIONS["gender"])
    zone = st.selectbox("Zone", model_helper.UI_OPTIONS["zone"])
    occupation = st.selectbox("Occupation", model_helper.UI_OPTIONS["occupation"])
    income_levels = st.selectbox("Income Levels", model_helper.UI_OPTIONS["income_levels"])

with col2:
    st.subheader("Consumption Habits")
    consume_frequency = st.selectbox("Consume Frequency (Weekly)", model_helper.UI_OPTIONS["consume_frequency(weekly)"])
    current_brand = st.selectbox("Current Brand", model_helper.UI_OPTIONS["current_brand"])
    preferable_consumption_size = st.selectbox("Preferable Consumption Size", model_helper.UI_OPTIONS["preferable_consumption_size"])
    typical_consumption_situations = st.selectbox("Typical Consumption Situations", model_helper.UI_OPTIONS["typical_consumption_situations"])
    health_concerns = st.selectbox("Health Concerns", model_helper.UI_OPTIONS["health_concerns"])

with col3:
    st.subheader("Preferences & Brand")
    awareness_of_other_brands = st.selectbox("Awareness of Other Brands", model_helper.UI_OPTIONS["awareness_of_other_brands"])
    reasons_for_choosing_brands = st.selectbox("Reasons for Choosing Brands", model_helper.UI_OPTIONS["reasons_for_choosing_brands"])
    flavor_preference = st.selectbox("Flavor Preference", model_helper.UI_OPTIONS["flavor_preference"])
    purchase_channel = st.selectbox("Purchase Channel", model_helper.UI_OPTIONS["purchase_channel"])
    packaging_preference = st.selectbox("Packaging Preference", model_helper.UI_OPTIONS["packaging_preference"])

st.markdown("---")

if st.button("Predict Optimal Price Range", type="primary", use_container_width=True):
    payload = {
        "age": age,
        "gender": gender,
        "zone": zone,
        "occupation": occupation,
        "income_levels": income_levels,
        "consume_frequency(weekly)": consume_frequency,
        "current_brand": current_brand,
        "preferable_consumption_size": preferable_consumption_size,
        "awareness_of_other_brands": awareness_of_other_brands,
        "reasons_for_choosing_brands": reasons_for_choosing_brands,
        "flavor_preference": flavor_preference,
        "purchase_channel": purchase_channel,
        "packaging_preference": packaging_preference,
        "health_concerns": health_concerns,
        "typical_consumption_situations": typical_consumption_situations
    }
    
    with st.spinner("Analyzing profile and predicting price range..."):
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                predicted_price = result["predicted_price_range"]
                confidence = result["confidence"]
                
                st.success("Prediction Complete!")
                
                # Display metrics prominently
                metric_col1, metric_col2, metric_col3 = st.columns([1, 1, 2])
                with metric_col1:
                    st.metric(label="Predicted Price Range", value=predicted_price)
                with metric_col2:
                    st.metric(label="Confidence", value=f"{confidence*100:.1f}%")
                
                # Plot probabilities
                with metric_col3:
                    st.markdown("**Class Probabilities**")
                    probs = result["class_probabilities"]
                    prob_df = pd.DataFrame({
                        "Price Range": list(probs.keys()),
                        "Probability": list(probs.values())
                    }).set_index("Price Range")
                    st.bar_chart(prob_df, y="Probability", color="#FF4B4B")
                
            else:
                st.error(f"Error from API: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the backend API. Is the FastAPI server running on http://localhost:8000?")
