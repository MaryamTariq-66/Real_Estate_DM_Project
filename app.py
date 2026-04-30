import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and encoders
model = joblib.load('property_model.pkl')
le_type = joblib.load('le_type.pkl')
le_neigh = joblib.load('le_neigh.pkl')

st.title("🏡 Karachi Real Estate Price Predictor")
st.markdown("Predict the market value of properties in Karachi using Data Mining.")

# Sidebar for Inputs
st.sidebar.header("Property Specifications")
prop_type = st.sidebar.selectbox("Property Type", le_type.classes_)
neighborhood = st.sidebar.selectbox("Neighborhood", le_neigh.classes_)
area = st.sidebar.number_input("Area (Sq. Yards)", min_value=50, max_value=5000, value=120)
beds = st.sidebar.slider("Bedrooms", 0, 10, 3)
baths = st.sidebar.slider("Bathrooms", 0, 10, 3)

# Amenity Flags
west_open = st.sidebar.checkbox("Is West Open?")
parking = st.sidebar.checkbox("Has Parking?")
furnished = st.sidebar.checkbox("Is Furnished?")
installment = st.sidebar.checkbox("On Installments?")

# Predict Button
if st.button("Predict Price"):
    # Preprocess inputs
    type_enc = le_type.transform([prop_type])[0]
    neigh_enc = le_neigh.transform([neighborhood])[0]
    
    input_data = np.array([[area, beds, baths, int(west_open), int(parking), 
                            int(furnished), int(installment), type_enc, neigh_enc]])
    
    prediction = model.predict(input_data)[0]
    
    st.success(f"### Estimated Market Price: PKR {prediction:,.2f}")
    
    # Contextual Insight
    if prediction > 50000000:
        st.info("This property is in the 'Premium' tier of the Karachi market.")
    else:
        st.info("This property is in the 'Mid-Range' tier.")