import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load Trained Model
# -------------------------------
model = joblib.load("house_price_model.pkl")

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

st.title("🏠 House Price Prediction System")
st.write("Enter the house details below to predict its price.")

# -------------------------------
# User Inputs
# -------------------------------
sqft = st.number_input(
    "Enter House Size (SqFt)",
    min_value=0.0,
    value=None
)

bedrooms = st.number_input(
    "Enter Number of Bedrooms",
    min_value=0,
    value=None
)

bathrooms = st.number_input(
    "Enter Number of Bathrooms",
    min_value=0,
    value=None
)

offers = st.number_input(
    "Enter Number of Offers",
    min_value=0,
    value=None
)

brick = st.selectbox(
    "Is the House made of Brick?",
    ["Select", "Yes", "No"],
    index=0
)

neighborhood = st.selectbox(
    "Select Neighborhood",
    ["Select", "East", "North", "West"],
    index=0
)

# -------------------------------
# Convert Categorical Values
# -------------------------------
brick_yes = 1 if brick == "Yes" else 0

north = 1 if neighborhood == "North" else 0
west = 1 if neighborhood == "West" else 0

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict House Price"):

    if brick == "Select" or neighborhood == "Select":
        st.error("Please select Brick and Neighborhood.")
    else:
        brick_yes = 1 if brick == "Yes" else 0

        north = 1 if neighborhood == "North" else 0
        west = 1 if neighborhood == "West" else 0

        new_data = pd.DataFrame({
            "SqFt": [sqft],
            "Bedrooms": [bedrooms],
            "Bathrooms": [bathrooms],
            "Offers": [offers],
            "Brick_Yes": [brick_yes],
            "Neighborhood_North": [north],
            "Neighborhood_West": [west]
        })

        prediction = model.predict(new_data)

        st.success(f"Predicted House Price: {prediction[0]:,.2f}$")
    st.subheader("House Details")

    st.write(f"**SqFt :** {sqft}")
    st.write(f"**Bedrooms :** {bedrooms}")
    st.write(f"**Bathrooms :** {bathrooms}")
    st.write(f"**Offers :** {offers}")
    st.write(f"**Brick :** {brick}")
    st.write(f"**Neighborhood :** {neighborhood}")