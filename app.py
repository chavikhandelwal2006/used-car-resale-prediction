
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# Load the saved model
MODEL_PATH = Path(__file__).parent / "random_forest_model.pkl"
model = joblib.load(MODEL_PATH)

# Page heading

st.sidebar.title("🚗 CarValue AI")
st.sidebar.write("Used-Car Resale Price Prediction")
st.sidebar.divider()
st.sidebar.info(
    "This project uses a Random Forest Regression model "
    "to estimate resale asking prices."
)

st.title("Used-Car Resale Price Predictor")
st.write(
    "Discover an estimated resale price using vehicle age, "
    "mileage, brand, model, fuel type, transmission, "
    "and ownership history."
)


st.divider()

# Input form
st.subheader("Enter Car Details")

with st.form("car_details"):
    col1, col2 = st.columns(2)

    with col1:
        brand = st.text_input("Brand", value="Hyundai")
        car_model = st.text_input("Model", value="Creta")
        age = st.number_input(
            "Car Age (years)", min_value=0, max_value=50, value=3
        )
        km_driven = st.number_input(
            "Kilometres Driven",
            min_value=0,
            max_value=1_000_000,
            value=35000,
            step=1000
        )

    with col2:
        fuel_type = st.selectbox(
            "Fuel Type",
            ["Petrol", "Diesel", "CNG", "LPG", "Electric", "Hybrid/CNG"]
        )
        transmission = st.selectbox(
            "Transmission", ["Manual", "Automatic"]
        )
        owner = st.selectbox(
            "Ownership",
            ["first", "second", "third", "fourth & above"]
        )

    submitted = st.form_submit_button(
        "Predict Resale Price",
        use_container_width=True
    )

if submitted:
    car_data = pd.DataFrame([{
        "Age": age,
        "kmDriven": km_driven,
        "Brand": brand.strip(),
        "model": car_model.strip(),
        "FuelType": fuel_type,
        "Transmission": transmission,
        "Owner": owner
    }])

    if not brand.strip() or not car_model.strip():
        st.error("Please enter both the car brand and model.")
    else:
        try:
            predicted_price = model.predict(car_data)[0]

            st.success("Prediction completed!")
            st.metric(
                "Estimated Resale Asking Price",
                f"₹{predicted_price:,.0f}"
            )
            st.caption(
                "This is a model-based estimate, not a guaranteed "
                "market or selling price."
            )
        except Exception as error:
            st.error(f"Prediction failed: {error}")
# Depreciation analysis
st.divider()
st.subheader("📉 Used-Car Depreciation Trend")

st.write(
    "Explore how median used-car asking prices "
    "change with vehicle age in our dataset."
)

graph_path = Path(__file__).parent / "depreciation_curve.png"

if graph_path.exists():
    st.image(
        str(graph_path),
        caption="Median asking price by car age",
        use_container_width=True
    )
else:
    st.warning("Depreciation graph not found.")


# Model performance
st.divider()
st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", "₹2.77 lakh")
col2.metric("RMSE", "₹9.32 lakh")
col3.metric("R² Score", "0.69")

st.caption(
    "Evaluation on the held-out test dataset of 1,743 records. "
    "R² indicates the proportion of target-price variance explained "
    "by the model; it does not mean 69% prediction accuracy."
)
