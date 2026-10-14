# Used-Car Resale Price Prediction

## Project Overview
This project predicts the estimated resale asking price of a used car using machine learning. It also explores how median asking prices vary with vehicle age.

## Objectives
- Estimate used-car resale prices using vehicle information.
- Analyze the relationship between car age and asking price.
- Evaluate the performance of a machine learning regression model.
- Provide a simple web interface for making predictions.

## Dataset
The project uses a used-car dataset containing details such as brand, model, year, age, kilometres driven, fuel type, transmission, ownership history, and asking price.

## Machine Learning Model
A Random Forest Regression model is used to predict resale asking prices.

### Input Features
- Car age
- Kilometres driven
- Brand and model
- Fuel type
- Transmission
- Ownership history

### Model Performance
- **MAE:** ₹2.77 lakh
- **RMSE:** ₹9.32 lakh
- **R² Score:** 0.69

These metrics were calculated on a held-out test dataset. The R² score does not represent 69% prediction accuracy.

## Depreciation Analysis
The project visualizes median asking prices by vehicle age. In the dataset, the median asking price is ₹14.5 lakh for one-year-old cars and ₹4.5 lakh for ten-year-old cars. This is a comparison across listings, not a guaranteed depreciation rate for an individual vehicle.

## Web Application
The Streamlit app allows users to enter vehicle details and obtain an estimated resale asking price. It also displays the depreciation graph and model evaluation metrics.

## How to Run
1. Install Python.
2. Install the required libraries:
   `pip install -r requirements.txt`
3. Start the application:
   `python -m streamlit run app.py`

## Limitations
Predictions depend on the dataset and its quality. Actual resale prices can differ due to vehicle condition, location, service history, market demand, and other factors. The model's errors are larger for some expensive vehicles.

## Disclaimer
Predictions are estimates for educational and informational purposes only. They are not guaranteed market prices.