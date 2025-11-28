🌱 Agro Analytica
AI-Powered Crop Recommendation & Plant Disease Detection System
Agro Analytica is a smart agriculture assistant that combines Machine Learning, Google Gemini AI, Weather API, Market Price API, and Flask + Streamlit technologies to help farmers make better agricultural decisions.
This system predicts:
✅ Best crop to grow based on soil & weather
✅ Live crop market prices (from data.gov.in)
✅ AI expert advice from Google Gemini
✅ Plant disease detection (via Flask portal or direct upload)

🚀 Features
1️⃣ Crop Recommendation System

Uses Random Forest ML Model
Inputs automatically fetched based on village:
Nitrogen (N)
Phosphorus (P)
Potassium (K)
pH Level
Temperature
Humidity
Rainfall
Top 3 crop suggestions with confidence
Fetches Live Market Price from Data.gov.in API
Provides AI Expert Analysis powered by Google Gemini

2️⃣ Plant Disease Detection

Two modes available:
A. Web Portal (Flask Server)
Runs a lightweight Flask server on port 5000
Allows plant leaf image upload
Returns:
Plant type
Disease name
Confidence score
Healthy / unhealthy status
B. Direct Upload (No Server Needed)
Upload an image directly inside Streamlit
Performs mock disease detection
Gives actionable recommendations
