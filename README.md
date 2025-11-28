🌱 Agro Analytica
AI-Powered Crop Recommendation & Plant Disease Detection System
<br>
Agro Analytica is a smart agriculture assistant that combines Machine Learning, Google Gemini AI, Weather API, Market Price API, and Flask + Streamlit technologies to help farmers make better agricultural decisions.
<br>
This system predicts:
<br>
.✅ Best crop to grow based on soil & weather
<br>
.✅ Live crop market prices (from data.gov.in)
<br>
.✅ AI expert advice from Google Gemini
<br>
.✅ Plant disease detection (via Flask portal or direct upload)
<br>
🚀 Features
<br>
1️⃣ Crop Recommendation System
<br>
.Uses Random Forest ML Model
<br>
.Inputs automatically fetched based on village:
<br>
.Nitrogen (N)
<br>
.Phosphorus (P)
<br>
.Potassium (K)
<br>
.pH Level
<br>
.Temperature
<br>
.Humidity
<br>
.Rainfall
<br>
.Top 3 crop suggestions with confidence
<br>
.Fetches Live Market Price from Data.gov.in API
<br>
.Provides AI Expert Analysis powered by Google Gemini
<br>
2️⃣ Plant Disease Detection
<br>
Two modes available:
<br>
A. Web Portal (Flask Server)
<br>
.Runs a lightweight Flask server on port 5000
<br>
.Allows plant leaf image upload
<br>
.Returns:
<br>
 .Plant type
<br>
 .Disease name
<br>
 .Confidence score
<br>
 .Healthy / unhealthy status
<br>
B. Direct Upload (No Server Needed)
<br>
 .Upload an image directly inside Streamlit
<br>
 .Performs mock disease detection
<br>
 .Gives actionable recommendations
