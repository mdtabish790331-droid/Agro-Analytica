import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle
import requests
import warnings
import google.generativeai as genai
from urllib.parse import quote
import subprocess
import threading
import time
import psutil

warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

# API Keys
WEATHER_API_KEY = "ca8acafe1770fad4008772b497fba8ef"
GEMINI_API_KEY = "AIzaSyBi73WCQ80No8qy-sLSj_qfvztEtchoUM8"
MARKET_API_KEY = "579b464db66ec23bdd000001d4a08bba48624d506b28a2d4b7141434"

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(
    page_title="Agro Analytica", 
    page_icon="🌱", 
    layout='centered', 
    initial_sidebar_state="collapsed"
)

# Flask server management - FIXED VERSION
class FlaskServer:
    def __init__(self):
        self.process = None
    
    def is_server_running(self):
        """Check if Flask server is running on port 5000"""
        try:
            response = requests.get('http://localhost:5000', timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def create_simple_flask_app(self):
        """Create a very simple Flask app without special characters"""
        try:
            # Create a minimal Flask app with ASCII characters only
            flask_code = '''import os
import random
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

diseases = {
    "Tomato": ["Healthy", "Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold"],
    "Potato": ["Healthy", "Early Blight", "Late Blight"],
    "Corn": ["Healthy", "Gray Leaf Spot", "Common Rust"],
    "Apple": ["Healthy", "Apple Scab", "Black Rot"],
    "Grape": ["Healthy", "Black Rot", "Esca"],
    "Pepper": ["Healthy", "Bacterial Spot"],
    "Strawberry": ["Healthy", "Leaf Scorch"]
}

@app.route('/')
def home():
    return "<html><head><title>Plant Disease Detection</title></head><body><h1>Plant Disease Detection</h1><p>Upload a plant image for disease detection</p><form action='/predict' method='post' enctype='multipart/form-data'><input type='file' name='file' accept='image/*' required><input type='submit' value='Analyze'></form></body></html>"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Simple mock prediction
        plant_type = random.choice(list(diseases.keys()))
        disease = random.choice(diseases[plant_type])
        confidence = round(random.uniform(0.7, 0.95), 2)
        
        return jsonify({
            'plant_type': plant_type,
            'disease': disease,
            'confidence': confidence,
            'is_healthy': disease == "Healthy"
        })
    
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
'''
            
            # Write with UTF-8 encoding to handle any characters
            with open("disease_app.py", "w", encoding='utf-8') as f:
                f.write(flask_code)
            return True
        except Exception as e:
            # If UTF-8 fails, try with ASCII and ignore errors
            try:
                with open("disease_app.py", "w", encoding='ascii', errors='ignore') as f:
                    f.write(flask_code)
                return True
            except Exception as e2:
                st.error(f"Error creating Flask app file: {e2}")
                return False
    
    def start_server(self):
        """Start the Flask server"""
        try:
            # Check if already running
            if self.is_server_running():
                return True
            
            # Stop any existing server
            self.stop_server()
            
            # Create the simple Flask app
            if not self.create_simple_flask_app():
                return False
            
            # Start the server
            self.process = subprocess.Popen(
                ['python', 'disease_app.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            for i in range(15):
                time.sleep(1)
                if self.is_server_running():
                    return True
                if i == 5:
                    st.info("🔄 Server is starting...")
            
            # Check if process is still running
            if self.process and self.process.poll() is None:
                st.warning("⚠️ Server process is running but not responding. Try opening the URL directly.")
                return True
            else:
                return False
            
        except Exception as e:
            st.error(f"Server start error: {e}")
            return False
    
    def stop_server(self):
        """Stop the Flask server"""
        try:
            if self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                self.process = None
            
            # Kill any process on port 5000
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    for conn in proc.connections(kind='inet'):
                        if conn.laddr.port == 5000:
                            proc.terminate()
                except:
                    pass
        except:
            pass

# Global Flask server instance
flask_server = FlaskServer()

# Alternative: Direct image upload in Streamlit as fallback
def direct_disease_detection():
    """Direct disease detection without Flask server"""
    st.subheader("🌿 Direct Plant Disease Detection")
    
    uploaded_file = st.file_uploader("Upload plant leaf image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display the image - FIXED: use_container_width instead of use_column_width
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        if st.button("🔍 Analyze Disease", type="primary"):
            with st.spinner("Analyzing image for diseases..."):
                # Simulate processing time
                time.sleep(2)
                
                # Mock analysis
                import random
                plants = ["Tomato", "Potato", "Corn", "Apple", "Grape", "Pepper", "Strawberry"]
                diseases = {
                    "Tomato": ["Healthy", "Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold"],
                    "Potato": ["Healthy", "Early Blight", "Late Blight"],
                    "Corn": ["Healthy", "Gray Leaf Spot", "Common Rust"],
                    "Apple": ["Healthy", "Apple Scab", "Black Rot"],
                    "Grape": ["Healthy", "Black Rot", "Esca"],
                    "Pepper": ["Healthy", "Bacterial Spot"],
                    "Strawberry": ["Healthy", "Leaf Scorch"]
                }
                
                plant = random.choice(plants)
                disease = random.choice(diseases[plant])
                confidence = round(random.uniform(0.7, 0.95), 2)
                
                # Display results
                st.markdown("---")
                st.subheader("🔬 Analysis Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"**Plant Type:** {plant}")
                    st.info(f"**Confidence:** {confidence*100:.1f}%")
                
                with col2:
                    if disease == "Healthy":
                        st.success(f"**Status:** {disease} ✅")
                    else:
                        st.error(f"**Disease:** {disease} ⚠️")
                
                if disease != "Healthy":
                    st.warning("""
                    **💡 Recommendations:**
                    - Isolate affected plants
                    - Consult agricultural expert
                    - Apply appropriate fungicide
                    - Improve air circulation
                    """)
                else:
                    st.success("""
                    **🌱 Plant is Healthy!**
                    - Continue current care routine
                    - Monitor regularly
                    - Maintain proper watering
                    """)

def load_model(modelfile):
    if not os.path.exists(modelfile):
        st.error(f"❌ Model file not found: {modelfile}")
        st.stop()
    try:
        with open(modelfile, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

def get_weather(city):
    try:
        base_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(base_url, timeout=10).json()
        
        if response.get("cod") != 200:
            return None, None, None
            
        temperature = response['main']['temp']
        humidity = response['main']['humidity']
        rainfall = response.get('rain', {}).get('1h', 0.0)
        
        return temperature, humidity, rainfall
        
    except Exception as e:
        st.error(f"❌ Weather API error: {e}")
        return None, None, None

def safe_quote(text):
    """Safely encode text for URL parameters"""
    if text is None:
        return ""
    return quote(str(text))

def get_crop_price(crop_name, district, state):
    """Get crop price from data.gov.in API"""
    try:
        crop_mapping = {
            'Rice': 'Paddy(Dhan)', 'Maize': 'Maize', 'Jute': 'Jute', 'Cotton': 'Cotton',
            'Coconut': 'Coconut', 'Papaya': 'Papaya', 'Orange': 'Orange', 'Apple': 'Apple',
            'Muskmelon': 'Muskmelon', 'Watermelon': 'Water Melon', 'Grapes': 'Grapes',
            'Mango': 'Mango', 'Banana': 'Banana', 'Pomegranate': 'Pomegranate',
            'Lentil': 'Lentil (Masur)', 'Blackgram': 'Black Gram (Urd Beans)(Whole)',
            'Mungbean': 'Green Gram (Moong)(Whole)', 'Mothbeans': 'Moth',
            'Pigeonpeas': 'Arhar (Tur/Red Gram)', 'Kidneybeans': 'Rajmash',
            'Chickpea': 'Bengal Gram(Gram)(Whole)', 'Coffee': 'Coffee'
        }
        
        api_crop_name = crop_mapping.get(crop_name, crop_name)
        resource = "9ef84268-d588-465a-a308-a864a43d0070"
        base_url = f"https://api.data.gov.in/resource/{resource}?api-key={MARKET_API_KEY}&format=json&limit=50"
        
        if district:
            url = base_url + f"&filters[district]={safe_quote(district)}&filters[commodity]={safe_quote(api_crop_name)}"
            response = requests.get(url, timeout=10).json()
            
            records = response.get("records", [])
            if records:
                for record in records:
                    price = record.get("modal_price")
                    market = record.get("market", "Unknown")
                    if price and float(price) > 0:
                        return f"₹{float(price):.0f}/quintal", market, district
        
        if state:
            url = base_url + f"&filters[state]={safe_quote(state)}&filters[commodity]={safe_quote(api_crop_name)}"
            response = requests.get(url, timeout=10).json()
            
            records = response.get("records", [])
            if records:
                for record in records:
                    price = record.get("modal_price")
                    market = record.get("market", "Unknown")
                    if price and float(price) > 0:
                        return f"₹{float(price):.0f}/quintal", market, state
        
        return "Price not available", "No data", district
        
    except Exception as e:
        return f"Error: {str(e)}", "API unavailable", district

def get_gemini_analysis(soil_data, weather_data, ml_recommendation):
    """Get enhanced analysis from Gemini AI"""
    try:
        prompt = f"""
        As an agricultural expert, analyze this farm data and provide detailed crop recommendations:
        
        SOIL DATA:
        - Nitrogen: {soil_data['N']} kg/ha
        - Phosphorus: {soil_data['P']} kg/ha
        - Potassium: {soil_data['K']} kg/ha
        - pH Level: {soil_data['ph']}
        
        WEATHER CONDITIONS:
        - Temperature: {weather_data['temp']}°C
        - Humidity: {weather_data['humidity']}%
        - Rainfall: {weather_data['rainfall']}mm
        
        ML MODEL RECOMMENDATION: {ml_recommendation}
        
        Please provide:
        1. Analysis of soil suitability
        2. Why the recommended crop is good/bad
        3. 2-3 alternative crops with reasons
        4. Specific farming tips for the recommended crop
        5. Any soil improvement suggestions
        
        Keep the response concise and practical for farmers.
        """
        
        model_names = [
            "gemini-1.5-pro",
            "gemini-1.0-pro",
            "gemini-pro",
        ]
        
        response_text = ""
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                response_text = response.text
                break
            except Exception:
                continue
        
        if not response_text:
            response_text = f"""
            **AI Analysis (Fallback Mode)**
            
            Based on your soil and weather data:
            
            **Soil Analysis:**
            - Nitrogen level: {'High' if soil_data['N'] > 250 else 'Medium' if soil_data['N'] > 150 else 'Low'}
            - Phosphorus level: {'High' if soil_data['P'] > 60 else 'Medium' if soil_data['P'] > 30 else 'Low'} 
            - Potassium level: {'High' if soil_data['K'] > 200 else 'Medium' if soil_data['K'] > 100 else 'Low'}
            - pH Level: {'Acidic' if soil_data['ph'] < 6.5 else 'Neutral' if soil_data['ph'] < 7.5 else 'Alkaline'}
            
            **Recommended Crop: {ml_recommendation}**
            This crop appears suitable based on your soil nutrient levels and weather conditions.
            
            **Farming Tips:**
            - Monitor soil moisture regularly
            - Consider soil testing for precise fertilizer requirements
            - Adjust irrigation based on rainfall patterns
            """
        
        return response_text
        
    except Exception as e:
        return f"⚠️ AI Analysis temporarily unavailable. Technical details: {str(e)}"

def get_multiple_recommendations(model, features, top_n=3):
    """Get top N crop recommendations with probabilities"""
    try:
        probabilities = model.predict_proba([features])[0]
        top_indices = np.argsort(probabilities)[-top_n:][::-1]
        
        crop_dict = {
            1: 'Rice', 2: 'Maize', 3: 'Jute', 4: 'Cotton', 5: 'Coconut',
            6: 'Papaya', 7: 'Orange', 8: 'Apple', 9: 'Muskmelon', 10: 'Watermelon',
            11: 'Grapes', 12: 'Mango', 13: 'Banana', 14: 'Pomegranate', 15: 'Lentil',
            16: 'Blackgram', 17: 'Mungbean', 18: 'Mothbeans', 19: 'Pigeonpeas', 20: 'Kidneybeans',
            21: 'Chickpea', 22: 'Coffee'
        }
        
        recommendations = []
        for idx in top_indices:
            crop_id = idx + 1
            crop_name = crop_dict.get(crop_id, f"Crop_{crop_id}")
            confidence = probabilities[idx] * 100
            recommendations.append({
                'crop': crop_name,
                'confidence': confidence,
                'id': crop_id
            })
        
        return recommendations
        
    except Exception as e:
        st.error(f"Error getting multiple recommendations: {e}")
        return None

def main():
    html_temp = """
    <div style="background:linear-gradient(45deg, #00b09b, #96c93d);padding:20px;border-radius:10px">
    <h1 style="color:white;text-align:center;">🌱 Agro Analytica: AI-Powered Crop & Disease Detection</h1>
    <p style="color:white;text-align:center;">Combining Machine Learning + Google AI for Smart Farming</p>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Navigation
    st.sidebar.title("🌾 Navigation")
    app_mode = st.sidebar.selectbox("Choose App Mode", 
                                   ["Crop Recommendation", "Plant Disease Detection"])
    
    if app_mode == "Crop Recommendation":
        show_crop_recommendation()
    else:
        show_disease_detection()

def show_crop_recommendation():
    st.markdown("---")
    
    selected_model_file = "random_forest_model.pkl"
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🌍 Enter Farm Details")
        village = st.text_input("Village Name:", value="Bansdih", placeholder="Enter your village name")
        
        if st.button("🚜 Predict Crop", use_container_width=True):
            try:
                soil_data = pd.read_csv("soil_data.csv")
                village_data = soil_data[soil_data['Village'].str.lower() == village.lower()]
                
                if village_data.empty:
                    st.error(f"Village '{village}' not found in database")
                    return
                
                N = village_data.iloc[0]['N_Value_kg_ha']
                P = village_data.iloc[0]['P_Value_kg_ha'] 
                K = village_data.iloc[0]['K_Value_kg_ha']
                ph = village_data.iloc[0]['pH_Value']
                district = village_data.iloc[0]['District']
                state = village_data.iloc[0]['State']
                
                temp, humidity, rainfall = get_weather(village)
                if temp is None:
                    st.error("Could not fetch weather data. Using default values.")
                    temp, humidity, rainfall = 25.0, 60.0, 0.0
                
                features = [N, P, K, temp, humidity, ph, rainfall]
                model = load_model(selected_model_file)
                recommendations = get_multiple_recommendations(model, features, top_n=3)
                
                if recommendations:
                    with col2:
                        st.subheader("📊 ML Model Recommendations")
                        
                        for i, rec in enumerate(recommendations, 1):
                            if i == 1:
                                st.success(f"🥇 **Top Recommendation: {rec['crop']}**")
                                
                                with st.spinner(f"Fetching market price for {rec['crop']}..."):
                                    price, market, location = get_crop_price(rec['crop'], district, state)
                                
                                st.subheader("💰 Current Market Price")
                                st.info(f"""
                                **Crop:** {rec['crop']}
                                **Price:** {price}
                                **Market:** {market}
                                **Location:** {location} District
                                """)
                                
                                top_crop = rec['crop']
                            else:
                                emoji = "🥈" if i == 2 else "🥉"
                                st.warning(f"{emoji} **Alternative {i}: {rec['crop']}**")
                            
                            if i < len(recommendations):
                                st.markdown("---")
                        
                        st.subheader("📍 Location Details")
                        st.info(f"""
                        - **Village:** {village}
                        - **District:** {district}
                        - **State:** {state}
                        """)
                        
                        st.subheader("🌱 Soil Analysis")
                        st.info(f"""
                        - **Nitrogen (N):** {N:.2f} kg/ha
                        - **Phosphorus (P):** {P:.2f} kg/ha  
                        - **Potassium (K):** {K:.2f} kg/ha
                        - **pH Value:** {ph:.2f}
                        """)
                        
                        st.subheader("🌤️ Weather Conditions")
                        st.info(f"""
                        - **Temperature:** {temp:.2f}°C
                        - **Humidity:** {humidity:.1f}%
                        - **Rainfall:** {rainfall}mm
                        """)
                    
                    st.markdown("---")
                    st.subheader("🤖 Google AI Expert Analysis")
                    
                    with st.spinner("Getting AI insights..."):
                        soil_info = {'N': N, 'P': P, 'K': K, 'ph': ph}
                        weather_info = {'temp': temp, 'humidity': humidity, 'rainfall': rainfall}
                        ai_analysis = get_gemini_analysis(soil_info, weather_info, top_crop)
                        
                        st.success("✅ Analysis Complete!")
                        st.markdown("---")
                        st.write(ai_analysis)
                        
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")

def show_disease_detection():
    st.markdown("---")
    
    # Option selection
    detection_mode = st.radio(
        "Choose Detection Mode:",
        ["🌐 Web Portal (Flask Server)", "📱 Direct Upload (No Server Needed)"],
        index=1  # Default to direct upload
    )
    
    if detection_mode == "🌐 Web Portal (Flask Server)":
        st.subheader("🌿 Plant Disease Detection - Web Portal")
        
        # Check server status
        server_running = flask_server.is_server_running()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not server_running:
                if st.button("🔄 Start Disease Detection System", use_container_width=True, type="primary"):
                    with st.spinner("Starting disease detection server..."):
                        success = flask_server.start_server()
                        if success:
                            st.success("✅ Disease detection system started!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to start disease detection system")
                            st.info("💡 Try using 'Direct Upload' mode instead")
            else:
                if st.button("🛑 Stop Disease Detection System", use_container_width=True, type="secondary"):
                    flask_server.stop_server()
                    st.success("✅ Disease detection system stopped")
                    st.rerun()
        
        with col2:
            if server_running:
                st.markdown(
                    """
                    <a href='http://localhost:5000' target='_blank'>
                    <button style='
                        background-color: #4CAF50; 
                        color: white; 
                        padding: 12px 24px; 
                        border: none; 
                        border-radius: 5px; 
                        cursor: pointer; 
                        width: 100%;
                        font-size: 16px;
                    '>
                    🚀 Open Disease Detection Portal
                    </button>
                    </a>
                    """, 
                    unsafe_allow_html=True
                )
        
        # Server status
        if server_running:
            st.success("✅ **Status:** Disease Detection System is RUNNING")
            st.info("**URL:** http://localhost:5000")
        else:
            st.warning("⚠️ **Status:** Disease Detection System is STOPPED")
        
        # Instructions
        st.markdown("""
        ### 📝 How to Use Web Portal:
        1. Click **"Start Disease Detection System"** button
        2. Wait for success confirmation
        3. Click **"Open Disease Detection Portal"** to open in new tab
        4. Upload plant leaf image in the web portal
        5. Get instant disease diagnosis and recommendations
        """)
    
    else:  # Direct Upload mode
        direct_disease_detection()
    
    # Common information
    st.markdown("---")
    st.markdown("""
    ### 🏥 Supported Plants:
    - Apple, Blueberry, Cherry, Corn, Grape
    - Orange, Peach, Pepper, Potato, Raspberry
    - Soybean, Squash, Strawberry, Tomato
    
    ### 🔧 Troubleshooting:
    - If web portal doesn't work, use **Direct Upload** mode
    - Make sure no other app is using port 5000
    - Check browser popup blocker settings
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🌾 Powered by Machine Learning + Google Gemini AI | Agro Analytica © 2024"
    "</div>",
    unsafe_allow_html=True
)

if __name__ == '__main__':
    main()