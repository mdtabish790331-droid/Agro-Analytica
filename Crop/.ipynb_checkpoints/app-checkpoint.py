import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle
import warnings
import requests
import openai


warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

# Set up Streamlit page config
st.set_page_config(page_title="Agro Analytica", page_icon="https://cdn.jsdelivr.net/gh/twitter/twemoji@master/assets/72x72/1f33f.png", layout='centered', initial_sidebar_state="collapsed")

# Function to interact with the ChatGPT API
def get_chat_response(prompt, model="gpt-3.5-turbo"):
    api_endpoint = "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"
    headers = {'Content-Type': 'application/json'}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(api_endpoint, headers=headers, json=data)
    if response.status_code == 200:
        return response.json().get('choices', [])[0].get('message', '')
    else:
        return f"Error: {response.status_code}"

# Function to load saved models
def load_model(modelfile):
    loaded_model = pickle.load(open(modelfile, 'rb'))
    return loaded_model

# Main application function
def main():
    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:center;"> Agro Analytica: Intelligent Crop Recommendation 🌱 </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    col = st.columns(1)[0]

    with col:
        st.subheader(" Find out the most suitable crop to grow in your farm 👨‍🌾")
        
        # Input fields for the user
        N = st.number_input("Nitrogen", 1,10000)
        P = st.number_input("Phosporus", 1,10000)
        K = st.number_input("Potassium", 1,10000)
        temp = st.number_input("Temperature",0.0,100000.0)
        humidity = st.number_input("Humidity in %", 0.0,100000.0)
        ph = st.number_input("Ph", 0.0,100000.0)
        rainfall = st.number_input("Rainfall in mm",0.0,100000.0)

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1,-1)

        # Example test data (replace with actual input for prediction)
        custom_test_data = pd.DataFrame({
            'N': [N],
            'P': [P],
            'K': [K],
            'temperature': [temp],
            'humidity': [humidity],
            'ph': [ph],
            'rainfall': [rainfall],
        })

        # Crop dictionary
        crop_dict = {
            1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut',
            6: 'papaya', 7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon',
            11: 'grapes', 12: 'mango', 13: 'banana', 14: 'pomegranate', 15: 'lentil',
            16: 'blackgram', 17: 'mungbean', 18: 'mothbeans', 19: 'pigeonpeas', 20: 'kidneybeans',
            21: 'chickpea', 22: 'coffee'
        }

        # Options for model selection
        options = ['Decision TreeModel', 'Random Forest Model']
        selected_option = st.selectbox('Select an option:', options)

        if selected_option == options[0]:
            sel_model = 1
        else:
            sel_model = 2

        if st.button('Predict'):
            if sel_model == 1:
                # Load and predict using Decision Tree Model
                loaded_model = load_model('descision_tree_model.pkl')
                prediction = loaded_model.predict(single_pred)
            else:
                # Load and predict using Random Forest Model
                loaded_model = load_model('random_forest_model.pkl')
                prediction = loaded_model.predict(single_pred)

            # Get the recommended crop
            recommended_crop = crop_dict.get(prediction[0], 'Unknown Crop')
    
            col.write('''## Results 🔍''')
            col.success(f"{'Recommended ' + recommended_crop if prediction[0] else 'Not recommended'} by the A.I for your farm.")

            # Craft the prompt to ask the AI
            chat_prompt = f"Explain why {recommended_crop} is best for your land as suggested by {options[sel_model - 1]}"
            chat_response = get_chat_response(chat_prompt)
        
            # Display the AI response
            col.write(f"Ai Response: {chat_response}")

    # Hide Streamlit's default menu and footer
    hide_menu_style = """
    <style>
    .block-container {padding: 2rem 1rem 3rem;}
    #MainMenu {visibility: hidden;}
    </style>
    """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

# Run the app
if __name__ == '__main__':
    main()
