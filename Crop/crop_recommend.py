import requests
import pandas as pd

# Load soil dataset
soil_data = pd.read_csv("soil_data.csv")

# Weather API key
API_KEY = "ca8acafe1770fad4008772b497fba8ef"

def get_weather(village):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={village},IN&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    
    if data.get("main"):
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        rainfall = data.get("rain", {}).get("1h", 0)  # rainfall ho to lega warna 0
        return temp, humidity, rainfall
    else:
        return None, None, None

def get_soil(village):
    row = soil_data[soil_data["Village"].str.lower() == village.lower()]
    if not row.empty:
        return row.iloc[0]["N_Medium_Percent"], row.iloc[0]["P_Medium_Percent"], row.iloc[0]["K_Medium_Percent"], row.iloc[0]["pH_Normal_Percent"]
    else:
        return None, None, None, None
