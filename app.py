from flask import Flask, render_template, request
import phonenumbers
from phonenumbers import geocoder, carrier
import pandas as pd
from opencage.geocoder import OpenCageGeocode
from datetime import datetime
import joblib
import os
from dotenv import load_dotenv
load_dotenv()


app = Flask(__name__)

# Load datasets
indian_df = pd.read_csv("dataset/phone_prefix_database.csv", dtype={'NDC': str})
country_df = pd.read_csv("dataset/country_codes_cleaned.csv")
country_df.dropna(subset=["Country", "Dial"], inplace=True)
country_df["Dial"] = country_df["Dial"].astype(str).str.replace("+", "").str.strip()

# Load all ML models
models = {
    "Decision Tree": joblib.load("models/operator_dt_model.pkl"),
    "Random Forest": joblib.load("models/operator_rf_model.pkl"),
    "Logistic Regression": joblib.load("models/operator_lr_model.pkl")
}

# Operator and Circle Mappings
operator_map = {
    "A": "Airtel", "B": "BSNL", "C": "Aircel", "D": "Videocon",
    "E": "Etisalat", "I": "Idea", "J": "Jio", "L": "Loop", "M": "MTNL",
    "Q": "S Tel", "R": "Reliance Comm", "T": "Tata Docomo", "U": "Uninor",
    "V": "Vodafone", "Y": "MTS"
}

circle_map = {
    "AP": "Andhra Pradesh", "AS": "Assam", "BR": "Bihar & Jharkhand",
    "CH": "Chennai", "DL": "Delhi", "GJ": "Gujarat", "HP": "Himachal Pradesh",
    "HR": "Haryana", "JK": "Jammu & Kashmir", "KL": "Kerala", "KN": "Karnataka",
    "KO": "Kolkata", "MH": "Maharashtra", "MP": "Madhya Pradesh & Chhattisgarh",
    "MU": "Mumbai", "NE": "North East", "OR": "Odisha", "PB": "Punjab",
    "RJ": "Rajasthan", "TN": "Tamil Nadu", "UE": "Uttar Pradesh (East)",
    "UW": "Uttar Pradesh (West) & Uttarakhand", "WB": "West Bengal"
}

# OpenCage API Key
API_KEY = os.getenv("OPENCAGE_API_KEY")
geo = OpenCageGeocode(API_KEY)

# Prepare dropdown data
country_codes = [{"country": row["Country"], "dial": row["Dial"]} for _, row in country_df.iterrows()]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", result=None, country_codes=country_codes)

@app.route("/map")
def show_map():
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    place = request.args.get("place", default="Location")
    return render_template("location.html", lat=lat, lon=lon, place=place)

@app.route("/track", methods=["POST"])
def track():
    number = request.form.get("number")
    country_code = request.form.get("country_code")
    selected_model = request.form.get("model_choice")

    if not number or not country_code or not selected_model:
        return render_template("index.html", result={"error": "Please fill all fields."}, country_codes=country_codes)

    full_number = f"+{country_code}{number}"

    try:
        parsed = phonenumbers.parse(full_number)
        location = geocoder.description_for_number(parsed, "en")
        service = carrier.name_for_number(parsed, "en")

        prefix = str(parsed.national_number)[:4]
        is_indian = country_code == "91"
        circle = operator = lat = lon = None
        operator_predictions = {}

        if is_indian:
            entry = indian_df[indian_df["NDC"] == prefix]
            if not entry.empty:
                prefix_digits = list(map(int, prefix.zfill(4)))

                # Predict using all models
                for model_name, model in models.items():
                    code = model.predict([prefix_digits])[0]
                    operator_predictions[model_name] = operator_map.get(code, "Unknown")

                # Use selected model's prediction
                operator = operator_predictions[selected_model]

                raw_circle = entry.iloc[0]["Telecom Circles"]
                circle = circle_map.get(raw_circle, raw_circle)

                geo_res = geo.geocode(f"{circle}, India")
                if geo_res:
                    lat = geo_res[0]["geometry"]["lat"]
                    lon = geo_res[0]["geometry"]["lng"]
        else:
            operator = service or "Unknown"
            circle = location or "Unknown"
            geo_res = geo.geocode(location)
            if geo_res:
                lat = geo_res[0]["geometry"]["lat"]
                lon = geo_res[0]["geometry"]["lng"]

        country = geo_res[0]["components"].get("country") if geo_res else "Unknown"

        result = {
            "country": country,
            "service_provider": service,
            "operator": operator,
            "circle": circle,
            "latitude": lat,
            "longitude": lon,
            "phone_number": full_number,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ml_model_used": selected_model,
            "operator_predictions": operator_predictions
        }

        return render_template("index.html", result=result, country_codes=country_codes)

    except Exception as e:
        return render_template("index.html", result={"error": str(e)}, country_codes=country_codes)

if __name__ == "__main__":
    app.run(debug=True)
