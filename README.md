# 📱 Phone Number Tracking Using Machine Learning

A Flask-based web application that analyzes phone numbers and predicts the telecom operator using Machine Learning models. The application also provides country, service provider, telecom circle, coordinates, and interactive map information.

---

## 📸 Application Preview

![Phone Number Tracking ML]("C:\Users\Swapnil\OneDrive\Documents\Phone-Number-Tracking-ML\screenshots")


## 🎯 Project Overview

The **Phone Number Tracking Using Machine Learning** project is a Python Flask web application designed to analyze Indian phone numbers.

The application uses phone number information, prefix-based data, and Machine Learning models to predict the telecom operator.

It also integrates geocoding and interactive maps to display geographical information.

---

## ✨ Features

- 📱 Phone number analysis
- 🌍 Country identification
- 📡 Service provider identification
- 📍 Telecom circle identification
- 🗺️ Latitude and longitude information
- 🌐 Interactive Leaflet map
- 📌 Google Maps integration
- 🤖 Machine Learning operator prediction
- 🌳 Decision Tree prediction
- 🌲 Random Forest prediction
- 📊 Logistic Regression prediction
- 🔄 Comparison of predictions from multiple ML models
- 👍 User feedback for prediction results

---

## 🧠 Machine Learning Models

The project uses three Machine Learning algorithms:

| Model | Purpose |
|---|---|
| Decision Tree | Telecom operator classification |
| Random Forest | Telecom operator classification |
| Logistic Regression | Telecom operator classification |

The models are trained using phone number prefix information.

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Framework

- Flask

### Machine Learning

- Scikit-learn
- Decision Tree
- Random Forest
- Logistic Regression

### Data Processing

- Pandas
- NumPy

### Phone Number Processing

- Python PhoneNumbers Library

### Geocoding & Maps

- OpenCage Geocoder
- Folium
- Leaflet.js
- Google Maps

### Other Tools

- Joblib
- HTML
- CSS
- Bootstrap

---

## 📂 Project Structure

```text
Phone-Number-Tracking-ML/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
│
├── dataset/
│   ├── phone_prefix_database.csv
│   └── country_codes_cleaned.csv
│
├── models/
│   ├── operator_dt_model.pkl
│   ├── operator_rf_model.pkl
│   └── operator_lr_model.pkl
│
├── screenshots/
│   └── phone_tracking_result.png
│
├── static/
│   ├── css/
│   └── images/
│
└── templates/
    ├── index.html
    └── location.html
