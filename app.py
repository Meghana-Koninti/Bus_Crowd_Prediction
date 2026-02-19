from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load Model and Encoders
model_path = 'models/hyderabad_bus_model.pkl'
encoders_path = 'models/hyderabad_encoders.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(encoders_path, 'rb') as f:
    encoders = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    prediction_class = ""
    
    # Get unique values for dropdowns from encoders dictionary
    routes = encoders['Route_No'].classes_
    stops = encoders['Current_Stop'].classes_
    weathers = encoders['Weather'].classes_
    days = encoders['Day_of_Week'].classes_

    # Default values
    selected_route = ""
    selected_stop = ""
    selected_hour = datetime.now().hour
    selected_day = datetime.now().strftime('%A')
    selected_weather = ""
    
    # Check if reset was requested (via query param or specific button, though button usually submits form)
    # We'll handle reset in frontend by clearing form or reloading page without POST.

    if request.method == 'POST':
        try:
            # Inputs
            route = request.form.get('route')
            stop = request.form.get('stop')
            
            # sticky inputs
            selected_route = route
            selected_stop = stop
            
            # Handle time defaults if empty (though required in HTML, good to have fallback)
            hour_input = request.form.get('hour')
            if hour_input:
                hour = int(hour_input)
                selected_hour = hour
            else:
                hour = datetime.now().hour
                selected_hour = hour

            day_input = request.form.get('day')
            if day_input:
                day = day_input
                selected_day = day
            else:
                day = datetime.now().strftime('%A')
                selected_day = day

            weather = request.form.get('weather')
            selected_weather = weather
            
            # Additional features
            # Model trained on: Hour, Day_of_Week, Is_Weekend, Route_No, Current_Stop, Weather, Buses_Available
            # Buses_Available: Default to 5 (low availability assumption for conservative prediction)
            buses_available = 5 
            
            # Derive Is_Weekend
            is_weekend = 1 if day in ['Saturday', 'Sunday'] else 0

            # Encode Inputs
            route_encoded = encoders['Route_No'].transform([route])[0]
            stop_encoded = encoders['Current_Stop'].transform([stop])[0]
            day_encoded = encoders['Day_of_Week'].transform([day])[0]
            weather_encoded = encoders['Weather'].transform([weather])[0]

            # Prepare Features
            # Order: Hour, Day_of_Week, Is_Weekend, Route_No, Current_Stop, Weather, Buses_Available
            features = np.array([[hour, day_encoded, is_weekend, route_encoded, stop_encoded, weather_encoded, buses_available]])

            # Predict
            pred_encoded = model.predict(features)[0]
            prediction = encoders['Crowding_Level'].inverse_transform([pred_encoded])[0]

            # Determine Class for Color Coding
            if prediction == 'Low':
                prediction_class = 'low'
            elif prediction == 'Medium':
                prediction_class = 'medium'
            elif prediction == 'High':
                prediction_class = 'high'
            
            print(f"Prediction (v3): {route} @ {stop}, {hour}:00, {day} -> {prediction}")

        except Exception as e:
            print(f"Error: {e}")
            prediction = f"Error: {str(e)}"
            prediction_class = "high"

    return render_template('index.html', 
                           routes=routes, 
                           stops=stops, 
                           weathers=weathers, 
                           days=days, 
                           prediction=prediction, 
                           prediction_class=prediction_class,
                           selected_route=selected_route,
                           selected_stop=selected_stop,
                           selected_hour=selected_hour,
                           selected_day=selected_day,
                           selected_weather=selected_weather)

if __name__ == '__main__':
    app.run(debug=True)
