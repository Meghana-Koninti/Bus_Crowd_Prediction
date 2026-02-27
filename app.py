from flask import Flask, render_template, request, jsonify
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

# ─────────────────────────────────────────────
# Real-world Route → Stop mapping for Hyderabad
# This prevents impossible route+stop combinations
# ─────────────────────────────────────────────
ROUTE_STOPS = {
    "9X":  ["Secunderabad Station", "SR Nagar", "Ameerpet", "Hitech City", "Madhapur", "Gachibowli"],
    "10H": ["LB Nagar", "Uppal", "Secunderabad Station", "Ameerpet", "Banjara Hills", "Jubilee Hills Checkpost", "Madhapur"],
    "47L": ["Charminar", "Koti", "Mehdipatnam", "Ameerpet", "SR Nagar", "Banjara Hills"],
    "5K":  ["Secunderabad Station", "Koti", "Charminar", "LB Nagar", "Uppal"],
    "127K":["Gachibowli", "Hitech City", "Madhapur", "Banjara Hills", "Ameerpet", "SR Nagar", "Secunderabad Station"],
    "218": ["Mehdipatnam", "Banjara Hills", "Jubilee Hills Checkpost", "Gachibowli", "Hitech City"],
}

@app.route('/get_stops/<route>', methods=['GET'])
def get_stops(route):
    """API endpoint: returns valid stops for a given route."""
    stops = ROUTE_STOPS.get(route, [])
    return jsonify({"stops": stops})

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    prediction_class = ""
    error_message = None

    routes = list(ROUTE_STOPS.keys())
    weathers = encoders['Weather'].classes_
    days = encoders['Day_of_Week'].classes_

    selected_route = ""
    selected_stop = ""
    selected_hour = datetime.now().hour
    selected_day = datetime.now().strftime('%A')
    selected_weather = ""
    available_stops = []

    if request.method == 'POST':
        try:
            route = request.form.get('route')
            stop = request.form.get('stop')

            selected_route = route
            selected_stop = stop
            available_stops = ROUTE_STOPS.get(route, [])

            # ── VALIDATION: Check if stop belongs to selected route ──
            if route not in ROUTE_STOPS:
                error_message = f"Route '{route}' is not a recognized route."
            elif stop not in ROUTE_STOPS[route]:
                valid_stops = ", ".join(ROUTE_STOPS[route])
                error_message = (
                    f"Stop '{stop}' is not on Route {route}. "
                    f"Valid stops for {route} are: {valid_stops}."
                )
            else:
                hour_input = request.form.get('hour')
                hour = int(hour_input) if hour_input else datetime.now().hour
                selected_hour = hour

                day = request.form.get('day') or datetime.now().strftime('%A')
                selected_day = day

                weather = request.form.get('weather')
                selected_weather = weather

                buses_available = int(request.form.get('buses_available', 5))
                is_weekend = 1 if day in ['Saturday', 'Sunday'] else 0

                route_encoded   = encoders['Route_No'].transform([route])[0]
                stop_encoded    = encoders['Current_Stop'].transform([stop])[0]
                day_encoded     = encoders['Day_of_Week'].transform([day])[0]
                weather_encoded = encoders['Weather'].transform([weather])[0]

                features = np.array([[hour, day_encoded, is_weekend,
                                      route_encoded, stop_encoded,
                                      weather_encoded, buses_available]])

                pred_encoded = model.predict(features)[0]
                prediction = encoders['Crowding_Level'].inverse_transform([pred_encoded])[0]
                prediction_class = prediction.lower()

                print(f"Prediction: {route} @ {stop}, {hour}:00, {day} -> {prediction}")

        except Exception as e:
            print(f"Error: {e}")
            error_message = f"Something went wrong: {str(e)}"

    return render_template('index.html',
                           routes=routes,
                           weathers=weathers,
                           days=days,
                           prediction=prediction,
                           prediction_class=prediction_class,
                           error_message=error_message,
                           selected_route=selected_route,
                           selected_stop=selected_stop,
                           selected_hour=selected_hour,
                           selected_day=selected_day,
                           selected_weather=selected_weather,
                           available_stops=available_stops,
                           route_stops=ROUTE_STOPS)

if __name__ == '__main__':
    app.run(debug=True)
