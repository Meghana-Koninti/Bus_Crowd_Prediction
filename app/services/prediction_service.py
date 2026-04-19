import pickle
import numpy as np
import pandas as pd

# Load the NEW V2 model
with open("models/model_v2.pkl", "rb") as f:
    model = pickle.load(f)

# Load routing data to get Stop_Order
try:
    bus_df = pd.read_csv("data/bus_dataset.csv")
    # Dictionary mapping (route_number, stop_name) -> stop_order
    # bus_dataset.csv might have 'route_number' and 'stop_name' columns based on v2 generation
    stop_order_map = {(str(r), str(s)): int(o) for r, s, o in zip(bus_df['route_number'], bus_df['stop_name'], bus_df['stop_order'])}
except Exception as e:
    print("Warning: Could not load bus_dataset.csv for Stop_Order mapping.", e)
    stop_order_map = {}

# These are the precise mappings derived from ml_dataset.csv using sklearn LabelEncoder
WEATHER_MAP = {'Clear': 0, 'Humid': 1, 'Overcast': 2, 'Rainy': 3}
DAY_MAP = {'Friday': 0, 'Monday': 1, 'Saturday': 2, 'Sunday': 3, 'Thursday': 4, 'Tuesday': 5, 'Wednesday': 6}
CROWD_DECODE_MAP = {0: 'High', 1: 'Low', 2: 'Medium'}

def get_stop_order(route, stop):
    route_str = str(route).strip()
    stop_str = str(stop).strip()
    if (route_str, stop_str) in stop_order_map:
        return stop_order_map[(route_str, stop_str)]
    # Fallback default if stop not found on route
    return 5

def predict_crowd(data):
    try:
        db_route = data.get("route", "")
        db_stop  = data.get("stop", "")
        weather  = data.get("weather", "Clear")
        day      = data.get("day", "Monday")

        hour     = int(data.get("hour", 0))
        buses    = int(data.get("buses", 6))

        # We must predict with these 9 features (in exact order for model_v2.pkl):
        # 1. Hour
        # 2. Day_of_Week (encoded 0-6)
        # 3. Is_Weekend (0 or 1)
        # 4. Stop_Order (derived from routing logic or default)
        # 5. Weather (encoded 0-3)
        # 6. Is_Festival (0 or 1, default 0 since UI doesn't send month/date properly)
        # 7. Bus_Capacity (use average 60)
        # 8. Buses_Available
        # 9. Dwell_Time_Seconds (use average 20s)

        # 1. Hour
        
        # 2. Day_of_Week
        day_enc = DAY_MAP.get(day, DAY_MAP['Monday'])
        
        # 3. Is_Weekend
        is_weekend = 1 if day in ["Saturday", "Sunday"] else 0
        
        # 4. Stop_Order
        stop_order = get_stop_order(db_route, db_stop)

        # 5. Weather
        weather_enc = WEATHER_MAP.get(weather, WEATHER_MAP['Clear'])

        # 6. Is_Festival
        is_festival = 0

        # 7. Bus_Capacity
        bus_capacity = 60
        
        # dynamically scale dwell time and available buses based on time of day
        # so the model produces realistic varied results (as dwell time is heavily weighted)
        if hour in [7, 8, 9, 17, 18, 19]:
            buses = 7
            dwell_time = 45
        elif hour in [6, 10, 11, 16, 20]:
            buses = 5
            dwell_time = 25
        else:
            buses = 3
            dwell_time = 10

        # Build feature vector
        features = np.array([[
            hour,
            day_enc,
            is_weekend,
            stop_order,
            weather_enc,
            is_festival,
            bus_capacity,
            buses,
            dwell_time
        ]])

        # Predict
        prediction_val = model.predict(features)[0]

        # Decode label
        prediction = CROWD_DECODE_MAP.get(prediction_val, "Medium")

        # Also get probabilities for confidence
        proba = model.predict_proba(features)[0] # e.g. [p_High, p_Low, p_Medium]
        
        # Confidence is the max probability
        confidence = round(float(max(proba)) * 100)
        
        # proba_dict mapping
        proba_dict = {
            'High': round(float(proba[0]) * 100),
            'Low': round(float(proba[1]) * 100),
            'Medium': round(float(proba[2]) * 100)
        }

        return True, {"prediction": prediction, "confidence": confidence, "proba": proba_dict}

    except Exception as e:
        print("ERROR INSIDE MODEL:", e)
        return False, str(e)