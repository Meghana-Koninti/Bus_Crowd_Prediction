from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
from app.services.prediction_service import predict_crowd
import datetime

prediction_bp = Blueprint("prediction", __name__)

# Weather options must match what the model was trained on
WEATHER_OPTIONS = ["Clear", "Rainy", "Overcast", "Humid"]

ROUTES = ["9X", "47L", "127K", "218", "10H", "5K", "216", "400"]
DAYS   = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


# ── UI route ─────────────────────────────────────────────────────────────────
@prediction_bp.route("/")
@login_required
def home():
    now = datetime.datetime.now()
    day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    return render_template(
        "index.html",
        user=current_user.username,
        stop_coords={},
        routes=ROUTES,
        days=DAYS,
        weathers=WEATHER_OPTIONS,   # ← Clear / Rainy / Overcast / Humid (matches training)
        current_hour=now.hour,
        current_day=day_names[(now.weekday() + 1) % 7],
        current_month=now.month
    )


# ── Prediction API ────────────────────────────────────────────────────────────
@prediction_bp.route("/predict", methods=["POST"])
@login_required
def predict_api():
    data = request.get_json()
    print("DATA RECEIVED:", data)

    success, output = predict_crowd(data)

    if not success:
        return jsonify({"error": output}), 200

    prediction = output["prediction"]
    confidence = output["confidence"]
    proba_dict = output["proba"]

    # Build a 7-hour forecast using the same inputs
    hour  = int(data.get("hour", 0))
    forecast = []
    for i in range(7):
        h = (hour + i) % 24
        payload = dict(data)
        payload["hour"] = h
        ok, res = predict_crowd(payload)
        if ok:
            forecast.append({
                "hour": h,
                "level": res["prediction"],
                "label": "Now" if i == 0 else f"+{i}h"
            })

    # Alternate stops (if High crowd, show other stops on the route)
    alternates = []
    if prediction == "High":
        from app.routes.prediction_routes import ROUTE_STOPS
        route = data.get("route")
        current_stop = data.get("stop")
        for alt_stop in ROUTE_STOPS.get(route, []):
            if alt_stop == current_stop:
                continue
            alt_payload = dict(data)
            alt_payload["stop"] = alt_stop
            ok, res = predict_crowd(alt_payload)
            if ok and res["prediction"] != "High":
                alternates.append({"stop": alt_stop, "level": res["prediction"], "dist": ""})

    return jsonify({
        "prediction": prediction,
        "confidence": confidence,
        "proba": proba_dict,
        "forecast": forecast,
        "alternates": alternates,
        "event_alert": None
    })


# Route → stops map (also used internally above)
ROUTE_STOPS = {
    "9X":   ["Miyapur", "Kukatpally", "SR Nagar", "Ameerpet", "Mehdipatnam"],
    "47L":  ["Secunderabad Station", "Ameerpet", "Jubilee Hills Checkpost", "Banjara Hills", "Gachibowli"],
    "127K": ["MGBS", "Koti", "Mehdipatnam", "SR Nagar", "Hitech City"],
    "218":  ["ECIL", "Uppal", "LB Nagar", "Dilsukhnagar", "MGBS"],
    "10H":  ["Secunderabad Station", "ECIL", "Uppal", "LB Nagar"],
    "5K":   ["Madhapur", "Hitech City", "Gachibowli", "Miyapur"],
    "216":  ["Charminar", "Koti", "MGBS", "Mehdipatnam", "Ameerpet"],
    "400":  ["Dilsukhnagar", "LB Nagar", "Uppal", "ECIL", "Secunderabad Station"],
}

@prediction_bp.route("/profile-data")
@login_required
def profile_data():
    from flask import jsonify
    return jsonify({
        "username": current_user.username,
        "email": current_user.email,
        "user_id": current_user.id
    })