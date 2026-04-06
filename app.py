from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import pandas as pd
import pickle
import numpy as np
from datetime import datetime

app = Flask(__name__)
#advitha-added this for login. without this session it wont work properly
app.secret_key = "advitha_secret_key"

with open('models/hyderabad_bus_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/hyderabad_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)
#advitha-added this for backend database making part. (this part will create tables and db)
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

STOP_COORDS = {
    'Hitech City': [17.4483, 78.3915],
    'Madhapur': [17.4411, 78.3911],
    'Ameerpet': [17.4375, 78.4482],
    'Jubilee Hills Checkpost': [17.4251, 78.4140],
    'Secunderabad Station': [17.4399, 78.4983],
    'Mehdipatnam': [17.3958, 78.4312],
    'Kukatpally': [17.4875, 78.3953],
    'Koti': [17.3850, 78.4867],
    'Charminar': [17.3616, 78.4747],
    'LB Nagar': [17.3469, 78.5469],
    'Uppal': [17.4016, 78.5597],
    'Gachibowli': [17.4400, 78.3489],
    'SR Nagar': [17.4509, 78.4413],
    'Banjara Hills': [17.4156, 78.4347],
    'MGBS': [17.3784, 78.4822],
    'Dilsukhnagar': [17.3688, 78.5268],
    'Miyapur': [17.4957, 78.3544],
    'ECIL': [17.4650, 78.5528]
}

# Route → stops mapping for alternate stop suggestions
ROUTE_STOPS = {
    '9X':  ['Miyapur', 'Kukatpally', 'SR Nagar', 'Ameerpet', 'Mehdipatnam'],
    '47L': ['Secunderabad Station', 'Ameerpet', 'Jubilee Hills Checkpost', 'Banjara Hills', 'Gachibowli'],
    '127K':['MGBS', 'Koti', 'Mehdipatnam', 'SR Nagar', 'Hitech City'],
    '218': ['ECIL', 'Uppal', 'LB Nagar', 'Dilsukhnagar', 'MGBS'],
    '10H': ['Secunderabad Station', 'ECIL', 'Uppal', 'LB Nagar'],
    '5K':  ['Madhapur', 'Hitech City', 'Gachibowli', 'Miyapur'],
    '216': ['Charminar', 'Koti', 'MGBS', 'Mehdipatnam', 'Ameerpet'],
    '400': ['Dilsukhnagar', 'LB Nagar', 'Uppal', 'ECIL', 'Secunderabad Station'],
}

# Hyderabad event calendar: (month, day) -> event info
HYDERABAD_EVENTS = {
    (1,14): {'name':'Sankranti', 'affected_stops':['Charminar','Koti','MGBS','Mehdipatnam'], 'boost':'High'},
    (1,15): {'name':'Sankranti', 'affected_stops':['Charminar','Koti','MGBS','Mehdipatnam'], 'boost':'High'},
    (1,16): {'name':'Sankranti', 'affected_stops':['Charminar','Koti','MGBS'], 'boost':'Medium'},
    (3,17): {'name':'Holi',      'affected_stops':['Charminar','Koti','MGBS','Ameerpet'], 'boost':'High'},
    (3,18): {'name':'Holi',      'affected_stops':['Charminar','Koti','MGBS'], 'boost':'Medium'},
    (7,15): {'name':'Bonalu',    'affected_stops':['Charminar','Koti','MGBS','Mehdipatnam','LB Nagar'], 'boost':'High'},
    (7,16): {'name':'Bonalu',    'affected_stops':['Charminar','Koti','MGBS','Mehdipatnam'], 'boost':'High'},
    (7,22): {'name':'Bonalu',    'affected_stops':['Secunderabad Station','ECIL','Uppal'], 'boost':'High'},
    (7,23): {'name':'Bonalu',    'affected_stops':['Secunderabad Station','ECIL'], 'boost':'High'},
    (7,29): {'name':'Bonalu',    'affected_stops':['Charminar','Koti','MGBS'], 'boost':'High'},
    (7,30): {'name':'Bonalu',    'affected_stops':['Charminar','Koti','MGBS'], 'boost':'High'},
    (8,5):  {'name':'Bonalu',    'affected_stops':['Mehdipatnam','SR Nagar','Ameerpet'], 'boost':'High'},
    (8,15): {'name':'Independence Day', 'affected_stops':['Secunderabad Station','MGBS','Charminar'], 'boost':'High'},
    (9,7):  {'name':'Ganesh Chaturthi','affected_stops':['Charminar','Koti','MGBS','Ameerpet','Banjara Hills'], 'boost':'High'},
    (9,17): {'name':'Ganesh Immersion','affected_stops':['Charminar','MGBS','Koti','Mehdipatnam','LB Nagar'], 'boost':'High'},
    (10,2): {'name':'Gandhi Jayanti',  'affected_stops':['Secunderabad Station','MGBS'], 'boost':'Medium'},
    (10,12):{'name':'Dussehra',  'affected_stops':['Charminar','Koti','MGBS','Ameerpet','Secunderabad Station'], 'boost':'High'},
    (10,13):{'name':'Dussehra', 'affected_stops':['Charminar','Koti','MGBS'], 'boost':'Medium'},
    (11,1): {'name':'Diwali',    'affected_stops':['Charminar','Koti','MGBS','Ameerpet','Jubilee Hills Checkpost'], 'boost':'High'},
    (11,2): {'name':'Diwali',    'affected_stops':['Charminar','MGBS','Koti'], 'boost':'Medium'},
    (12,25):{'name':'Christmas', 'affected_stops':['Banjara Hills','Jubilee Hills Checkpost','Hitech City','Madhapur'], 'boost':'Medium'},
}

# IPL: April-May, Uppal area
IPL_STOPS = ['Uppal', 'ECIL', 'Secunderabad Station']
IPL_PEAK_HOURS = list(range(16, 23))

def get_event_alert(month, day, stop, hour):
    key = (month, day)
    if key in HYDERABAD_EVENTS:
        ev = HYDERABAD_EVENTS[key]
        if stop in ev['affected_stops']:
            return ev['name'], ev['boost']
    if month in [4, 5] and stop in IPL_STOPS and hour in IPL_PEAK_HOURS:
        return 'IPL Match', 'High'
    return None, None

def encode_and_predict(route, stop, hour, day, weather, buses, month=None, is_festival=0):
    if month is None:
        month = datetime.now().month
    route_enc = encoders['Route_No'].transform([route])[0]
    stop_enc  = encoders['Current_Stop'].transform([stop])[0]
    weather_enc = encoders['Weather'].transform([weather])[0]
    day_enc   = encoders['Day_of_Week'].transform([day])[0]
    is_weekend = 1 if day in ['Saturday', 'Sunday'] else 0
    features = np.array([[hour, day_enc, is_weekend, route_enc, stop_enc, weather_enc, buses, month, is_festival]])
    proba = model.predict_proba(features)[0]
    pred_enc = model.predict(features)[0]
    pred_label = encoders['Crowding_Level'].inverse_transform([pred_enc])[0]
    classes = encoders['Crowding_Level'].classes_
    proba_dict = {c: float(round(p*100)) for c, p in zip(classes, proba)}
    confidence = float(round(max(proba)*100))
    return pred_label, confidence, proba_dict

routes = encoders['Route_No'].classes_
stops  = encoders['Current_Stop'].classes_
weathers = encoders['Weather'].classes_
days   = encoders['Day_of_Week'].classes_

@app.route('/')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))

    now = datetime.now()
    return render_template('index.html',
        routes=list(routes), stops=list(stops),
        weathers=list(weathers), days=list(days),
        stop_coords=STOP_COORDS,
        current_hour=now.hour,
        current_day=now.strftime('%A'),
        current_month=now.month
    )
#advitha- added this. for login and signup. 
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])

        conn = sqlite3.connect('users.db')
        c = conn.cursor()

        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except:
            return "Username already exists ❌"

    return render_template('signup.html')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session['user'] = username
            return redirect(url_for('home'))
        else:
            return "Invalid credentials ❌"

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    route   = data['route']
    stop    = data['stop']
    hour    = int(data['hour'])
    day     = data['day']
    weather = data['weather']
    buses   = int(data['buses'])
    month   = int(data.get('month', datetime.now().month))

    # Check for event override
    event_name, event_boost = get_event_alert(month, datetime.now().day, stop, hour)
    is_festival = 1 if event_name else 0

    pred, confidence, proba = encode_and_predict(route, stop, hour, day, weather, buses, month, is_festival)

    # Event override: boost to High if event says so
    event_alert = None
    if event_boost == 'High' and pred != 'High':
        pred = 'High'
        confidence = max(confidence, 75)
        event_alert = f"⚠️ {event_name} detected near this stop — crowd boosted to High"
    elif event_boost == 'Medium' and pred == 'Low':
        pred = 'Medium'
        confidence = max(confidence, 70)
        event_alert = f"🎉 {event_name} nearby — expect moderate crowd increase"

    # 6-hour forecast
    forecast = []
    for h_offset in range(0, 7):
        fh = (hour + h_offset) % 24
        fp, fc, _ = encode_and_predict(route, stop, fh, day, weather, buses, month, is_festival)
        # Check event for each future hour
        fe_name, fe_boost = get_event_alert(month, datetime.now().day, stop, fh)
        if fe_boost == 'High' and fp != 'High':
            fp = 'High'
        elif fe_boost == 'Medium' and fp == 'Low':
            fp = 'Medium'
        label = 'Now' if h_offset == 0 else f'+{h_offset}h'
        forecast.append({'label': label, 'hour': fh, 'level': fp, 'confidence': fc})

    # Alternate stops (only if High)
    alternates = []
    if pred == 'High' and route in ROUTE_STOPS:
        route_stop_list = ROUTE_STOPS[route]
        for alt_stop in route_stop_list:
            if alt_stop != stop and alt_stop in list(stops):
                try:
                    ap, ac, _ = encode_and_predict(route, alt_stop, hour, day, weather, buses, month, is_festival)
                    if ap in ['Low', 'Medium']:
                        dist_info = ''
                        if alt_stop in STOP_COORDS and stop in STOP_COORDS:
                            s1 = STOP_COORDS[stop]
                            s2 = STOP_COORDS[alt_stop]
                            dist_km = round(((s1[0]-s2[0])**2 + (s1[1]-s2[1])**2)**0.5 * 111, 1)
                            dist_info = f'{dist_km} km away'
                        alternates.append({'stop': alt_stop, 'level': ap, 'dist': dist_info})
                    if len(alternates) >= 2:
                        break
                except:
                    pass

    return jsonify({
        'prediction': pred,
        'confidence': confidence,
        'proba': proba,
        'forecast': forecast,
        'alternates': alternates,
        'event_alert': event_alert,
        'stop_coords': STOP_COORDS.get(stop, [17.385, 78.4867])
    })

if __name__ == '__main__':
    app.run(debug=True)