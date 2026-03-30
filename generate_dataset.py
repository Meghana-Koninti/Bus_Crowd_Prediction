import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

ROUTES = ['9X', '47L', '127K', '218', '10H', '5K', '216', '400']
STOPS = [
    'Hitech City', 'Madhapur', 'Ameerpet', 'Jubilee Hills Checkpost',
    'Secunderabad Station', 'Mehdipatnam', 'Kukatpally', 'Koti',
    'Charminar', 'LB Nagar', 'Uppal', 'Gachibowli', 'SR Nagar',
    'Banjara Hills', 'MGBS', 'Dilsukhnagar', 'Miyapur', 'ECIL'
]
WEATHER_OPTIONS = ['Clear', 'Rainy', 'Overcast', 'Humid']
DAYS = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

# Stops with high IT/office crowd
IT_STOPS = ['Hitech City', 'Madhapur', 'Gachibowli', 'Kukatpally', 'SR Nagar']
# Stops near old city / markets
MARKET_STOPS = ['Koti', 'Charminar', 'MGBS', 'Mehdipatnam']
# Terminal/transit hubs
HUB_STOPS = ['Secunderabad Station', 'Ameerpet', 'LB Nagar', 'Dilsukhnagar', 'ECIL', 'Miyapur']

# Hyderabad festival/event days (month-day format)
FESTIVAL_DAYS = {
    (1, 14): 'Sankranti',   (1, 15): 'Sankranti',   (1, 16): 'Sankranti',
    (3, 17): 'Holi',        (3, 18): 'Holi',
    (4, 14): 'Ambedkar Jayanti',
    (7, 15): 'Bonalu',      (7, 16): 'Bonalu',       (7, 22): 'Bonalu',
    (7, 23): 'Bonalu',      (7, 29): 'Bonalu',       (7, 30): 'Bonalu',
    (8, 5):  'Bonalu',      (8, 6):  'Bonalu',
    (8, 15): 'Independence Day',
    (9, 7):  'Ganesh Chaturthi', (9, 8): 'Ganesh Chaturthi',
    (9, 17): 'Ganesh Immersion',
    (10, 2): 'Gandhi Jayanti',
    (10, 12): 'Dussehra',   (10, 13): 'Dussehra',
    (11, 1):  'Diwali',     (11, 2):  'Diwali',
    (12, 25): 'Christmas',
}

# IPL months (April-May) near Uppal
IPL_MONTHS = [4, 5]

def get_crowd_probability(hour, day, stop, weather, is_festival, is_ipl_area, month):
    """Return (low_prob, medium_prob, high_prob) for realistic Hyderabad patterns"""
    is_weekend = day in ['Saturday', 'Sunday']
    is_weekday = not is_weekend
    is_it_stop = stop in IT_STOPS
    is_market = stop in MARKET_STOPS
    is_hub = stop in HUB_STOPS

    # Base crowd by hour
    if hour in [0,1,2,3,4]:
        base = [0.98, 0.02, 0.00]  # Dead night
    elif hour == 5:
        base = [0.85, 0.13, 0.02]  # Early morning
    elif hour == 6:
        base = [0.60, 0.32, 0.08]  # Starts picking up
    elif hour in [7, 8, 9]:        # Morning peak
        if is_it_stop and is_weekday:
            base = [0.05, 0.30, 0.65]
        elif is_hub:
            base = [0.08, 0.35, 0.57]
        elif is_market and is_weekday:
            base = [0.20, 0.50, 0.30]
        else:
            base = [0.15, 0.45, 0.40]
    elif hour in [10, 11]:         # Mid morning
        if is_weekday:
            base = [0.25, 0.55, 0.20]
        else:
            base = [0.30, 0.50, 0.20]
    elif hour == 12:               # Lunch
        if is_market:
            base = [0.15, 0.45, 0.40]
        else:
            base = [0.35, 0.50, 0.15]
    elif hour in [13, 14, 15]:     # Afternoon
        if is_weekend and is_market:
            base = [0.15, 0.45, 0.40]
        else:
            base = [0.45, 0.42, 0.13]
    elif hour in [16, 17, 18, 19]: # Evening peak
        if is_it_stop and is_weekday:
            base = [0.04, 0.25, 0.71]
        elif is_hub and is_weekday:
            base = [0.06, 0.30, 0.64]
        elif is_market:
            base = [0.10, 0.40, 0.50]
        elif is_weekend:
            base = [0.20, 0.45, 0.35]
        else:
            base = [0.10, 0.38, 0.52]
    elif hour == 20:
        base = [0.30, 0.48, 0.22]
    elif hour == 21:
        base = [0.55, 0.35, 0.10]
    elif hour == 22:
        base = [0.75, 0.22, 0.03]
    elif hour == 23:
        base = [0.90, 0.09, 0.01]
    else:
        base = [0.50, 0.35, 0.15]

    low, med, high = base

    # Weather boost (rain = more buses needed, more crowd per bus)
    if weather == 'Rainy':
        high += 0.15
        med += 0.05
        low -= 0.20
    elif weather == 'Overcast':
        high += 0.05
        low -= 0.05
    elif weather == 'Humid':
        high += 0.08
        low -= 0.08

    # Festival boost
    if is_festival:
        high += 0.25
        med += 0.05
        low -= 0.30

    # IPL boost near Uppal/ECIL during evening
    if is_ipl_area and hour in [16,17,18,19,20]:
        high += 0.20
        low -= 0.20

    # Normalize
    total = low + med + high
    return [low/total, med/total, high/total]

def get_passenger_count(crowd_level, hour):
    if crowd_level == 'Low':
        return random.randint(5, 30)
    elif crowd_level == 'Medium':
        return random.randint(31, 65)
    else:
        return random.randint(66, 100)

rows = []
start_date = datetime(2023, 1, 1)

for i in range(10000):
    # Pick a random date in 2023-2024
    rand_days = random.randint(0, 730)
    dt = start_date + timedelta(days=rand_days)
    hour = random.randint(0, 23)
    dt = dt.replace(hour=hour, minute=random.randint(0, 59))

    day = DAYS[dt.weekday()]
    is_weekend = 1 if day in ['Saturday', 'Sunday'] else 0
    month = dt.month
    day_of_month = dt.day

    route = random.choice(ROUTES)
    stop = random.choice(STOPS)
    weather = np.random.choice(WEATHER_OPTIONS, p=[0.55, 0.15, 0.20, 0.10])

    is_festival = (month, day_of_month) in FESTIVAL_DAYS
    is_ipl_area = stop in ['Uppal', 'ECIL', 'Secunderabad Station'] and month in IPL_MONTHS

    probs = get_crowd_probability(hour, day, stop, weather, is_festival, is_ipl_area, month)
    probs = np.clip(probs, 0, 1)
    probs = probs / probs.sum()

    crowd = np.random.choice(['Low', 'Medium', 'High'], p=probs)
    passenger_count = get_passenger_count(crowd, hour)

    # Buses available: fewer during peak, more off-peak
    if hour in [7,8,9,17,18,19]:
        buses = random.randint(3, 8)
    elif hour in [0,1,2,3,4]:
        buses = random.randint(1, 3)
    else:
        buses = random.randint(4, 12)

    dwell = random.randint(10, 60) if crowd == 'High' else random.randint(8, 35)

    rows.append({
        'Timestamp': dt.strftime('%Y-%m-%d %H:%M:%S'),
        'Hour': hour,
        'Day_of_Week': day,
        'Is_Weekend': is_weekend,
        'Route_No': route,
        'Current_Stop': stop,
        'Weather': weather,
        'Buses_Available': buses,
        'Dwell_Time_Seconds': dwell,
        'Passenger_Count': passenger_count,
        'Crowding_Level': crowd,
        'Is_Festival': int(is_festival),
        'Month': month
    })

df = pd.DataFrame(rows)
print("Shape:", df.shape)
print("\nCrowd distribution:")
print(df['Crowding_Level'].value_counts())
print("\nHour vs crowd (peak hours):")
peak = df[df['Hour'].isin([8,9,17,18,19])]
print(peak.groupby('Hour')['Crowding_Level'].value_counts().unstack().fillna(0).astype(int))

df.to_csv('/tmp/Mini_Project/data/Hyderabad_Bus_Crowding_Realistic.csv', index=False)
print("\nSaved!")
