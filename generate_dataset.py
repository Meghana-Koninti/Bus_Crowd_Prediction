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
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

IT_STOPS     = ['Hitech City', 'Madhapur', 'Gachibowli', 'Kukatpally', 'SR Nagar']
MARKET_STOPS = ['Koti', 'Charminar', 'MGBS', 'Mehdipatnam']
HUB_STOPS    = ['Secunderabad Station', 'Ameerpet', 'LB Nagar', 'Dilsukhnagar', 'ECIL', 'Miyapur']

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

IPL_MONTHS = [4, 5]

ROUTE_STOP_MAP = {
    '9X':   ['Miyapur', 'Kukatpally', 'SR Nagar', 'Ameerpet', 'Mehdipatnam'],
    '47L':  ['Secunderabad Station', 'Ameerpet', 'Jubilee Hills Checkpost', 'Banjara Hills', 'Gachibowli'],
    '127K': ['MGBS', 'Koti', 'Mehdipatnam', 'SR Nagar', 'Hitech City'],
    '218':  ['ECIL', 'Uppal', 'LB Nagar', 'Dilsukhnagar', 'MGBS'],
    '10H':  ['Secunderabad Station', 'ECIL', 'Uppal', 'LB Nagar'],
    '5K':   ['Madhapur', 'Hitech City', 'Gachibowli', 'Miyapur'],
    '216':  ['Charminar', 'Koti', 'MGBS', 'Mehdipatnam', 'Ameerpet'],
    '400':  ['Dilsukhnagar', 'LB Nagar', 'Uppal', 'ECIL', 'Secunderabad Station'],
}


def get_crowd_probability(hour, day, stop, weather, is_festival, is_ipl_area, month):
    """
    KEY DESIGN PRINCIPLE:
    - Low    = clearly off-peak hours, no ambiguity
    - Medium = ONLY genuine transition zones: hour 6 (ramp-up), 20-21 (wind-down), weekends
    - High   = clearly peak hours 7-9 and 16-19
    Medium must NOT appear in off-peak weekday hours 10-15.
    That was the main cause of low accuracy in previous versions.
    """
    is_weekend = day in ['Saturday', 'Sunday']
    is_weekday = not is_weekend
    is_it_stop = stop in IT_STOPS
    is_market  = stop in MARKET_STOPS
    is_hub     = stop in HUB_STOPS

    # ── DEAD NIGHT: clearly Low ──
    if hour in [0, 1, 2, 3, 4]:
        base = [0.98, 0.02, 0.00]

    # ── EARLY MORNING 5AM: genuine ramp-up ──
    elif hour == 5:
        base = [0.75, 0.22, 0.03]

    # ── PRE-PEAK 6AM: Medium zone at busy stops, Low elsewhere ──
    elif hour == 6:
        if is_it_stop or is_hub:
            base = [0.20, 0.60, 0.20]
        else:
            base = [0.55, 0.38, 0.07]

    # ── MORNING PEAK 7-9: clearly High ──
    elif hour in [7, 8, 9]:
        if is_it_stop and is_weekday:
            base = [0.02, 0.08, 0.90]
        elif is_hub and is_weekday:
            base = [0.03, 0.12, 0.85]
        elif is_market and is_weekday:
            base = [0.10, 0.30, 0.60]
        elif is_weekend:
            base = [0.45, 0.40, 0.15]
        else:
            base = [0.08, 0.22, 0.70]

    # ── MID MORNING 10-11: strongly Low on weekdays ──
    elif hour in [10, 11]:
        if is_weekday:
            base = [0.90, 0.08, 0.02]
        elif is_market and is_weekend:
            base = [0.20, 0.55, 0.25]
        else:
            base = [0.82, 0.15, 0.03]

    # ── LUNCH 12: Low except markets ──
    elif hour == 12:
        if is_market:
            base = [0.08, 0.37, 0.55]
        elif is_it_stop and is_weekday:
            base = [0.60, 0.32, 0.08]
        else:
            base = [0.85, 0.12, 0.03]

    # ── AFTERNOON 13-15: strongly Low except weekend markets ──
    elif hour in [13, 14, 15]:
        if is_weekend and is_market:
            base = [0.08, 0.52, 0.40]
        elif is_weekend and is_hub:
            base = [0.30, 0.52, 0.18]
        else:
            base = [0.88, 0.10, 0.02]

    # ── EVENING PEAK 16-19: clearly High ──
    elif hour in [16, 17, 18, 19]:
        if is_it_stop and is_weekday:
            base = [0.01, 0.09, 0.90]
        elif is_hub and is_weekday:
            base = [0.02, 0.13, 0.85]
        elif is_market and is_weekday:
            base = [0.05, 0.28, 0.67]
        elif is_weekend:
            base = [0.18, 0.52, 0.30]
        else:
            base = [0.05, 0.25, 0.70]

    # ── POST-PEAK 20: genuine Medium wind-down zone ──
    elif hour == 20:
        if is_weekday:
            base = [0.25, 0.62, 0.13]
        else:
            base = [0.35, 0.52, 0.13]

    # ── LATE EVENING 21: transition to Low ──
    elif hour == 21:
        base = [0.58, 0.35, 0.07]

    # ── NIGHT 22-23: clearly Low ──
    elif hour == 22:
        base = [0.80, 0.18, 0.02]
    elif hour == 23:
        base = [0.92, 0.07, 0.01]
    else:
        base = [0.60, 0.30, 0.10]

    low, med, high = base

    # Weather modifiers
    if weather == 'Rainy':
        high += 0.18
        med  += 0.04
        low  -= 0.22
    elif weather == 'Overcast':
        high += 0.06
        low  -= 0.06
    elif weather == 'Humid':
        high += 0.09
        low  -= 0.09

    # Festival modifier
    if is_festival:
        high += 0.30
        med  += 0.05
        low  -= 0.35

    # IPL modifier
    if is_ipl_area and hour in [16, 17, 18, 19, 20]:
        high += 0.25
        low  -= 0.25

    low  = max(low,  0.0)
    med  = max(med,  0.0)
    high = max(high, 0.0)
    total = low + med + high
    return [low / total, med / total, high / total]


def get_passenger_count(crowd_level):
    if crowd_level == 'Low':
        return random.randint(3, 28)
    elif crowd_level == 'Medium':
        return random.randint(32, 62)
    else:
        return random.randint(65, 100)


rows = []
start_date = datetime(2023, 1, 1)

for i in range(35000):
    rand_days    = random.randint(0, 730)
    dt           = start_date + timedelta(days=rand_days)
    hour         = random.randint(0, 23)
    dt           = dt.replace(hour=hour, minute=random.randint(0, 59))

    day          = DAYS[dt.weekday()]
    is_weekend   = 1 if day in ['Saturday', 'Sunday'] else 0
    month        = dt.month
    day_of_month = dt.day

    route = random.choice(ROUTES)

    # 85% of rows: stop belongs to the route — makes Route_No a useful feature
    if random.random() < 0.85:
        stop = random.choice(ROUTE_STOP_MAP[route])
    else:
        stop = random.choice(STOPS)

    weather = np.random.choice(WEATHER_OPTIONS, p=[0.55, 0.15, 0.20, 0.10])

    is_festival = (month, day_of_month) in FESTIVAL_DAYS
    is_ipl_area = stop in ['Uppal', 'ECIL', 'Secunderabad Station'] and month in IPL_MONTHS

    probs = get_crowd_probability(hour, day, stop, weather, is_festival, is_ipl_area, month)
    probs = np.clip(probs, 0, 1)
    probs = probs / probs.sum()

    crowd           = np.random.choice(['Low', 'Medium', 'High'], p=probs)
    passenger_count = get_passenger_count(crowd)

    if hour in [7, 8, 9, 17, 18, 19]:
        buses = random.randint(3, 8)
    elif hour in [0, 1, 2, 3, 4]:
        buses = random.randint(1, 3)
    else:
        buses = random.randint(4, 12)

    if crowd == 'High':
        dwell = random.randint(15, 65)
    elif crowd == 'Medium':
        dwell = random.randint(10, 40)
    else:
        dwell = random.randint(5, 25)

    rows.append({
        'Timestamp':          dt.strftime('%Y-%m-%d %H:%M:%S'),
        'Hour':               hour,
        'Day_of_Week':        day,
        'Is_Weekend':         is_weekend,
        'Route_No':           route,
        'Current_Stop':       stop,
        'Weather':            weather,
        'Buses_Available':    buses,
        'Dwell_Time_Seconds': dwell,
        'Passenger_Count':    passenger_count,
        'Crowding_Level':     crowd,
        'Is_Festival':        int(is_festival),
        'Month':              month
    })

df = pd.DataFrame(rows)

print("Shape:", df.shape)
print("\nCrowd distribution:")
print(df['Crowding_Level'].value_counts())
print("\nHour vs crowd (peak hours 8, 9, 17, 18):")
peak = df[df['Hour'].isin([8, 9, 17, 18])]
print(peak.groupby('Hour')['Crowding_Level'].value_counts().unstack().fillna(0).astype(int))
print("\nHour vs crowd (off-peak hours 10, 11, 13, 14):")
off = df[df['Hour'].isin([10, 11, 13, 14])]
print(off.groupby('Hour')['Crowding_Level'].value_counts().unstack().fillna(0).astype(int))

df.to_csv('data/DataSet.csv', index=False)
print("\nSaved to data/DataSet.csv")