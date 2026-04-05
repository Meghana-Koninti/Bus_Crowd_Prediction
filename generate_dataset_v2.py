import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# reproducibility
random.seed(42)
np.random.seed(42)

# =========================
# LOAD ROUTE MASTER
# =========================
df = pd.read_csv("data/bus_dataset.csv")

# group by route
routes = df.groupby("route_number")

# =========================
# CONFIG
# =========================
WEATHER_OPTIONS = ['Clear', 'Rainy', 'Overcast', 'Humid']
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

PEAK_HOURS = [7, 8, 9, 17, 18, 19]
HOURS = [6, 7, 8, 9, 12, 14, 16, 17, 18, 19]

START_DATE = datetime(2023, 1, 1)
NUM_DAYS = 30

# =========================
# HELPER FUNCTIONS
# =========================

def get_weather():
    return np.random.choice(WEATHER_OPTIONS, p=[0.55, 0.15, 0.20, 0.10])

def get_is_festival(month, day):
    # simple random festivals (you can refine later)
    return 1 if random.random() < 0.05 else 0

def get_boarding(stop_order, total_stops, is_peak):
    if stop_order <= 3:
        return random.randint(10, 30) if is_peak else random.randint(5, 15)
    elif stop_order >= total_stops - 2:
        return random.randint(2, 10)
    else:
        return random.randint(5, 20)

def get_alighting(stop_order, total_stops):
    if stop_order >= total_stops - 2:
        return random.randint(10, 30)
    elif stop_order <= 3:
        return random.randint(0, 5)
    else:
        return random.randint(3, 15)

def get_crowd_level(load, capacity):
    ratio = load / capacity
    if ratio < 0.4:
        return "Low"
    elif ratio < 0.75:
        return "Medium"
    else:
        return "High"

# =========================
# DATA GENERATION
# =========================

rows = []

for route, group in routes:

    group = group.sort_values("stop_order")
    total_stops = len(group)

    for day_offset in range(NUM_DAYS):

        current_date = START_DATE + timedelta(days=day_offset)
        day_name = DAYS[current_date.weekday()]
        is_weekend = 1 if day_name in ['Saturday', 'Sunday'] else 0

        for hour in HOURS:

            is_peak = 1 if hour in PEAK_HOURS else 0
            weather = get_weather()
            is_festival = get_is_festival(current_date.month, current_date.day)

            capacity = random.choice([50, 60, 70])
            buses = random.randint(3, 8) if is_peak else random.randint(1, 5)

            current_load = 0  # reset for each trip

            for _, row in group.iterrows():

                stop_order = row["stop_order"]
                stop = row["stop_name"]

                if stop_order <= total_stops // 3:
                    boarding = random.randint(15, 30) if is_peak else random.randint(8, 20)
                    alighting = random.randint(0, 5)
                elif stop_order >= (2 * total_stops) // 3:
                    boarding = random.randint(2, 8)
                    alighting = random.randint(10, 25)
                else:
                    boarding = random.randint(5, 15)
                    alighting = random.randint(5, 15)       

                # update load
                current_load = current_load + boarding - alighting
                current_load -= random.randint(0, 5)
                current_load = max(0, min(current_load, capacity))

                crowd = get_crowd_level(current_load, capacity)

                dwell = random.randint(15, 60) if crowd == "High" else \
                        random.randint(10, 40) if crowd == "Medium" else \
                        random.randint(5, 25)

                rows.append({
                    "Date": current_date.strftime("%Y-%m-%d"),
                    "Hour": hour,
                    "Day_of_Week": day_name,
                    "Is_Weekend": is_weekend,

                    "Route_No": route,
                    "Stop_Name": stop,
                    "Stop_Order": stop_order,

                    "Weather": weather,
                    "Is_Festival": is_festival,

                    "Bus_Capacity": capacity,
                    "Buses_Available": buses,

                    "Boarding": boarding,
                    "Alighting": alighting,
                    "Current_Load": current_load,
                    "Dwell_Time_Seconds": dwell,

                    "Crowding_Level": crowd
                })

# =========================
# SAVE DATASET
# =========================

final_df = pd.DataFrame(rows)

print("Dataset shape:", final_df.shape)
print("\nCrowd distribution:")
print(final_df["Crowding_Level"].value_counts())

final_df.to_csv("data/ml_dataset.csv", index=False)

print("\nSaved to data/ml_dataset.csv")