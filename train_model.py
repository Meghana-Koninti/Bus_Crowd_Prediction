import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load Data
data_path = 'data/Hyderabad_Bus_Crowding_Data_v2.csv'
df = pd.read_csv(data_path)

# Feature Engineering
# Features: Hour, Day_of_Week, Is_Weekend, Route_No, Current_Stop, Weather, Buses_Available, Dwell_Time_Seconds
# Target: Crowding_Level

categorical_cols = ['Route_No', 'Current_Stop', 'Weather', 'Day_of_Week', 'Crowding_Level']
encoders = {}

# Apply Label Encoding and save to dict
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"Encoded {col}: {le.classes_}")

# Define Features (X) and Target (y)
# Exclude Dwell_Time_Seconds and Passenger_Count as per refined requirements
feature_cols = ['Hour', 'Day_of_Week', 'Is_Weekend', 'Route_No', 'Current_Stop', 'Weather', 'Buses_Available']
X = df[feature_cols]
y = df['Crowding_Level']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
# n_estimators=150, max_depth=None
# Manual class weights to favor High crowding detection in critical areas
class_weights = {'High': 3, 'Medium': 2, 'Low': 1}
# Note: LabelEncoder might have encoded classes. We need to map these strings to the encoded integers if passing a dict with ints, 
# or pass 'balanced' or 'balanced_subsample'.
# However, sklearn handles string keys if y is string? No, y is encoded integers (0, 1, 2).
# We need to map 'High', 'Medium', 'Low' to their encoded integers.

le_crowd = encoders['Crowding_Level']
# Start with default weights
weights_formatted = {}
for label, weight in class_weights.items():
    try:
        encoded_label = le_crowd.transform([label])[0]
        weights_formatted[encoded_label] = weight
    except:
        pass

print("Class Weights:", weights_formatted)

rf = RandomForestClassifier(n_estimators=150, max_depth=None, random_state=42, class_weight=weights_formatted)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Validation Check (Specific Scenario)
# Scenario: Route 9X, Stop Hitech City, Hour 18, Day Thursday, Weather Clear.
# Expected: High.

# Helper to encode single value
def encode_val(col, val):
    return encoders[col].transform([val])[0]

try:
    val_route = encode_val('Route_No', '9X')
    val_stop = encode_val('Current_Stop', 'Hitech City')
    val_day = encode_val('Day_of_Week', 'Thursday')
    val_weather = encode_val('Weather', 'Clear')
    val_hour = 18
    val_weekend = 0 # Thursday is not weekend
    
    # Buses_Available default low for peak?
    val_buses = 5 # Low availability
    
    # Features: Hour, Day_of_Week, Is_Weekend, Route_No, Current_Stop, Weather, Buses_Available
    val_features = np.array([[val_hour, val_day, val_weekend, val_route, val_stop, val_weather, val_buses]])
    
    pred_encoded = rf.predict(val_features)[0]
    pred_label = encoders['Crowding_Level'].inverse_transform([pred_encoded])[0]
    
    print(f"\n--- Validation Check ---")
    print(f"Scenario: 9X, Hitech City, 18:00, Thu, Clear, Buses: {val_buses}")
    print(f"Prediction: {pred_label}")
    
    if pred_label == 'High':
        print("Validation PASSED: Prediction is High.")
    else:
        print("Validation WARNING: Prediction is not High. Consider adjusting weights/data.")

except Exception as e:
    print(f"Validation Error: {e}")

# Save Model and Encoders
model_path = 'models/hyderabad_bus_model.pkl'
encoders_path = 'models/hyderabad_encoders.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(rf, f)

with open(encoders_path, 'wb') as f:
    pickle.dump(encoders, f)

print(f"\nModel saved to {model_path}")
print(f"Encoders dict saved to {encoders_path}")
