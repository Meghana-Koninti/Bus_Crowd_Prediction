import pandas as pd
import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('data/DataSet.csv')

categorical_cols = ['Route_No', 'Current_Stop', 'Weather', 'Day_of_Week', 'Crowding_Level']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

feature_cols = ['Hour', 'Day_of_Week', 'Is_Weekend', 'Route_No', 'Current_Stop',
                'Weather', 'Buses_Available', 'Month', 'Is_Festival']
X = df[feature_cols]
y = df['Crowding_Level']

le_crowd = encoders['Crowding_Level']
class_weights_dict = {'High': 4, 'Medium': 2, 'Low': 1}
weights_formatted = {le_crowd.transform([k])[0]: v for k, v in class_weights_dict.items()}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=18,
    class_weight=weights_formatted,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred, target_names=le_crowd.classes_))

with open('models/hyderabad_bus_model.pkl', 'wb') as f:
    pickle.dump(rf, f)
with open('models/hyderabad_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("Model saved. Feature columns:", feature_cols)
