import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/ml_dataset.csv")

print("Dataset shape:", df.shape)
print("\nCrowd distribution:\n", df["Crowding_Level"].value_counts())

# =========================
# ENCODE
# =========================
le_weather = LabelEncoder()
df["Weather"] = le_weather.fit_transform(df["Weather"])

le_day = LabelEncoder()
df["Day_of_Week"] = le_day.fit_transform(df["Day_of_Week"])

le_crowd = LabelEncoder()
df["Crowding_Level"] = le_crowd.fit_transform(df["Crowding_Level"])

# =========================
# DROP NON-USEFUL COLUMNS
# =========================
df = df.drop(columns=[
    "Date",
    "Route_No",
    "Stop_Name",
    "Current_Load",
    "Boarding",
    "Alighting"
])
# =========================
# FEATURES
# =========================
X = df.drop(columns=["Crowding_Level"])
y = df["Crowding_Level"]

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# MODEL
# =========================
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================
# EVALUATE
# =========================
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=le_crowd.classes_))

# =========================
# SAVE MODEL
# =========================
with open("models/model_v2.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved successfully.")