import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/ml_dataset.csv")

print("Original shape:", df.shape)

# =========================
# DROP UNUSED COLUMNS
# =========================
df = df.drop(columns=["Date"])  # remove raw date
df = df.drop(columns=["Route_No", "Stop_Name"])

# =========================
# ENCODE CATEGORICAL FEATURES
# =========================
le_weather = LabelEncoder()
df["Weather"] = le_weather.fit_transform(df["Weather"])

le_day = LabelEncoder()
df["Day_of_Week"] = le_day.fit_transform(df["Day_of_Week"])

le_crowd = LabelEncoder()
df["Crowding_Level"] = le_crowd.fit_transform(df["Crowding_Level"])

# =========================
# FEATURES & TARGET
# =========================
X = df.drop(columns=["Crowding_Level"])
y = df["Crowding_Level"]

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# =========================
# SAVE PROCESSED DATA
# =========================
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("Preprocessing complete ✅")