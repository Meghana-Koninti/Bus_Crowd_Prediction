import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# 1. LOAD
df = pd.read_csv('data/DataSet.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nCrowding distribution:\n{df['Crowding_Level'].value_counts()}\n")

# 2. ENCODE
categorical_cols = ['Route_No', 'Current_Stop', 'Weather', 'Day_of_Week', 'Crowding_Level']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# 3. FEATURES
feature_cols = ['Hour', 'Day_of_Week', 'Is_Weekend', 'Route_No', 'Current_Stop',
                'Weather', 'Buses_Available', 'Month', 'Is_Festival']
X = df[feature_cols]
y = df['Crowding_Level']

# 4. SPLIT — stratify keeps class ratios identical in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. SAMPLE WEIGHTS — Medium boosted hard to fix its low recall
le_crowd = encoders['Crowding_Level']
weight_map = {
    le_crowd.transform(['High'])[0]:   5,
    le_crowd.transform(['Medium'])[0]: 8,   # boosted — Medium was f1=0.26
    le_crowd.transform(['Low'])[0]:    1,
}
sample_weights = np.array([weight_map[label] for label in y_train])

# 6. MODEL
model = XGBClassifier(
    n_estimators=800,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.2,
    reg_alpha=0.1,
    reg_lambda=1.5,
    objective='multi:softmax',
    num_class=3,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=30,
    verbose=100)

# 7. EVALUATE
y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.3f}")
print(classification_report(y_test, y_pred, target_names=le_crowd.classes_))

# Cross-validation
cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"5-Fold CV Accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

# Feature importance
print("\nFeature Importance:")
imp_df = pd.DataFrame({
    'feature':    feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(imp_df.to_string(index=False))

# 8. SAVE
with open('models/hyderabad_bus_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/hyderabad_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("\nModel saved successfully.")
print("Feature columns:", feature_cols)