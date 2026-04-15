# cropprediction.py

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# Load dataset
df = pd.read_csv("data/produce.csv")

df.columns = df.columns.str.strip()

# Remove text columns
df = df.drop(columns=[
    'Particulars',
    'Frequency',
    'Unit'
])

# Convert to numeric
df = df.apply(pd.to_numeric, errors='coerce')

df = df.fillna(df.mean())


# -----------------------------
# USE ONLY LAST 6 YEARS
# (5 inputs → predict 6th year)
# -----------------------------

df = df.iloc[:, -6:]


# Features (last 5 years)

X = df.iloc[:, :-1]

# Target (next year)

y = df.iloc[:, -1]


# Train/test split

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)


# Train model

model = RandomForestRegressor(

    n_estimators=200,
    random_state=42

)

model.fit(X_train, y_train)


# Evaluate

y_pred = model.predict(X_test)

score = r2_score(y_test, y_pred)

print(f"✅ Model Accuracy (R2 Score): {score:.2f}")


# Save model

joblib.dump(model, "model/crop_model.pkl")

print("✅ Model saved successfully!")