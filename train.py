import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv('/tmp/data/automobile.csv')

feature_names = ['engine-size', 'horsepower', 'curb-weight', 'city-mpg',
                  'highway-mpg', 'wheel-base', 'length', 'width']

df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())

X = df[feature_names]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
model.fit(X_train_scaled, y_train)

pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
print(f"R2: {r2:.4f}")
print(f"MAE: {mae:.2f}")

os.makedirs('/tmp/work/Regression/model_files', exist_ok=True)
joblib.dump(model, '/tmp/work/Regression/model_files/rf_model.pkl')
joblib.dump(scaler, '/tmp/work/Regression/model_files/scaler.pkl')
joblib.dump(feature_names, '/tmp/work/Regression/model_files/feature_names.pkl')

ranges = {c: (float(X[c].min()), float(X[c].max()), float(X[c].mean())) for c in feature_names}
for k, v in ranges.items():
    print(k, v)
