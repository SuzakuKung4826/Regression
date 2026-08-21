import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42),
}

results = []
os.makedirs('/tmp/work/Regression/model_files', exist_ok=True)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    results.append({"model": name, "r2": round(float(r2), 4), "mae": round(float(mae), 1), "rmse": round(float(rmse), 1)})
    print(f"{name:20s} R2={r2:.4f}  MAE={mae:,.1f}  RMSE={rmse:,.1f}")
    # save the random forest as the deployed model (keep same filenames as before)
    if name == "Random Forest":
        joblib.dump(model, '/tmp/work/Regression/model_files/rf_model.pkl')

joblib.dump(scaler, '/tmp/work/Regression/model_files/scaler.pkl')
joblib.dump(feature_names, '/tmp/work/Regression/model_files/feature_names.pkl')

with open('/tmp/work/Regression/model_files/comparison_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(json.dumps(results, indent=2))
