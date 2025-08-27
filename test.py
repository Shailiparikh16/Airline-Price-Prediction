import pandas as pd
import yaml
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import os
import numpy as np

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

model = joblib.load("model.pkl")
X_test, y_test = joblib.load("test_data.pkl")

y_pred = model.predict(X_test)

if config["preprocessing"].get("log_target", False):
    y_test = np.expm1(y_test)  
    y_pred = np.expm1(y_pred)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test Results: MAE={mae:.2f}, MSE={mse:.2f}, R²={r2:.4f}")

os.makedirs("experiments", exist_ok=True)
log_file = "experiments/results.csv"

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
row = {
    "timestamp": timestamp,
    "model": config["model"]["type"],
    "params": config["model"]["params"],
    "log_target": config["preprocessing"].get("log_target", False),
    "MAE": mae,
    "MSE": mse,
    "R2": r2
}

df_row = pd.DataFrame([row])
if not os.path.exists(log_file):
    df_row.to_csv(log_file, index=False)
else:
    df_row.to_csv(log_file, mode="a", header=False, index=False)

print(f"Results logged in {log_file}")
