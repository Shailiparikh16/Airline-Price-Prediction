import pandas as pd
import yaml
import numpy as np
import os

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

df = pd.read_csv("Data/raw.csv")

if "index" in df.columns:
    df = df.drop(columns=["index"])

df = pd.get_dummies(df, drop_first=True)

if config["preprocessing"].get("log_target", False):
    df["price"] = np.log1p(df["price"]) 

os.makedirs("Data", exist_ok=True)
df.to_csv("Data/clean.csv", index=False)
print("Preprocessing done. Clean data saved to Data/clean.csv")
