import pandas as pd
import yaml
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import GradientBoostingRegressor

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

df = pd.read_csv("Data/clean.csv")

X = df.drop(columns=["price"])
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config["training"]["test_size"],
    random_state=config["training"]["random_state"]
)

params = config["model"]["params"]

if config["model"]["type"] == "LinearRegression":
    model = LinearRegression(**params)
elif config["model"]["type"] == "RandomForestRegressor":
    model = RandomForestRegressor(**params)
elif config["model"]["type"] == "GradientBoostingRegressor":
    model = GradientBoostingRegressor(**params)
else:
    raise ValueError(f"Model {config['model']['type']} not supported yet.")

model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
joblib.dump((X_test, y_test), "test_data.pkl")
print(f"{config['model']['type']} trained and saved as model.pkl")
