✈️ Airline Ticket Price Prediction (MLOps Project)

Overview

This project predicts airline ticket prices using machine learning and an MLOps-style pipeline.  

It includes preprocessing, model training, testing, and experiment logging.


Dataset

The dataset is \*\*not included\*\* in this repository due to size limits.  

You can download it from \[https://www.kaggle.com/datasets/rohitgrewal/airlines-flights-data] and place it in the `Data/` folder as `raw.csv`.



Columns include:

\- airline, source\_city, destination\_city, class, duration, days\_left, etc.

\- Target variable: `price`.



 Pipeline

\- Config-driven design (`config.yaml`).

\- Preprocessing (`preprocess.py`).

\- Training (`train.py`).

\- Testing and logging (`test.py` → saves results to `experiments/results.csv`).


Experiments and Models tested:

\- Linear Regression (baseline).

\- Random Forest Regressor.

\- Gradient Boosting Regressor.

\- Variations with log-transform of target.



**Best model**: _RandomForestRegressor with log-transform_

\- MAE ≈ 2114  
\- R² ≈ 0.969  


📊 Results
All experiment logs are saved in: _experiments/results.csv_
This file contains:

\-Timestamp of each run
\-Model type and parameters
\-Whether log-transform was applied
\-Evaluation metrics (MAE, MSE, R²)

👉 If you only want to check results without downloading the dataset or running the pipeline, simply open experiments/results.csv.


🚀 How to Run

1\. Clone the repository:

&nbsp;  ```bash

&nbsp;  git clone https://github.com/your-username/Airline-Price-Prediction.git

&nbsp;  cd Airline-Price-Prediction

2\. Download the dataset from \[https://www.kaggle.com/datasets/rohitgrewal/airlines-flights-data].

&nbsp;  Save it in the Data/ folder as:

   ***Data/raw.csv***

3\. Run preprocessing:

&nbsp;  type this cmd in the terminal:- python preprocess.py

4\. Train the model:

&nbsp;  type this cmd in the terminal:- python train.py

5\. Test and log results:

&nbsp;  type this cmd in the terminal:- python test.py
