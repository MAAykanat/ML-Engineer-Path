#######################################################
# End-to-End Telco Churn Machine Learning Pipeline II #
#######################################################

import pandas as pd
from helpers import *

def main():
    
    # Load Data
    df = pd.read_csv("Machine Learning/datasets/telco/Telco-Customer-Churn.csv")
    
    # Data Preprocessing & Feature Engineering
    X_train, X_test, y_train, y_test = data_prep(df)
    
    # Base Models
    base_models(df)
    
    # Automated Hyperparameter Optimization
    hyperparameter_optimization(df)
    
    # Stacking & Ensemble Learning
    stacking_ensemble(df)
    
    # Prediction for a New Observation
    prediction(df)