#######################################################
# End-to-End Telco Churn Machine Learning Pipeline II #
#######################################################

from helpers import *

def main():
    
    # Load Data
    df = pd.read_csv("Machine Learning/datasets/telco/Telco-Customer-Churn.csv")
    
    # Data Preprocessing & Feature Engineering
    X_train, X_test, y_train, y_test = data_prep(dataframe=df,target="Churn")

    # Base Models
    base_models(X_train, y_train)
    
    # Automated Hyperparameter Optimization
    hyperparameter_optimization(df)
    
    # Stacking & Ensemble Learning
    stacking_ensemble(df)
    
    # Prediction for a New Observation
    prediction(df)

if __name__ == "__main__":
    print("Pipeline Started")
    main()
    print("Pipeline Completed")