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
    base_models(X_train, y_train, cv=3, scoring="roc_auc")
    
    # Automated Hyperparameter Optimization
    knn_params = {"n_neighbors": range(2, 50)}

    cart_params = {'max_depth': range(1, 20),
                "min_samples_split": range(2, 30)}

    rf_params = {"max_depth": [8, 15, None],
                "max_features": [5, 7, "auto"],
                "min_samples_split": [15, 20],
                "n_estimators": [200, 300]}

    xgboost_params = {"learning_rate": [0.1, 0.01],
                    "max_depth": [5, 8],
                    "n_estimators": [100, 200]}

    lightgbm_params = {"learning_rate": [0.01, 0.1],
                    "n_estimators": [300, 500]}

    classifiers = [('KNN', KNeighborsClassifier(), knn_params),
                ("CART", DecisionTreeClassifier(), cart_params),
                ("RF", RandomForestClassifier(), rf_params),
                    ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
                ('LightGBM', LGBMClassifier(), lightgbm_params)]

    best_models = hyperparameter_optimization(X_train,y_train,classifiers,cv=3,scoring="roc_auc")
    
    # Stacking & Ensemble Learning
    voting_classifier(best_models=best_models, X=X_train, y=y_train,cv=3,voting="soft")
    
if __name__ == "__main__":
    print("Pipeline Started")
    main()
    print("Pipeline Completed")