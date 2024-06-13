import joblib
import pandas as pd
from helpers import data_prep

df = pd.read_csv("Machine Learning/datasets/telco/Telco-Customer-Churn.csv")

X_train, X_test, y_train, y_test = data_prep(dataframe=df,target="Churn")

random_user = X_test.sample(1, random_state=45)

model = joblib.load("telco_voting_clf.pkl")

print(model.predict(random_user))