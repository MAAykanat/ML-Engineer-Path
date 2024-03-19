import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

def check_df(dataframe, head=5):
    print("Dataframe shape: ", dataframe.shape)
    print("*"*50)
    print("Dataframe coloumn types: ", dataframe.dtypes)
    print("*"*50)
    print("Dataframe head: \n", dataframe.head(head))
    print("*"*50)
    print("Dataframe tail: \n", dataframe.tail(head))
    print("*"*50)
    print("Dataframe info: \n", dataframe.info())
    print("*"*50)
    print("Dataframe describe: \n", dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("*"*50)
    print("Dataframe missing values: \n", dataframe.isnull().sum())

# Importing the dataset
df = pd.read_csv('Machine Learning/datasets/advertising.csv')

check_df(df)

##############################
## Simple Linear Regression ##
##############################

X = df[["TV"]]
y = df[["sales"]]

print(type(X))

reg_model = LinearRegression().fit(X, y)
y_pred = reg_model.predict(X)

# Formula: y_hat = b0 + b1*X

print("b0 (bias) is ",reg_model.intercept_)
print("b1 (weight) is ",reg_model.coef_)

##############################
##### Model Evaluation #######
##############################

#MSE (Mean Squared Error)
mse = mean_squared_error(y, y_pred)
print("MSE is ", mse)
# 10.512652915656757

#RMSE (Root Mean Squared Error)
rmse = np.sqrt(mse)
print("RMSE is ", rmse)
# 3.2423221486546887

#MAE (Mean Absolute Error)
mae = mean_absolute_error(y, y_pred)
print("MAE is ", mae)
# 2.549806038927486

#R-Squared (Coefficient of Determination) - Means how much of change in y can be explained by the change in X
r2 = reg_model.score(X, y)
print("R-Squared is ", r2)
# 0.611875050850071

##############################
####### Visualization ########
##############################

figure = sns.regplot(x=X, y=y, data=df, 
                     scatter_kws={'color': 'b', 's': 9}, 
                     ci=False, color="r") # ci=False: Confidence Interval
figure.set_title(f"Model Equation: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
figure.set_ylabel("Sales")
figure.set_xlabel("TV Advertising")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()

