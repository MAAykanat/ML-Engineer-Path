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
# plt.show()

################################
## Multiple Linear Regression ##
################################

X = df.drop('sales', axis=1)
y = df[["sales"]]

## Model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

reg_model = LinearRegression().fit(X_train, y_train)
y_pred = reg_model.predict(X_test) 

# bias (b)
print("Bias is ", reg_model.intercept_[0])

# weights (w)
print("Weights are ", reg_model.coef_[0])
print("#"*50)
##############################
##### Model Evaluation #######
##############################

y_pred_train = reg_model.predict(X_train)

### Train Data
#MSE (Mean Squared Error)
mse_train = mean_squared_error(y_train, y_pred_train)
print("MSE-train is ", mse_train)

#RMSE (Root Mean Squared Error)
rmse_train = np.sqrt(mse_train)
print("RMSE-train is ", rmse_train)

#MAE (Mean Absolute Error)
mae_train = mean_absolute_error(y_train, y_pred_train)
print("MAE-train is ", mae_train)

#R-Squared (Coefficient of Determination) - Means how much of change in y can be explained by the change in X
r2_train = reg_model.score(X_train, y_train)
print("R-Squared-train is ", r2_train)

"""
MSE-train is  3.0168306076596774
RMSE-train is  1.736902590147092
MAE-train is  1.3288502460998388
R-Squared-train is  0.8959372632325174
"""

print("#"*50)
### Test Data
#MSE (Mean Squared Error)
mse_test = mean_squared_error(y_test, y_pred)
print("MSE-test is ", mse_test)

#RMSE (Root Mean Squared Error)
rmse_test = np.sqrt(mse_test)
print("RMSE-test is ", rmse_test)

#MAE (Mean Absolute Error)
mae_test = mean_absolute_error(y_test, y_pred)
print("MAE-test is ", mae_test)

#R-Squared (Coefficient of Determination) - Means how much of change in y can be explained by the change in X
r2_test = reg_model.score(X_test, y_test)
print("R-Squared-test is ", r2_test)

"""
MSE-test is  1.9918855518287906
RMSE-test is  1.4113417558581587
MAE-test is  1.0402154012924718
R-Squared-test is  0.8927605914615384
"""

######################################################
# Simple Linear Regression with Gradient Descent from Scratch
######################################################

#Cost Funtion MSE
def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0

    for i in range(0,m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) **2
    mse = sse / m
    return mse

# Update Weights
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_derivative_sum = 0
    w_derivative_sum = 0

    for i in range(0,m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_derivative_sum += (y_hat-y)
        w_derivative_sum += (y_hat-y) * X[i]
    new_b = b - (learning_rate * (1 / (m * b_derivative_sum)))
    new_w = w - (learning_rate * (1 / (m * w_derivative_sum)))

    return new_b, new_w