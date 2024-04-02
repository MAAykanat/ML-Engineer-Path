import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.simplefilter(action='ignore', category=Warning)
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv("Machine Learning/datasets/hitter/hitters.csv")
print(df.head())

#######################################
### EXPLORATORY DATA ANALYSIS - EDA ###
#######################################

# 1. General Picture of the Dataset
# 2. Catch Numeric and Categorical Value
# 3. Catetorical Variables Analysis
# 4. Numeric Variables Analysis
# 5. Target Variable Analysis (Dependent Variable) - Categorical
# 6. Target Variable Analysis (Dependent Variable) - Numeric
# 7. Outlier Detection
# 8. Missing Value Analysis
# 9. Correlation Matrix

# 1. General Picture of the Dataset

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)
# Shape: (322,20)
# Target: Salary (Numeric) - float64
# League, Division, NewLeague (Categorical) - object
# Others: 16 Numeric - int64 

# Only salary has missing values (59)
# There can be outliers!!!

# 2. Catch Numeric and Categorical Value
"""
Observations: 322
Variables: 20
categorical_cols: 3
num_cols: 17
categorical_but_cardinal: 0
numeric_but_categorical: 0
"""

def grap_column_names(dataframe, categorical_th=10, cardinal_th=20):
    """
    It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
    Note: Categorical variables with numerical appearance are also included."""

    """
    Cardinal Variables: Variables that are categorical and do not carry information,
    that is, have too many classes, are called variables with high cardinality.
    """

    """
    Returns
    ------
        categorical_cols: list
                Categorical variable list
        num_cols: list
                Numeric variable list
        categorical_but_cardinal: list
                Categorical variables with high cardinality list
    """
    # categorical_cols, categorical_but_cardinal
    categorical_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    numeric_but_categorical = [col for col in dataframe.columns if dataframe[col].nunique() < categorical_th and
                   dataframe[col].dtypes != "O"]
    categorical_but_cardinal = [col for col in dataframe.columns if dataframe[col].nunique() > cardinal_th and
                   dataframe[col].dtypes == "O"]
    categorical_cols = categorical_cols + numeric_but_categorical
    categorical_cols = [col for col in categorical_cols if col not in categorical_but_cardinal]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in numeric_but_categorical]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'categorical_cols: {len(categorical_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'categorical_but_cardinal: {len(categorical_but_cardinal)}')
    print(f'numeric_but_categorical: {len(numeric_but_categorical)}')

    return categorical_cols, num_cols, categorical_but_cardinal

cat_cols, num_cols, cat_but_car = grap_column_names(df)

print("Categorical Columns: \n\n", cat_cols)
print("Numeric Columns: \n\n", num_cols)
[print("Categorical but Cardinal EMPTY!!!\n\n") if cat_but_car == [] else print("Categorical but Cardinal: \n", cat_but_car)]
print("#"*50)

# 3. Catetorical Variables Analysis

def cat_summary(dataframe, col_name, plot=False):
    """
    This function shows the frequency of categorical variables.

    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    col_name : str
        The name of the column to be analyzed.
    plot : bool, optional
        The default is False.
    Returns
    -------
    None.
    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df,col)
print("#"*50)

# 4. Numeric Variables Analysis

def numerical_col_summary(dataframe, col_name, plot=False):

    """
    This function shows the frequency of numerical variables.

    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    col_name : str
        The name of the column to be analyzed.
    plot : bool, optional
        The default is False.
    Returns
    -------
    None.
    """
    print(dataframe[col_name].describe([0.01, 0.05, 0.75, 0.90, 0.99]).T)
    print("##########################################")
    if plot:
        sns.histplot(dataframe[col_name], kde=True)
        plt.xlabel(col_name)
        plt.title(f"{col_name} Distribution")
        plt.show()

for col in num_cols:
    numerical_col_summary(df,col)
print("#"*50)

# 5. Target Variable Analysis (Dependent Variable) - Categorical
"""
It looks Division has significant effect on Salary, but League and NewLeague does not.
"""

def target_summary_with_cat(dataframe, target, categorical_col):
    """
    This function shows the mean of the target variable according to the categorical variable.

    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    target : str
        The name of the target variable.
    categorical_col : str
        The name of the categorical variable.
    Returns
    -------
    None.
    """
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Salary", col)

print("#"*50)

# 6. Target Variable Analysis (Dependent Variable) - Numeric
"""
Nonsensical results. Since, target variable is also numeric.

"""

def target_summary_with_num(dataframe, target, numerical_col):
    """
    This function shows the average of numerical variables according to the target variable.
    
    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    target : str
        The name of the target variable.
    numerical_col : str
        The name of the numerical variable.
    Returns
    -------
    None.
    """
    print(dataframe.groupby(target).agg({numerical_col: ["mean", "median", "count"]}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Salary", col)
print("#"*50)

# 7. Outlier Detection

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    """
    This function calculates the lower and upper limits for the outliers.

    Calculation:
    Interquantile range = q3 - q1
    Up limit = q3 + 1.5 * interquantile range
    Low limit = q1 - 1.5 * interquantile range

    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    col_name : str
        The name of the column to be analyzed.
    q1 : float, optional
        The default is 0.05.
    q3 : float, optional
        The default is 0.95.
    Returns
    -------
    low_limit, up_limit : float
        The lower and upper limits for the outliers.
    """

    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)

    interquantile_range = quartile3 - quartile1

    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range

    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    """
        This function checks dataframe has outlier or not.

    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    col_name : str
        The name of the column to be analyzed.
    Returns
    -------
    bool
        True if the dataframe has outlier, False otherwise.
    """

    lower_limit, upper_limit = outlier_thresholds(dataframe=dataframe, col_name=col_name)

    if dataframe[(dataframe[col_name] > upper_limit) | (dataframe[col_name] < lower_limit)].any(axis=None):
        print(f'{col_name} have outlier')
        return True
    else:
        return False
    
def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    """
    This function replaces the outliers with the lower and upper limits.

    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    variable : str
        The name of the column to be analyzed.
    q1 : float, optional
        The default is 0.05.
    q3 : float, optional
        The default is 0.95.
    Returns 
    -------
    None
    """

    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# There is no outlier in any numeric variables.
for col in num_cols:
    print(f"{col}: {check_outlier(df, col)}")

print("#"*50)

# 8. Missing Value Analysis

print(df.isnull().sum())

def missing_values_table(dataframe, null_columns_name = False):
    """
    This function returns the number and percentage of missing values in a dataframe.
    """
    """
    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    null_columns_name : bool, optional
        The default is False.
    Returns
    -------
    missing_values_table : pandas dataframe
        A dataframe that contains the number and percentage of missing values in the dataframe.
    """
    # Calculate total missing values in each column
    null_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    number_of_missing_values = dataframe[null_columns].isnull().sum().sort_values(ascending=False)
    percentage_of_missing_values = (dataframe[null_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_values_table = pd.concat([number_of_missing_values, np.round(percentage_of_missing_values, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_values_table)

    if null_columns_name:
        return null_columns  

missing_values_table(df, True)

# There are 59 missing values in only Salary column.

df["Salary"].fillna(df["Salary"].mean(), inplace=True)

missing_values_table(df, True)

print("#"*50)

# 9. Correlation Matrix
"""
There are too many high correlated variables. 
Model will be considered by dropping some of them and not.
"""


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    """
    This function returns the columns that have a correlation higher than the threshold value.

    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    plot : bool, optional
        The default is False.
    corr_th : float, optional
        The default is 0.90.
    Returns
    -------
    drop_list : list
        The list of columns that have a correlation higher than the threshold value.
    """
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

df_corr = df.corr()
"""
f, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df_corr, annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Heatmap", color="black", fontsize=20)
plt.show()
"""
drop_list = high_correlated_cols(df, False, 0.90)
print(drop_list)
# ['Hits', 'Runs', 'CAtBat', 'CHits', 'CRuns', 'CRBI', 'CWalks']

df_drop = df.drop(drop_list, axis=1)
print(df.columns.shape) # 20
print("*"*50)
print(df_drop.columns.shape) # 13

#######################################
######### FEATURE ENGINEERING #########
#######################################

# There are 6 steps to be taken in the Feature Engineering process.
# 1. Missing Values
# 2. Outlier Values Analysis
# 3. Feature Generation
# 4. Encoding
# 5. Standardization
# 6. Save the Dataset

cat_cols, num_cols, cat_but_car = grap_column_names(df)
print(" "*50)
cat_cols_drop, num_cols_drop, cat_but_car_drop = grap_column_names(df_drop)

# 1. Missing Values
"""
    It has been filled in the previous section. EDA - 8. Missing Value Analysis
    df is filled with the mean of the target variable.
"""

# 2. Outlier Values Analaysis
"""
    It has been filled in the previous section. EDA - 7. Missing Value Analysis
    df is filled with the mean of the target variable.
"""

# 3. Feature Generation
"""
    At first implementation of model,
    New features will not generated.
"""
# 4. Encoding

# 4.1 Label Encoding

def label_encoder(dataframe, binary_col):
    """
    This function encodes the binary variables to numericals.

    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    binary_col : str
        The name of the column to be encoded.
    Returns
    -------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    """
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    print(binary_col, "is encoded.")
    return dataframe

binary_col = [col for col in df.columns if df[col].dtypes=='O' and df[col].nunique() == 2]
binary_col_drop = [col for col in df_drop.columns if df_drop[col].dtypes=='O' and df_drop[col].nunique() == 2]

print("BINARY COLS",binary_col)

for col in binary_col:
    df = label_encoder(df, col)

for col in binary_col_drop:
    df_drop = label_encoder(df_drop, col)
# 4.2 One-Hot Encoding
"""
    There is no categorical variable to be one-hot encoded.
"""

# 5. Standardization
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
print(df.head())

df_drop[num_cols_drop] = scaler.fit_transform(df_drop[num_cols_drop])

# 6. Save the Dataset
# df.to_csv("Machine Learning/datasets/hitter/hitters_preprocessed.csv", index=False)
# df_drop.to_csv("Machine Learning/datasets/hitter/hitters_preprocessed_drop.csv", index=False)

#######################################
########### MODEL IMPLEMENT ###########
#######################################

# 1. Train-Test Split
# 2. Model Selection and Evaluation
# 3. Hyperparameter Optimization
# 4. Final Model Implementation and Evaluation
# 5. Feature Importance
# 6. Analyzing Model Complexity with Learning Curves
# 7. Prediction and Model Deployment

# 1. Train-Test Split

X = df.drop("Salary", axis=1)
y = df["Salary"]

X_drop = df_drop.drop("Salary", axis=1)
y_drop = df_drop["Salary"]

# 2. Model Selection and Evaluation

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, n_jobs=-1, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")
    f = open('Estimators.txt', 'a')
    f.writelines(f"RMSE: {round(rmse, 4)} ({name})\n")
    f.close()
"""
RMSE: 0.8124 (LR)
RMSE: 0.8017 (Ridge)
RMSE: 0.9897 (Lasso)
RMSE: 0.9842 (ElasticNet)
RMSE: 0.7657 (KNN)
RMSE: 0.9644 (CART)
RMSE: 0.7028 (RF)
RMSE: 0.743 (SVR)
RMSE: 0.704 (GBM)
RMSE: 0.7342 (XGBoost)
RMSE: 0.7019 (LightGBM)
RMSE: 0.7179 (CatBoost)

"""

print("################DROPED DATAFRAME################ ")

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X_drop, y_drop, cv=5, n_jobs=-1, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")
    f = open('Estimators_drop.txt', 'a')
    f.writelines(f"RMSE: {round(rmse, 4)} ({name})\n")
    f.close()

"""
RMSE: 0.8519 (LR)
RMSE: 0.8508 (Ridge)
RMSE: 0.9897 (Lasso)
RMSE: 0.9892 (ElasticNet)
RMSE: 0.8093 (KNN)
RMSE: 1.0881 (CART)
RMSE: 0.7575 (RF)
RMSE: 0.7914 (SVR)
RMSE: 0.789 (GBM)
RMSE: 0.8326 (XGBoost)
RMSE: 0.7484 (LightGBM)
RMSE: 0.7398 (CatBoost)
"""

"""
It looks LightGBM is the best model for this dataset.
"""