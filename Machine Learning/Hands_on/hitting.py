import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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