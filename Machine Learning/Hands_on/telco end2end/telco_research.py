################################################
# End-to-End Diabetes Machine Learning Pipeline I
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Base Models
# 4. Automated Hyperparameter Optimization
# 5. Stacking & Ensemble Learning
# 6. Prediction for a New Observation
# 7. Pipeline Main Function

################################################
####################NOTES#######################
################################################

# PRE-PROCESSING
# 1. Convert TotatlCharges to numeric
# 2. Convert Churn to 1 if "Yes", 0 if "No"
# 3. Implement Outlier Handle


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("Machine Learning/datasets/telco/Telco-Customer-Churn.csv")
print(df.head())

################################################
# 1. Exploratory Data Analysis (EDA)
################################################

# 1.1 General Picture of the Dataset
# 1.2 Catch Numeric and Categorical Value
# 1.3 Catetorical Variables Analysis
# 1.4 Numeric Variables Analysis
# 1.5 Target Variable Analysis (Dependent Variable) - Categorical
# 1.6 Target Variable Analysis (Dependent Variable) - Numeric
# 1.7 Outlier Analysis
# 1.8 Missing Value Analysis
# 1.9 Correlation Matrix

# 1.1 General Picture of the Dataset

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
"""
Target Variable: Churn
Shape = (7043, 21)
Types = object(18), float64(2), int64(1)
NA = No missing value
"""

check_df(df)

# PUT IT ALSO PRE-PROCESSING PART
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce") # Convert to numeric and if error occurs, convert it to NaN
df["Churn"] = df["Churn"].apply(lambda x: 1 if x =="Yes" else 0) # Convert to 1 if "Yes", 0 if "No"

print("#"*50)

# 1.2 Catch Numeric and Categorical Value

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

"""
Observations: 7043
Variables: 21
categorical_cols: 17
num_cols: 2
categorical_but_cardinal: 2
numeric_but_categorical: 1
"""

print("Categorical Columns: \n\n", cat_cols)
print("Numeric Columns: \n\n", num_cols)
[print("Categorical but Cardinal EMPTY!!!\n\n") if cat_but_car == [] else print("Categorical but Cardinal: \n", cat_but_car)]
print("#"*50)

# 1.3 Catetorical Variables Analysis

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

# 1.4 Numeric Variables Analysis

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

# 1.5 Target Variable Analysis (Dependent Variable) - Categorical

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
    target_summary_with_cat(df, "Churn", col)

print("#"*50)

# 1.6 Target Variable Analysis (Dependent Variable) - Numeric

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
    target_summary_with_num(df, "Churn", col)
print("#"*50)

# 1.7 Outlier Analysis
"""
For this dataset, There is no need to implement outlier handle in preprocessing part.
However, End to end pipeline should include outlier handle in any case.
"""

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

for col in num_cols:
    print(f"{col}: {check_outlier(df, col)}")
    
print("#"*50)

# 1.8 Missing Value Analysis

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
