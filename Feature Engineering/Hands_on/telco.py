import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import missingno as msno

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

PATH ="D:\!!!MAAykanat Dosyalar\Miuul\Feature Engineering\Görevler\Görev-2 Telco Dataset"
df = pd.read_csv(PATH + "\Telco-Customer-Churn.csv")

#######################################
### EXPLORATORY DATA ANALYSIS - EDA ###
#######################################


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

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce") # Convert to numeric and if error occurs, convert it to NaN
df["Churn"] = df["Churn"].apply(lambda x: 1 if x =="Yes" else 0) # Convert to 1 if "Yes", 0 if "No"
check_df(df, 20)
print("#"*50)

# 2. Catch Numeric and Categorical Value

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

print("Categorical Columns: \n", cat_cols)
print("Numeric Columns: \n", num_cols)
[print("Categorical but Cardinal EMPTY!!!\n") if cat_but_car == [] else print("Categorical but Cardinal: \n", cat_but_car)]
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

# 6. Target Variable Analysis (Dependent Variable) - Numeric
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

# No outlier
for col in num_cols:
    print(f"{col}: {check_outlier(df, col)}")

print("#"*50)

# 8. Missing Value Analysis

print(df.isnull().sum())

df_copy = df.copy()

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
    """
    # Calculate total missing values in each column
    null_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    number_of_missing_values = dataframe[null_columns].isnull().sum().sort_values(ascending=False)
    percentage_of_missing_values = (dataframe[null_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_values_table = pd.concat([number_of_missing_values, np.round(percentage_of_missing_values, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_values_table)
    
    if null_columns_name:
        return null_columns  

na_col = missing_values_table(dataframe=df_copy, null_columns_name=True)

def missing_vs_target(dataframe, target, na_columns):
    """
    This function examines the relationship between the target variable and the missing values.
    
    Alternative of Library: missingno.matrix(dataframe)

    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    target : str
        The name of the target variable.
    na_columns : list
        The name of the columns to be analyzed.
    Returns
    -------
    None.

    """
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(dataframe=df_copy, target="Churn", na_columns=na_col)

msno.matrix(df_copy)
plt.title("Missing Values Matrix- Before Filling")
# plt.show()

for col in num_cols:
    # Fill all null values with mean of target variable (Outcome)
    df_copy.loc[(df_copy[col].isnull()) & (df_copy["Churn"]==0), col] = df_copy.groupby("Churn")[col].mean()[0]
    df_copy.loc[(df_copy[col].isnull()) & (df_copy["Churn"]==1), col] = df_copy.groupby("Churn")[col].mean()[1]
print(df_copy.head())

msno.matrix(df_copy)
plt.title("Missing Values Matrix- After Filling")
plt.show()

# 9. Correlation Matrix

# Correlation Matrix (Heatmap)
df_corr = df.corr()

f, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df_corr, annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Heatmap", color="black", fontsize=20)
# plt.show()