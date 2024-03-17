import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import missingno as msno


PATH ="D:\!!!MAAykanat Dosyalar\Miuul\Feature Engineering\Görevler\Görev-1 Diabetes Dataset"
df=pd.read_csv(PATH + "\diabetes.csv")

print(df.head())

#######################################
### EXPLORATORY DATA ANALYSIS - EDA ###
#######################################

# There are 6 steps to be taken in the EDA process.

# 1. General Picture of the Dataset
# 2. Catch Numeric and Categorical Values
# 3. Categoical Variable Analysis
# 4. Numeric Variable Analysis
# 5. Target Variable Analysis (Dependent Variable) - Numerical
# 6. Korrelation Analysis

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
print("#"*50)
# 2. Catch Numeric and Categorical Values

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

print("Categorical Columns: ", cat_cols)
print("Numeric Columns: ", num_cols)
[print("Categorical but Cardinal EMPTY!!!") if cat_but_car == [] else print("Categorical but Cardinal: ", cat_but_car)]
print("#"*50)

# 3. Categoical Variable Analysis
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

# 4. Numeric Variable Analysis
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

# 5. Target Variable Analysis (Dependent Variable) - Numerical

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
    target_summary_with_num(df, "Outcome", col)
print("#"*50)

# 6. Korrelation Analysis

# Correlation Matrix (Heatmap)

df_corr = df.corr()

f, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df_corr, annot=True, fmt=".2f", ax=ax, cmap="viridis")
ax.set_title("Correlation Heatmap", color="blue", fontsize=20)
# plt.show()

#######################################
######### FEATURE ENGINEERING #########
#######################################
# 1. Missing Values
# 1.1 Find Missing Value Table
df_copy = df.copy()

# Insulin, SkinThickness, BloodPressure, BMI, Glucose, DiabetesPedigreeFunction, Age
# These columns have 0 values. They cannot be 0. We will convert them to NaN.

print("Before Zero-NaN:\n ", df_copy.head())


#######################################
# First Method - Convert 0 to NaN
"""
counter = df_copy.shape[0]

for col in num_cols:
    for i in range(counter):
        if df_copy[col][i] == 0:
            df_copy[col][i] = None
"""
#######################################
# Second Method - Convert 0 to NaN
zero_columns = [col for col in df_copy.columns if (df_copy[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]
print(zero_columns)

for col in zero_columns:
    df_copy[col] = np.where(df_copy[col] == 0, np.nan, df_copy[col])
#######################################
print("After Zero-NaN:\n ", df_copy.head())
# print(df_copy.head())

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

# 1.2 Missing Values - Target Variable Relationship
# We will examine the relationship between the target variable and the missing values.
# If there is a relationship, we will fill in the missing values with the median of the target variable.

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

missing_vs_target(dataframe=df_copy, target="Outcome", na_columns=na_col)

msno.matrix(df_copy)
plt.title("Missing Values Matrix- Before Filling")
# plt.show()

# 1.3 Missing Values - Filling
# We will fill in the missing values with the median of the target-Outcome (0-1) variable.

for col in num_cols:
    # Fill all null values with mean of target variable (Outcome)
    df_copy.loc[(df_copy[col].isnull()) & (df_copy["Outcome"]==0), col] = df_copy.groupby("Outcome")[col].mean()[0]
    df_copy.loc[(df_copy[col].isnull()) & (df_copy["Outcome"]==1), col] = df_copy.groupby("Outcome")[col].mean()[1]
print(df_copy.head())

msno.matrix(df_copy)
plt.title("Missing Values Matrix- After Filling")
plt.show()