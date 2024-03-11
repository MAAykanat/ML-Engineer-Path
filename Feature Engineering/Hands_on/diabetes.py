import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import missingno as msno

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

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    # Outlier thresholds for any attribute
    # Interquantile range = q3 - q1
    # Up limit = q3 + 1.5 * interquantile range
    # Low limit = q1 - 1.5 * interquantile range
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)

    interquantile_range = quartile3 - quartile1

    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range

    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    lower_limit, upper_limit = outlier_thresholds(dataframe=dataframe, col_name=col_name)

    if dataframe[(dataframe[col_name] > upper_limit) | (dataframe[col_name] < lower_limit)].any(axis=None):
        print(f'{col_name} have outlier')
        return True
    else:
        return False

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

PATH ="D:\!!!MAAykanat Dosyalar\Miuul\Feature Engineering\Görevler\Görev-1 Diabetes Dataset"
df=pd.read_csv(PATH + "\diabetes.csv")

###############################
#####CHECK GENARAL PICTURE#####
###############################
print(df.head())
"""
print(df.head())
print("*"* 50)
print("Shape of dataset: ", df.shape)
print("*"* 50)          
print("Number of null\n", df.isnull().sum())
"""

categorical_col, numeric_col, cat_but_cardinal_col = grap_column_names(dataframe=df)
print(numeric_col)
print("*"*50)
print(categorical_col)

for col in numeric_col:
    print(f"Column is {col}")
    print(df.groupby("Outcome")[col].mean())
    print("Is there outlier or not?")
    print(check_outlier(dataframe=df, col_name=col))
    print("*"*50)

# No missing variable something is wrong
print(missing_values_table(dataframe=df, null_columns_name=True))

# msno.bar(df)
# plt.show()
###############################
#### Correlation Heat Map
###############################
"""
correlation = df.corr("kendall")
sns.heatmap(correlation, annot=True, fmt=".2f")
plt.show()
"""

print(df.head())
print(categorical_col)

counter = df.shape[0]
df_copy = df.copy()

#Correct Dataset convert 0 to NaN
for col in numeric_col:
    for i in range(counter):
        if df_copy[col][i] == 0:
            df_copy[col][i] = None

print(df_copy.head())

print(missing_values_table(dataframe=df_copy, null_columns_name=True))

# sns.boxplot(x="Pregnancies", data=df_copy)
msno.matrix(df_copy)
# plt.show()

###############################
#### Handling Missing Values###
###############################
print(df_copy.head())
df_copy.loc[(df_copy["Insulin"].isnull()) & (df_copy["Outcome"]==0), "Insulin"] = df_copy.groupby("Outcome")["Insulin"].mean()[0]
df_copy.loc[(df_copy["Insulin"].isnull()) & (df_copy["Outcome"]==1), "Insulin"] = df_copy.groupby("Outcome")["Insulin"].mean()[1]
print(df_copy.head())

df_copy.correlation = df_copy.corr("kendall")
sns.heatmap(df_copy.correlation, annot=True, fmt=".2f")
plt.show()

