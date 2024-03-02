import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

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


PATH ="D:\!!!MAAykanat Dosyalar\Miuul\Diabetes Dataset"
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
    print("*"*50)