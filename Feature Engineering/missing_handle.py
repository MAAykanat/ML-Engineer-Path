import dataset_handle as dh

import pandas as pd
import numpy as np

def missing_values_table(dataframe, null_columns_name = False):
    null_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    number_of_missing_values = dataframe[null_columns].isnull().sum().sort_values(ascending=False)
    percentage_of_missing_values = (dataframe[null_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_values_table = pd.concat([number_of_missing_values, np.round(percentage_of_missing_values, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_values_table)
    
    if null_columns_name:
        return null_columns    

df_titanic = dh.load_dataset("titanic.csv")

df_titanic.loc[[3]] # Specific row(3th) of the dataset 

df_titanic.isnull() # True-False matrix for null values
df_titanic.notnull() # True-False matrix for non-null values

df_titanic.isnull().sum() # number of null values in each column
df_titanic.notnull().sum() # number of non-null values in each column

df_titanic.isnull().sum().sort_values(ascending=False) # Sort the number of null values in each column
((df_titanic.isnull().sum() / df_titanic.shape[0])*100).sort_values(ascending=False) # Sort the percentage of null values in each column

na_columns = [col for col in df_titanic.columns if df_titanic[col].isnull().sum() > 0]

print(missing_values_table(df_titanic))

# print(n_miss)
# print("*"*50)
# print(ratio)