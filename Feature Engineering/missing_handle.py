import dataset_handle as dh

df_titanic = dh.load_dataset("titanic.csv")

df_titanic.loc[[3]] # Specific row(3th) of the dataset 

df_titanic.isnull() # True-False matrix for null values
df_titanic.notnull() # True-False matrix for non-null values

df_titanic.isnull().sum() # number of null values in each column
df_titanic.notnull().sum() # number of non-null values in each column

df_titanic.isnull().sum().sort_values(ascending=False) # Sort the number of null values in each column
((df_titanic.isnull().sum() / df_titanic.shape[0])*100).sort_values(ascending=False) # Sort the percentage of null values in each column

na_columns = [col for col in df_titanic.columns if df_titanic[col].isnull().sum() > 0]