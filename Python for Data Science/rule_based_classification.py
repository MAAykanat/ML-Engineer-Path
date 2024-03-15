import pandas as pd
import seaborn as sns
import numpy as np

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

df= pd.read_csv(".\Python for Data Science\datasets\persona.csv")
check_df(df)

###################
# STEP-1
###################

print(df["SOURCE"].unique())
print(df["SOURCE"].value_counts())

print("#"*100)

print(df["PRICE"].unique())
print(df["PRICE"].value_counts())

print("#"*100)

print(df["COUNTRY"].value_counts())
print(df.groupby(by="COUNTRY").agg({"PRICE": ["sum", "mean"]}))

print(df.groupby(by="SOURCE").agg({"PRICE": "sum"}))

print(df.groupby(by=["COUNTRY","SOURCE"]).agg({"PRICE": "mean"}))

#####################
# STEP-2
#####################

print(df.groupby(by=["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}))