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
#################################
# ASSIGNEMNT-1
#################################
df = pd.read_excel(".\Python for Data Science\datasets\miuul_gezinomi.xlsx")
#Task-1: Details of the dataset
print("*"*50)
check_df(df)

#Task-2: Unique values of the SaleCityName object variable
print("*"*50)
print(df["SaleCityName"].unique())
"""
['Antalya' 'İzmir' 'Diğer' 'Aydın' 'Muğla' 'Girne']
"""
print(df["SaleCityName"].value_counts())
"""
Antalya    31649
Muğla      10662
Aydın      10646
Diğer       3245
İzmir       2507
Girne        455
"""
#Task-3: Unique values of the ConceptName object variable
# print(len(df["ConceptName"].unique()))
print("*"*50)
#Task-4: Number of unique values of the ConceptName object variable
print(df["ConceptName"].value_counts())

#Task-5: Total earnings from sales by each city
print("*"*50)
print(df.groupby("SaleCityName").agg({"Price": "sum"}))

#Task-6: Total earnings from sales by each concept
print("*"*50)
print(df.groupby("ConceptName").agg({"Price": "sum"}))

#Task-7: Average PRICE by cities
print("*"*50)
print(df.groupby(by=['SaleCityName']).agg({"Price": "mean"}))

#Task-8: Average PRICE by concepts
print("*"*50)
print(df.groupby(by=['ConceptName']).agg({"Price": np.mean})) #np.mean="mean"

#Task-9: Average PRICE by city-concept
print("*"*50)
print(df.groupby(by=["SaleCityName", 'ConceptName']).agg({"Price": "mean"}))