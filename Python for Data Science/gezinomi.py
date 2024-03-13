import pandas as pd
import seaborn as sns

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

df = pd.read_excel(".\Python for Data Science\datasets\miuul_gezinomi.xlsx")
#Task-1: Details of the dataset
#check_df(df)

#Task-2: Unique values of the SaleCityName object variable
print(df["SaleCityName"].unique())