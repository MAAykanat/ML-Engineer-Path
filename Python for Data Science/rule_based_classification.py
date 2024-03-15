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

#####################
# STEP-3
#####################

agg_df=df.groupby(by=["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)
print(agg_df.head())

#####################
# STEP-4
#####################

agg_df.reset_index(inplace=True)
print(agg_df.head())

#####################
# STEP-5
#####################

bins = [-1,18,19,23,24,30,31,40,41,70]

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins=bins)
# agg_df["AGE_CAT"] = agg_df["AGE_CAT"].str.replace(r'[(),]', '_')
print(agg_df.head())

agg_df["AGE_CAT"]=agg_df["AGE_CAT"].agg(lambda x: str(x).replace(r', ', '_').replace(r'(', '').replace(r']', ''))

print(agg_df.head())

#####################
# STEP-6
#####################

agg_df["customer_level_based"]=agg_df[["COUNTRY","SOURCE", "SEX", "AGE_CAT"]].agg(lambda x: '_'.join(x).upper(), axis=1)

print(agg_df.head())

#####################
# STEP-7
#####################

agg_df["Segment"]=pd.qcut(agg_df["PRICE"], 4, ["D","C","B","A"])

print(agg_df.head())

#####################
# STEP-8
#####################
new_user= "TUR_ANDROID_FEMALE_31_40"

print(agg_df[agg_df["customer_level_based"]==new_user])