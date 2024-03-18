import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

check_df(df, 20)
print("#"*50)