import dataset_handle as dh
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df_titanic = dh.load_dataset("titanic.csv")

print(df_titanic.head())

le=LabelEncoder()
le.fit_transform(df_titanic["Sex"])

print(df_titanic.head())