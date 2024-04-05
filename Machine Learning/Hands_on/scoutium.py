import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df_attribute = pd.read_csv("Machine Learning/datasets/scoutium/scoutium_attributes.csv", sep=";")
df_labels = pd.read_csv("Machine Learning/datasets/scoutium/scoutium_potential_labels.csv", sep=";")


df = pd.merge(df_attribute, df_labels, on=["player_id", "evaluator_id","match_id","task_response_id"])
print(df.head())
print(df.shape)
# (10730, 9)

df.drop(df[df["position_id"]==1].index, inplace=True)
print(df["potential_label"].value_counts())
# above_average = 136

print(df["potential_label"].value_counts()["below_average"]/df["potential_label"].value_counts().sum())
# 0.013 below_average ~1% of the data will be removed
df.drop(df[df["potential_label"]=="below_average"].index, inplace=True)

print(df.shape)
# (9894, 9)