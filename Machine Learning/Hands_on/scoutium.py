import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df_attribute = pd.read_csv("Machine Learning/datasets/scoutium/scoutium_attributes.csv", sep=";")
df_labels = pd.read_csv("Machine Learning/datasets/scoutium/scoutium_potential_labels.csv", sep=";")

print(df_attribute.head())
print(df_labels.head())

df = pd.merge(df_attribute, df_labels, on=["player_id", "evaluator_id","match_id","task_response_id"])
print(df.head())
