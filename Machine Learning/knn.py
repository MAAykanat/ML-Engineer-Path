import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cleaned dataeset import
df = pd.read_csv("Machine Learning/datasets/diabetes_cleaned.csv")
print(df.head())
df = df.reindex(sorted(df.columns), axis=1)