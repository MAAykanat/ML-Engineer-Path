import numpy as np
import pandas as pd
import seaborn as sns

df_car_crush = sns.load_dataset("car_crashes")

print(df_car_crush.head())
print(df_car_crush.info())
print(df_car_crush.describe().T)
