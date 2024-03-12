import numpy as np
import pandas as pd
import seaborn as sns

############################################
# Assignment-1:
"""Capitalize the names of numeric variables in the car_crashes data by using the List Comprehension structure.
convert it to letter and add NUM at the beginning.
"""
###########################################

df_car_crush = sns.load_dataset("car_crashes")

print(df_car_crush.head())
print(df_car_crush.info())
print(df_car_crush.describe().T)

df_car_crush_columns=[("NUM_"+col).upper() for col in df_car_crush.columns]

print(df_car_crush_columns)

############################################
# Assignment-2:
"""Using the List Comprehension structure, the car_crashes data does not contain "no" in its name.
Write "FLAG" at the end of the variable names.
"""
###########################################