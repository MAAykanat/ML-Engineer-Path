import numpy as np
import pandas as pd
import seaborn as sns

############################################
# Assignment-1:
"""Capitalize the names of numeric variables in the car_crashes data by using the List Comprehension structure.
convert it to letter and add NUM at the beginning.
"""
###########################################
print("*"*100)

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
print("*"*100)

print([(col + "_FLAG").upper() if "NO" not in col.upper() else col.upper() for col in df_car_crush.columns ])

############################################
# Assignment-3:
"""Using the List Comprehension structure, 
you can find the variable names that are DIFFERENT from the ones given below.
Select the names of the variables and create a new dataframe.
og_list = ['abbrev', 'no_previoues']
"""
###########################################
print("*"*100)

og_list = ['abbrev', 'no_previous']

print([col for col in df_car_crush.columns if col not in og_list])

df_new = df_car_crush.loc[:, [col for col in df_car_crush.columns if col not in og_list]]

print(df_new.head())