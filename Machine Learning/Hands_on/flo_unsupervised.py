import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import datetime as dt


from shutil import get_terminal_size
import warnings

from helpers import *
from display_helpers import *

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None) # Show all the columns
pd.set_option('display.max_rows', None) # Show all the rows
pd.set_option('max_colwidth', None) # Show all the text in the columns
pd.set_option('display.float_format', lambda x: '%.3f' % x) # Show all the decimals
pd.set_option('display.width', get_terminal_size()[0]) # Get bigger terminal display width

df = pd.read_csv("Machine Learning/datasets/flo/flo_data_20K.csv")
print(df.head())

data_cols=df.columns[df.columns.str.contains("date")]
df[data_cols] = df[data_cols].apply(pd.to_datetime)

#########################################
### 1.EXPLORATORY DATA ANALYSIS - EDA ###
#########################################

# 1.1. General Picture of the Dataset
# 1.2. Catch Numeric and Categorical Value
# 1.3. Catetorical Variables Analysis
# 1.4. Numeric Variables Analysis
# 1.5. Target Variable Analysis (Dependent Variable) - Categorical
# 1.6. Target Variable Analysis (Dependent Variable) - Numeric
# 1.7. Outlier Detection
# 1.8. Missing Value Analysis
# 1.9. Correlation Matrix

# 1.1. General Picture of the Dataset

check_df(df)

print("\n\n")
# 1.2. Catch Numeric and Categorical Value
"""
Observations: 19945
Variables: 12
categorical_cols: 2
num_cols: 4
categorical_but_cardinal: 6
numeric_but_categorical: 0
"""

cat_cols, num_cols, cat_but_car = grap_column_names(df)

print("Categorical Columns: \n\n", cat_cols)
print("Numeric Columns: \n\n", num_cols)
[print("Categorical but Cardinal EMPTY!!!\n\n") if cat_but_car == [] else print("Categorical but Cardinal: \n", cat_but_car)]
print("#"*50)

# 1.3. Catetorical Variables Analysis

for col in cat_cols:
    cat_summary(df,col, plot=False)

print("#"*50)

# 1.4. Numerical Variable Analysis

for col in num_cols:
    numerical_col_summary(df,num_cols,plot=False)
print("#"*50)