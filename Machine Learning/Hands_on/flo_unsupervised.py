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

print(df.columns.str.contains("date"))

data_cols=df.columns[df.columns.str.contains("date")]
print(data_cols)
df[data_cols] = df[data_cols].apply(pd.to_datetime)

##########################################
##### EXPLORATORY DATA ANALYSIS - EDA ####
### UNSUPERVISED LEARNING - CLUSTERING ###
"""
    Steps can be different for supervised and unsupervised learning.
    Since, There is no target variable
"""
##########################################

# 1.1. General Picture of the Dataset
# 1.2. Catch Numeric and Categorical Value
# 1.3. Catetorical Variables Analysis
# 1.4. Numeric Variables Analysis
# 1.5. Outlier Detection
# 1.6. Missing Value Analysis

# 1.1. General Picture of the Dataset

check_df(df)

"""
##################### Types #####################
master_id                                    object
order_channel                                object
last_order_channel                           object
first_order_date                     datetime64[ns]
last_order_date                      datetime64[ns]
last_order_date_online               datetime64[ns]
last_order_date_offline              datetime64[ns]
order_num_total_ever_online                 float64
order_num_total_ever_offline                float64
customer_value_total_ever_offline           float64
customer_value_total_ever_online            float64
interested_in_categories_12                  object
dtype: object
"""

print("\n\n")
# 1.2. Catch Numeric and Categorical Value
"""
Observations: 19945
Variables: 12
categorical_cols: 2
num_cols: 8
categorical_but_cardinal: 2
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

# 1.5. Outlier Detection

"""
Have Outliers: order_num_total_ever_online
Have Outliers: order_num_total_ever_offline
Have Outliers: customer_value_total_ever_offline
Have Outliers: customer_value_total_ever_online

Outliers are handled with the threshold values. IQR method is used.

"""

for col in num_cols:
    print(col, ":", check_outlier(df, col))

print("#"*50)

for col in num_cols:
    replace_with_thresholds(df, col)

print("#"*50)

for col in num_cols:
    print(col, ":", check_outlier(df, col))

print("#"*50)
