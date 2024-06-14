import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

# 1.2. Catch Numeric and Categorical Value

cat_cols, num_cols, cat_but_car = grap_column_names(df)
