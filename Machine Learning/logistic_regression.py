import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv('Machine Learning/datasets/diabetes.csv')

print(df.head())