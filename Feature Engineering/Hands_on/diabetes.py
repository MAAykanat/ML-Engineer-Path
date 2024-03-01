import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

PATH ="D:\!!!MAAykanat Dosyalar\Miuul\Diabetes Dataset"
df=pd.read_csv(PATH + "\diabetes.csv")

print(df.head())
print("*"* 50)
print("Shape of dataset: ", df.shape)
print("*"* 50)          
print("Number of null\n", df.isnull().sum())
