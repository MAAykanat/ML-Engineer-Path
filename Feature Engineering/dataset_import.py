import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

PATH = "D:\!!!MAAykanat Dosyalar\Miuul\Feature Engineering\\feature_engineering\datasets"

def load_dataset(dataset_name, path=PATH):
    df = pd.read_csv(path + "\\" + dataset_name)
    return df

def dataset_details(df):
    print("Dataset shape: ", df.shape)
    print("*"*50)
    print("Dataset columns: ", df.columns)
    print("*"*50)
    print("Dataset describe: ", df.describe())
    print("*"*50)
    print("Dataset head: \n", df.head())
    
print("Dataset imported successfully!")

