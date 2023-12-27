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

def plot_hist(df, col_name, bin_type = None):

    if bin_type == None:
        plt.hist(df[col_name], color="red", alpha=0.75, linewidth=4., joinstyle='miter')
        plt.title(col_name + " Attribute Distribution\nDefault binning-10")
    elif bin_type == 1:
        # Square root binning
        plt.hist(df[col_name], bins=int(np.sqrt(len(df[col_name]))))
        plt.title(col_name + " Attribute Distribution\nSquare root binning")
    elif bin_type == 2:
        # Sturges' formula
        plt.hist(df[col_name], bins=int(np.log2(len(df[col_name])))+1)
        plt.title(col_name + " Attribute Distribution\nSturges' formula")
    elif bin_type == 3:
        # Rice Rule
        plt.hist(df[col_name], bins=2*int(np.power(len(df[col_name]), 1/3)))
        plt.title(col_name + " Attribute Distribution\nRice Rule")
    elif bin_type == 4:
        # Doane's formula
        plt.hist(df[col_name], bins=int(1 + np.log2(len(df[col_name])) + np.log2(1 + np.abs(skew(df[col_name]))/se(df[col_name]))))
        plt.title(col_name + " Attribute Distribution\nDoane's formula")
    elif bin_type == 5:
        # Scott's normal reference rule
        plt.hist(df[col_name], bins=int((max(df[col_name])-min(df[col_name]))/(3.5*se(df[col_name])*np.power(len(df[col_name]), -1/3))))
        plt.title(col_name + " Attribute Distribution\nScott's normal reference rule")
    elif bin_type == 6:
        # Freedman-Diaconis' choice
        plt.hist(df[col_name], bins=int((max(df[col_name])-min(df[col_name]))/(2*iqr(df[col_name])*np.power(len(df[col_name]), -1/3))))
        plt.title(col_name + " Attribute Distribution\nFreedman-Diaconis' choice")
    elif bin_type == 7:
        # Shimazaki and Shinomoto's choice
        plt.hist(df[col_name], bins=int((max(df[col_name])-min(df[col_name]))/(2*iqr(df[col_name])*np.power(len(df[col_name]), -1/5))))
        plt.title(col_name + " Attribute Distribution\nShimazaki and Shinomoto's choice")
    elif bin_type == 8:
        # Knuth's choice
        plt.hist(df[col_name], bins=int(np.log2(len(df[col_name]))))
        plt.title(col_name + " Attribute Distribution\nKnuth's binning")
    elif bin_type == 9:
        # Bayesian blocks
        plt.hist(df[col_name], bins="auto")
        plt.title(col_name + " Attribute Distribution\nBayesian blocks binning")
    else:
        pass

    plt.xlabel(col_name)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def plot_boxplot(df, col_name):

    sns.boxplot(x=df[col_name])
    plt.title(col_name + " Attribute Boxplot")
    plt.show()

print("Dataset imported successfully!")

