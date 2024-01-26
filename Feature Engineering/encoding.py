import dataset_handle as dh
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def label_encoder(dataframe, column_name):
    """
    This function encodes the categorical variables to numericals.
    """
    """
    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    column_name : str
        The name of the column to be encoded.
    """
    le = LabelEncoder().fit(dataframe[column_name])
    dataframe[column_name] = le.transform(dataframe[column_name])
    return dataframe

def check_binary_col(dataframe):
    """
    This function checks the binary columns in the dataset.
    """
    """
    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    """
    binary_columns = [col for col in dataframe.columns if dataframe[col].nunique() == 2 and dataframe[col].dtypes == "O"]
    return binary_columns

df_titanic = dh.load_dataset("titanic.csv")

print(df_titanic.head())

le=LabelEncoder()
le.fit_transform(df_titanic["Sex"])

print(check_binary_col(df_titanic))