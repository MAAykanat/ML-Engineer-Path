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

def one_hot_encoder(dataframe, categorical_columns, drop_first=True):
    """
    This function encodes the categorical variables to numericals.
    """
    """
    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    categorical_columns : list
        The name of the column to be encoded.
    drop_first : bool, optional
        The default is True.
    Returns
    -------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    """
    dataframe = pd.get_dummies(dataframe, columns=categorical_columns, drop_first=drop_first)
    return dataframe

df_titanic = dh.load_dataset("titanic.csv")
df_application = dh.load_dataset("application_train.csv")   

binarycol_titanic = check_binary_col(df_titanic)
binarycol_application = check_binary_col(df_application)

# print(check_binary_col(df_titanic))
print(df_application[binarycol_application].head())

#######################
#NOTES:
# We can use the label_encoder function to encode the binary columns.
# But, we should be careful about the binary columns.
# Because, some binary columns are not binary in fact.
# For example, the binary columns in the application_train dataset are not binary.
# They are categorical.
# So, we should encode them with label_encoder function.
# We can use the check_binary_col function to check the binary columns.
# Then, we can encode them with label_encoder function.
# LaberEncoder function includes also NONE value. CAREFUL!
#######################
for col in binarycol_application:
    df_application = label_encoder(df_application, col)

print(df_application[binarycol_application].head())

#######################
#Be NOTICE:
#######################
df_application=dh.load_dataset("application_train.csv")

print(df_application["EMERGENCYSTATE_MODE"].value_counts())
print(df_application["EMERGENCYSTATE_MODE"].nunique()) # 2 categories Yes and No
print(df_application["EMERGENCYSTATE_MODE"].unique()) # Be aware of NONE value

# Eliminate binart columns from the dataset and check the number of unique values in the remaining columns
ohe_cols = [col for col in df_application.columns if 10 >= df_application[col].nunique() > 2]

one_hot_encoder(df_application, ohe_cols, drop_first=True) #Result
