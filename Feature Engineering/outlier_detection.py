import numpy as np

import dataset_handle as dh

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    # Outlier thresholds for any attribute
    # Interquantile range = q3 - q1
    # Up limit = q3 + 1.5 * interquantile range
    # Low limit = q1 - 1.5 * interquantile range
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)

    interquantile_range = quartile3 - quartile1

    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range

    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    lower_limit, upper_limit = outlier_thresholds(dataframe=dataframe, col_name=col_name)

    if dataframe[(dataframe[col_name] > upper_limit) | (dataframe[col_name] < lower_limit)].any(axis=None):
        print(f'{col_name} have outlier')
        return True
    else:
        return False

def grap_column_names(dataframe, categorical_th=10, cardinal_th=20):
    """
    It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
    Note: Categorical variables with numerical appearance are also included."""

    """
    Cardinal Variables: Variables that are categorical and do not carry information,
    that is, have too many classes, are called variables with high cardinality.
    """

    """
    Returns
    ------
        categorical_cols: list
                Categorical variable list
        num_cols: list
                Numeric variable list
        categorical_but_cardinal: list
                Categorical variables with high cardinality list
    """
    # categorical_cols, categorical_but_cardinal
    categorical_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    numeric_but_categorical = [col for col in dataframe.columns if dataframe[col].nunique() < categorical_th and
                   dataframe[col].dtypes != "O"]
    categorical_but_cardinal = [col for col in dataframe.columns if dataframe[col].nunique() > cardinal_th and
                   dataframe[col].dtypes == "O"]
    categorical_cols = categorical_cols + numeric_but_categorical
    categorical_cols = [col for col in categorical_cols if col not in categorical_but_cardinal]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in numeric_but_categorical]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'categorical_cols: {len(categorical_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'categorical_but_cardinal: {len(categorical_but_cardinal)}')
    print(f'numeric_but_categorical: {len(numeric_but_categorical)}')
    return categorical_cols, num_cols, categorical_but_cardinal

def grap_outliers(dataframe, col_name, index=False):
    # It graps and prints the outliers in the data set.
    # If index = True, it returns the indexes of the outliers.
    """Returns
    ------
        outlier_index: list
                List of outlier indexes
    """
    low_limit, up_limit = outlier_thresholds(dataframe=dataframe, col_name=col_name)

    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].shape[0] > 10:
        print(dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].head())
    else:
        print(dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)])
    if index:
        outlier_index = dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].index
        return outlier_index

def remove_outliers(dataframe, col_name):
    # Removes outliers from the data set.
    """Returns
    ------
        df_without_outliers: dataframe
                Dataframe without outliers
    """
    low_limit, up_limit = outlier_thresholds(dataframe=dataframe, col_name=col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]

    return df_without_outliers

def replace_with_thresholds(dataframe, variable):

    low_limit, up_limit = outlier_thresholds(dataframe=dataframe, col_name=variable)

    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def categorical_col_summary(dataframe, col_name, plot=False):
    """
    This function shows the frequency of categorical variables.

    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    col_name : str
        The name of the column to be analyzed.
    plot : bool, optional
        The default is False.
    Returns
    -------
    None.
    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

df_titanic = dh.load_dataset("titanic.csv")

dh.dataset_details(df_titanic)
#############################################
# Outlier Detection with Graphs
# Boxplot
# Histogram
#############################################
# Boxplot is a graphical method to visualize the dhstribution of data based on
# the five-number summary: minimum, first quartile, medhan, third quartile, and maximum.
#############################################
# dh.plot_boxplot(df_titanic, "Fare")
# dh.plot_hist(df_titanic, "Fare")
#############################################
# Outlier Detection
#############################################

low, up = outlier_thresholds(df_titanic, "Fare")
print("Low limit: ", low)
print("Up limit: ", up)

print(df_titanic[(df_titanic["Age"] < low)])
categorical_cols, num_cols, categorical_but_cardinal = grap_column_names(df_titanic)
outlier_index = grap_outliers(df_titanic, "Age", index=True)
print(len(outlier_index))


# Grap and remove outliers
# Crate new data set without outliers
for col in num_cols:
    print(col, grap_outliers(df_titanic, col, index=True))
    new_df_titanic = remove_outliers(df_titanic, col)

print(df_titanic.shape[0] - new_df_titanic.shape[0])

# Replace outliers with thresholds
for col in num_cols:
    print(col, grap_outliers(df_titanic, col, index=True))
    check_outlier(df_titanic, col)
    replace_with_thresholds(df_titanic, col)

check_outlier(df_titanic, "Age")


#############################################
# Multivariate Outlier Analysis: Local Outlier Factor
#############################################
# Local Outlier Factor (LOF) is a score that tells us how isolated a certain data point is based on the
# density of the local neighborhood of the data point. The higher the LOF value for an observation, the more
# likely it is to be an outlier.
#############################################

from sklearn.neighbors import LocalOutlierFactor

# df_titanic = dh.load_dataset("titanic.csv")
df_diamons = sns.load_dataset("diamonds")

df_diamons = df_diamons.select_dtypes(include=["float64", "int64"])
df_diamons = df_diamons.dropna()

lof = LocalOutlierFactor(n_neighbors=20)
lof.fit_predict(df_diamons)

df_scores = lof.negative_outlier_factor_
np.sort(df_scores)[0:30]

# Plotting the outlier scores
# Elbow analysis
# Plot to define the threshold value
scores= pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0,20], style=".-")
# plt.show() # Exaple: in 3th index, the value is -4.98 and there is elbow. Highly steep slope.

th = np.sort(df_scores)[3]

df_diamons[df_scores < th]

print(df_diamons.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T) # 0.01, 0.05, 0.75, 0.90, 0.99 are quantiles

print(df_diamons[df_scores < th].index) # Outlier index according to LOF

print(df_diamons[df_scores < th].drop(axis=0, labels = df_diamons[df_scores < th].index)) # Drop outliers according to LOF