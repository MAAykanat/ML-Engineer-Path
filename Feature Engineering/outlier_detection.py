import dataset_handle as dh
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
    
    if dataframe[dataframe[col_name] <lower_limit] | dataframe[dataframe[col_name] > upper_limit]:
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

grap_column_names(df_titanic)
outlier_index = grap_outliers(df_titanic, "Age", index=True)
print(len(outlier_index))
# print(df_titanic[(df_titanic["Fare"] > up) | (df_titanic["Fare"] < low)])