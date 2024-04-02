import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Machine Learning/datasets/hitter/hitters.csv")
print(df.head())

#######################################
### EXPLORATORY DATA ANALYSIS - EDA ###
#######################################

# 1. General Picture of the Dataset
# 2. Catch Numeric and Categorical Value
# 3. Catetorical Variables Analysis
# 4. Numeric Variables Analysis
# 5. Target Variable Analysis (Dependent Variable) - Categorical
# 6. Target Variable Analysis (Dependent Variable) - Numeric
# 7. Outlier Detection
# 8. Missing Value Analysis
# 9. Correlation Matrix

# 1. General Picture of the Dataset

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)
# Shape: (322,20)
# Target: Salary (Numeric) - float64
# League, Division, NewLeague (Categorical) - object
# Others: 16 Numeric - int64 

# Only salary has missing values (59)
# There can be outliers!!!

# 2. Catch Numeric and Categorical Value
"""
Observations: 322
Variables: 20
categorical_cols: 3
num_cols: 17
categorical_but_cardinal: 0
numeric_but_categorical: 0
"""

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

cat_cols, num_cols, cat_but_car = grap_column_names(df)

print("Categorical Columns: \n\n", cat_cols)
print("Numeric Columns: \n\n", num_cols)
[print("Categorical but Cardinal EMPTY!!!\n\n") if cat_but_car == [] else print("Categorical but Cardinal: \n", cat_but_car)]
print("#"*50)

# 3. Catetorical Variables Analysis

def cat_summary(dataframe, col_name, plot=False):
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

for col in cat_cols:
    cat_summary(df,col)
print("#"*50)

# 4. Numeric Variables Analysis

def numerical_col_summary(dataframe, col_name, plot=False):

    """
    This function shows the frequency of numerical variables.

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
    print(dataframe[col_name].describe([0.01, 0.05, 0.75, 0.90, 0.99]).T)
    print("##########################################")
    if plot:
        sns.histplot(dataframe[col_name], kde=True)
        plt.xlabel(col_name)
        plt.title(f"{col_name} Distribution")
        plt.show()

for col in num_cols:
    numerical_col_summary(df,col)
print("#"*50)

# 5. Target Variable Analysis (Dependent Variable) - Categorical
"""
It looks Division has significant effect on Salary, but League and NewLeague does not.
"""

def target_summary_with_cat(dataframe, target, categorical_col):
    """
    This function shows the mean of the target variable according to the categorical variable.

    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    target : str
        The name of the target variable.
    categorical_col : str
        The name of the categorical variable.
    Returns
    -------
    None.
    """
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Salary", col)

print("#"*50)

# 6. Target Variable Analysis (Dependent Variable) - Numeric
"""
Nonsensical results. Since, target variable is also numeric.

"""

def target_summary_with_num(dataframe, target, numerical_col):
    """
    This function shows the average of numerical variables according to the target variable.
    
    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    target : str
        The name of the target variable.
    numerical_col : str
        The name of the numerical variable.
    Returns
    -------
    None.
    """
    print(dataframe.groupby(target).agg({numerical_col: ["mean", "median", "count"]}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Salary", col)
print("#"*50)

# 7. Outlier Detection

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    """
    This function calculates the lower and upper limits for the outliers.

    Calculation:
    Interquantile range = q3 - q1
    Up limit = q3 + 1.5 * interquantile range
    Low limit = q1 - 1.5 * interquantile range

    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    col_name : str
        The name of the column to be analyzed.
    q1 : float, optional
        The default is 0.05.
    q3 : float, optional
        The default is 0.95.
    Returns
    -------
    low_limit, up_limit : float
        The lower and upper limits for the outliers.
    """

    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)

    interquantile_range = quartile3 - quartile1

    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range

    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    """
        This function checks dataframe has outlier or not.

    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    col_name : str
        The name of the column to be analyzed.
    Returns
    -------
    bool
        True if the dataframe has outlier, False otherwise.
    """

    lower_limit, upper_limit = outlier_thresholds(dataframe=dataframe, col_name=col_name)

    if dataframe[(dataframe[col_name] > upper_limit) | (dataframe[col_name] < lower_limit)].any(axis=None):
        print(f'{col_name} have outlier')
        return True
    else:
        return False
    
def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    """
    This function replaces the outliers with the lower and upper limits.

    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    variable : str
        The name of the column to be analyzed.
    q1 : float, optional
        The default is 0.05.
    q3 : float, optional
        The default is 0.95.
    Returns 
    -------
    None
    """

    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# There is no outlier in any numeric variables.
for col in num_cols:
    print(f"{col}: {check_outlier(df, col)}")

print("#"*50)

# 8. Missing Value Analysis

print(df.isnull().sum())

def missing_values_table(dataframe, null_columns_name = False):
    """
    This function returns the number and percentage of missing values in a dataframe.
    """
    """
    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    null_columns_name : bool, optional
        The default is False.
    Returns
    -------
    missing_values_table : pandas dataframe
        A dataframe that contains the number and percentage of missing values in the dataframe.
    """
    # Calculate total missing values in each column
    null_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    number_of_missing_values = dataframe[null_columns].isnull().sum().sort_values(ascending=False)
    percentage_of_missing_values = (dataframe[null_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_values_table = pd.concat([number_of_missing_values, np.round(percentage_of_missing_values, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_values_table)

    if null_columns_name:
        return null_columns  

missing_values_table(df, True)

# There are 59 missing values in only Salary column.

df["Salary"].fillna(df["Salary"].mean(), inplace=True)

missing_values_table(df, True)

print("#"*50)

# 9. Correlation Matrix

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    """
    This function returns the columns that have a correlation higher than the threshold value.

    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    plot : bool, optional
        The default is False.
    corr_th : float, optional
        The default is 0.90.
    Returns
    -------
    drop_list : list
        The list of columns that have a correlation higher than the threshold value.
    """
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

df_corr = df.corr()

f, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df_corr, annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Heatmap", color="black", fontsize=20)
plt.show()