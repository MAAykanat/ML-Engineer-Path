import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler

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

def label_encoder(dataframe, binary_col):
    """
    This function encodes the binary variables to numericals.

    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    binary_col : str
        The name of the column to be encoded.
    Returns
    -------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    """
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    print(binary_col, "is encoded.")
    return dataframe

def one_hot_encoder(dataframe, categorical_columns, drop_first=True):
    """
    This function encodes the categorical variables to numericals.

    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    categorical_columns : list
        The name of the column to be encoded.
    drop_first : bool, optional
        Dummy trap. The default is True.
    Returns
    -------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    """
    dataframe = pd.get_dummies(dataframe, columns=categorical_columns, drop_first=drop_first)
    return dataframe

def telco_data_prep(dataframe):
    target=["CHURN"]

    dataframe.columns = [col.upper() for col in dataframe.columns]
    
    dataframe["TOTALCHARGES"] = pd.to_numeric(dataframe["TOTALCHARGES"], errors="coerce")
    dataframe["CHURN"] = dataframe["CHURN"].apply(lambda x: 1 if x =="Yes" else 0)

    dataframe.drop("CUSTOMERID", axis=1, inplace=True)

    cat_cols, num_cols, cat_but_car = grap_column_names(dataframe)

    for col in num_cols:
        replace_with_thresholds(dataframe, col)
        dataframe[col].fillna(dataframe[col].median(), inplace=True)
    
    ########################
    ### FEATURE CREATION ###
    ########################
    # Creating an annual categorical variable from a tenure variable
    dataframe.loc[(dataframe["TENURE"]>=0) & (dataframe["TENURE"]<=12),"NEW_TENURE_YEAR"] = "0-1 Year"
    dataframe.loc[(dataframe["TENURE"]>12) & (dataframe["TENURE"]<=24),"NEW_TENURE_YEAR"] = "1-2 Year"
    dataframe.loc[(dataframe["TENURE"]>24) & (dataframe["TENURE"]<=36),"NEW_TENURE_YEAR"] = "2-3 Year"
    dataframe.loc[(dataframe["TENURE"]>36) & (dataframe["TENURE"]<=48),"NEW_TENURE_YEAR"] = "3-4 Year"
    dataframe.loc[(dataframe["TENURE"]>48) & (dataframe["TENURE"]<=60),"NEW_TENURE_YEAR"] = "4-5 Year"
    dataframe.loc[(dataframe["TENURE"]>60) & (dataframe["TENURE"]<=72),"NEW_TENURE_YEAR"] = "5-6 Year"

    # Specify customers with 1 or 2 years of contract as Engaged
    dataframe["NEW_ENGAGED"] = dataframe["CONTRACT"].apply(lambda x: 1 if x in ["One year","Two year"] else 0)

    # People who do not receive any support, backup or protection
    dataframe["NEW_NOPROT"] = dataframe.apply(lambda x: 1 if (x["ONLINEBACKUP"] != "Yes") or (x["DEVICEPROTECTION"] != "Yes") or (x["TECHSUPPORT"] != "Yes") else 0, axis=1)

    # Customers who have monthly contracts and are young
    dataframe["NEW_YOUNG_NOT_ENGAGED"] = dataframe.apply(lambda x: 1 if (x["NEW_ENGAGED"] == 0) and (x["SENIORCITIZEN"] == 0) else 0, axis=1)

    # Total number of services received by the person
    dataframe['NEW_TOTALSERVICES'] = (dataframe[['PHONESERVICE', 'INTERNETSERVICE', 'ONLINESECURITY',
                                        'ONLINEBACKUP', 'DEVICEPROTECTION', 'TECHSUPPORT',
                                        'STREAMINGTV', 'STREAMINGMOVIES']]== 'Yes').sum(axis=1)

    # People who receive any streaming service
    dataframe["NEW_FLAG_ANY_STREAMING"] = dataframe.apply(lambda x: 1 if (x["STREAMINGTV"] == "Yes") or (x["STREAMINGMOVIES"] == "Yes") else 0, axis=1)

    # Does the person make automatic payments?
    dataframe["NEW_FLAG_AUTOPAYMENT"] = dataframe["PAYMENTMETHOD"].apply(lambda x: 1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)

    # Average monthly payment
    dataframe["NEW_AVG_CHARGES"] = dataframe["TOTALCHARGES"] / (dataframe["TENURE"] + 1)

    # Rate of change the current price compared to the average price
    dataframe["NEW_INCREASE"] = dataframe["NEW_AVG_CHARGES"] / dataframe["MONTHLYCHARGES"]

    # Fee per service
    dataframe["NEW_AVG_SERVICE_FEE"] = dataframe["MONTHLYCHARGES"] / (dataframe['NEW_TOTALSERVICES'] + 1)

    cat_cols, num_cols, cat_but_car = grap_column_names(dataframe)

    ###ENCODING###
    binary_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtypes == "O"]

    for col in binary_cols:
        df = label_encoder(df, col)
    
    cat_cols = [col for col in cat_cols if col not in binary_cols and col not in target]
    df = one_hot_encoder(df, cat_cols, drop_first=True)

    ###STANDARDIZATION###
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])


