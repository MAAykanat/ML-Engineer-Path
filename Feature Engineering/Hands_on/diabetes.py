import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import missingno as msno

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
# pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

PATH ="D:\!!!MAAykanat Dosyalar\Miuul\Feature Engineering\Görevler\Görev-1 Diabetes Dataset"
df=pd.read_csv(PATH + "\diabetes.csv")

print(df.head())

#######################################
### EXPLORATORY DATA ANALYSIS - EDA ###
#######################################

# There are 6 steps to be taken in the EDA process.

# 1. General Picture of the Dataset
# 2. Catch Numeric and Categorical Values
# 3. Categoical Variable Analysis
# 4. Numeric Variable Analysis
# 5. Target Variable Analysis (Dependent Variable) - Numerical
# 6. Korrelation Analysis

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
print("#"*50)
# 2. Catch Numeric and Categorical Values

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

print("Categorical Columns: ", cat_cols)
print("Numeric Columns: ", num_cols)
[print("Categorical but Cardinal EMPTY!!!") if cat_but_car == [] else print("Categorical but Cardinal: ", cat_but_car)]
print("#"*50)

# 3. Categoical Variable Analysis
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

# 4. Numeric Variable Analysis
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

# 5. Target Variable Analysis (Dependent Variable) - Numerical

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
    target_summary_with_num(df, "Outcome", col)
print("#"*50)

# 6. Korrelation Analysis

# Correlation Matrix (Heatmap)

df_corr = df.corr()

f, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df_corr, annot=True, fmt=".2f", ax=ax, cmap="viridis")
ax.set_title("Correlation Heatmap", color="blue", fontsize=20)
# plt.show()

#######################################
######### FEATURE ENGINEERING #########
#######################################

# There are 6 steps to be taken in the Feature Engineering process.
# 1. Missing Values
# 2. Outlier Values Analysis
# 3. Feature Generation
# 4. Encoding
# 5. Standardization
# 6. Save the Dataset

# 1. Missing Values
# 1.1 Find Missing Value Table
df_copy = df.copy()

# Insulin, SkinThickness, BloodPressure, BMI, Glucose, DiabetesPedigreeFunction, Age
# These columns have 0 values. They cannot be 0. We will convert them to NaN.

print("Before Zero-NaN:\n ", df_copy.head())


#######################################
# First Method - Convert 0 to NaN
"""
counter = df_copy.shape[0]

for col in num_cols:
    for i in range(counter):
        if df_copy[col][i] == 0:
            df_copy[col][i] = None
"""
#######################################
# Second Method - Convert 0 to NaN
zero_columns = [col for col in df_copy.columns if (df_copy[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]
print(zero_columns)

for col in zero_columns:
    df_copy[col] = np.where(df_copy[col] == 0, np.nan, df_copy[col])
#######################################
print("After Zero-NaN:\n ", df_copy.head())
# print(df_copy.head())

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
    """
    # Calculate total missing values in each column
    null_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    number_of_missing_values = dataframe[null_columns].isnull().sum().sort_values(ascending=False)
    percentage_of_missing_values = (dataframe[null_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_values_table = pd.concat([number_of_missing_values, np.round(percentage_of_missing_values, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_values_table)
    
    if null_columns_name:
        return null_columns  

na_col = missing_values_table(dataframe=df_copy, null_columns_name=True)

# 1.2 Missing Values - Target Variable Relationship
# We will examine the relationship between the target variable and the missing values.
# If there is a relationship, we will fill in the missing values with the median of the target variable.

def missing_vs_target(dataframe, target, na_columns):
    """
    This function examines the relationship between the target variable and the missing values.
    
    Alternative of Library: missingno.matrix(dataframe)

    Parameters
    ----------
    dataframe : pandas dataframe
        The dataframe to be analyzed.
    target : str
        The name of the target variable.
    na_columns : list
        The name of the columns to be analyzed.
    Returns
    -------
    None.

    """
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(dataframe=df_copy, target="Outcome", na_columns=na_col)

msno.matrix(df_copy)
plt.title("Missing Values Matrix- Before Filling")
# plt.show()

# 1.3 Missing Values - Filling
# We will fill in the missing values with the median of the target-Outcome (0-1) variable.

for col in num_cols:
    # Fill all null values with mean of target variable (Outcome)
    df_copy.loc[(df_copy[col].isnull()) & (df_copy["Outcome"]==0), col] = df_copy.groupby("Outcome")[col].mean()[0]
    df_copy.loc[(df_copy[col].isnull()) & (df_copy["Outcome"]==1), col] = df_copy.groupby("Outcome")[col].mean()[1]
print(df_copy.head())

msno.matrix(df_copy)
plt.title("Missing Values Matrix- After Filling")
# plt.show()

# 2. Outlier Values Analysis
# We will detect and suppres the outliers in the dataset.
# Possibble solutions: 3 sigma, boxplot, IQR, Z-score, Local Outlier Factor (LOF), Isolation Forest, One-Class SVM, Minimum Covariance Determinant (MCD), Robust Covariance
# We will use the IQR method.

#######################################
########## IMPORTANT Note #############
# 3 Functions are written for Outlier Analysis.
# outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95) : This function calculates the lower and upper limits for the outliers.
# check_outlier(dataframe, col_name) : This function checks dataframe has outlier or not.
# replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95) : This function replaces the outliers with the lower and upper limits.
 

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

for col in df_copy.columns:
    # Check outliers and replace with thresholds if there is any
    print(f"{col}: {check_outlier(df_copy, col)}")
    if check_outlier(df_copy, col):
        replace_with_thresholds(df_copy, col)

print("#"*50)

for col in df_copy.columns:
    print(f"{col}: {check_outlier(df_copy, col)}")

# 3. Feature Generation
# We will generate new features from the existing features.
# We will use the domain knowledge and the relationship between the variables to generate new features.
# We will create new variables by using the mathematical operations on the variables.

# New Category - NEW_AGE_CAT - Numeric to Categorical
df_copy.loc[((df_copy["Age"] >= 20) & (df_copy["Age"] < 50) ), "NEW_AGE_CAT"] = "mature"
df_copy.loc[((df_copy["Age"] >= 50)), "NEW_AGE_CAT"] = "senior"

# New Category - NEW_BMI_LEVEL - Numeric to Categorical
df_copy["NEW_BMI_LEVEL"] = pd.cut(df_copy["BMI"], [0, 18.5, 24.9, 29.9, 34.9, 100], labels=["Underweight", "Normal", "Overweight", "Obese", "Extremely Obese"])

# New Category - NEW_GLUCOSE - Numeric to Categorical
df_copy["NEW_GLUCOSE_CAT"] = pd.cut(df_copy["Glucose"], [0, 100, 125, df_copy["Glucose"].max()], labels=["Normal", "Prediabetes" ,"Diabetes"])

# Age-BMI Interaction
df_copy.loc[(df_copy["BMI"]<18.5) & ((df_copy["Age"] >= 20) & (df_copy["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "UnderweightMature"
df_copy.loc[(df_copy["BMI"]<18.5) & (df_copy["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "UnderweightSenior"

df_copy.loc[(df_copy["BMI"]>=18.5) & (df_copy["BMI"]<24.9) & ((df_copy["Age"] >= 20) & (df_copy["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "NormalMature"
df_copy.loc[(df_copy["BMI"]>=18.5) & (df_copy["BMI"]<24.9) & (df_copy["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "NormalSenior"

df_copy.loc[(df_copy["BMI"]>=24.9) & (df_copy["BMI"]<29.9) & ((df_copy["Age"] >= 20) & (df_copy["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "OverweightMature"
df_copy.loc[(df_copy["BMI"]>=24.9) & (df_copy["BMI"]<29.9) & (df_copy["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "OverweightSenior"

df_copy.loc[(df_copy["BMI"]>=29.9) & (df_copy["BMI"]<34.9) & ((df_copy["Age"] >= 20) & (df_copy["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "ObeseMature"
df_copy.loc[(df_copy["BMI"]>=29.9) & (df_copy["BMI"]<34.9) & (df_copy["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "ObeseSenior"

df_copy.loc[(df_copy["BMI"]>=34.9) & (df_copy["BMI"]<100) & ((df_copy["Age"] >= 20) & (df_copy["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "ExtremelyObeseMature"
df_copy.loc[(df_copy["BMI"]>=34.9) & (df_copy["BMI"]<100) & (df_copy["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "ExtremelyObeseSenior"

# Age-Glucoce Interaction
df_copy.loc[(df_copy["Glucose"]<100) & ((df_copy["Age"] >= 20) & (df_copy["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "NormalMature"
df_copy.loc[(df_copy["Glucose"]<100) & (df_copy["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "NormalSenior"

df_copy.loc[(df_copy["Glucose"]>=100) & (df_copy["Glucose"]<125) & ((df_copy["Age"] >= 20) & (df_copy["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "PrediabetesMature"
df_copy.loc[(df_copy["Glucose"]>=100) & (df_copy["Glucose"]<125) & (df_copy["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "PrediabetesSenior"

df_copy.loc[(df_copy["Glucose"]>=125) & (df_copy["Age"] >= 20) & (df_copy["Age"] < 50), "NEW_AGE_GLUCOSE_NOM"] = "DiabetesMature"
df_copy.loc[(df_copy["Glucose"]>=125) & (df_copy["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "DiabetesSenior"

def set_insulin(dataframe, col= "Insulin"):
    # Binary Category - Insulin
    if 16 <= dataframe[col] <= 166:
        return "Normal"
    else:
        return "Abnormal"
    
df_copy["NEW_INSULIN_CAT"] = df_copy.apply(set_insulin, axis=1)
df_copy["NEW_INSULIN*GLUCOSE"] = df_copy["Insulin"] * df_copy["Glucose"]

print(df_copy.head())
print(df_copy.shape)

# 4. Encoding
print("Old DataFrame")
cat_cols, num_cols, cat_but_car = grap_column_names(df)
print("New DataFrame")
cat_cols_copy, num_cols_copy, cat_but_car_copy = grap_column_names(df_copy)

# 4.1 Label Encoding

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
    return dataframe

# 2 Possible Solution to find Binary Columns
# binary_col = [col for col in df_copy.columns if df_copy[col].dtypes=='O' and df_copy[col].nunique() == 2]
binary_col = [col for col in cat_cols_copy if df_copy[col].nunique() == 2 and col not in ["Outcome"]]
print(binary_col)

print("Before Label Encoder:\n",df_copy.head())

for col in binary_col:
    label_encoder(df_copy, col)

print("After Label Encoder:\n",df_copy.head())

cat_cols_copy = [col for col in cat_cols_copy if col not in binary_col and col not in ["Outcome"]]
print(cat_cols_copy)

# 4.2 One-Hot Encoding
# We will use the get_dummies method to perform one-hot encoding.

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

df_copy = one_hot_encoder(df_copy, cat_cols_copy, drop_first=True)

print(df_copy.head())
print(df_copy.shape)

# 5. Standardization
# We will standardize the variables to make robust to model.

print(num_cols_copy)

scaler = StandardScaler()
df_copy[num_cols_copy] = scaler.fit_transform(df_copy[num_cols_copy])
print(df_copy.head())

# 6. Save the Dataset
df_copy.to_csv(PATH + "\diabetes_cleaned.csv", index=False)

#######################################
#### MODEL BUILDING AND EVALUATION ####
#######################################
# We will build a model and evaluate the model.

# 1. Train-Test Split
# 2. Model Building
# 3. Model Evaluation

# 1. Train-Test Split
# We will split the dataset into two parts as train and test.

y = df_copy["Outcome"]
X = df_copy.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

# 2. Model Building
# We will build a model using the train dataset.

rf_model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# 3. Model Evaluation
# We will evaluate the model using the test dataset.

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

def plot_importance(model, features, num=len(X), save=False):
    """
        Show to feature importance of the model.

    Parameters
    ----------
    model : model
        The model to be analyzed.
    features : pandas dataframe
        The dataframe to be analyzed.
    num : int, optional
        The default is len(X).
    save : bool, optional
        To save the plot.
        The default is False.
    Returns
    -------
    None.
    """

    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

# plot_importance(rf_model, X)
print("#"*50)

#######################################
############ COMPARISON ###############
#######################################
# We will compare CREATED MODELS from the old and new datasets.


# 1. Train-Test Split
# We will split the dataset into two parts as train and test.

y_original = df["Outcome"]
X_original = df.drop(["Outcome"], axis=1)

X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(X_original, y_original, test_size=0.20, random_state=17)

# 2. Model Building
# We will build a model using the train dataset.

rf_model_original = RandomForestClassifier(random_state=42).fit(X_train_original, y_train_original)
y_pred_original = rf_model_original.predict(X_test_original)

# 3. Model Evaluation
# We will evaluate the model using the test dataset.

print(f"Accuracy: {round(accuracy_score(y_pred_original, y_test_original), 2)}")
print(f"Recall: {round(recall_score(y_pred_original, y_test_original),3)}")
print(f"Precision: {round(precision_score(y_pred_original, y_test_original), 2)}")
print(f"F1: {round(f1_score(y_pred_original, y_test_original), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred_original, y_test_original), 2)}")
