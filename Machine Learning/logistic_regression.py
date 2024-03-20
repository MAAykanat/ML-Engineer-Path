import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import missingno as msno

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report


pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv('Machine Learning/datasets/diabetes.csv')

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

# 2. Catch Numeric and Categorical Value

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

# Only one Categorical Variable Outcome (TARGET)

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

# No NEED, Only TARGET variable is Categorical

 # 6. Target Variable Analysis (Dependent Variable) - Numeric
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

# There is OUTLIER --> Insulin
for col in num_cols:
    print(f"{col}: {check_outlier(df, col)}")

for col in df.columns:
    # Check outliers and replace with thresholds if there is any
    print(f"{col}: {check_outlier(df, col)}")
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

# There is OUTLIER --> Insulin
for col in num_cols:
    print(f"{col}: {check_outlier(df, col)}")
print("#"*50)

# 8. Missing Value Analysis
"""
There is no missing value, but there is something wrong Insulin glucose cannot be zero.
It should be fixed and changed to NaN
"""
# Convert 0 to NaN

counter = df.shape[0]

for col in num_cols:
    for i in range(counter):
        if df[col][i] == 0:
            df[col][i] = None

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
    """
    # Calculate total missing values in each column
    null_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    number_of_missing_values = dataframe[null_columns].isnull().sum().sort_values(ascending=False)
    percentage_of_missing_values = (dataframe[null_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_values_table = pd.concat([number_of_missing_values, np.round(percentage_of_missing_values, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_values_table)
    
    if null_columns_name:
        return null_columns  

na_col = missing_values_table(dataframe=df, null_columns_name=True)

print(na_col)

# Filling Missing Values
# We will fill in the missing values with the median of the target-Outcome (0-1) variable.

for col in num_cols:
    df.loc[(df[col].isnull()) & (df["Outcome"]==0), col] = df.groupby("Outcome")[col].mean()[0]
    df.loc[(df[col].isnull()) & (df["Outcome"]==1), col] = df.groupby("Outcome")[col].mean()[1]

print(df.head())

msno.matrix(df)
plt.title("Missing Values Matrix - After Filling")
# plt.show()

# 9. Correlation Matrix

df_corr = df.corr()

f, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df_corr, annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Heatmap", color="black", fontsize=20)
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
"""
    It has been filled in the previous section. EDA - 8. Missing Value Analysis
    df is filled with the mean of the target variable.
"""

# 2. Outlier Values Analaysis
"""
    It has been filled in the previous section. EDA - 7. Missing Value Analysis
    df is filled with the mean of the target variable.
"""

# 3. Feature Generation
# Generate new features from existing ones.

# New Category - NEW_AGE_CAT - Numeric to Categorical
df.loc[((df["Age"] >= 20) & (df["Age"] < 50) ), "NEW_AGE_CAT"] = "mature"
df.loc[((df["Age"] >= 50)), "NEW_AGE_CAT"] = "senior"

# New Category - NEW_BMI_LEVEL - Numeric to Categorical
df["NEW_BMI_LEVEL"] = pd.cut(df["BMI"], [0, 18.5, 24.9, 29.9, 34.9, 100], labels=["Underweight", "Normal", "Overweight", "Obese", "Extremely Obese"])

# New Category - NEW_GLUCOSE - Numeric to Categorical
df["NEW_GLUCOSE_CAT"] = pd.cut(df["Glucose"], [0, 100, 125, df["Glucose"].max()], labels=["Normal", "Prediabetes" ,"Diabetes"])

# Age-BMI Interaction
df.loc[(df["BMI"]<18.5) & ((df["Age"] >= 20) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "UnderweightMature"
df.loc[(df["BMI"]<18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "UnderweightSenior"

df.loc[(df["BMI"]>=18.5) & (df["BMI"]<24.9) & ((df["Age"] >= 20) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "NormalMature"
df.loc[(df["BMI"]>=18.5) & (df["BMI"]<24.9) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "NormalSenior"

df.loc[(df["BMI"]>=24.9) & (df["BMI"]<29.9) & ((df["Age"] >= 20) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "OverweightMature"
df.loc[(df["BMI"]>=24.9) & (df["BMI"]<29.9) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "OverweightSenior"

df.loc[(df["BMI"]>=29.9) & (df["BMI"]<34.9) & ((df["Age"] >= 20) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "ObeseMature"
df.loc[(df["BMI"]>=29.9) & (df["BMI"]<34.9) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "ObeseSenior"

df.loc[(df["BMI"]>=34.9) & (df["BMI"]<100) & ((df["Age"] >= 20) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "ExtremelyObeseMature"
df.loc[(df["BMI"]>=34.9) & (df["BMI"]<100) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "ExtremelyObeseSenior"

# Age-Glucoce Interaction
df.loc[(df["Glucose"]<100) & ((df["Age"] >= 20) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "NormalMature"
df.loc[(df["Glucose"]<100) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "NormalSenior"

df.loc[(df["Glucose"]>=100) & (df["Glucose"]<125) & ((df["Age"] >= 20) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "PrediabetesMature"
df.loc[(df["Glucose"]>=100) & (df["Glucose"]<125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "PrediabetesSenior"

df.loc[(df["Glucose"]>=125) & (df["Age"] >= 20) & (df["Age"] < 50), "NEW_AGE_GLUCOSE_NOM"] = "DiabetesMature"
df.loc[(df["Glucose"]>=125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "DiabetesSenior"

def set_insulin(dataframe, col= "Insulin"):
    # Binary Category - Insulin
    if 16 <= dataframe[col] <= 166:
        return "Normal"
    else:
        return "Abnormal"
    
df["NEW_INSULIN_CAT"] = df.apply(set_insulin, axis=1)
df["NEW_INSULIN*GLUCOSE"] = df["Insulin"] * df["Glucose"]

print(df.head())
print(df.shape)

#Take New Coloumns 
cat_cols, num_cols, cat_but_car = grap_column_names(df)

# 4. Encoding
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

binary_col = [col for col in df.columns if df[col].dtypes=='O' and df[col].nunique() == 2]

print("BINARY COLS",binary_col)

for col in binary_col:
    label_encoder(df, col)

# 4.2 One-Hot Encoding

# Catch Categorical Variables After Binary Coloumns
cat_cols = [col for col in cat_cols if col not in binary_col and col not in ["Outcome"]]
print(cat_cols)

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

df = one_hot_encoder(df, cat_cols, drop_first=True)

print(df.head())

# 5. Standardization
# We will standardize the variables to make robust to model.

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
print(df.head())

 # 6. Save Dataset
# df.to_csv("Machine Learning/datasets/diabetes_cleaned.csv", index=False)

#######################################
#### MODEL BUILDING AND EVALUATION ####
#######################################
# We will build a model and evaluate the model.
def plot_confusion_matrix(y, y_pred):

    acc = round(accuracy_score(y, y_pred), 3)
    cm = confusion_matrix(y, y_pred)

    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred', size=10)
    plt.ylabel('y', size=10)
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

# 1. Train-Test Split
# 2. Model Building
# 3. Model Evaluation

# 1. Train-Test Split
# We will split the dataset into two parts as train and test.

df_cleaned = pd.read_csv("Machine Learning/datasets/diabetes_cleaned.csv")
print(df_cleaned.head())

y = df_cleaned["Outcome"]
X = df_cleaned.drop(["Outcome"], axis=1)

# Hold-out Method
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=17)

# Model creation
log_model = LogisticRegression().fit(X=X_train, y=y_train)

print("Bias: \n",log_model.intercept_)
print("\nWeights: \n",log_model.coef_)

# Prediction
y_pred = log_model.predict(X_test)

print(classification_report(y_true=y_test, y_pred=y_pred))

# xticklabels=["Not Diabetes", "Diabetes"], 
                # yticklabels=["Not Diabetes", "Diabetes"] -- Can Be Added

# plot_confusion_matrix(y_test, y_pred)


