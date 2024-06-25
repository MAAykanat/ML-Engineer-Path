import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import datetime as dt


from shutil import get_terminal_size
import warnings

from helpers import *
from display_helpers import *

from yellowbrick.cluster import KElbowVisualizer

from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None) # Show all the columns
pd.set_option('display.max_rows', None) # Show all the rows
pd.set_option('max_colwidth', None) # Show all the text in the columns
pd.set_option('display.float_format', lambda x: '%.3f' % x) # Show all the decimals
pd.set_option('display.width', get_terminal_size()[0]) # Get bigger terminal display width

df = pd.read_csv("Machine Learning/datasets/flo/flo_data_20K.csv")
print(df.head())

print(df.columns.str.contains("date"))

data_cols=df.columns[df.columns.str.contains("date")]
print(data_cols)
df[data_cols] = df[data_cols].apply(pd.to_datetime)

############################################
##### 1. EXPLORATORY DATA ANALYSIS - EDA ###
#### UNSUPERVISED LEARNING - CLUSTERING ####
"""
    Steps can be different for supervised and unsupervised learning.
    Since, There is no target variable
"""
############################################

# 1.1. General Picture of the Dataset
# 1.2. Catch Numeric and Categorical Value
# 1.3. Catetorical Variables Analysis
# 1.4. Numeric Variables Analysis
# 1.5. Outlier Detection
# 1.6. Missing Value Analysis

# 1.1. General Picture of the Dataset

check_df(df)

"""
##################### Types #####################
master_id                                    object
order_channel                                object
last_order_channel                           object
first_order_date                     datetime64[ns]
last_order_date                      datetime64[ns]
last_order_date_online               datetime64[ns]
last_order_date_offline              datetime64[ns]
order_num_total_ever_online                 float64
order_num_total_ever_offline                float64
customer_value_total_ever_offline           float64
customer_value_total_ever_online            float64
interested_in_categories_12                  object
dtype: object
"""

print("\n\n")
# 1.2. Catch Numeric and Categorical Value
"""
Observations: 19945
Variables: 12
categorical_cols: 2
num_cols: 8
categorical_but_cardinal: 2
numeric_but_categorical: 0
"""

cat_cols, num_cols, cat_but_car = grap_column_names(df)

print("Categorical Columns: \n\n", cat_cols)
print("Numeric Columns: \n\n", num_cols)
[print("Categorical but Cardinal EMPTY!!!\n\n") if cat_but_car == [] else print("Categorical but Cardinal: \n", cat_but_car)]
print("#"*50)

# 1.3. Catetorical Variables Analysis

for col in cat_cols:
    cat_summary(df,col, plot=False)

print("#"*50)

# 1.4. Numerical Variable Analysis

for col in num_cols:
    numerical_col_summary(df,num_cols,plot=False)
print("#"*50)

# 1.5. Outlier Detection

"""
Have Outliers: order_num_total_ever_online
Have Outliers: order_num_total_ever_offline
Have Outliers: customer_value_total_ever_offline
Have Outliers: customer_value_total_ever_online

"""

for col in num_cols:
    print(col, ":", check_outlier(df, col))

print("#"*50)

# 1.6. Missing Value Analysis

"""
    There is no missing value in the dataset.
"""

for col in df.columns:
    print(col, ":", df[col].isnull().sum())

print("#"*50)

##################################
##### 2. FEATURE ENGINEERING #####
##################################
# 2.1. Missing Values
# 2.2. Outlier Values Analysis
# 2.3. Feature Generation
# 2.4. Encoding
# 2.5. Standardization
# 2.6. Save the Dataset

# 2.1. Missing Values
    
"""
    There is no missing value in the dataset.
"""

# 2.2. Outlier Values Analysis
"""
Outliers are handled with the threshold values. IQR method is used.
"""

print(df["order_num_total_ever_offline"].unique())

for col in num_cols:
    replace_with_thresholds(df, col)

print("#"*50)

for col in num_cols:
    print(col, ":", check_outlier(df, col))

print(df["order_num_total_ever_offline"].unique())


print("#"*50)

# 2.3. Feature Generation

print(df["last_order_date"].max()) # 2021-05-30
analysis_date = dt.datetime(2021,6,1)
# analysis_date = dt.datetime.now() # More accurate if SQL is exist


df["recency"] = (analysis_date-df["first_order_date"]).astype('timedelta64[D]')
df["tenure"] = (df["last_order_date"]-df["first_order_date"]).astype('timedelta64[D]')

cat_cols, num_cols, cat_but_car = grap_column_names(df)

print("Categorical Columns: \n\n", cat_cols)
print("Numeric Columns: \n\n", num_cols)
[print("Categorical but Cardinal EMPTY!!!\n\n") if cat_but_car == [] else print("Categorical but Cardinal: \n", cat_but_car)]

cat_cols = [col for col in cat_cols if col not in ["order_num_total_ever_offline"]]
num_cols = num_cols + ["order_num_total_ever_offline"]
print(num_cols)
# 2.4. Encoding

"""
No need for label encoding (No binary columns)
"""
binary_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtypes == "O"]

print("Binary Columns: \n\n", binary_cols)

print(cat_cols)

df=one_hot_encoder(df, cat_cols, drop_first=True)

print(df.head())
print(df.shape)

# 2.5. Standardization

# Check the skewness of the data

df.reset_index(inplace=True)

model_df = df[["order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure"]]

def check_skew(df_skew, column, plot=False):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column],color = "g")
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    if plot:
        plt.show()
    return

plt.figure(figsize=(9, 9))
plt.subplot(6, 1, 1)
check_skew(model_df,'order_num_total_ever_online')
plt.subplot(6, 1, 2)
check_skew(model_df,'order_num_total_ever_offline')
plt.subplot(6, 1, 3)
check_skew(model_df,'customer_value_total_ever_offline')
plt.subplot(6, 1, 4)
check_skew(model_df,'customer_value_total_ever_online')
plt.subplot(6, 1, 5)
check_skew(model_df,'recency')
plt.subplot(6, 1, 6)
check_skew(model_df,'tenure')
plt.tight_layout()
plt.show()

# Normalization of the data by using log transformation
for col in model_df.columns:
    df[col] = np.log1p(df[col])

print(df.head())

# Scaling

scaler = MinMaxScaler((0, 1))
model_scaling = scaler.fit_transform(model_df)
model_df=pd.DataFrame(model_scaling,columns=model_df.columns)
print(model_df.head())

# 2.6. Save the Dataset

# df.to_csv("Machine Learning/datasets/flo/flo_data_20K_cleaned.csv", index=False)

########################
##### 3. MODELLING #####
########################
model_df = df[["order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure"]]

# 3.1. K-Means Clustering
# 3.2. Hierarchical Clustering
# 3.3. DBSCAN Clustering
# 3.4. Evaluation of the Clusters

# 3.1. K-Means Clustering

# 3.1.1. Optimum Number of Clusters
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(model_df)
elbow.show()

# 3.1.2. Model Building
k_means = KMeans(n_clusters=7, random_state=42).fit(model_df)
segment = k_means.labels_
print(segment)

final_df = df[["master_id","order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure"]]
final_df["segment"] = segment
print(final_df.head())

agg_df = final_df.groupby("segment").agg({"order_num_total_ever_online":["mean","min","max"],
                                  "order_num_total_ever_offline":["mean","min","max"],
                                  "customer_value_total_ever_offline":["mean","min","max"],
                                  "customer_value_total_ever_online":["mean","min","max"],
                                  "recency":["mean","min","max"],
                                  "tenure":["mean","min","max","count"]})
agg_df.reset_index(inplace=True)

print(agg_df)

# 3.2. Hierarchical Clustering
# 3.2.1. Optimum Number of Clusters

hc_complete = linkage(model_df, 'complete')

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_complete,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=1.2, color='r', linestyle='--')
plt.show()

