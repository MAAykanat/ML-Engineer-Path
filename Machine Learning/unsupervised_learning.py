import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

from yellowbrick.cluster import KElbowVisualizer



df = pd.read_csv("Machine Learning/datasets/USArrests.csv", index_col=0)

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

# 50 samples 4 columns

# 2. Standartization

scaler = MinMaxScaler()
df = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=4, random_state=17).fit(df) 

print(kmeans.get_params())
print(kmeans.n_iter_)
print(kmeans.cluster_centers_)
print(kmeans.labels_) # 4 cluster labels for 50 samples
print(kmeans.inertia_)

# 3. Determining the Optimum Number of Clusters


################################
# To see the elbow point plot for the optimum number of clusters

kmeans = KMeans()
ssd = []

for i in range(1,30):
    kmeans = KMeans(n_clusters=i).fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(range(1,30), ssd, "bx-")
plt.ylabel("SSE/SSR/SSD")
plt.xlabel("Different K Values")
plt.title("Elbow Method for Optimum Number of Clusters")
# plt.show()
################################

###############################
##### YELLOWBRICK LIBRARY #####
# To see more clear elbow point plot for the optimum number of clusters

kmeans = KMeans(random_state=17)
elbow = KElbowVisualizer(kmeans, k=(2,20)) # k is the range of the number of clusters
elbow.fit(df)
elbow.show()

print(elbow.elbow_value_)
print(elbow.elbow_score_)
print(elbow.estimator)

# 4. Creating the Final Clusters

kmeans = KMeans(n_clusters=elbow.elbow_value_, random_state=17).fit(df)

print("Number of Clusters: \n",kmeans.n_clusters)
print("Center of cluster: \n",kmeans.cluster_centers_)
print("Output labels of samples: \n",kmeans.labels_)

cluster_kmeans = kmeans.labels_

df = pd.read_csv("Machine Learning/datasets/USArrests.csv", index_col=0)

df["cluster"] = cluster_kmeans
# print(df.head())

df["cluster"] = df["cluster"] + 1

# print(df[df["cluster"]==5])

print(df.groupby("cluster").agg(["count","mean","median"]))

# 5. Visualizing the Clusters

centroid = kmeans.cluster_centers_

centroids_x = centroid[:,0]
centroids_y = centroid[:,1]

# plt.scatter(df.iloc[:,0], df.iloc[:,1], c=cluster_kmeans, s=50, cmap="viridis")
sns.scatterplot(data=df, x="Murder", y="Assault", hue=kmeans.labels_)
plt.scatter(centroids_x, centroids_y, 
            marker="X", c="r", s=80, label="centroids")
plt.legend()
plt.show()

###########################
# Hierarchical Clustering #
###########################

df = pd.read_csv("Machine Learning/datasets/USArrests.csv", index_col=0)

scaler = MinMaxScaler((0,1))
df = scaler.fit_transform(df)

hc_average = linkage(df, "average")

print(hc_average)

plt.figure(figsize=(10,5))
plt.title("Hierarcical Denogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=20,
           show_contracted=True,
           leaf_font_size=10)
plt.show(block=True)

############################
# Hierarchical Clustering  #
# Decision on # of Cluster #
############################

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.axhline(y=0.6, color='b', linestyle='--')
plt.show()

###########################
# Hierarchical Clustering #
####### Final Model #######
###########################

"""
    Question: Why Kmeans and Hierarchical Clustering are not the same?
    Answer: Because the algorithms are different. Kmeans is a centroid-based algorithm,
    Hierarchical Clustering is a distance-based algorithm.
"""

df = pd.read_csv("Machine Learning/datasets/USArrests.csv", index_col=0)

cluster = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="average")

cluster = cluster.fit_predict(df)

print(cluster)
print(cluster.shape)

df["Kmeans_Cluster"] = cluster_kmeans

df["Kmeans_Cluster"] = df["Kmeans_Cluster"] + 1   


df["Hierarchical_Cluster"] = cluster
df["Hierarchical_Cluster"] = df["Hierarchical_Cluster"] + 1


print(df.head())