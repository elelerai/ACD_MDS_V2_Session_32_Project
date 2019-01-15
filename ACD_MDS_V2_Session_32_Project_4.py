# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 20:12:39 2019

@author: Eliud Lelerai
"""
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('data_stocks.csv')

dataset.head()
X = dataset.iloc[:, 1:].values
dataset1= dataset.iloc[:, 1:]


# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X = pca.fit_transform(X)

df_comp = pd.DataFrame(pca.components_,columns=dataset1.columns)
import seaborn as sns

plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma')

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
# Within cluster sum of square or SSE
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
