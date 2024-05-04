# k-means clustering
!pip install pandas

# importing the libraries  
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns 
# run before importing kmeans
import os
os.environ["OMP_NUM_THREADS"] = '1'

# importing the data set 
dataset = pd.read_csv('tripadvisor_review.csv')

dataset.head()

dataset.info()

dataset.describe()

sns.pairplot(dataset.iloc[:,[1,2,3,4,5,6,7,9,10]])


# min max scaling using iloc() function
from sklearn.preprocessing import StandardScaler
X = dataset.iloc[:, [2,5]].values
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# elbow method to assess the optimal number of clusters using K-means
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('the elbow method')
plt.xlabel('number of clusters')
plt.ylabel('WCSS')
plt.show()


# fitting k-means to the data set
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# visualising the clusters
plt.figure(figsize=(8,8))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0,1], s = 100, c = 'red', label = 'cluster 1')
plt.scatter(X[y_kmeans == 1, 1], X[y_kmeans == 1,1], s = 100, c = 'blue', label = 'cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2,1], s = 100, c = 'green', label = 'cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3,1], s = 100, c = 'cyan', label = 'cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4,1], s = 100, c = 'magenta', label = 'cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'centroids')
plt.title('clusters of ratings')
plt.xlabel('category 2 (scaled)')
plt.ylabel('category 5 (scaled)')
plt.legend()
plt.show()

# Clustering Data with Higher Dimensionality
# utilise previous work throughs to complete this meyhod of clustering
dataset.info()

dataset.describe()

dataset.head()

X = dataset.iloc[:,1:10]
sns.pairplot(X)


# scaling the data 
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# elbow method to assess the optimal number of clusters using clustering with higher dimensionality
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('the elbow method')
plt.xlabel('number of clusters')
plt.ylabel('WCSS')
plt.show()


# fitting k-means to the data set
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# reducing dimensionality before we can visualise

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

pca.explained_variance_ratio_

# sum of original variance explained by new dimensions

sum(pca.explained_variance_ratio_)

# visualising the clusters
colours = ['red', 'blue', 'green', 'cyan', 'magenta']
plt.figure(figsize=(8,8))
for i in range(3):
    plt.scatter(X_reduced[y_kmeans == i, 0], X_reduced[y_kmeans == i, 1],
           s = 100, c = colours [i], label = 'cluster' +str(i+1))
plt.title('clusters of customers')
plt.xlabel('dimension 1')
plt.ylabel('dimension 2')
plt.legend()
plt.show()
