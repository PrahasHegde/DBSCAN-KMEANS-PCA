import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA



iris = sns.load_dataset('iris')
print(iris.head())
print(iris.shape)
print(iris.info())

# #plots wrt to different features
# sns.scatterplot(iris, x='sepal_length', y='sepal_width', hue='species')
# plt.show()

# sns.scatterplot(iris, x='petal_length', y='petal_width', hue='species')
# plt.show()

# sns.scatterplot(iris, x='sepal_length', y='petal_length', hue='species')
# plt.show()

# sns.scatterplot(iris, x='sepal_width', y='petal_width', hue='species')
# plt.show()

#I picked up two features from the “Iris” dataset and trained a KMeans model giving the n_clusters parameter the value of 3
X = iris[['sepal_length', 'petal_length']].values
km = KMeans(n_clusters=3)
km.fit(X)
print(km.predict(X))

#instead of giving the hue parameter the “species” columns from the “Iris” dataset we are going to give it the km.predict(X)
#clusters made by our KMeans model

sns.scatterplot(iris, x='sepal_length', y='petal_length', hue=km.predict(X))
plt.show()



#let’s try DBSCAN and see what type of visualization it will give.
dbscan = DBSCAN(eps=1, min_samples=6)
dbscan.fit(X)
print(dbscan.labels_)

sns.scatterplot(iris, x='sepal_length', y='petal_length', hue=dbscan.labels_)
plt.show()

"""first we will chose a random point in the plane then considering that random point as center 
we will draw a 1cm circle (that is the value of our eps see the parameter in the DBSCAN()).
After drawing the circle we will check how many points are there within the circle if there are 6 points within the circle
(that is the value of our min_samples see the parameter in the DBSCAN()) Those 6 points forms a cluster. And it will not stop there"""


#PCA (Principle Component Analysis)
"""As our goal with PCA is to reduce the dimension choosing a dataset with more features would be the right choice -> MNIST """

mnist = fetch_openml('mnist_784', version=1)

X = mnist.data / 255.0  # Scale pixel values to between 0 and 1
y = mnist.target

# Reduce dimensionality using PCA
pca = PCA(n_components=2)  # Reduce to 2 principal components
X_pca = pca.fit_transform(X)
print(X_pca)


#MNIST Dataset Scatter Plot (PCA)
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y.astype(int), cmap='tab10', alpha=0.5)
plt.colorbar(label='Digit Label')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('MNIST Dataset Scatter Plot (PCA)')
plt.grid(True)
plt.show()
