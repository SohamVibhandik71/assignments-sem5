
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load and preprocess dataset
data = load_iris()
X = StandardScaler().fit_transform(data.data)

# Elbow Method
wcss = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    wcss.append(model.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Dendrogram (sample 50 rows for simplicity)
Z = linkage(X[:50], method='ward')
dendrogram(Z)
plt.title("Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

# Apply K-Means and Hierarchical Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
k_labels = kmeans.fit_predict(X)

hierarchy = AgglomerativeClustering(n_clusters=3)
h_labels = hierarchy.fit_predict(X)

# Visualize clusters using first two features
plt.scatter(X[:, 0], X[:, 1], c=k_labels)
plt.title("K-Means Clustering")
plt.xlabel("Sepal Length (scaled)")
plt.ylabel("Sepal Width (scaled)")
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=h_labels)
plt.title("Hierarchical Clustering")
plt.xlabel("Sepal Length (scaled)")
plt.ylabel("Sepal Width (scaled)")
plt.show()

print("K-Means cluster sizes:")
print(__import__('numpy').bincount(k_labels))

print("\nHierarchical cluster sizes:")
print(__import__('numpy').bincount(h_labels))
