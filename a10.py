import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load high-dimensional dataset
data = load_breast_cancer()
X, y = data.data, data.target

print("Features before PCA:", X.shape[1])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Classification before PCA
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
accuracy_before = accuracy_score(y_test, model.predict(X_test))

# Apply PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("Features after PCA:", X_train_pca.shape[1])
print("Variance retained:", round(sum(pca.explained_variance_ratio_) * 100, 2), "%")

# Classification after PCA
model_pca = LogisticRegression(max_iter=1000)
model_pca.fit(X_train_pca, y_train)
accuracy_after = accuracy_score(y_test, model_pca.predict(X_test_pca))

print("Accuracy before PCA:", round(accuracy_before * 100, 2), "%")
print("Accuracy after PCA:", round(accuracy_after * 100, 2), "%")

# Visualize reduced dimensions
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization")
plt.colorbar(label="Class")
plt.show()