# Install once if needed:
# pip install pandas numpy matplotlib scikit-learn
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA = Path("datasets")
if not DATA.exists():
    DATA = Path("../datasets")

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv(DATA / "kmeans_points.csv")
X = df[["X","Y"]]
model = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = model.fit_predict(X)
print("Cluster labels:", labels)
print("Cluster centres:\\n", model.cluster_centers_)
print("Silhouette score:", round(silhouette_score(X, labels), 3))
plt.scatter(X["X"], X["Y"], c=labels, cmap="viridis", s=80)
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], c="red", marker="X", s=200, label="Centroids")
plt.xlabel("X"); plt.ylabel("Y"); plt.title("K-Means Clustering"); plt.legend(); plt.show()
