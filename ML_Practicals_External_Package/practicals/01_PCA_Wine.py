# Install once if needed:
# pip install pandas numpy matplotlib scikit-learn
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA = Path("datasets")
if not DATA.exists():
    DATA = Path("../datasets")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv(DATA / "wine.csv")
X = df.drop(columns=["target"])
y = df["target"]
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Original shape:", X.shape)
print("Reduced shape:", X_pca.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance retained:", round(pca.explained_variance_ratio_.sum()*100, 2), "%")
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="viridis")
plt.xlabel("Principal Component 1"); plt.ylabel("Principal Component 2")
plt.title("PCA on Wine Dataset"); plt.colorbar(label="Wine class"); plt.show()
