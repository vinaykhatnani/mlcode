# Install once if needed:
# pip install pandas numpy matplotlib scikit-learn
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA = Path("datasets")
if not DATA.exists():
    DATA = Path("../datasets")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

df = pd.read_csv(DATA / "handwritten_digits.csv")
X = df.drop(columns=["label"]); y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=5, gamma="scale"))
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("SVM Accuracy:", round(accuracy_score(y_test, pred)*100, 2), "%")
print("\\nClassification Report:\\n", classification_report(y_test, pred))
ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap="Blues")
plt.title("SVM Confusion Matrix"); plt.show()
sample = X_test.iloc[[0]]
print("Actual digit:", y_test.iloc[0], "| Predicted digit:", model.predict(sample)[0])
