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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

df = pd.read_csv(DATA / "car_safety_demo.csv")
X = df.drop(columns=["safety"]); y = df["safety"]
cat_cols = ["road_type", "weather"]
preprocess = ColumnTransformer([("categorical", OneHotEncoder(handle_unknown="ignore"), cat_cols)], remainder="passthrough")
model = make_pipeline(preprocess, RandomForestClassifier(n_estimators=150, random_state=42))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Random Forest Accuracy:", round(accuracy_score(y_test, pred)*100, 2), "%")
print("\\nClassification Report:\\n", classification_report(y_test, pred))
ConfusionMatrixDisplay.from_predictions(y_test, pred); plt.title("Car Safety Confusion Matrix"); plt.show()
new_car = pd.DataFrame({"speed_kmph":[55], "driver_age":[30], "experience_years":[8], "road_type":["City"], "weather":["Clear"]})
print("New car safety prediction:", model.predict(new_car)[0])
