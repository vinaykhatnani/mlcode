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
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv(DATA / "uber_fares_demo.csv")
print("Dataset preview:\n", df.head())
X = df[["distance_km", "passenger_count", "pickup_hour"]]
y = df["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
models = {"Linear Regression": LinearRegression(), "Ridge": Ridge(alpha=1.0), "Lasso": Lasso(alpha=0.1)}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\\n{name}")
    print("MAE:", round(mean_absolute_error(y_test, pred), 2))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 2))
    print("R2:", round(r2_score(y_test, pred), 3))
model = models["Linear Regression"]
sample = pd.DataFrame({"distance_km":[5], "passenger_count":[2], "pickup_hour":[18]})
print("\\nPredicted sample fare:", round(model.predict(sample)[0], 2))
plt.scatter(y_test, model.predict(X_test), alpha=.6)
plt.xlabel("Actual fare"); plt.ylabel("Predicted fare"); plt.title("Actual vs Predicted Uber Fare")
plt.show()
