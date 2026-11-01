import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
D=Path("../datasets")
a=pd.read_csv(D/"sales_data.csv"); b=pd.read_excel(D/"sales_data.xlsx"); c=pd.read_json(D/"sales_data.json")
df=pd.concat([a,b,c],ignore_index=True).drop_duplicates()
df["Sales_Amount"]=df.Quantity*df.Unit_Price
print(df.head()); print(df.isnull().sum()); print(df.groupby("Product").Sales_Amount.sum())
df.groupby("Product").Sales_Amount.sum().plot(kind="bar",title="Sales by Product"); plt.tight_layout(); plt.show()
df.to_csv(D/"sales_cleaned.csv",index=False)
