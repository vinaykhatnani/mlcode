import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
D=Path("../datasets"); df=pd.read_csv(D/"Retail_Sales_Data.csv",parse_dates=["Date"])
region=df.groupby("Region").Sales_Amount.sum().sort_values(ascending=False); print(region)
print(df.groupby("Product_Category").Sales_Amount.sum())
region.plot(kind="bar",title="Sales by Region"); plt.tight_layout(); plt.show()
pd.pivot_table(df,values="Sales_Amount",index="Region",columns="Product_Category",aggfunc="sum",fill_value=0).plot(kind="bar",stacked=True,title="Sales by Region and Category")
plt.tight_layout(); plt.show(); region.to_csv(D/"Region_Sales_Summary.csv")
