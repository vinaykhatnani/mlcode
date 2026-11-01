import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
D=Path("../datasets"); df=pd.read_csv(D/"RealEstate_Prices.csv")
df.columns=df.columns.str.strip().str.lower().str.replace(" ","_",regex=False); df=df.drop_duplicates()
df["price"]=df.price.fillna(df.price.median()); df["price_per_sqft"]=df.price/df.area_sqft
df=pd.get_dummies(df,columns=["location","property_type"],dtype=int)
print(df.head()); df.groupby("bedrooms").price.mean().plot(kind="bar",title="Average Price by Bedrooms")
plt.tight_layout(); plt.show(); df.to_csv(D/"RealEstate_Prices_Cleaned.csv",index=False)
