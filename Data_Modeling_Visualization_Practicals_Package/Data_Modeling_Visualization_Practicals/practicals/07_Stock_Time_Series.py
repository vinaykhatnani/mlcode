import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
D=Path("../datasets"); df=pd.read_csv(D/"Stock_Prices.csv",parse_dates=["Date"]).set_index("Date").sort_index()
df["MA_7"]=df.Close.rolling(7).mean(); df["MA_30"]=df.Close.rolling(30).mean()
print(df.head()); print(df[["Close","Volume"]].corr())
df[["Close","MA_7","MA_30"]].plot(title="Stock Price and Moving Averages",figsize=(10,5)); plt.tight_layout(); plt.show()
df.Close.resample("ME").mean().plot(kind="bar",title="Monthly Average Close"); plt.tight_layout(); plt.show()
print("Simple next-period baseline:",round(df.Close.tail(7).mean(),2))
