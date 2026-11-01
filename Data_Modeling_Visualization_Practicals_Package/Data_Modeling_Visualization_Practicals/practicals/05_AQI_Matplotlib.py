import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
D=Path("../datasets"); df=pd.read_csv(D/"City_Air_Quality.csv",parse_dates=["Date"])
print(df.describe()); plt.plot(df.Date,df.AQI); plt.title("AQI Trend"); plt.xticks(rotation=45); plt.tight_layout(); plt.show()
df[["PM2_5","PM10","CO"]].plot(title="Pollutant Levels"); plt.tight_layout(); plt.show()
plt.scatter(df.PM2_5,df.AQI); plt.title("PM2.5 vs AQI"); plt.xlabel("PM2.5"); plt.ylabel("AQI"); plt.show()
