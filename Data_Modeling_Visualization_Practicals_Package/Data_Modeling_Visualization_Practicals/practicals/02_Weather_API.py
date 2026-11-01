import os, requests, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
D=Path("../datasets"); key=os.getenv("OPENWEATHER_API_KEY"); city="Pune"
if key:
 j=requests.get("https://api.openweathermap.org/data/2.5/weather",params={"q":city,"appid":key,"units":"metric"},timeout=20).json()
 df=pd.DataFrame([{"date":pd.Timestamp.now(),"temperature_c":j["main"]["temp"],"humidity":j["main"]["humidity"],"wind_speed":j["wind"]["speed"],"pressure":j["main"]["pressure"],"rain_mm":j.get("rain",{}).get("1h",0)}])
 print("Live API data")
else:
 df=pd.read_csv(D/"weather_sample.csv",parse_dates=["date"]); print("Offline sample data (set OPENWEATHER_API_KEY for live data)")
print(df); df.plot(x="date",y="temperature_c",marker="o",title="Temperature"); plt.tight_layout(); plt.show()
df.to_csv(D/"weather_result.csv",index=False)
