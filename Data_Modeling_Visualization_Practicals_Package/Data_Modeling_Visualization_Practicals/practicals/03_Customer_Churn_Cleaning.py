import pandas as pd
from pathlib import Path
D=Path("../datasets"); df=pd.read_csv(D/"Telecom_Customer_Churn.csv")
print("Before:",df.shape); print(df.isnull().sum()); df=df.drop_duplicates()
for c in df.select_dtypes("number"): df[c]=df[c].fillna(df[c].median())
for c in df.select_dtypes("object"): df[c]=df[c].fillna(df[c].mode()[0])
df.columns=df.columns.str.strip().str.replace(" ","_",regex=False)
print("After:",df.shape); print(df.Churn.value_counts()); df.to_csv(D/"Telecom_Customer_Churn_Cleaned.csv",index=False)
