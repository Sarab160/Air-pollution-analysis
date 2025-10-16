import pandas as pd
df=pd.read_csv("airpollution.csv")
#print(df)
# print(df.isnull().sum())

# condition=df[df["City"]=="Granville"]
# print(condition)

# print(df["Country"].mode())
# print(df.duplicated().sum())
df["Country"].fillna(df["Country"].mode()[0],inplace=True)
df["City"].fillna(df["City"].mode(),inplace=True)

print(df.isnull().sum())   

print(df.head())