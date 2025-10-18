import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

# sns.boxplot(data=df)
# plt.show()
print(df.info())
print(df.shape)

q1=df["PM2.5 AQI Value"].quantile(0.25)
q3=df["PM2.5 AQI Value"].quantile(0.75)

iqr=q3-q1
min=q1-(1.5*iqr)
max=q3+(1.5*iqr)
filtered_data = df[(df["PM2.5 AQI Value"] >= min) & (df["PM2.5 AQI Value"] <= max)]
print(filtered_data.shape)

sns.boxplot(data=filtered_data)
plt.show()

q11=df["NO2 AQI Value"].quantile(0.25)
q21=df["NO2 AQI Value"].quantile(0.75)

iqr1=q21-q11
min1=q11-(1.5*iqr1)
max1=q21+(1.5*iqr1)

filtered_data1= filtered_data[(filtered_data["NO2 AQI Value"] >= min) & (filtered_data["NO2 AQI Value"] <= max)]
print(df.shape)
