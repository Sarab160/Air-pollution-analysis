import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,OrdinalEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score


df=pd.read_csv("airpollution.csv")

print(df.columns)

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

print(filtered_data.head(10))
print(filtered_data.columns)

filtered_data = filtered_data.dropna(subset=[
    "AQI Value","CO AQI Value","Ozone AQI Value","NO2 AQI Value","PM2.5 AQI Value",
    "Country","City",
    "AQI Category","CO AQI Category","Ozone AQI Category","NO2 AQI Category",
    "PM2.5 AQI Category"
]).reset_index(drop=True)

x=filtered_data[["AQI Value","CO AQI Value","Ozone AQI Value","NO2 AQI Value","PM2.5 AQI Value"]]
le=LabelEncoder()
y=le.fit_transform(filtered_data["PM2.5 AQI Category"])

oe=OrdinalEncoder()
feature=filtered_data[["Country","City"]]
encode_data=oe.fit_transform(feature)
col=oe.get_feature_names_out(feature.columns)
encode_dataframe=pd.DataFrame(encode_data,columns=col)

X=pd.concat([x,encode_dataframe],axis=1)

fe1=filtered_data[["AQI Category","CO AQI Category","Ozone AQI Category","NO2 AQI Category"]]
oh=OneHotEncoder(sparse_output=False,drop="first")
encode=oh.fit_transform(fe1)
col1=oh.get_feature_names_out(fe1.columns)
data=pd.DataFrame(encode,columns=col1)

X_final=pd.concat([X,data],axis=1)

wcss=[]

for i in range(1,20):
    km=KMeans(n_clusters=i,init="k-means++")
    km.fit(X_final)
    wcss.append(km.inertia_)

plt.plot([i for i in range(1,20)],wcss,marker="o")
plt.show()

kmn=KMeans(n_clusters=3)
kmn.fit_predict(X_final)

ymeans=kmn.labels_
score=silhouette_score(X_final,ymeans)
print(score)
