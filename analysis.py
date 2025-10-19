import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score,confusion_matrix,precision_score,f1_score
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

# sns.boxplot(data=filtered_data)
# plt.show()

# q11=df["NO2 AQI Value"].quantile(0.25)
# q21=df["NO2 AQI Value"].quantile(0.75)

# iqr1=q21-q11
# min1=q11-(1.5*iqr1)
# max1=q21+(1.5*iqr1)

# filtered_data1= filtered_data[(filtered_data["NO2 AQI Value"] >= min) & (filtered_data["NO2 AQI Value"] <= max)]
# print(df.shape)
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

x_train,x_test,y_train,y_test=train_test_split(X_final,y,test_size=0.2,random_state=42)

lr=LogisticRegression()
lr.fit(x_train,y_train)

print(lr.score(x_test,y_test))