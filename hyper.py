import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,OrdinalEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix

df=pd.read_csv("airpollution.csv")

# sns.pairplot(data=df)
# plt.show()

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

x_train,x_test,y_train,y_test=train_test_split(X_final,y,test_size=0.2,random_state=42)

knc=KNeighborsClassifier(n_neighbors=5)
knc.fit(x_train,y_train)

print(knc.score(x_train,y_train))
print(knc.score(x_test,y_test))

print(precision_score(y_test,knc.predict(x_test),average="macro"))
print(recall_score(y_test,knc.predict(x_test),average="macro"))
print(f1_score(y_test,knc.predict(x_test),average="macro"))

s=confusion_matrix(y_test,knc.predict(x_test))
sns.heatmap(data=s,annot=True)
plt.show()

param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10, 20, 30, 40, 50],
    'metric': ['minkowski', 'euclidean', 'manhattan'],
    'p': [1, 2]
}

gd=GridSearchCV(KNeighborsClassifier(),param_grid=param_grid)
gd.fit(x_train,y_train)

print(gd.best_score_)
print(gd.best_params_)