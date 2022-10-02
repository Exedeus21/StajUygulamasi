# -*- coding: utf-8 -*-
#%% Importing Library
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import re

#%% Read Data
df= pd.read_excel(r"C:\Users\PC\Desktop\sample data - rnv.xlsx",sheet_name="Ecommerce Customers")

#%% Exploratory Data Analysis

print(df.info())
ddf=df.describe()

df["State"]="Null"
for i in range(len(df)):
    df["State"].iloc[i]=re.findall("(?:^|\s)[A-Z]{2}(?:^|\s)",df["Address"].iloc[i])
df['State'] = df['State'].str.get(0)

plt.figure(figsize=(15,10))
df["State"].value_counts().plot(kind="barh")

sns.pairplot(df)

f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)},figsize=(15,15))
sns.boxplot(df["Avg. Session Length"], ax=ax_box)
sns.histplot(data=df, x="Avg. Session Length", ax=ax_hist)

f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)},figsize=(15,15))
sns.boxplot(df["Time on App"], ax=ax_box)
sns.histplot(data=df, x="Time on App", ax=ax_hist)


f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)},figsize=(15,15))
sns.boxplot(df["Time on Website"], ax=ax_box)
sns.histplot(data=df, x="Time on Website", ax=ax_hist)

f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)},figsize=(15,15))
sns.boxplot(df["Length of Membership"], ax=ax_box)
sns.histplot(data=df, x="Length of Membership", ax=ax_hist)

f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)},figsize=(15,15))
sns.boxplot(df["Yearly Amount Spent"], ax=ax_box)
sns.histplot(data=df, x="Yearly Amount Spent", ax=ax_hist)

#%% Outlier Handling
for x in df.columns:
    if df[x].dtypes in ["int64","float64"]:
        Q1= np.percentile(df[x],25)
        Q3=np.percentile(df[x],75)
        IQR = Q3 -Q1
        out1 = Q1 - 1.5*IQR
        out2 = 1.5*IQR + Q3
        for i in range(len(df[x])):
            if df[x].iloc[i] < out1:
                df[x].iloc[i] = out1
            elif df[x].iloc[i] > out2:
                df[x].iloc[i] = out2

#%% Separating Train Test Datasets
y = df[['Yearly Amount Spent']]
X = df.drop(['Yearly Amount Spent','Email', 'Address', 'Avatar','State'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

#%% Lineer Regresyon Modelinin Kurulması
from sklearn.linear_model import LinearRegression
lim=LinearRegression()
lim.fit(X_train,y_train)
pred2=lim.predict(X_test)

print('MAE',mean_absolute_error(y_test,pred2))
print('MSE',mean_squared_error(y_test,pred2))
print('RMSE',np.sqrt(mean_squared_error(y_test,pred2)))
print("R2",r2_score(y_test,pred2))

#%% Coeff matris bulma
coefdf=pd.DataFrame(lim.coef_[0],['Avg. Session Length', 'Time on App', 'Time on Website','Length of Membership'])

coefdf.plot(kind='barh', figsize=(9, 7))
plt.title("Coefficiatn Matrix",fontsize=30)

# save the model to disk
import pickle

filename = 'finalized_model.model'
pickle.dump(lim, open(filename, 'wb'))