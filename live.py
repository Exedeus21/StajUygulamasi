
# Importing necessary libs
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pickle


x="https://www.springboard.com/library/static/f069aa8aaa8fd3e4b11fdc9a959fb4cb/801ec/mobile-app-vs-web-apps.jpg"
st.image(x,width=800)

df= pd.read_excel("sample data - rnv.xlsx",sheet_name="Ecommerce Customers")
df.drop(['Yearly Amount Spent','Email', 'Address', 'Avatar'],inplace=True, axis=1)


s1=st.sidebar.slider("Avg. Session Length",min_value=29.53, max_value=36.14)
s2=st.sidebar.slider("Time on App",min_value=8.51, max_value=15.13)
s3=st.sidebar.slider("Time on Website",min_value=33.91, max_value=40.01)
s4=st.sidebar.slider("Length of Membership",min_value=0.27, max_value=6.92)

col_list=["Avg. Session Length","Time on App","Time on Website","Length of Membership"]
l=[s1,s2,s3,s4]
df2=pd.DataFrame(data=[l],columns=col_list)

filename = 'finalized_model.model'
loaded_model = pickle.load(open(filename, 'rb'))

ypred = loaded_model.predict(df2)
st.title("Yearly Amoung Spent Prediction")
st.title(round(ypred[0][0],2))


