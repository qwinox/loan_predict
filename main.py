import streamlit as st

import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

import matplotlib.pyplot as plt

import pydotplus as pypi
from IPython.display import Image

def predict(input):
       filename = 'https://raw.githubusercontent.com/qwinox/aml/main/loan.csv'
       
       dat = pd.read_csv(filename, sep = ',')
       
       dat.head(15)
       
       df_encoded = dat.copy()
       df_encoded.shape
       
       df_encoded.isnull().sum()
       
       df_encoded.Gender.value_counts()
       
       df_encoded.head(15)
       
       
       le = LabelEncoder()
       
       df_encoded['Gender'] = le.fit_transform(df_encoded['Gender'].values)
       df_encoded['Married'] = le.fit_transform(df_encoded['Married'].values)
       df_encoded['Dependents'] = le.fit_transform(df_encoded['Dependents'].values)
       df_encoded['Self_Employed'] = le.fit_transform(df_encoded['Self_Employed'].values)
       df_encoded['Education'] = le.fit_transform(df_encoded['Education'].values)
       df_encoded['Property_Area'] = le.fit_transform(df_encoded['Property_Area'].values)
       df_encoded['Loan_Status'] = le.fit_transform(df_encoded['Loan_Status'].values)
       
       df_encoded.head(15)
       
       df_encoded.isnull().sum()
       
       df_encoded.columns
       
       feature_cols= ['Gender', 'Married', 'Dependents', 'Education',
              'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
              'Loan_Amount_Term', 'Credit_History', 'Property_Area']
       
       X = df_encoded[feature_cols]
       Y = df_encoded['Loan_Status']
       
       X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.1,random_state=1)
       
       fig, ax = plt.subplots(figsize=(20, 10))
       tree.plot_tree(clf, fontsize=10, filled=True, rounded=True)
       
       model = BaggingClassifier(clf)
       model.fit(X_train, y_train)
       model.score(X_test, y_test)

       return model.predict(input)

st.title("Модель предсказания возможности получения займа")

user_input = st.text_input("Введите текст", "Введите описание изображения")

button_pressed = st.button("Распознать")

if button_pressed:
       user_input = pd.DataFrame({'Gender' : [1], 'Married' : 	[0], 'Dependents' : [0], 'Education' : 	[0], 'Self_Employed' : 	[0], 'ApplicantIncome' : [5849], 'CoapplicantIncome' :	[0.0], 'LoanAmount' : [146.412162], 'Loan_Amount_Term' :	[360.0], 'Credit_History' : [1.0], 'Property_Area' : [2]})
       st.write("Класс изображения:", predict(user_input)[0])
