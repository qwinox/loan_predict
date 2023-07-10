import streamlit as st

import os
import pandas as pd
import numpy as np

import sklearn

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


import matplotlib.pyplot as plt

import pydotplus as pypi
from IPython.display import Image

def predict(input):
       filename = 'https://raw.githubusercontent.com/qwinox/aml/main/loan.csv'
       dat = pd.read_csv(filename, sep = ',')

       df_encoded = dat.copy()
       df_encoded.Gender.value_counts()
       le = LabelEncoder()
       
       df_encoded['Gender'] = le.fit_transform(df_encoded['Gender'].values)
       df_encoded['Married'] = le.fit_transform(df_encoded['Married'].values)
       df_encoded['Dependents'] = le.fit_transform(df_encoded['Dependents'].values)
       df_encoded['Self_Employed'] = le.fit_transform(df_encoded['Self_Employed'].values)
       df_encoded['Education'] = le.fit_transform(df_encoded['Education'].values)
       df_encoded['Property_Area'] = le.fit_transform(df_encoded['Property_Area'].values)
       df_encoded['Loan_Status'] = le.fit_transform(df_encoded['Loan_Status'].values)
       
       feature_cols= ['Gender', 'Married', 'Dependents', 'Education',
              'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
              'Loan_Amount_Term', 'Credit_History', 'Property_Area']
       
       X = df_encoded[feature_cols]
       Y = df_encoded['Loan_Status']
       
       X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.1,random_state=1)
       
       # fig, ax = plt.subplots(figsize=(20, 10))
       # tree.plot_tree(clf, fontsize=10, filled=True, rounded=True)

       np.random.seed(42)

       clf = DecisionTreeClassifier(criterion='gini')
       clf = clf.fit(X_train, y_train)
       y_pred = clf.predict(X_test)
       
       r2 = accuracy_score(y_test, y_pred)
       
       model = BaggingClassifier(clf)
       model.fit(X_train, y_train)
       model.score(X_test, y_test)

       return model.predict(input)

st.title("Модель предсказания возможности получения займа в банке")

text_input = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

options = ['Мужчина', 'Женщина']
text_input[0] = st.selectbox('Ваш пол?', options)
text_input[0] = (1.0 if text_input[0].lower() == 'мужчина' else 0.0)

options = ['Да', 'Нет']
text_input[1] = st.selectbox("Состоите в браке?", options)
text_input[1] = (1.0 if text_input[1].lower() == 'да' else 0.0)

text_input[2] = st.text_input("Сколько у Вас детей?", "Нужно ввести количестов")

options = ['Да', 'Нет']
text_input[3] = st.selectbox("Есть ли у Вас высшее образование?", options)
text_input[3] = (1.0 if text_input[3].lower() == 'да' else 0.0)

options = ['Да', 'Нет']
text_input[4] = st.selectbox("Вы самозанятый?", options)
text_input[4] = (1.0 if text_input[4].lower() == 'да' else 0.0)


text_input[5] = st.text_input("Какой у Вас месячный доход в рублях?", "Нужно ввести количестов")
text_input[6] = st.text_input("Какой у Вашего созаявителя месячный доход в рублях?", "Нужно ввести количестов")
text_input[7] = st.text_input("Cколько рублей вы хотели бы взять?", "Нужно ввести количестов")
text_input[8] = st.text_input("На сколько дней хотели бы взять займ?", "Нужно ввести количестов")

options = ['Да', 'Нет']
text_input[9] = st.selectbox("Есть ли у Вас кредитная история?", options)
text_input[9] = (1.0 if text_input[9].lower() == 'да' else 0.0)


options = ['Город', 'Село', "Посёлок городского типа"]
text_input[10] = st.selectbox("Где вы проживаете?", options)

if text_input[10] == "город":
       text_input[10] = 2
elif text_input[10] == "село":
       text_input[10] = 0
else:
       text_input[10] = 1

button_pressed = st.button("Рассчитать возможность получения займа")

if button_pressed:
       if text_input[2].isdigit() and text_input[5].isdigit() and text_input[6].isdigit() and text_input[7].isdigit():
              text_input[2] = int(text_input[2])
              text_input[5] = float(text_input[5]) * 12 // 91.26
              text_input[6] = float(text_input[6]) * 12 // 91.26
              text_input[7] = float(text_input[7]) // 91.26
       
       
                     
              
              user_input = pd.DataFrame({'Gender' : [text_input[0]], 'Married' : [text_input[1]], 'Dependents' : [text_input[2]], 'Education' : [text_input[3]], 
                                         'Self_Employed' : [text_input[4]], 'ApplicantIncome' : [text_input[5]], 
                                         'CoapplicantIncome' : [text_input[6]], 'LoanAmount' : [text_input[7]], 
                                         'Loan_Amount_Term' : [text_input[8]], 'Credit_History' : [text_input[9]], 'Property_Area' : [text_input[10]]})
       
              if predict(user_input)[0] == 1:
                     st.subheader("Модель одобрила Вам займ!")
                     st.balloons()
              else:
                     st.subheader("Модель выявила, что по какому-то из показателей вы нежелательный заёмщик")
       else:
              st.subheader("В поля ввода текст данные введены неправильно")

       st.write("Введённые вами данные:")
       st.write(user_input)
