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

text_input[0] = (1.0 if st.text_input("Введите пол?", "Нужно ввести Мужчина или Женщина").lower() == 'мужчина' else 0.0)
text_input[1] = (1.0 if st.text_input("Состоите в браке?", "Нужно ввести Да или Нет").lower() == 'да' else 0.0)
text_input[2] = st.text_input("Сколько у Вас детей?", "Нужно ввести количестов")
text_input[3] = (1.0 if st.text_input("Есть ли у Вас высшее образование?", "Нужно ввести Да или Нет").lower() == 'Да' else 0.0)
text_input[4] = (1.0 if st.text_input("Вы самозанятый?", "Нужно ввести Да или Нет").lower() == 'да' else 0.0)
text_input[5] = st.text_input("Какой у Вас месячный доход в рублях?", "Нужно ввести количестов")
text_input[6] = st.text_input("Какой у Вашего созаявителя месячный доход в рублях?", "Нужно ввести количестов")
text_input[7] = st.text_input("Cколько рублей вы хотели бы взять?", "Нужно ввести количестов")
text_input[8] = st.text_input("На сколько дней хотели бы взять займ?", "Нужно ввести количестов")
text_input[9] = (1.0 if st.text_input("Есть ли у Вас кредитная история?", "Нужно ввести Да или Нет").lower() == 'Да' else 0.0)
text_input[10] = (1.0 if st.text_input("Где вы проживаете?", "В городе, селе или в посёлке городского типа").lower() == 'Да' else 0.0)

button_pressed = st.button("Рассчитать возможность получения займа")

if button_pressed:
       st.write("Gender",  text_input)
       user_input = pd.DataFrame({'Gender' : [1], 'Married' : [0], 'Dependents' : [0], 'Education' : [0], 'Self_Employed' : [0], 'ApplicantIncome' : [5849], 
                                  'CoapplicantIncome' : [0.0], 'LoanAmount' : [146.412162], 'Loan_Amount_Term' : [360.0], 'Credit_History' : [1.0], 'Property_Area' : [2]})

       st.write(user_input)
       
       if predict(user_input)[0] == 1:
              st.subheader("Модель одобрила Вам займ!")
              st.balloons()
       else:
              st.subheader("Модель выявила, что по какому-то из показателей вы нежелательный заёмщик")
