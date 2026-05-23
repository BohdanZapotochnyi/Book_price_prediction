!pip install scikit-learn
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, StandardScaler
import re
import matplotlib.pyplot as plt
import io
import base64
from IPython.display import display

#def load_and_preprocess_data(url: str):
#    try:
#        df = pd.read_csv(url, encoding='latin1', sep=';')
#        return df
#    except FileNotFoundError:
#        st.error(f"Помилка: файл '{url}' не знайдено. Переконайтеся, що він знаходиться в правильному шляху.")
#        st.stop()

# Виклик функції
data_url = "https://github.com/m67074/Book_price_prediction/raw/refs/heads/main/train.csv"
#df = load_and_preprocess_data(data_url)
df = pd.read_csv(data_url, encoding='latin1', sep=';')


st.title('Book price prediction')

st.subheader('Enter book information:')

title = st.text_input('Book title')
author = st.text_input('Book author')
edition  =  st.text_input('Book edition ') 
genre = st.text_input('Book genre')
reviews = st.number_input('Book reviews', min_value=0.0, value=4.4, step=0.1)
ratings = st.number_input('Book ratings',step=1)
Synopsis  =  st.text_input('Book synopsis ')   
BookCategory  =  st.text_input('Book category ') 

if st.button('Прогнозувати ціну'):
    input_data = pd.DataFrame({
        'Reviews': [reviews],
        'Ratings': [ratings],
        'Author': [author],
        'Genre': [genre]
    })

st.subheader('Прогнозовані ціни:')
t.metric(label="Linear Regression", value=f"{linear_pred:.2f} ")

