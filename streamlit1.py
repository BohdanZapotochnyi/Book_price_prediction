
import streamlit as st
import pandas as pd
import numpy as np

def load_and_preprocess_data(train.csv):
    try:
        df = pd.read_csv(train.csv, encoding='latin1', sep=';', engine='python', on_bad_lines='warn')
    except FileNotFoundError:
        st.error(f"Помилка: файл '{file_path}' не знайдено. Переконайтеся, що він знаходиться в правильному шляху.")
        st.stop()

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

