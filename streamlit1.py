import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, StandardScaler
import re
import io
import base64
from IPython.display import display

def load_and_preprocess_data(url: str):
    try:
        df = pd.read_csv(url, encoding='latin1', sep=';')
        return df
    except FileNotFoundError:
        st.error(f"Помилка: файл '{url}' не знайдено. Переконайтеся, що він знаходиться в правильному шляху.")
        st.stop()

# Виклик функції
data_url = "https://github.com/m67074/Book_price_prediction/raw/refs/heads/main/train.csv"
df = load_and_preprocess_data(data_url)

df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df['Reviews'] = df['Reviews'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
df['Ratings'] = df['Ratings'].astype(str).str.extract(r'(\d+)').astype(float)
global_mean_price = df['Price'].mean()
mean_prices_by_author = df.groupby('Author')['Price'].transform('mean')
mean_prices_by_genre = df.groupby('Genre')['Price'].transform('mean')
df['Author_Encoded'] = mean_prices_by_author
df['Genre_Encoded'] = mean_prices_by_genre
df['Author_Encoded'] = df['Author_Encoded'].fillna(global_mean_price)
df['Genre_Encoded'] = df['Genre_Encoded'].fillna(global_mean_price)
X = df[['Reviews', 'Ratings', 'Author_Encoded', 'Genre_Encoded']]
y = df['Price']
y.dropna(inplace=True)
X = X.loc[y.index]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
numerical_features = ['Reviews', 'Ratings']
categorical_features = ['Author_Encoded', 'Genre_Encoded']
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numerical_features])
X_test_scaled = scaler.transform(X_test[numerical_features])
X_train_combined = np.hstack((X_train_scaled, X_train[categorical_features].values))
X_test_combined = np.hstack((X_test_scaled, X_test[categorical_features].values))

model = LinearRegression()
model.fit(X_train, y_train)
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

