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

poly_model = LinearRegression()
poly_model.fit(X_train_combined, y_train)

elastic_net_model = ElasticNet(random_state=42)
elastic_net_model.fit(X_train_combined, y_train)

ridge_net_model = Ridge(random_state=42)
ridge_net_model.fit(X_train_combined, y_train)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbr_model.fit(X_train, y_train)

st.title('Book price prediction')

st.subheader('Enter book information:')

title = st.text_input('Book title')
author = st.text_input('Book author')
edition  =  st.text_input('Book edition ') 
genre = st.text_input('Book genre')
reviews = st.number_input('Book reviews', min_value=0.0, value=4.4, step=0.1)
ratings = st.number_input('Book ratings',step=1)
synopsis  =  st.text_input('Book synopsis ')   
bookcategory  =  st.text_input('Book category ') 

if st.button('Прогнозувати ціну'):
    input_data = pd.DataFrame({
        'Reviews1': [reviews],
        'Ratings1': [ratings],
        'Author1': [author],
        'Genre1': [genre]
    })
    
    #input_data['Reviews'] = input_data['Reviews'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
    #input_data['Ratings'] = input_data['Ratings'].astype(str).str.extract(r'(\d+)').astype(float)
    #input_data['Author_Encoded'] = input_data['Author'].map(mean_prices_by_author.fillna(global_mean_price))
    #input_data['Genre_Encoded'] = input_data['Genre'].map(mean_prices_by_genre.fillna(global_mean_price))
    #input_data['Author_Encoded'] = input_data['Author_Encoded'].fillna(global_mean_price)
    #input_data['Genre_Encoded'] = input_data['Genre_Encoded'].fillna(global_mean_price)
    #X_predict = input_data[['Reviews', 'Ratings', 'Author_Encoded', 'Genre_Encoded']]
    #X_predict['Reviews'] = X_predict['Reviews'].fillna(X_train['Reviews'].mean())
    #X_predict['Ratings'] = X_predict['Ratings'].fillna(X_train['Ratings'].mean())
    #X_predict_scaled_numerical = scaler.transform(X_predict[numerical_features])
    #X_predict_combined = np.hstack((X_predict_scaled_numerical, X_predict[categorical_features].values))

    input_data['Author_Encoded1'] = input_data['Author1'].map(mean_prices_by_author)
    input_data['Genre_Encoded1'] = input_data['Genre1'].map(mean_prices_by_genre)
    input_data['Author_Encoded1'] = input_data['Author_Encoded1'].fillna(global_mean_price)
    input_data['Genre_Encoded1'] = input_data['Genre_Encoded1'].fillna(global_mean_price)
    X_predict_input = input_data[['Reviews1', 'Ratings1', 'Author_Encoded1', 'Genre_Encoded1']]
    for col in ['Reviews1', 'Ratings1']:
        col_mean_input = X_predict_input[col].mean() 
        X_predict_input.loc[:, col] = X_predict_input.loc[:, col].fillna(col_mean_input if not pd.isna(col_mean_input) else 0)
    numerical_features1 = ['Reviews1', 'Ratings1']
    categorical_features1 = ['Author_Encoded1', 'Genre_Encoded1']
    scaler = StandardScaler()
    X_predict_scaled_numerical1 = scaler.transform(X_predict_input[numerical_features1])
    X_predict_combined1 = np.hstack((X_predict_scaled_numerical1, X_predict_input[categorical_features1].values))

    
    linear_pred = model.predict(X_predict_combined1)[0]
    poly_pred = poly_model.predict(X_predict_combined1)[0]
    elastic_test_pred = elastic_net_model.predict(X_predict_combined1)[0]
    ridge_test_pred = ridge_net_model.predict(X_predict_combined1)[0]
    rf_test_pred = model.predict(X_predict_combined1)[0] 
    gbr_test_pred = gbr_model.predict(X_predict_combined1)[0]

   
    

    st.subheader('Прогнозовані ціни:')
    st.metric(label="Linear Regression", value=f"{linear_pred:.2f} ")
    st.metric(label="Polynomial Regression", value=f"{poly_pred:.2f} ")
    st.metric(label="Elastic Net Regression", value=f"{elastic_test_pred :.2f} ")
    st.metric(label="Ridge Regression", value=f"{ridge_test_pred:.2f} ")
    st.metric(label="Random Forest Regression", value=f"{rf_test_pred:.2f} ")
    st.metric(label="Gradient Boosting Regression", value=f"{gbr_test_pred:.2f} ")

