import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# 1. Завантаження даних (Використовуємо локальний файл або RAW посилання)
data_file = 'Predict_Book_Prices_Actual_USD.csv' # Сконвертований файл у USD із роздільником ';'

try:
    df = pd.read_csv(data_file, sep=';', encoding='utf-8')
except FileNotFoundError:
    # Запасний варіант, якщо файлу немає локально (прибираємо blob на raw)
    raw_url = "https://raw.githubusercontent.com/BohdanZapotochnyi/Book_price_prediction/main/train.csv"
    df = pd.read_csv(raw_url, encoding='latin1', sep=';')

# 2. Очищення та обробка базових ознак
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df['Reviews'] = df['Reviews'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
df['Ratings'] = df['Ratings'].astype(str).str.extract(r'(\d+)').astype(float)

# Створюємо словники реальних середніх цін (Target Encoding) для авторів та жанрів
global_mean_price = df['Price'].mean()
author_map = df.groupby('Author')['Price'].mean().to_dict()
genre_map = df.groupby('Genre')['Price'].mean().to_dict()

# Кодуємо колонки в датафреймі
df['Author_Encoded'] = df['Author'].map(author_map).fillna(global_mean_price)
df['Genre_Encoded'] = df['Genre'].map(genre_map).fillna(global_mean_price)

# Видаляємо пропуски в цільовій змінній
df.dropna(subset=['Price'], inplace=True)

# 3. Розділення на ознаки та таргет
X = df[['Reviews', 'Ratings', 'Author_Encoded', 'Genre_Encoded']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Масштабування ознак
numerical_features = ['Reviews', 'Ratings']
categorical_features = ['Author_Encoded', 'Genre_Encoded']

scaler = StandardScaler()
# Навчаємо scaler на числових ознаках тренувальної вибірки
X_train_num_scaled = scaler.fit_transform(X_train[numerical_features])
X_test_num_scaled = scaler.transform(X_test[numerical_features])

# Об'єднуємо масштабовані числові та категоріальні ознаки
X_train_combined = np.hstack((X_train_num_scaled, X_train[categorical_features].values))
X_test_combined = np.hstack((X_test_num_scaled, X_test[categorical_features].values))

# Створюємо поліноміальні ознаки (степінь 2)
poly_trans = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_trans.fit_transform(X_train_combined)
X_test_poly = poly_trans.transform(X_test_combined)

# 5. Навчання моделей (ВСІ моделі вчимо на однакових комбінованих даних!)
lr_model = LinearRegression()
lr_model.fit(X_train_combined, y_train)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

elastic_net_model = ElasticNet(alpha=1.0, random_state=42)
elastic_net_model.fit(X_train_combined, y_train)

ridge_model = Ridge(alpha=10.0, random_state=42) # Збільшено alpha для уникнення аномалій
ridge_model.fit(X_train_combined, y_train)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_combined, y_train)

gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbr_model.fit(X_train_combined, y_train)

# 6. Інтерфейс Streamlit
st.title('Book Price Prediction (in USD)')
st.subheader('Enter book information:')

author = st.text_input('Book author', value="Unknown")
genre = st.text_input('Book genre', value="Fiction")
reviews = st.number_input('Book reviews (Stars out of 5)', min_value=0.0, max_value=5.0, value=4.4, step=0.1)
ratings = st.number_input('Book ratings (Count)', min_value=0, value=100, step=1)

if st.button('Прогнозувати ціну'):
    # Організація вхідних даних користувача
    user_author_encoded = author_map.get(author, global_mean_price)
    user_genre_encoded = genre_map.get(genre, global_mean_price)
    
    # Створюємо масив числових ознак та масштабуємо його
    user_num = np.array([[reviews, ratings]])
    user_num_scaled = scaler.transform(user_num)
    
    # Створюємо фінальний вектор ознак для прогнозів
    X_predict_combined = np.hstack((user_num_scaled, [[user_author_encoded, user_genre_encoded]]))
    X_predict_poly = poly_trans.transform(X_predict_combined)
    
    # Робимо прогнози
    linear_pred = lr_model.predict(X_predict_combined)[0]
    poly_pred = poly_model.predict(X_predict_poly)[0]
    elastic_pred = elastic_net_model.predict(X_predict_combined)[0]
    ridge_pred = ridge_model.predict(X_predict_combined)[0]
    rf_pred = rf_model.predict(X_predict_combined)[0]
    gbr_pred = gbr_model.predict(X_predict_combined)[0]
    
    # Виведення результатів
    st.subheader('Прогнозовані ціни ($):')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Linear Regression", value=f"${max(0.0, linear_pred):.2f}")
        st.metric(label="Polynomial Regression", value=f"${max(0.0, poly_pred):.2f}")
    with col2:
        st.metric(label="Elastic Net", value=f"${max(0.0, elastic_pred):.2f}")
        st.metric(label="Ridge Regression", value=f"${max(0.0, ridge_pred):.2f}")
    with col3:
        st.metric(label="Random Forest", value=f"${max(0.0, rf_pred):.2f}")
        st.metric(label="Gradient Boosting", value=f"${max(0.0, gbr_pred):.2f}")
