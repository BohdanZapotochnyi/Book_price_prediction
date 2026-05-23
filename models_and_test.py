# Імпорт бібліотеки NumPy для числових операцій
import numpy as np
# Імпорт бібліотеки Pandas для роботи з табличними даними
import pandas as pd
# Імпорт метрик для оцінки моделі
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
# Імпорт моделей LinearRegression, ElasticNet і Ridge
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
# Імпорт функції для розділення даних на тренувальні та тестові набори
from sklearn.model_selection import train_test_split
# Імпорт ансамблевих моделей регресії
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# Імпорт інструментів для попередньої обробки даних
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, StandardScaler
# Імпорт модуля для роботи з регулярними виразами
import re
# Імпорт бібліотеки Matplotlib для візуалізації даних
import matplotlib.pyplot as plt
# Імпорт модуля для роботи з потоками вводу/виводу
import io
# Імпорт модуля для кодування/декодування Base64
import base64
# Імпорт функції display з IPython.display
from IPython.display import display

# -----------------------

df = pd.read_csv('/content/Book price/train.csv', encoding='latin1', sep=';', engine='python', on_bad_lines='warn')



# --- Початок доданої попередньої обробки даних для самостійного виконання ---
# Перетворення стовпця 'Price' на числовий формат, NaN для некоректних значень
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Очищення стовпців 'Reviews' та 'Ratings' для вилучення числових значень
df['Reviews'] = df['Reviews'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
df['Ratings'] = df['Ratings'].astype(str).str.extract(r'(\d+)').astype(float)

# Calculate mean prices by Author and Genre for target encoding
# Note: Applying target encoding on the full dataframe before splitting can lead to data leakage.
# For a more robust ML pipeline, target encoding should ideally be calculated on training data only.
# However, to replicate the kernel state's 'Author_Encoded' and 'Genre_Encoded' as floats,
# we apply it here for self-containment.

# Розрахунок глобального середнього значення ціни для заповнення пропущених значень
global_mean_price = df['Price'].mean()

# Розрахунок середніх цін за автором та жанром для кодування
mean_prices_by_author = df.groupby('Author')['Price'].transform('mean')
mean_prices_by_genre = df.groupby('Genre')['Price'].transform('mean')

# Створення нових стовпців для закодованих автора та жанру на основі середніх цін
df['Author_Encoded'] = mean_prices_by_author
df['Genre_Encoded'] = mean_prices_by_genre

# Заповнення будь-яких пропущених значень у закодованих стовпцях глобальним середнім значенням ціни
df['Author_Encoded'] = df['Author_Encoded'].fillna(global_mean_price)
df['Genre_Encoded'] = df['Genre_Encoded'].fillna(global_mean_price)

# Визначення ознак (X) та цільової змінної (y) за допомогою оброблених даних
X = df[['Reviews', 'Ratings', 'Author_Encoded', 'Genre_Encoded']]
y = df['Price']

# Видалення рядків, де y (Price) є NaN, оскільки вони не можуть бути використані для навчання
# Забезпечення однакових індексів X та y після видалення NaN
y.dropna(inplace=True)
X = X.loc[y.index]

# Розділення даних на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Визначення числових та категоріальних ознак для подальшої обробки
numerical_features = ['Reviews', 'Ratings']
categorical_features = ['Author_Encoded', 'Genre_Encoded']

# Масштабування числових ознак
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numerical_features])
X_test_scaled = scaler.transform(X_test[numerical_features])

# Об'єднання масштабованих числових ознак з категоріальними
X_train_combined = np.hstack((X_train_scaled, X_train[categorical_features].values))
X_test_combined = np.hstack((X_test_scaled, X_test[categorical_features].values))
# --- Кінець доданої попередньої обробки даних для самостійного виконання ---

# -----------------------

# Ініціалізація моделі лінійної регресії
model = LinearRegression()

# Навчання моделі
model.fit(X_train, y_train)

# Прогнозування на тестовому наборі
y_pred = model.predict(X_test)

# -----------------------

# -----------------------
# Ініціалізація моделі лінійної регресії для поліноміальних ознак
poly_model = LinearRegression()

# Навчання моделі з комбінованими ознаками
poly_model.fit(X_train_combined, y_train)

# Прогнозування на тестовому наборі
y_pred_poly = poly_model.predict(X_test_combined)


# -----------------------

# -----------------------

# Initialize and train Elastic Net model
elastic_net_model = ElasticNet(random_state=42)
elastic_net_model.fit(X_train_combined, y_train)

# Predict on the test data using the Elastic Net model
y_pred_elastic = elastic_net_model.predict(X_test_combined)



# -----------------------

# -----------------------

# Initialize and train Ridge model
ridge_net_model = Ridge(random_state=42)
ridge_net_model.fit(X_train_combined, y_train)

# Predict on the test data using the Ridge model
y_pred_ridge = ridge_net_model.predict(X_test_combined)


# -----------------------

# -----------------------

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions with the trained model
y_pred_rf = model.predict(X_test)


# -----------------------

# -----------------------

# Train the GradientBoostingRegressor model
gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbr_model.fit(X_train, y_train)

# Make predictions with the trained model
y_pred_gbr = gbr_model.predict(X_test)

# -----------------------

# -----------------------

# Прогнозування цін на книги за допомогою test.csv

print('Завантаження та попередня обробка файлу test.csv')

# Завантаження тестових даних
df_test = pd.read_csv('/content/Book price/test.csv', encoding='latin1', sep=';', engine='python', on_bad_lines='warn')

# Перетворення стовпця 'Price' на числовий формат (якщо він існує і має бути оброблений)
# У тестовому наборі 'Price' зазвичай відсутній або містить NaN, але для консистентності обробки
# з тренувальним набором, ми застосовуємо pd.to_numeric, якщо 'Price' присутній.
if 'Price' in df_test.columns:
    df_test['Price'] = pd.to_numeric(df_test['Price'], errors='coerce')

# Очищення стовпців 'Reviews' та 'Ratings' для вилучення числових значень
df_test['Reviews'] = df_test['Reviews'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
df_test['Ratings'] = df_test['Ratings'].astype(str).str.extract(r'(\d+)').astype(float)

# Застосування цільового кодування, використовуючи середні значення з тренувального набору
# Це запобігає витоку даних з тестового набору
# Обробка нових авторів/жанрів у тестовому наборі: заповнюємо їх глобальним середнім значенням з тренувального набору

df_test['Author_Encoded'] = df_test['Author'].map(mean_prices_by_author.fillna(global_mean_price))
df_test['Genre_Encoded'] = df_test['Genre'].map(mean_prices_by_genre.fillna(global_mean_price))

# Заповнення будь-яких NaN, які могли виникнути через нові категорії в тестовому наборі, глобальним середнім значенням
df_test['Author_Encoded'] = df_test['Author_Encoded'].fillna(global_mean_price)
df_test['Genre_Encoded'] = df_test['Genre_Encoded'].fillna(global_mean_price)


# Визначення ознак (X_test_predict) для прогнозування
X_test_predict = df_test[['Reviews', 'Ratings', 'Author_Encoded', 'Genre_Encoded']]

# Заповнення можливих NaN у числових ознаках, які могли виникнути під час екстракції
X_test_predict['Reviews'] = X_test_predict['Reviews'].fillna(X_train['Reviews'].mean())
X_test_predict['Ratings'] = X_test_predict['Ratings'].fillna(X_train['Ratings'].mean())

# Масштабування числових ознак за допомогою СКЕЙЛЕРА, навченого на тренувальних даних
X_test_predict_scaled_numerical = scaler.transform(X_test_predict[numerical_features])

# Об'єднання масштабованих числових ознак з категоріальними
X_test_predict_combined = np.hstack((X_test_predict_scaled_numerical, X_test_predict[categorical_features].values))

display(df_test.head())

# -----------------------

# -----------------------

# Прогнозування цін на книги за допомогою навчених моделей

print('Прогнозування за допомогою Linear Regression')
y_pred_linear_test = model.predict(X_test_predict)
df_test['Predicted_Price_Linear_Regression'] = y_pred_linear_test

print('Прогнозування за допомогою Polynomial Regression')
y_pred_poly_test = poly_model.predict(X_test_predict_combined)
df_test['Predicted_Price_Polynomial_Regression'] = y_pred_poly_test

print('Прогнозування за допомогою Elastic Net')
y_pred_elastic_test = elastic_net_model.predict(X_test_predict_combined)
df_test['Predicted_Price_Elastic_Net'] = y_pred_elastic_test

print('Прогнозування за допомогою Ridge Regression')
y_pred_ridge_test = ridge_net_model.predict(X_test_predict_combined)
df_test['Predicted_Price_Ridge_Regression'] = y_pred_ridge_test

print('Прогнозування за допомогою RandomForestRegressor')
y_pred_rf_test = model.predict(X_test_predict) # Re-using 'model' from RandomForestRegressor training block
df_test['Predicted_Price_RandomForestRegressor'] = y_pred_rf_test

print('Прогнозування за допомогою GradientBoostingRegressor')
y_pred_gbr_test = gbr_model.predict(X_test_predict)
df_test['Predicted_Price_GradientBoostingRegressor'] = y_pred_gbr_test

print('\nПрогнозовані ціни для тестового набору:')
display(df_test[['Title', 'Author', 'Predicted_Price_Linear_Regression', 'Predicted_Price_Polynomial_Regression', 'Predicted_Price_Elastic_Net', 'Predicted_Price_Ridge_Regression', 'Predicted_Price_RandomForestRegressor', 'Predicted_Price_GradientBoostingRegressor']].head())

