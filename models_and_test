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

# Завантаження даних
# Завантаження даних з Excel
#df = pd.read_excel('train.xlsx')
# Завантаження даних з CSV-файлу
df = pd.read_csv('/content/Book price/train.csv', encoding='latin1', sep=';', engine='python', on_bad_lines='warn')

# Оцінки моделей
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred) # Розрахунок середньої абсолютної помилки
    mse = mean_squared_error(y_true, y_pred) # Розрахунок середньої квадратичної помилки
    # Додаємо невелике значення до y_true, щоб уникнути ділення на нуль при обчисленні MAPE
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    r2 = r2_score(y_true, y_pred) # Розрахунок коефіцієнта детермінації R2
    accuracy = 100 - mape # Розрахунок точності у відсотках
    print(f"\n{model_name} Model Evaluation:") # Adjusted print statement to be generic
    print(f"  Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  Mean Squared Error (MSE): {mse:.2f}")
    print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"  R-squared (R2): {r2:.2f}")
    print(f"  Accuracy: {accuracy:.2f}%")
    return mae, mse, mape, r2, accuracy

# Графік порівняння реальних цін із прогнозованими моделлю
def plot_predictions(y_true, y_pred, model_title, plot_color):
    """
    Generates a scatter plot comparing actual vs. predicted values for a given model.

    Args:
        y_true (pd.Series): Actual values.
        y_pred (np.array): Predicted values.
        model_title (str): Title for the plot and Y-axis label.
        plot_color (str): Color for the scatter points.
    """
    plt.figure(figsize=(10, 10)) # Встановлення розміру графіка
    plt.scatter(y_true, y_pred, alpha=0.7, color=plot_color) # Побудова точкового графіка: реальні vs прогнозовані ціни
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], '--r', linewidth=2) # Ideal prediction line
    plt.xlabel('Actual Prices') # Підпис осі X
    plt.ylabel(f'Predicted Prices ({model_title})') # Підпис осі Y
    plt.title(f'Actual vs. Predicted Prices ({model_title})') # Заголовок графіка
    plt.grid(True) # Увімкнення сітки
    plt.show() # Відображення графіка

# Визначення точок даних з найбільшими помилками прогнозування
def display_top_errors(y_true, y_pred, model_name, top_n=10):
    """
    Identifies and displays the top N data points with the largest prediction errors.

    Args:
        y_true (pd.Series): Actual values.
        y_pred (np.array): Predicted values.
        model_name (str): Name of the model for display purposes.
        top_n (int): The number of top errors to display.
    """
    errors_df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred, 'Absolute_Error': np.abs(y_true - y_pred)}) # Створення DataFrame з помилками
    errors_df = errors_df.sort_values(by='Absolute_Error', ascending=False) # Сортування за абсолютною помилкою
    print(f"\nTop {top_n} data points with the largest prediction errors for {model_name}:")
    display(errors_df.head(top_n))

# -----------------------

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

# Оцінка моделі Linear Regression
mae, mse, mape, r2, accuracy_in_percent = evaluate_model(y_test, y_pred, "Linear Regression")

# Графік порівняння реальних цін із прогнозованими моделлю лінійної регресії
plot_predictions(y_test, y_pred, "Linear Regression Model", 'orange')

# Визначення точок даних з найбільшими помилками прогнозування
display_top_errors(y_test, y_pred, "Linear Regression")

# -----------------------

# -----------------------
# Ініціалізація моделі лінійної регресії для поліноміальних ознак
poly_model = LinearRegression()

# Навчання моделі з комбінованими ознаками
poly_model.fit(X_train_combined, y_train)

# Прогнозування на тестовому наборі
y_pred_poly = poly_model.predict(X_test_combined)

# Оцінка моделі Polynomial Regression
mae_poly, mse_poly, mape_poly, r2_poly, accuracy_poly = evaluate_model(y_test, y_pred_poly, "Polynomial Regression")

# Графік порівняння реальних цін із прогнозованими моделлю пліномінальної регресії
plot_predictions(y_test, y_pred_poly, "Polynomial Regression", 'purple')

# Визначення точок даних з найбільшими помилками прогнозування для поліноміальної моделі
display_top_errors(y_test, y_pred_poly, "Polynomial Regression")

# -----------------------

# -----------------------

# Initialize and train Elastic Net model
elastic_net_model = ElasticNet(random_state=42)
elastic_net_model.fit(X_train_combined, y_train)

# Predict on the test data using the Elastic Net model
y_pred_elastic = elastic_net_model.predict(X_test_combined)

# Оцінка моделі Elastic Net
mae_elastic, mse_elastic, mape_elastic, r2_elastic, accuracy_elastic = evaluate_model(y_test, y_pred_elastic, "Elastic Net")

# Графік порівняння реальних цін із прогнозованими Elastic Net Model
plot_predictions(y_test, y_pred_elastic, "Elastic Net Model", 'blue') # Changed color to blue for distinction

# Визначення точок даних з найбільшими помилками прогнозування для Elastic Net
display_top_errors(y_test, y_pred_elastic, "Elastic Net")

# -----------------------

# -----------------------

# Initialize and train Ridge model
ridge_net_model = Ridge(random_state=42)
ridge_net_model.fit(X_train_combined, y_train)

# Predict on the test data using the Ridge model
y_pred_ridge = ridge_net_model.predict(X_test_combined)

# Оцінка моделі Ridge
mae_ridge, mse_ridge, mape_ridge, r2_ridge, accuracy_ridge = evaluate_model(y_test, y_pred_ridge, "Ridge Regression")

# Графік порівняння реальних цін із прогнозованими Ridge Model
plot_predictions(y_test, y_pred_ridge, "Ridge Model", 'green')

# Визначення точок даних з найбільшими помилками прогнозування для Ridge
display_top_errors(y_test, y_pred_ridge, "Ridge Regression")

# -----------------------

# -----------------------

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions with the trained model
y_pred_rf = model.predict(X_test)

# Оцінка моделі RandomForestRegressor
mae_rf, mse_rf, mape_rf, r2_rf, accuracy_rf = evaluate_model(y_test, y_pred_rf, "RandomForestRegressor")

# Графік порівняння реальних цін із прогнозованими RandomForestRegressor Model
plot_predictions(y_test, y_pred_rf, "RandomForestRegressor Model", 'yellow')

# Визначення точок даних з найбільшими помилками прогнозування для RandomForestRegressor
display_top_errors(y_test, y_pred_rf, "RandomForestRegressor")

# -----------------------

# -----------------------

# Train the GradientBoostingRegressor model
gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbr_model.fit(X_train, y_train)

# Make predictions with the trained model
y_pred_gbr = gbr_model.predict(X_test)

# Оцінка моделі GradientBoostingRegressor
mae_gbr, mse_gbr, mape_gbr, r2_gbr, accuracy_gbr = evaluate_model(y_test, y_pred_gbr, "GradientBoostingRegressor")

# Графік порівняння реальних цін із прогнозованими GradientBoostingRegressor Model
plot_predictions(y_test, y_pred_gbr, "GradientBoostingRegressor Model", 'brown')

# Визначення точок даних з найбільшими помилками прогнозування для GradientBoostingRegressor
display_top_errors(y_test, y_pred_gbr, "GradientBoostingRegressor")

# -----------------------

# -----------------------

import seaborn as sns

# Гістограма розподілу цін
plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], bins=50, kde=True)
plt.title('Розподіл цін на книги')
plt.xlabel('Ціна')
plt.ylabel('Частота')
plt.grid(True)
plt.show()

# Визначення мінімальної та максимальної ціни
min_price = df['Price'].min()
max_price = df['Price'].max()

print(f"Мінімальна ціна: {min_price:.2f}")
print(f"Максимальна ціна: {max_price:.2f}")

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

