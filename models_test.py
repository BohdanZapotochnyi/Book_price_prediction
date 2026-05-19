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
# Імпорт модуля для відображення об'єктів в IPython
import IPython.display as display

# -----------------------

# Завантаження даних з CSV-файлу
df = pd.read_csv('/content/Book price/train.csv', encoding='latin1', sep=';')
df_test = pd.read_csv('/content/Book price/test.csv', encoding='latin1', sep=';')
df_sample_submission = pd.read_csv('/content/Book price/sample_submission.csv', encoding='latin1', sep=';')

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
    display.display(errors_df.head(top_n))

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
# These are `Series` produced by `.transform()`, which are applied to `df` to create `Author_Encoded` and `Genre_Encoded`
# for the training set. For the test set, we need actual mappings.
author_price_map = df.groupby('Author')['Price'].mean().to_dict()
genre_price_map = df.groupby('Genre')['Price'].mean().to_dict()

df['Author_Encoded'] = df['Author'].map(author_price_map).fillna(global_mean_price)
df['Genre_Encoded'] = df['Genre'].map(genre_price_map).fillna(global_mean_price)

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

# Preprocess df_test in the same way as df for training
df_test_processed = df_test.copy()

# Clean 'Reviews' and 'Ratings' for df_test_processed
df_test_processed['Reviews'] = df_test_processed['Reviews'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
df_test_processed['Ratings'] = df_test_processed['Ratings'].astype(str).str.extract(r'(\d+)').astype(float)

# Apply target encoding to df_test_processed using mappings from training data
df_test_processed['Author_Encoded'] = df_test_processed['Author'].map(author_price_map).fillna(global_mean_price)
df_test_processed['Genre_Encoded'] = df_test_processed['Genre'].map(genre_price_map).fillna(global_mean_price)

# Select the features for the final test set prediction
X_test_final = df_test_processed[['Reviews', 'Ratings', 'Author_Encoded', 'Genre_Encoded']]

# Make predictions on the preprocessed test data
# y_pred_1 = model.predict(X_test_final)
y_pred_1 = np.maximum(0, model.predict(X_test_final_combined_for_linear))

# Оцінка моделі Linear Regression
mae, mse, mape, r2, accuracy_in_percent = evaluate_model(df_sample_submission['Price'], y_pred_1, "Linear Regression for test")

# Графік порівняння реальних цін із прогнозованими моделлю лінійної регресі
plot_predictions(df_sample_submission['Price'], y_pred_1, "Linear Regression Model for test", 'orange')

# Визначення точок даних з найбільшими помилками прогнозування
display_top_errors(df_sample_submission['Price'], y_pred_1, "Linear Regression for test")

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

# Графік порівняння реальних цін із прогнозованими моделлю пліноміальної регресії
plot_predictions(y_test, y_pred_poly, "Polynomial Regression", 'purple')

# Визначення точок даних з найбільшими помилками прогнозування для поліноміальної моделі
display_top_errors(y_test, y_pred_poly, "Polynomial Regression")

# -----------------------

# Preprocess df_test in the same way as df for training
df_test_processed = df_test.copy()

# Clean 'Reviews' and 'Ratings' for df_test_processed
df_test_processed['Reviews'] = df_test_processed['Reviews'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
df_test_processed['Ratings'] = df_test_processed['Ratings'].astype(str).str.extract(r'(\d+)').astype(float)

# Apply target encoding to df_test_processed using mappings from training data
df_test_processed['Author_Encoded'] = df_test_processed['Author'].map(author_price_map).fillna(global_mean_price)
df_test_processed['Genre_Encoded'] = df_test_processed['Genre'].map(genre_price_map).fillna(global_mean_price)

# Scaling numerical features for df_test_processed using the same scaler
X_test_processed_scaled_numerical = scaler.transform(df_test_processed[numerical_features])

# Combine scaled numerical features with encoded categorical features for df_test_processed
X_test_final_combined_for_poly = np.hstack((X_test_processed_scaled_numerical, df_test_processed[categorical_features].values))

# Make predictions on the preprocessed test data
y_pred_poly_1 = np.maximum(0, poly_model.predict(X_test_final_combined_for_poly))

# Оцінка моделі Polynomial Regression
mae_poly, mse_poly, mape_poly, r2_poly, accuracy_poly = evaluate_model(df_sample_submission['Price'], y_pred_poly_1, "Polynomial Regression for test")

# Графік порівняння реальних цін із прогнозованими моделлю пліноміальної регресії
plot_predictions(df_sample_submission['Price'], y_pred_poly_1, "Polynomial Regression for test", 'purple')

# Визначення точок даних з найбільшими помилками прогнозування
display_top_errors(df_sample_submission['Price'], y_pred_poly_1, "Polynomial Regression for test")
