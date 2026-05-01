# Book_price_prediction
ML project. Authors: Запоточний Богдан, Коцеловська Марія

# -------------------------------------------------------

import numpy as np
import pandas as pd

# Load your data
#df = pd.read_csv('train.csv')
df = pd.read_csv('/content/Book price/train.csv', encoding='latin1', sep=';')
df.info()

df #display it

# Вибираємо всі ознаки, які впливають на ціну
features = ['Title', 'Author', 'Edition', 'Reviews', 'Ratings', 'Synopsis', 'Genre', 'BookCategory']
X = df[features]

# Цільова змінна (те, що прогнозуємо)
y = df['Price']

# Розділяємо дані
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ---------------------------------------
# RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import re

df = pd.read_csv('/content/Book price/train.csv', encoding='latin1', sep=';')

# Ensure 'Price' column is numeric
df['Price'] = df['Price'].astype(str).str.replace(',', '.', regex=False).astype(float)

# Preprocess 'Reviews' column to extract numerical part
df['Reviews'] = df['Reviews'].astype(str).apply(lambda x: float(re.search(r'\d+\.?\d*', x).group()) if re.search(r'\d+\.?\d*', x) else 0.0)

# Preprocess 'Ratings' column to extract numerical part
df['Ratings'] = df['Ratings'].astype(str).apply(lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)

# Define features and target
features = ['Title', 'Author', 'Edition', 'Reviews', 'Ratings', 'Synopsis', 'Genre', 'BookCategory']
X = df[features]
y = df['Price']

# Apply Label Encoding to remaining categorical features in X
for column in ['Title', 'Author', 'Edition', 'Synopsis', 'Genre', 'BookCategory']:
    if column in X.columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Added random_state for reproducibility

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --------------------------------------
# Додаток для опису диних
# --------------------------------------
# Перевірка пропущених значень
missing_values = df.isnull().sum()
display(missing_values)

# ---------------------------------------
# Вивести описову статистику
display(df.describe())

# Ось описова статистика фрейму даних:

# Count (Кількість): 6237 записів для всіх стовпців, що підтверджує відсутність пропущених значень.
# Unique (Унікальних значень): Для стовпця 'Title' є 5568 унікальних назв, для 'Author' — 3679 унікальних авторів, а для 'Reviews' — 36 унікальних відгуків. Це вказує на велике різноманіття даних у текстових стовпцях.
# Top (Найчастіше значення):
# Найпопулярніша назва книги: 'A Game of Thrones (A Song of Ice and Fire)' (зустрічається 4 рази).
# Найпопулярніший автор: 'Agatha Christie' (зустрічається 69 разів).
# Найпопулярніше видання: 'Paperback,– 5 Oct 2017' (зустрічається 48 разів).
# Найчастіший відгук: '5.0 out of 5 stars' (зустрічається 1375 разів).
# Найпопулярніший рейтинг: '1 customer review' (зустрічається 1040 разів).
# Найчастіший жанр і категорія: 'Action & Adventure (Books)' і 'Action & Adventure' відповідно.
# Найчастіша ціна: 299 (зустрічається 108 разів).
# Freq (Частота): Показує кількість появ найчастішого значення для кожного стовпця.
# Ці дані дають гарне уявлення про розподіл і характеристики даних у вашому наборі.

# ----------------------------------------

# Перетворення стовпця 'Price' на числовий тип
df['Price'] = df['Price'].str.replace(',', '.', regex=False).astype(float)

# Розрахунок середньої ціни для кожної категорії книги
average_price_by_category = df.groupby('BookCategory')['Price'].mean().sort_values(ascending=False).reset_index()

# Візуалізація середньої ціни за категоріями
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 7))
sns.barplot(x='Price', y='BookCategory', data=average_price_by_category, palette='viridis')
plt.title('Середня ціна книги за категоріями')
plt.xlabel('Середня ціна')
plt.ylabel('Категорія книги')
plt.show()

# -----------------------------------------

# Розрахунок топ-10 найпопулярніших авторів за кількістю книг
top_10_authors = df['Author'].value_counts().head(10).reset_index()
top_10_authors.columns = ['Author', 'BookCount']

# Візуалізація топ-10 авторів
plt.figure(figsize=(12, 7))
sns.barplot(x='BookCount', y='Author', hue='Author', data=top_10_authors, palette='magma', legend=False)
plt.title('Топ-10 найпопулярніших авторів за кількістю книг')
plt.xlabel('Кількість книг')
plt.ylabel('Автор')
plt.show()

# -------------------------------------------

# Аналіз залежності ціни від кількості відгуків ???
# Щоб візуалізувати залежність ціни книги від кількості відгуків, ми використаємо діаграму розсіювання. Це допоможе нам визначити, чи є якась кореляція між цими двома показниками. 

plt.figure(figsize=(12, 7))
sns.scatterplot(x='Reviews', y='Price', data=df, alpha=0.6)
plt.title('Залежність ціни від кількості відгуків')
plt.xlabel('Кількість відгуків')
plt.ylabel('Ціна')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# або ?

plt.figure(figsize=(12, 7))
sns.scatterplot(x='Reviews', y='Price', data=df, alpha=0.6)
plt.title('Залежність ціни від кількості відгуків')
plt.xlabel('Кількість відгуків')
plt.ylabel('Ціна')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45) # Rotate x-axis labels for readability
plt.tight_layout() # Adjust layout to prevent labels from being cut off
plt.show()

# ---------------------------------------------
# Створення цінових діапазонів
# Переконайтеся, що стовпець 'Price' є числовим, перетворюючи нечислові значення на NaN
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Ви можете налаштувати `bins` (кількість діапазонів) та `labels` (мітки для діапазонів)
num_bins = 5 # Кількість діапазонів
df['Price_Range'] = pd.cut(df['Price'], bins=num_bins, labels=[f'Range {i+1}' for i in range(num_bins)])

# Вивід перших кількох рядків з новим стовпцем
display(df[['Price', 'Price_Range']].head())

# ----------------------------------------------

# Підрахунок кількості книг у кожному ціновому діапазоні
price_range_counts = df['Price_Range'].value_counts().sort_index()

# Візуалізація розподілу книг за ціновими діапазонами
plt.figure(figsize=(10, 6))
sns.barplot(x=price_range_counts.index, y=price_range_counts.values, hue=price_range_counts.index, palette='coolwarm', legend=False)
plt.title('Розподіл книг за ціновими діапазонами')
plt.xlabel('Ціновий діапазон')
plt.ylabel('Кількість книг')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
