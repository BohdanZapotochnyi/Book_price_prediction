# Book_price_prediction
ML project. Authors: Запоточний Богдан, Коцеловська Марія
import numpy as np
import pandas as pd

# Load your data
df = pd.read_csv('train.csv')
#df = pd.read_csv('/content/Book price/train.csv', encoding='latin1', sep=';')

df #display it

# Вибираємо всі ознаки, які впливають на ціну
features = ['Title', 'Author', 'Edition', 'Reviews', 'Ratings', 'Synopsis', 'Genre', 'BookCategory']
X = df[features]

# Перевірка пропущених значень
missing_values = df.isnull().sum()
display(missing_values)

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
