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

# Перевірка пропущених значень
missing_values = df.isnull().sum()
display(missing_values)
