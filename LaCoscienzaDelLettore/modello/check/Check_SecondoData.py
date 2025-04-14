import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Caricare il dataset
df = pd.read_csv("C:\\Users\\user\\LaCoscienzaDelLettore\\LaCoscienzaDelLettore\\LaCoscienzaDelLettore\\goodreads\\books.csv")

# Controllare le prime righe del dataset
print(df.head())

# Verificare informazioni generali (valori nulli, tipi di dati)
print(df.info())

# Controllare il numero di duplicati
print(f"Duplicati trovati: {df.duplicated().sum()}")

# Dividere il dataset in training (80%) e test (20%) mantenendo la casualità
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Feature Engineering SOLO sul training set
train_df['isbn'] = train_df['isbn'].fillna("N/A")
train_df['isbn13'] = train_df['isbn13'].astype(str).fillna("N/A")  # Converti float64 in string prima di riempire
train_df['original_publication_year'] = train_df['original_publication_year'].fillna(train_df['original_publication_year'].median())
train_df['original_title'] = train_df['original_title'].fillna(train_df['title'])
train_df['language_code'] = train_df['language_code'].fillna(train_df['language_code'].mode()[0])

# Controllo dei valori mancanti
print("Valori mancanti nel training set:")
print(train_df.isnull().sum())

print("Valori mancanti nel test set:")
print(test_df.isnull().sum())  # Il test set rimane inalterato
