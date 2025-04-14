import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

file_path = "LaCoscienzaDelLettore\\goodreads\\books_enriched.csv"

# Caricare il dataset
df = pd.read_csv(file_path)

# Controllare le prime righe del dataset
print(df.head())

# Verificare informazioni generali (valori nulli, tipi di dati)
print(df.info())

#colonne con valori duplicati
duplicates = df.duplicated().sum()
print("Duplicati:", duplicates)

# Identificare colonne con valori nulli
null_counts = df.isnull().sum()
print("Colonne con valori nulli:\n", null_counts[null_counts > 0])
zero_counts = (df["pageCount"]==0).sum()
print("Libri con pageCount a 0:", zero_counts)

# Feature Engineering SOLO sul training set
df['categories'] = df['categories'].fillna("N/A")
df['pageCount'] = df['pageCount'].fillna(0)  # Converti float64 in string prima di riempire
df['description'] = df['description'].fillna("N/A")


import requests

def get_isbn_from_google(title, author):
    query = f"{title} {author}"
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}"
    
    response = requests.get(url)
    data = response.json()
    
    if "items" in data:
        for item in data["items"]:
            volume_info = item["volumeInfo"]
            if "industryIdentifiers" in volume_info:
                for identifier in volume_info["industryIdentifiers"]:
                    if identifier["type"] in ["ISBN_10", "ISBN_13"]:
                        return identifier["identifier"]
    return "UNKNOWN-ISBN"

# Applicare la funzione solo ai valori mancanti
df.loc[df['isbn'].isna(), 'isbn'] = df[df['isbn'].isna()].apply(
    lambda row: get_isbn_from_google(row['title'], row['authors']), axis=1
)


# Salvataggio finale
df.to_csv(file_path, index=False)
