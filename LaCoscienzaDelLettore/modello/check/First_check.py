import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Caricare il dataset
df = pd.read_csv("LaCoscienzaDelLettore\modello\data.csv")

# Controllare le prime righe del dataset
print(df.head())

# Verificare informazioni generali (valori nulli, tipi di dati)
print(df.info())

# Controllare il numero di duplicati
print(f"Duplicati trovati: {df.duplicated().sum()}")


from sklearn.model_selection import train_test_split

# Dividere il dataset (esempio: 80% training, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"Training set: {train_df.shape}")
print(f"Test set: {test_df.shape}")

# Per colonne numeriche: sostituire i NaN con la media
train_df['num_pages'] = train_df['num_pages'].fillna(train_df['num_pages'].mean())
train_df['ratings_count'] = train_df['ratings_count'].fillna(train_df['ratings_count'].mean())
train_df['average_rating'] = train_df['average_rating'].fillna(train_df['average_rating'].mean())

# Per colonne categoriche: sostituire i NaN con la moda
train_df['categories'] = train_df['categories'].fillna(train_df['categories'].mode()[0])

# Per colonne testuali: sostituire i NaN con valori predefiniti
train_df['description'] = train_df['description'].fillna("No description available")
train_df['thumbnail'] = train_df['thumbnail'].fillna("https://placehold.co/128x206?text=no+cover+available&font=roboto")
train_df['authors'] = train_df['authors'].fillna("Unknown")

# Test set rimane inalterato
print(train_df.isnull().sum())  # Controllo valori mancanti

# Concatenare le colonne testuali per migliorare la similarità
# Unendo descrizione, categoria e autore otteniamo una rappresentazione più ricca del contenuto
train_df["content"] = train_df["title"] + " " + train_df["authors"] + " " + train_df["categories"] + " " + train_df["description"]

# Creare il vettorizzatore TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
content_matrix = vectorizer.fit_transform(train_df["content"])

# Calcolare la matrice di similarità
cosine_sim = cosine_similarity(content_matrix, content_matrix)

# Funzione per ottenere raccomandazioni
def recommend_books(book_title, df, cosine_sim, top_n=5):
    # Trova l'indice del libro richiesto
    idx = train_df[train_df['title'].str.lower() == book_title.lower()].index
    if len(idx) == 0:
        return "Libro non trovato nel dataset."
    idx = idx[0]
    
    # Ottieni la lista di similarità per quel libro
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Ordina in base alla similarità
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Prendi i top_n libri più simili (escludendo il primo, che è il libro stesso)
    sim_scores = sim_scores[1:top_n+1]
    
    # Ottieni gli indici dei libri consigliati
    book_indices = [i[0] for i in sim_scores]
    
    return train_df.iloc[book_indices][["title", "authors", "categories"]]

# Esempio di utilizzo
title_to_search = "The Mad Ship"
recommendations = recommend_books(title_to_search, df, cosine_sim)
print(recommendations)
