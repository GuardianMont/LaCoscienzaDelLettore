import pandas as pd
import numpy as np
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from surprise import Dataset, Reader, SVD, KNNBasic
import matplotlib.pyplot as plt
from surprise.model_selection import train_test_split, cross_validate
from surprise import accuracy
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
import sys 
import csv
import os

USER_FILE = "C:\\Users\\user\\LaCoscienzaDelLettore\\LaCoscienzaDelLettore\\LaCoscienzaDelLettore\\" + "users.csv"


# Aggiungi il percorso della cartella principale del progetto
sys.path.append(os.path.abspath("C:\\Users\\user\\LaCoscienzaDelLettore\\LaCoscienzaDelLettore\\LaCoscienzaDelLettore"))

from SVD.recommender import recommend_books, get_similar_books


nltk.download('stopwords')
nltk.download('punkt_tab')

# Caricare i dataset di Goodreads
DATA_PATH = "C:\\Users\\user\\LaCoscienzaDelLettore\\LaCoscienzaDelLettore\\LaCoscienzaDelLettore\\goodreads\\"
books = pd.read_csv(DATA_PATH + "books_enriched.csv")
ratings = pd.read_csv(DATA_PATH + "ratings.csv")
personal = pd.read_csv(DATA_PATH + "personalLibrary.csv")

user_reading_habits = {
    53425: 30,  # Utente che legge 30 pagine al giorno
    12345: 50,  # Utente che legge 50 pagine al giorno
}

#validazioni nulle o pari a 0 in rating vengono rimosse
ratings = ratings[ratings["rating"] > 0]
personal = personal[personal["rating"] > 0]
#combino i due
combined_ratings = pd.concat([ratings[["user_id", "book_id", "rating"]], 
                              personal[["user_id", "book_id", "rating"]]])

#Ridurre il numero di utenti per ottimizzare KNN 
#applico il filtro sulla base dell'attività degli utenti con almeno 5 valutazioni
num_users = 10000
if len (combined_ratings["user_id"].unique()) > num_users:
    active_users = combined_ratings["user_id"].value_counts()
    sampled_users = active_users[active_users > 5].index[:10000]  # Utenti con almeno 5 valutazioni
    combined_ratings = combined_ratings[combined_ratings["user_id"].isin(sampled_users)]


# Pulizia del testo: rimozione di stopwords, numeri e punteggiatura
def clean_text(text):
    text = str(text).lower()  # Minuscolo
    text = re.sub(r'\d+', '', text)  # Rimuove numeri
    text = re.sub(r'[^\w\s]', '', text)  # Rimuove punteggiatura
    tokens = word_tokenize(text)  # Tokenizzazione
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Stopwords
    return " ".join(tokens)

# --- Pulizia del dataset PRIMA di Surprise ---
books["categories"] = books["categories"].astype(str).fillna("N/A")
books["categories"] = books["categories"].str.lower().str.replace("&", "and", regex=False).str.strip()
books["pageCount"] = books.groupby("categories")["pageCount"].transform(lambda x: x.fillna(x.mean()))
books["pageCount"] = books["pageCount"].fillna(books["pageCount"].median()).astype(int)
books["description"] = books["description"].fillna("Descrizione non disponibile.")
books["isbn"] = books["isbn"].fillna("UNKNOWN-ISBN")
books["clean_description"] = books["description"].apply(clean_text)

# Creare dataset Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(combined_ratings, reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)


# Addestrare modello SVD
model = SVD(n_factors=100, n_epochs=20, random_state=42)
model.fit(trainset)

# Calcola la matrice TF-IDF una sola volta all'inizio
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(books["description"].fillna(""))
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# Funzione per ottenere raccomandazioni
def recommend_books_interface(user_id: int, num_books: int):
    try:
        user_id = int(user_id)
        recommended_books = recommend_books(user_id, model, books, ratings, personal, user_reading_habits, num_books)
        return "\n".join(recommended_books) if recommended_books else "Nessuna raccomandazione trovata."
    except ValueError:
        return "Inserisci un ID utente valido."
    

def register_user(name, email):
    """
    Registra un nuovo utente assegnando un ID unico.
    """
    try:
        # Controlla se il file esiste
        if not os.path.exists(USER_FILE):
            with open(USER_FILE, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["user_id", "name", "email"])  # Intestazione

        # Legge gli ID esistenti per assegnare un nuovo ID
        existing_ids = []
        with open(USER_FILE, "r") as file:
            reader = csv.reader(file)
            next(reader, None)  # Salta l'intestazione
            for row in reader:
                existing_ids.append(int(row[0]))

        new_user_id = max(existing_ids) + 1 if existing_ids else 100001  # Primo ID disponibile

        # Aggiunge il nuovo utente
        with open(USER_FILE, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([new_user_id, name, email])

        return f"Registrazione completata! Il tuo ID utente è {new_user_id}."
    except Exception as e:
        return f"Errore durante la registrazione: {str(e)}"
    
def find_similar_books(book_title):
    """
    Funzione per trovare libri simili a partire da un titolo.
    """
    try:
        book_id = books[books["title"].str.lower() == book_title.lower()]["book_id"].values
        if len(book_id) == 0:
            return "Libro non trovato nel database."
        
        similar_books = get_similar_books(book_id[0], books_df=books, top_n=5)
        return "\n".join(similar_books) if similar_books else "Nessun libro simile trovato."
    except Exception as e:
        return f"Errore: {str(e)}"

# 📌 Creazione interfaccia Gradio
with gr.Blocks() as demo:
    gr.Markdown("## 📚 La Coscienza del Lettore - Sistema di Raccomandazione")
    
    with gr.Tab("📖 Consiglia Libri"):
        gr.Markdown("Inserisci il tuo ID utente per ricevere raccomandazioni personalizzate.")
        user_id_input = gr.Textbox(label="ID Utente", placeholder="Inserisci il tuo ID numerico")
        num_books_input = gr.Slider(1, 10, step=1, value=5, label="Numero di Raccomandazioni")
        recommend_button = gr.Button("Consiglia 📚")
        recommendation_output = gr.Textbox(label="Libri Raccomandati", interactive=False)

        recommend_button.click(fn=recommend_books_interface, inputs=[user_id_input, num_books_input], outputs=recommendation_output)

    with gr.Tab("🔍 Trova Libri Simili"):
        gr.Markdown("Inserisci il titolo di un libro per trovare libri simili.")
        book_title_input = gr.Textbox(label="Titolo del Libro", placeholder="Es. Harry Potter")
        find_button = gr.Button("Trova Simili 🔎")
        similar_books_output = gr.Textbox(label="Libri Simili", interactive=False)

        find_button.click(fn=find_similar_books, inputs=book_title_input, outputs=similar_books_output)

# 📌 Avvia l'interfaccia
demo.launch()