import pandas as pd
import numpy as np
import re
import nltk
import csv
import os
import gradio as gr
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download nltk resources
nltk.download('stopwords')
nltk.download('punkt')

# Percorsi dei file
data_path = "C:\\Users\\user\\LaCoscienzaDelLettore\\LaCoscienzaDelLettore\\LaCoscienzaDelLettore\\goodreads\\"
user_file = "C:\\Users\\user\\LaCoscienzaDelLettore\\LaCoscienzaDelLettore\\LaCoscienzaDelLettore\\users.csv"
library_file = "C:\\Users\\user\\LaCoscienzaDelLettore\\LaCoscienzaDelLettore\\LaCoscienzaDelLettore\\books_enriched.csv"

# Caricamento dei dataset
books = pd.read_csv(data_path + "books_enriched.csv")
ratings = pd.read_csv(data_path + "ratings.csv")
personal = pd.read_csv(data_path + "personalLibrary.csv")

# Pulizia e combinazione delle valutazioni
ratings = ratings[ratings["rating"] > 0]
personal = personal[personal["rating"] > 0]
combined_ratings = pd.concat([ratings[["user_id", "book_id", "rating"]], 
                              personal[["user_id", "book_id", "rating"]]])

# Riduzione degli utenti per migliorare l'efficienza
num_users = 10000
if len(combined_ratings["user_id"].unique()) > num_users:
    active_users = combined_ratings["user_id"].value_counts()
    sampled_users = active_users[active_users > 5].index[:num_users]
    combined_ratings = combined_ratings[combined_ratings["user_id"].isin(sampled_users)]

# Pulizia delle descrizioni dei libri
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

books["clean_description"] = books["description"].fillna("Descrizione non disponibile.").apply(clean_text)

# Creazione dataset per Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(combined_ratings, reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Addestramento del modello SVD
model = SVD(n_factors=100, n_epochs=20, random_state=42)
model.fit(trainset)

# Creazione della matrice TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(books["clean_description"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Funzione per raccomandazioni personalizzate
def recommend_books(user_id, num_books):
    try:
        user_id = int(user_id)
        predictions = [model.predict(user_id, book_id) for book_id in books["book_id"]]
        sorted_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:num_books]
        recommended_books = [books[books["book_id"] == pred.iid]["title"].values[0] for pred in sorted_predictions]
        return "\n".join(recommended_books) if recommended_books else "Nessuna raccomandazione trovata."
    except Exception as e:
        return f"Errore: {str(e)}"

# Funzione per aggiungere libri alla libreria personale
def add_book_to_library(user_id, book_title, rating, pages_read):
    book = books[books["title"].str.lower() == book_title.lower()]
    if book.empty:
        return "Libro non trovato."
    book_id = book["book_id"].values[0]
    with open(library_file, "a", newline="") as file:
        csv.writer(file).writerow([user_id, book_id, book_title, rating, pages_read])
    return f"{book_title} aggiunto alla libreria personale."


# Funzione per trovare libri simili
def find_similar_books(book_title):
    try:
        book_id = books[books["title"].str.lower() == book_title.lower()]["book_id"].values
        if len(book_id) == 0:
            return "Libro non trovato."
        similarities = list(enumerate(cosine_sim[book_id[0]]))
        similar_books = sorted(similarities, key=lambda x: x[1], reverse=True)[1:6]
        return "\n".join([books.iloc[i[0]]["title"] for i in similar_books])
    except Exception as e:
        return f"Errore: {str(e)}"

# Funzione per aggiungere un libro alla libreria personale
def add_to_personal_library(user_id, book_title, rating, pages_read):
    book_row = books[books["title"].str.lower() == book_title.lower()]
    if book_row.empty:
        return "Libro non trovato."
    book_id = book_row.iloc[0]["book_id"]
    with open(library_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([user_id, book_id, rating, pages_read])
    return "Libro aggiunto alla libreria personale!"

# Funzione per registrazione utenti
def register_user(name, email):
    try:
        if not os.path.exists(user_file):
            with open(user_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["user_id", "name", "email"])

        existing_ids = []
        with open(user_file, "r") as file:
            reader = csv.reader(file)
            next(reader, None)
            for row in reader:
                existing_ids.append(int(row[0]))

        new_user_id = max(existing_ids) + 1 if existing_ids else 100001
        with open(user_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([new_user_id, name, email])

        return f"Registrazione completata! Il tuo ID utente è {new_user_id}."
    except Exception as e:
        return f"Errore durante la registrazione: {str(e)}"

# Creazione interfaccia Gradio
with gr.Blocks() as demo:
    gr.Markdown("## 📚 La Coscienza del Lettore - Sistema di Raccomandazione")
    
    with gr.Tab("📖 Consiglia Libri"):
        gr.Markdown("Inserisci il tuo ID utente per ricevere raccomandazioni.")
        user_id_input = gr.Textbox(label="ID Utente", placeholder="Es. 12345")
        num_books_input = gr.Slider(1, 10, step=1, value=5, label="Numero di Raccomandazioni")
        recommend_button = gr.Button("Consiglia 📚")
        recommendation_output = gr.Textbox(label="Libri Raccomandati")
        recommend_button.click(fn=recommend_books, inputs=[user_id_input, num_books_input], outputs=recommendation_output)

    with gr.Tab("🔍 Trova Libri Simili"):
        gr.Markdown("Inserisci il titolo di un libro per trovare libri simili.")
        book_title_input = gr.Textbox(label="Titolo del Libro")
        find_button = gr.Button("Trova Simili 🔎")
        similar_books_output = gr.Textbox(label="Libri Simili")
        find_button.click(fn=find_similar_books, inputs=book_title_input, outputs=similar_books_output)

    with gr.Tab("📚 Libreria Personale"):
        user_id_library = gr.Textbox(label="ID Utente")
        book_title_search = gr.Dropdown(choices=books["title"].tolist(), label="Seleziona Libro")
        rating_input = gr.Slider(1, 5, step=1, label="Valutazione")
        pages_read_input = gr.Number(label="Pagine Letti al Giorno")
        add_book_button = gr.Button("Aggiungi alla Libreria")
        add_book_output = gr.Textbox(label="Esito")
        add_book_button.click(fn=add_to_personal_library, inputs=[user_id_library, book_title_search, rating_input, pages_read_input], outputs=add_book_output)
    
    with gr.Tab("📝 Registrazione Utente"):
        gr.Markdown("Registrati per ottenere un ID utente unico.")
        name_input = gr.Textbox(label="Nome Completo")
        email_input = gr.Textbox(label="Email")
        register_button = gr.Button("Registrati ✅")
        register_output = gr.Textbox(label="Esito Registrazione")
        register_button.click(fn=register_user, inputs=[name_input, email_input], outputs=register_output)

# Avvio dell'interfaccia
if __name__ == "__main__":
    demo.launch()