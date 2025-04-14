import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, KNNBasic
import matplotlib.pyplot as plt
from surprise.model_selection import train_test_split, cross_validate
from surprise import accuracy
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow
from keras.layers import Input, Embedding, Flatten, Dense
from keras.models import Model
from sklearn.preprocessing import LabelEncoder

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


def get_read_books(user_id, ratings_df, personal_df):
    """ Restituisce l'insieme dei book_id letti dall'utente. """
    user_ratings = ratings_df[ratings_df["user_id"] == user_id]["book_id"]
    user_personal = personal_df[personal_df["user_id"] == user_id]["book_id"]
    return set(pd.concat([user_ratings, user_personal]))

def get_preferred_genres(user_id, books_df, read_books):
    user_books = books_df[books_df["book_id"].isin(read_books)]
    
    # Controlla se la colonna 'categories' esiste
    if "categories" not in user_books.columns:
        print("Errore: 'categories' non presente nel dataset books_enriched.")
        print("Colonne disponibili:", user_books.columns)
        return []
    
    return list(user_books["categories"].value_counts().index[:5])

def recommend_books_ncf(user_id, model, books_df, ratings_df, personal_df, user_reading_habits, n=10):
    read_books = get_read_books(user_id, ratings_df, personal_df)
    preferred_genres = get_preferred_genres(user_id, books_df, read_books)
    daily_pages = user_reading_habits.get(user_id, 50)

    candidate_books = books_df[books_df["categories"].isin(preferred_genres)]
    max_pages = daily_pages * 10
    candidate_books = candidate_books[candidate_books["pageCount"] <= max_pages]

    # Previsione dei rating per i libri candidati
    candidate_user_ids = np.full(len(candidate_books), user_encoder.transform([user_id])[0])
    candidate_book_ids = book_encoder.transform(candidate_books["book_id"].values)
    
    predictions = model.predict([candidate_user_ids, candidate_book_ids])
    predictions = [(book_id, prediction) for book_id, prediction in zip(candidate_books["book_id"], predictions)]

    # Ordina per rating predetto e restituisci i migliori libri
    predictions.sort(key=lambda x: x[1], reverse=True)
    recommended_books = [books_df[books_df["book_id"] == book_id]["title"].values[0] for book_id, _ in predictions[:n]]

    return recommended_books

# Pre-processamento dei dati per Neural Collaborative Filtering
# 1. Codifica le informazioni di user e item con LabelEncoder
user_encoder = LabelEncoder()
book_encoder = LabelEncoder()

combined_ratings['user'] = user_encoder.fit_transform(combined_ratings['user_id'])
combined_ratings['book'] = book_encoder.fit_transform(combined_ratings['book_id'])

n_users = len(user_encoder.classes_)
n_books = len(book_encoder.classes_)

# Dividiamo il dataset in train e test
train_data = combined_ratings.sample(frac=0.8, random_state=42)
test_data = combined_ratings.drop(train_data.index)

X_train = [train_data['user'], train_data['book']]
y_train = train_data['rating']

X_test = [test_data['user'], test_data['book']]
y_test = test_data['rating']

# Costruzione del modello NCF
user_input = Input(shape=(1,), name='user')
book_input = Input(shape=(1,), name='book')

# Creiamo gli embedding
user_embedding = Embedding(input_dim=n_users, output_dim=20)(user_input)
book_embedding = Embedding(input_dim=n_books, output_dim=20)(book_input)

# Flatten degli embeddings
user_vec = Flatten()(user_embedding)
book_vec = Flatten()(book_embedding)

# Concatenare gli embeddings
concat = tensorflow.keras.layers.concatenate([user_vec, book_vec])

# Aggiungiamo un layer denso
dense1 = Dense(128, activation='relu')(concat)
dense2 = Dense(64, activation='relu')(dense1)

# Output finale con sigmoide
output = Dense(1)(dense2)

# Creiamo il modello
ncf_model = Model(inputs=[user_input, book_input], outputs=output)

# Compilazione del modello
ncf_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Sommario del modello
ncf_model.summary()

# Allenamento del modello
history = ncf_model.fit(
    [X_train[0], X_train[1]], y_train,
    epochs=10,
    batch_size=64,
    validation_data=([X_test[0], X_test[1]], y_test)
)

# Valutazione sul set di test
test_loss, test_mae = ncf_model.evaluate([X_test[0], X_test[1]], y_test, verbose=0)
print(f"Test Loss: {test_loss}")
print(f"Test MAE: {test_mae}")
