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

# Cross-validation K-Fold
cv_results = cross_validate(model, data, measures=["RMSE", "MAE"], cv=5, verbose=True)

# Valutazione del modello
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)

# Analisi della distribuzione degli errori
def plot_error_distribution(predictions):
    errors = [abs(pred.r_ui - pred.est) for pred in predictions]
    plt.hist(errors, bins=30, edgecolor="black")
    plt.xlabel("Errore Assoluto")
    plt.ylabel("Frequenza")
    plt.title("Distribuzione degli Errori del Modello SVD")
    plt.show()

plot_error_distribution(predictions)

def penalize_repetitive_genres(recommended_books, books_df):
    genre_count = Counter(books_df.loc[books_df["book_id"].isin(recommended_books)]["categories"])
    
    penalized_books = {}
    for book in recommended_books:
        book_genres = books_df.loc[books_df["book_id"] == book, "categories"]
        if len(book_genres) > 0:
            book_genre = book_genres.iloc[0]  
        else:
            book_genre = "Unknown"  
        penalty = genre_count[book_genre] * 0.1  
        penalized_books[book] = recommended_books[book] - penalty  

    return penalized_books  # Aggiunto return


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

# Funzione per raccomandare libri basati su generi preferiti e pagine giornaliere
def recommend_books(user_id, model, books_df, ratings_df, personal_df, user_reading_habits, n=10):
    read_books = get_read_books(user_id, ratings_df, personal)
    preferred_genres = get_preferred_genres(user_id, books_df, read_books)
    daily_pages = user_reading_habits.get(user_id, 50)  # Valore di default: 50 pagine al giorno

    # Filtriamo i libri che appartengono ai generi preferiti
    candidate_books = books_df[books_df["categories"].isin(preferred_genres)]
    
    # Filtriamo i libri in base al numero di pagine
    max_pages = daily_pages * 10
    candidate_books = candidate_books[candidate_books["pageCount"] <= max_pages]

    

    # Prediciamo i rating per i libri candidati
    predictions = [(book_id, model.predict(user_id, book_id).est)
                   for book_id in candidate_books["book_id"].unique() if book_id not in read_books]
    predictions.sort(key=lambda x: x[1], reverse=True)
    # Restituisce i migliori n libri
    
    # Penalizza generi ripetitivi
    diversified_recommendations = penalize_repetitive_genres(predictions, books_df)

    return [books_df[books_df["book_id"] == book_id]["title"].values[0] for book_id in diversified_recommendations[:n]]

def get_similar_books(book_id, books_df, top_n=5):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(books_df["description"].fillna(""))

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    book_idx = books_df.index[books_df["book_id"] == book_id].tolist()
    if not book_idx:
        return []
    
    book_idx = book_idx[0]
    similar_books = sorted(list(enumerate(cosine_sim[book_idx])), key=lambda x: x[1], reverse=True)[1:top_n+1]

    return books_df.iloc[[i[0] for i in similar_books]]["title"].tolist()


# Benchmark con un modello di baseline (media globale dei rating)
mean_rating = ratings["rating"].mean()
baseline_rmse = np.sqrt(np.mean((ratings["rating"] - mean_rating) ** 2))
print(f"RMSE del modello di baseline (media globale): {baseline_rmse}")

# Benchmark con modello KNN per il collaborative filtering
knn = KNNBasic()
knn.fit(trainset)
knn_predictions = knn.test(testset)
knn_rmse = accuracy.rmse(knn_predictions)
print(f"RMSE KNN: {knn_rmse}")

# Confronto finale
print("\nConfronto dei risultati:")
print(f"RMSE SVD: {rmse}")
print(f"RMSE KNN: {knn_rmse}")
print(f"RMSE Baseline: {baseline_rmse}")

# Esempio di utilizzo
user_id = 53425
recommended = recommend_books(user_id, model, books, ratings, personal , user_reading_habits,  n=5)
print(f"Libri consigliati per l'utente {user_id}: {recommended}")

