import pandas as pd

DATA_PATH = "C:\\Users\\user\\LaCoscienzaDelLettore\\LaCoscienzaDelLettore\\LaCoscienzaDelLettore\\goodreads\\"

# Caricamento dataset
books = pd.read_csv(DATA_PATH + "books_enriched.csv")
ratings = pd.read_csv(DATA_PATH + "ratings.csv")
personal = pd.read_csv(DATA_PATH + "personalLibrary.csv")

# Filtriamo solo valutazioni positive
ratings = ratings[ratings["rating"] > 0]

def get_read_books(user_id, ratings_df, personal_df):
    """ Restituisce l'insieme dei book_id letti dall'utente. """
    user_ratings = ratings_df[ratings_df["user_id"] == user_id]["book_id"]
    user_personal = personal_df[personal_df["user_id"] == user_id]["book_id"]
    return set(pd.concat([user_ratings, user_personal]))

def get_preferred_genres(user_id, books_df, read_books):
    """ Identifica i generi più letti dall'utente basandosi su books_enriched.csv. """
    user_books = books_df[books_df["book_id"].isin(read_books)]
    
    # Controlla se la colonna 'categories' esiste
    if "categories" not in user_books.columns:
        print("Errore: 'categories' non presente nel dataset books_enriched.")
        print("Colonne disponibili:", user_books.columns)
        return []
    
    # Conta le categorie più frequenti
    return list(user_books["categories"].value_counts().index[:5])

user_reading_habits = {53425: 30}  # Utente che legge 30 pagine al giorno

# Controlla i libri letti
read_books = get_read_books(53425, ratings, personal)
print(f"L'utente 53425 ha letto {len(read_books)} libri")

# Trova i generi preferiti
preferred_genres = get_preferred_genres(53425, books, read_books)
print(f"Generi preferiti per l'utente 53425: {preferred_genres}")

# Filtra i libri candidati in base ai generi preferiti
candidate_books = books[books["categories"].isin(preferred_genres)]
print(f"Libri candidati dopo il filtro: {len(candidate_books)}")

# Filtra i libri in base al numero di pagine
daily_pages = user_reading_habits.get(53425, 50)
max_pages = daily_pages * 10
candidate_books = candidate_books[candidate_books["pageCount"] <= max_pages]
print(f"Libri dopo filtro pagine: {len(candidate_books)}")
