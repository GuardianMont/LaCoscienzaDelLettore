import requests
import pandas as pd
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


# Caricare il dataset esistente
file_path = "LaCoscienzaDelLettore/goodreads/books_enriched.csv"
if os.path.exists(file_path):
    books = pd.read_csv(file_path)
else:
    books = pd.DataFrame(columns=["title", "authors", "original_publication_year", "isbn", "description", "pageCount", "categories"])
    books["description"] = "N/A"
    books["pageCount"] = 0
    books["categories"] = "N/A"

OPEN_LIBRARY_URL = "https://openlibrary.org/search.json"

PARALLEL_REQUESTS = 2  # Numero massimo di richieste in parallelo
MAX_RETRIES = 2  # Numero massimo di tentativi in caso di errore API

def get_book_info_open_library(title, author):
    """Prova a trovare il libro su Open Library."""
    if pd.isna(title) or pd.isna(author):
        return None  # Se mancano dati, non fare richieste API
    
    query = f"title={title}&author={author}"
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(f"{OPEN_LIBRARY_URL}?{query}", timeout=10)
            response.raise_for_status()
            data = response.json()
            print (data)
            if "docs" in data and len(data["docs"]) > 0:
                book_data = data["docs"][0]  # Prendiamo il primo risultato
                return {
                    "description": book_data.get("first_sentence", ["N/A"])[0] if "first_sentence" in book_data else "N/A",
                    "pageCount": book_data.get("number_of_pages_median", 0),
                    "categories": ", ".join(book_data.get("subject", ["N/A"]))
                }
        except requests.exceptions.RequestException as e:
            print(f"⚠ Errore Open Library su '{title}' (tentativo {attempt+1}): {e}")
            time.sleep(2)  # Ritardo tra i retry
    return None

def update_book_data(index, book_info):
    """Aggiorna il dataset con i dati ottenuti."""
    if book_info:
        books.at[index, "description"] = book_info["description"]
        books.at[index, "pageCount"] = book_info["pageCount"]
        books.at[index, "categories"] = book_info["categories"]

# Filtra i libri che necessitano di aggiornamento
books_to_fetch = books[
    (books["description"].isna() | (books["description"] == "N/A")) &
    (books["pageCount"] == 0) &
    ((books["categories"] == "N/A") | (books["categories"].isna()))
].head(10)

print(f"🔹 Processiamo {len(books_to_fetch)} libri su Open Library API...")

# Avvia richieste in parallelo con miglior gestione degli errori
with ThreadPoolExecutor(max_workers=PARALLEL_REQUESTS) as executor:
    futures = {
        executor.submit(get_book_info_open_library, row["title"], row["authors"]): idx
        for idx, row in books_to_fetch.iterrows()
    }
    
    for future in as_completed(futures):
        update_book_data(futures[future], future.result())

# Salvataggio incrementale per evitare perdita di dati
books.to_csv(file_path, index=False)
print("✅ Dataset aggiornato e salvato!")