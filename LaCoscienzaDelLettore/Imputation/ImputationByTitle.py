import requests
import pandas as pd
import time
import os
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

# geraldinemontella2@gmail.com   AIzaSyDP_s5vAoonrv0V5VWZ1ylGndxWprHxp1Q
# tedx                           AIzaSyBBVQhyaErTei9w1qprI8v88_flap1daCE out
# the hater                      AIzaSyDl2KjuUJHLG09oYpdfp_wDJ2qIVpuk-jc  xx
# mamma                          AIzaSyCHFb9OaKDWbp7PgN34A1kyj3vpbozz1SM xx

# Caricare il dataset esistente
file_path = "LaCoscienzaDelLettore\\goodreads\\books_enriched.csv"
if os.path.exists(file_path):
    books = pd.read_csv(file_path)
else:
    books = pd.read_csv("LaCoscienzaDelLettore\\goodreads\\books_enriched.csv")
    books["description"] = "N/A"
    books["pageCount"] = 0
    books["categories"] = "N/A"

# Impostazioni API
API_KEY = "AIzaSyDP_s5vAoonrv0V5VWZ1ylGndxWprHxp1Q"  # Sostituisci con la tua API Key
BASE_URL = "https://www.googleapis.com/books/v1/volumes?q="

PARALLEL_REQUESTS = 2  # Numero massimo di richieste in parallelo
MAX_RETRIES = 2 # Numero massimo di tentativi in caso di errore API

def get_book_info(index, isbn, originalTit, title, author, pub_year):
    """Recupera informazioni sui libri dalla Google Books API con gestione degli errori e retry."""
    
    def make_request(query):
        """Effettua la richiesta API e gestisce errori."""
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(BASE_URL + query + f"&key={API_KEY}", timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"⚠ Errore API su '{title}' (tentativo {attempt+1}): {e}")
                time.sleep(2)  # Ritardo tra i retry
        return None  # Se fallisce anche dopo i retry
    
        # proviamo con Titolo 
    if pd.notna(originalTit):
        query = f'intitle:"{(str(originalTit).replace(" ", "+"))}"'
        data = make_request(query)
        if data and "items" in data:
            print(f"✅ Trovato tramite titolo Originale: {originalTit}")
            return index, extract_book_info(data)
    else :
        query = f'intitle:"{(str(title).replace(" ", "+"))}"'
        data = make_request(query)
        if data and "items" in data:
            print(f"✅ Trovato tramite Titolo: {title}")
            return index, extract_book_info(data)
        

    # 1️⃣ Proviamo prima con ISBN
    if pd.notna(isbn) and len(str(isbn)) in [10, 13]:
        query = f"isbn:{isbn}"
        data = make_request(query)
        if data and "items" in data:
            print(f"✅ Trovato tramite ISBN: {isbn}")
            return index, extract_book_info(data)
        
    
    # 2️⃣ Se ISBN non ha funzionato, proviamo con Titolo + Autore
    query = f'intitle:"{quote(str(title))}"+inauthor:"{quote(str(author))}"'
    data = make_request(query)
    if data and "items" in data:
        print(f"✅ Trovato tramite Titolo+Autore: {title} - {author}")
        return index, extract_book_info(data)

    # 3️⃣ Se ancora nessun risultato, tentiamo con Titolo + Anno
    query = f'intitle:"{quote(str(title))}"+inpublisher:{pub_year}'
    data = make_request(query)
    if data and "items" in data:
        print(f"✅ Trovato tramite Titolo+Anno: {title} ({pub_year})")
        return index, extract_book_info(data)

    print(f"❌ Nessun risultato per '{title}' ({author})")
    return index, {"description": "N/A", "pageCount": 0, "categories": "N/A"}

def extract_book_info(data):
    """Estrae le informazioni di un libro dalla risposta API."""
    volume_info = data["items"][0]["volumeInfo"]
    return {
        "description": volume_info.get("description", "N/A"),
        "pageCount": volume_info.get("pageCount", 0),
        "categories": ", ".join(volume_info.get("categories", ["N/A"]))
    }

def update_book_data(result):
    """Aggiorna il dataset con i dati ottenuti."""
    if result:
        index, book_info = result
        if index in books.index:  # Assicuriamoci che l'indice sia valido
            books.loc[index, "description"] = book_info["description"]
            books.loc[index, "pageCount"] = book_info["pageCount"]
            books.loc[index, "categories"] = book_info["categories"]
            print(f"🔹 Aggiornato '{books.loc[index, 'title']}' con nuove info.")
        else:
            print(f"⚠ Errore: indice {index} non trovato nel DataFrame.")


# Filtra i libri che necessitano di aggiornamento
books_to_fetch = books[
    (books["description"].isna() | (books["description"] == "N/A")) |
    (books["pageCount"] == 0) |
    ((books["categories"] == "N/A") | (books["categories"].isna()))
].head(10).copy()

print(f"🔹 Processiamo {len(books_to_fetch)} libri su Google Books API...")
 
# Avvia richieste in parallelo con miglior gestione degli errori
with ThreadPoolExecutor(max_workers=PARALLEL_REQUESTS) as executor:
    futures = {
        executor.submit(get_book_info, idx, row["isbn"], row["original_title"], row["title"],  row["authors"], row["original_publication_year"]): idx
        for idx, row in books_to_fetch.iterrows()
    }
    
    for future in as_completed(futures):
        update_book_data(future.result())

# Salvataggio incrementale per evitare perdita di dati

books.to_csv("LaCoscienzaDelLettore\\goodreads\\books_enriched_2.csv", index=False)

books_after = pd.read_csv(file_path)
print(books_after[books_after["description"] != "N/A"].head(10))  # Controlla se i dati sono cambiati
print("✅ Dataset aggiornato e salvato!")
