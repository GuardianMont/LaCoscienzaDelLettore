import requests
import pandas as pd
import time
import os
from concurrent.futures import ThreadPoolExecutor

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
API_KEY = "AIzaSyBBVQhyaErTei9w1qprI8v88_flap1daCE"  # Sostituisci con la tua API Key
BASE_URL = "https://www.googleapis.com/books/v1/volumes?q="

PARALLEL_REQUESTS = 5  # Numero massimo di richieste in parallelo
REQUEST_DELAY = 1  # Ritardo tra le richieste per evitare blocchi

# Contatore richieste
request_count = 0

def get_book_info(index, isbn, title):
    """Fetch dei dati dalla Google Books API"""
    global request_count

    # Proviamo prima con l'ISBN-13
    if pd.notna(isbn):  
        # Controlla se l'ISBN è già un ISBN-13 (lunghezza 13)
        if len(str(isbn)) == 13:
            query = f"isbn:{isbn}"
            print(f"🔍 Cerco libro con ISBN-13: {isbn}")
        else:
            # Se l'ISBN è di 10 caratteri, lo convertiamo in ISBN-13
            isbn13 = convert_isbn10_to_isbn13(isbn)
            query = f"isbn:{isbn13}"
            print(f"🔍 Cerco libro con ISBN-13 (convertito): {isbn13}")
    else:
        query = f"intitle:{title}"
        print(f"🔍 Cerco libro con titolo: {title}")

    try:
        response = requests.get(BASE_URL + query + f"&key={API_KEY}", timeout=10)
        response.raise_for_status()
        data = response.json()

        # Se troviamo il libro, restituiamo i dati
        if "items" in data:
            volume_info = data["items"][0]["volumeInfo"]
            book_info = {
                "description": volume_info.get("description", "N/A"),
                "pageCount": volume_info.get("pageCount", 0),
                "categories": ", ".join(volume_info.get("categories", ["N/A"]))
            }
            request_count += 1
            return index, book_info
        else:
            print(f"⚠ Nessun libro trovato per ISBN {isbn} / Titolo {title}, tentando con il titolo...")

            # Fallback: Proviamo con il titolo se ISBN non ha funzionato
            query = f"intitle:{title}"
            response = requests.get(BASE_URL + query + f"&key={API_KEY}", timeout=10)
            response.raise_for_status()
            data = response.json()

            if "items" in data:
                volume_info = data["items"][0]["volumeInfo"]
                book_info = {
                    "description": volume_info.get("description", "N/A"),
                    "pageCount": volume_info.get("pageCount", 0),
                    "categories": ", ".join(volume_info.get("categories", ["N/A"]))
                }
                request_count += 1
                return index, book_info
            else:
                print(f"⚠ Nessun libro trovato nemmeno con il titolo {title}.")

    except requests.exceptions.RequestException as e:
        print(f"⚠ Errore API su '{title}': {e}")
    
    return index, {"description": "N/A", "pageCount": 0, "categories": "N/A"}

def convert_isbn10_to_isbn13(isbn10):
    """Converte un ISBN-10 in ISBN-13"""
    isbn10 = str(isbn10)
    if len(isbn10) == 10:
        isbn13 = '978' + isbn10[:-1]
        check_digit = calculate_check_digit(isbn13)
        return isbn13 + check_digit
    return isbn10  # Se l'ISBN non è valido, restituiamo quello che abbiamo

def calculate_check_digit(isbn13):
    """Calcola il carattere di controllo per un ISBN-13"""
    total = 0
    for i, digit in enumerate(isbn13):
        weight = 1 if i % 2 == 0 else 3
        total += int(digit) * weight
    remainder = total % 10
    return str((10 - remainder) % 10)



def update_book_data(result):
    """Aggiorna il dataset con i dati ottenuti"""
    if result:
        index, book_info = result
        books.at[index, "description"] = book_info["description"]
        books.at[index, "pageCount"] = book_info["pageCount"]
        books.at[index, "categories"] = book_info["categories"]

# Filtra i libri che hanno descrizione "N/A" o valori NaN e pageCount = 0
books_to_fetch = books[(books["description"].isna() | (books["description"] == "N/A")) & (books["pageCount"] == 0)].head(200)

print(f"🔹 Processiamo {len(books_to_fetch)} libri su Google Books API...")

# Avvia richieste in parallelo
with ThreadPoolExecutor(max_workers=PARALLEL_REQUESTS) as executor:
    futures = [executor.submit(get_book_info, idx, row["isbn"], row["title"]) for idx, row in books_to_fetch.iterrows()]
    
    for future in futures:
        update_book_data(future.result())
        time.sleep(REQUEST_DELAY)  # Evitiamo di superare i limiti API

# Salvataggio finale
books.to_csv(file_path, index=False)
print("✅ Dataset aggiornato e salvato!")
