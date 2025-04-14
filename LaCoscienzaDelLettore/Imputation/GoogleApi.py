import requests
import pandas as pd
import time

# Caricare il dataset esistente
books = pd.read_csv("LaCoscienzaDelLettore\\goodreads\\books.csv")

# Funzione per ottenere dati da Google Books API
def get_book_info(isbn, title):
    api_key = "AIzaSyDl2KjuUJHLG09oYpdfp_wDJ2qIVpuk-jc"  # <-- Sostituisci con la tua API Key di Google Books
    base_url = "https://www.googleapis.com/books/v1/volumes?q="

    if pd.notna(isbn):  # Se l'ISBN è disponibile, cerchiamo con esso
        query = f"isbn:{isbn}"
    else:  # Altrimenti, cerchiamo per titolo
        query = f"intitle:{title}"

    response = requests.get(base_url + query + f"&key={api_key}")
    data = response.json()

    if "items" in data:
        volume_info = data["items"][0]["volumeInfo"]
        return {
            "description": volume_info.get("description", "N/A"),
            "pageCount": volume_info.get("pageCount", 0),
            "categories": ", ".join(volume_info.get("categories", ["N/A"]))
        }
    return {"description": "N/A", "pageCount": 0, "categories": "N/A"}

# Creare nuove colonne per il dataset
books["description"] = "N/A"
books["pageCount"] = 0
books["categories"] = "N/A"

# Iterare sui libri e arricchire il dataset
for index, row in books.iterrows():
    print(f"Fetching info for: {row['title']}")
    
    book_info = get_book_info(row["isbn"], row["title"])
    
    books.at[index, "description"] = book_info["description"]
    books.at[index, "pageCount"] = book_info["pageCount"]
    books.at[index, "categories"] = book_info["categories"]
    
    time.sleep(1)  # Per evitare di superare i limiti di richiesta dell'API

# Salvare il dataset arricchito
books.to_csv("books_enriched.csv", index=False)
print("Dataset aggiornato e salvato come 'books_enriched.csv'")
