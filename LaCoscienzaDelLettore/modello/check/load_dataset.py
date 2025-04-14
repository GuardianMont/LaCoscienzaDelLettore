import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carica il dataset
file_path = "LaCoscienzaDelLettore\modello\data.csv"  # Cambia con il nome corretto del tuo file
df = pd.read_csv(file_path)

# Mostra le prime righe
print("\n Prime righe del dataset:")
print(df.head())

# Informazioni sulle colonne
print("\n🔍 Info sul dataset:")
print(df.info())

# Controllo valori nulli
print("\n Valori nulli per colonna:")
print(df.isnull().sum())

# Analisi delle valutazioni
plt.figure(figsize=(10,5))
sns.histplot(df['average_rating'].dropna(), bins=20, kde=True)
plt.title("Distribuzione delle valutazioni dei libri")
plt.xlabel("Valutazione media")
plt.ylabel("Frequenza")
plt.show()

# Sostituzione valori nulli
df['subtitle'].fillna("", inplace=True)
df['authors'].fillna("Autore sconosciuto", inplace=True)
df['categories'].fillna("Senza categoria", inplace=True)
df['description'].fillna("Nessuna descrizione disponibile", inplace=True)

# Immagine placeholder per libri senza copertina
df['thumbnail'].fillna("https://www.mrw.it/wp-content/uploads/2020/12/0iwkf4_1609360688.jpg", inplace=True)

# Sostituiamo l'anno di pubblicazione mancante con la mediana
df['published_year'].fillna(df['published_year'].median(), inplace=True)

# Sostituiamo i valori nulli nelle colonne numeriche con la media
num_cols = ['average_rating', 'num_pages', 'ratings_count']
for col in num_cols:
    df[col].fillna(df[col].mean(), inplace=True)

# Controlliamo se ci sono ancora valori nulli
print("\n Controllo dopo la pulizia:")
print(df.isnull().sum())

# Salviamo il dataset pulito
df.to_csv("dataset_clean.csv", index=False)
print("\n Dataset pulito salvato come 'dataset_clean.csv'")

