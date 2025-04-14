# 📚 La Coscienza del Lettore

**La Coscienza del Lettore** è un sistema di raccomandazione intelligente progettato per **stimolare l’abitudine alla lettura** e offrire **suggerimenti personalizzati** in base a:

- 🎯 **Gusti letterari dell'utente**
- ⏱️ **Abitudini di lettura** (es. pagine lette al giorno)
- 📚 **Libreria personale** e libri già letti

Il progetto integra tecniche di **Collaborative Filtering (SVD)** e **Content-Based Filtering (TF-IDF)** per generare raccomandazioni efficaci, varie e realistiche.  
L’obiettivo non è solo consigliare libri, ma creare un’esperienza **coinvolgente e su misura** per ogni lettore.

---

## 🔗 GitHub del progetto

[🔗 Visita il repository](https://github.com/GuardianMont/LaCoscienzaDelLettore)

---

---

## 📄 Documentazione del Progetto

La documentazione completa del progetto (analisi, modello, dataset, interfaccia, valutazione) è disponibile nella cartella [`documentazione`](https://github.com/GuardianMont/LaCoscienzaDelLettore/tree/main/LaCoscienzaDelLettore/documentazione) del repository.

Contiene:

- 📘 Relazione descrittiva in PDF
- 🧠 Approfondimenti sul modello ibrido
- 📊 Valutazioni e risultati

📥 [Scarica la relazione completa](https://github.com/GuardianMont/LaCoscienzaDelLettore/blob/main/documentazione/La_Coscienza_Del_Lettore.pdf)

---


## 🚀 Funzionalità principali

- **Consigli personalizzati** basati sul modello SVD (Surprise)
- **Raccomandazioni simili** tramite analisi testuale TF-IDF + Cosine Similarity
- **Gestione della libreria personale**: aggiunta di libri con valutazioni e pagine lette
- **Registrazione utenti** con assegnazione ID univoco
- **Interfaccia utente interattiva** sviluppata con Gradio

---

## 🧠 Tecnologie utilizzate

- Python
- Pandas, NumPy
- NLTK
- Scikit-learn
- Surprise (SVD)
- Gradio
- Google Books API (per arricchimento dati)

---

## 📌 Obiettivi del progetto

- Offrire **suggerimenti su misura**
- Considerare **tempo disponibile e motivazione**
- **Evitare ripetitività** nei generi consigliati
- Promuovere la **lettura come abitudine quotidiana**

---

## 📥 Dataset

- **Goodbooks-10k** (da Kaggle)
- Arricchimento automatico tramite **Google Books API**

---

## 🛠️ Come eseguirlo

1. Clona il repository:
   ```bash
   git clone https://github.com/tuo-username/la-coscienza-del-lettore.git
   cd la-coscienza-del-lettore
