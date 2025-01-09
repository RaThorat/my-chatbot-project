import os
import faiss
import sqlite3
from sentence_transformers import SentenceTransformer

# Laad hetzelfde model dat gebruikt werd voor de FAISS-index
model = SentenceTransformer('all-MiniLM-L6-v2')

# Padconfiguratie
BASE_PATH = "/home/RaThorat/my-chatbot-project/faiss_data"
FAISS_INDEX_PATH = os.path.join(BASE_PATH, "faiss_index.bin")
DOC_IDS_PATH = os.path.join(BASE_PATH, "doc_ids.txt")
DB_PATH = "/home/RaThorat/my-chatbot-project/documents.db"

def search_faiss_with_content(query, top_k=5):
    """
    Zoek relevante documenten met FAISS en retourneer de titel (eerste regel) en afstand.
    """
    try:
        # Controleer of bestanden bestaan
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(f"FAISS-index niet gevonden op {FAISS_INDEX_PATH}")
        if not os.path.exists(DOC_IDS_PATH):
            raise FileNotFoundError(f"doc_ids.txt niet gevonden op {DOC_IDS_PATH}")
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(f"Database niet gevonden op {DB_PATH}")

        # Laad FAISS-index en document-ID's
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(DOC_IDS_PATH, "r") as f:
            doc_ids = [int(line.strip()) for line in f.readlines()]

        # Genereer embedding van de zoekopdracht
        query_embedding = model.encode([query], convert_to_numpy=True)

        # Voer FAISS-zoekopdracht uit
        distances, indices = index.search(query_embedding, top_k)

        # Haal documentinhoud op uit database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        results = []
        seen_titles = set()  # Om duplicaat-titels te detecteren
        seen_docs = set()  # Om duplicaat-documenten te detecteren

        for i, idx in enumerate(indices[0]):
            if idx < len(doc_ids) and doc_ids[idx] not in seen_docs:
                cursor.execute("SELECT content FROM docs WHERE rowid = ?", (doc_ids[idx],))
                content = cursor.fetchone()
                if content:
                    # Haal de eerste regel als titel
                    title = content[0].split("\n", 1)[0].strip()
                    title = title if title else f"Document zonder titel (ID: {doc_ids[idx]})"
                    
                    # Controleer op duplicaten op basis van de titel
                    if title not in seen_titles:
                        distance = distances[0][i] if distances[0][i] else 1.0  # Gebruik standaard waarde
                        results.append((title, distance))
                        seen_titles.add(title)  # Voeg titel toe aan de lijst van unieke titels
                else:
                    results.append((f"Geen inhoud beschikbaar voor document ID {doc_ids[idx]}", float("inf")))
                seen_docs.add(doc_ids[idx])  # Voeg document-ID toe om duplicaten te voorkomen
            else:
                print(f"Invalid or duplicate FAISS index {idx}.")

        conn.close()
        return results
    except Exception as e:
        logging.error(f"Error in search_faiss_with_content: {e}")
        return []

