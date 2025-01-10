import os
import faiss
import sqlite3
from sentence_transformers import SentenceTransformer

# Laad hetzelfde model dat gebruikt werd voor de FAISS-index
model = SentenceTransformer('all-MiniLM-L6-v2')

# Padconfiguratie
BASE_PATH = "/home/RaThorat/my-chatbot-project/faiss_data"
FAISS_INDEX_PATH = os.path.join(BASE_PATH, "faiss_index.bin")
DOC_CHUNKS_PATH = os.path.join(BASE_PATH, "doc_chunks.txt")
DB_PATH = "/home/RaThorat/my-chatbot-project/documents.db"

def rerank_results(results, query):
    """
    Herordent resultaten op basis van semantische relevantie met de vraag.
    - results: Lijst van FAISS-resultaten met 'distance'.
    - query: De originele zoekopdracht.
    """
    # Voorbeeld: Herordenen op basis van afstand (hoe lager, hoe beter)
    reranked_results = sorted(results, key=lambda x: x['distance'])
    return reranked_results

def search_faiss_with_content(query, top_k=5):
    """
    Zoek relevante documenten met FAISS en retourneer de titel (eerste regel) en afstand.
    """
    try:
        # Controleer of bestanden bestaan
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(f"FAISS-index niet gevonden op {FAISS_INDEX_PATH}")
        if not os.path.exists(DOC_CHUNKS_PATH):
            raise FileNotFoundError(f"doc_chunks.txt niet gevonden op {DOC_CHUNKS_PATH}")
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(f"Database niet gevonden op {DB_PATH}")

        # Laad FAISS-index en document-chunks
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(DOC_CHUNKS_PATH, "r", encoding="utf-8") as f:
            doc_chunks = [line.strip() for line in f.readlines()]

        # Genereer embedding van de zoekopdracht
        query_embedding = model.encode([query], convert_to_numpy=True)

        # Voer FAISS-zoekopdracht uit
        distances, indices = index.search(query_embedding, top_k)

        # Haal documentinhoud en Markdown op uit database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        results = []

        for i, idx in enumerate(indices[0]):
            if idx < len(doc_chunks):
                doc_chunk = doc_chunks[idx]
                # Splits op om de chunk en originele document-ID te scheiden
                doc_id, chunk_content = doc_chunk.split("\nChunk: ", 1)
                doc_id = int(doc_id.replace("Doc ID: ", "").strip())

                cursor.execute("SELECT content, markdown FROM docs WHERE rowid = ?", (doc_id,))
                row = cursor.fetchone()
                if row:
                    content, markdown = row
                    title = markdown.split("\n", 1)[0].strip() if markdown else content.split("\n", 1)[0].strip()
                    title = title if title else f"Document zonder titel (ID: {doc_id})"

                    # Voeg Markdown toe aan het resultaat voor presentatie
                    results.append({
                        "title": title,
                        "content": markdown if markdown else chunk_content,
                        "distance": distances[0][i]
                    })

        conn.close()

        # Reranking stap toevoegen
        results = rerank_results(results, query)

        return results
    except Exception as e:
        print(f"Error in search_faiss_with_content: {e}")
        return []

# Usage example
if __name__ == "__main__":
    query = "Wat is de rol van DUS-I in subsidies?"
    results = search_faiss_with_content(query, top_k=5)
    for res in results:
        print(f"Title: {res['title']}\nContent: {res['content']}\nDistance: {res['distance']}\n")
