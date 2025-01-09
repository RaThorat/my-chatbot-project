import faiss
import sqlite3
from sentence_transformers import SentenceTransformer

# Laad hetzelfde model dat gebruikt werd voor de FAISS-index
model = SentenceTransformer('all-MiniLM-L6-v2')

def search_faiss_with_content(query, top_k=5):
    """
    Zoek relevante documenten met FAISS en retourneer inhoud en afstanden.
    """
    # Laad FAISS-index en document-ID's
    index = faiss.read_index("/home/RaThorat/my-chatbot-project/faiss_index.bin")
    with open("doc_ids.txt", "r") as f:
        doc_ids = [int(line.strip()) for line in f.readlines()]
    
    # Genereer embedding van de zoekopdracht
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # Voer FAISS-zoekopdracht uit
    distances, indices = index.search(query_embedding, top_k)
    
    # Haal documentinhoud op uit database
    conn = sqlite3.connect("/home/RaThorat/my-chatbot-project/documents.db")
    cursor = conn.cursor()
    results = []
    seen_docs = set()

    for i, idx in enumerate(indices[0]):
        if idx < len(doc_ids) and doc_ids[idx] not in seen_docs:  # Controleer validiteit en duplicaten
            cursor.execute("SELECT content FROM docs WHERE rowid = ?", (doc_ids[idx],))
            content = cursor.fetchone()
            if content:
                results.append((content[0], distances[0][i]))  # Voeg inhoud en afstand toe
                seen_docs.add(doc_ids[idx])

    conn.close()
    return results
