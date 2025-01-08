import faiss
from sentence_transformers import SentenceTransformer

# Laad hetzelfde model dat gebruikt werd voor de FAISS-index
model = SentenceTransformer('all-MiniLM-L6-v2')

def search_faiss(query, top_k=5):
    """
    Zoek relevante documenten in de FAISS-index.
    """
    try:
        # Laad FAISS-index en document-ID's
        index = faiss.read_index("/home/RaThorat/my-chatbot-project/faiss_index.bin")
        with open("/home/RaThorat/my-chatbot-project/doc_ids.txt", "r") as f:
            doc_ids = [int(line.strip()) for line in f.readlines()]

        # Genereer embedding van de zoekopdracht
        query_embedding = model.encode([query], convert_to_numpy=True)

        # Zoek in de FAISS-index
        distances, indices = index.search(query_embedding, top_k)

        # Verwerk resultaten
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(doc_ids):  # Controleer of de index geldig is
                results.append((doc_ids[idx], distances[0][i]))
        return results
    except Exception as e:
        print(f"Error in search_faiss: {e}")
        return []
