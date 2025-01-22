import os
import faiss
from sentence_transformers import SentenceTransformer
import logging
import json

# Configure logging
logging.basicConfig(
    filename="faiss_search.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load the same model used for the FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')

# Path configuration
BASE_PATH = "faiss_data"
FAISS_INDEX_PATH = os.path.join(BASE_PATH, "faiss_index.bin")
DOC_CHUNKS_PATH = os.path.join(BASE_PATH, "doc_chunks.json")

def rerank_results(results, query):
    """
    Rerank results based on semantic relevance to the query.
    """
    # Filter out results without a 'distance' key
    valid_results = [r for r in results if 'Distance' in r]
    # Sort the valid results by the 'distance' value
    return sorted(valid_results, key=lambda x: x['Distance'])

def load_faiss_index(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index not found at {path}")
    return faiss.read_index(path)

def load_doc_chunks(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"doc_chunks.json not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def search_faiss_with_content(query, top_k=5):
    try:
        # Laad index en documenten
        index = load_faiss_index(FAISS_INDEX_PATH)
        doc_chunks = load_doc_chunks(DOC_CHUNKS_PATH)

        # Verwerk documenten naar platte structuur
        flat_chunks = []
        for doc in doc_chunks["Documents"]:
            for chunk in doc["Chunks"]:
                flat_chunks.append({
                    "Doc ID": doc["Doc ID"],
                    "Title": doc["Title"],
                    "Chunk": chunk["Chunk"]
                })

        if len(flat_chunks) != index.ntotal:
            raise ValueError("Misalignment tussen FAISS index en documenten.")

        # Genereer query-embedding
        query_embedding = model.encode([query], convert_to_numpy=True)

        # FAISS-zoekopdracht
        distances, indices = index.search(query_embedding, top_k)
        results = []

        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(flat_chunks):
                continue  # Ongeldig index
            chunk_entry = flat_chunks[idx]
            results.append({
                "Doc ID": chunk_entry.get("Doc ID"),
                "Title": chunk_entry.get("Title"),
                "Content": chunk_entry.get("Chunk"),
                "Distance": distances[0][i]
            })

        return rerank_results(results, query)

    except FileNotFoundError as fnf:
        logging.error(f"Bestand niet gevonden: {fnf}")
    except Exception as e:
        logging.error(f"Fout in FAISS-zoekopdracht: {traceback.format_exc()}")

    return []

# Usage example
if __name__ == "__main__":
    try:
        query = "Wat is de rol van DUS-I in subsidies?"
        results = search_faiss_with_content(query, top_k=5)
        for res in results:
            print(f"Doc ID: {res['Doc ID']}\nTitle: {res['Title']}\nContent: {res['Content']}\nDistance: {res['Distance']}\n")
    except Exception as e:
        print(f"Er is een fout opgetreden: {e}")
