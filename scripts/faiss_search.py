import os
import sqlite3
import faiss
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(
    filename="document_level_search.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load model and FAISS index
model = SentenceTransformer('all-mpnet-base-v2')
faiss_index_path = "faiss_data/faiss_index_documents.bin"
document_map_path = "faiss_data/document_map.txt"
db_path = "documents.db"


def load_document_map(document_map_path):
    """
    Load the mapping between document IDs and content.
    """
    document_map = {}
    try:
        with open(document_map_path, "r", encoding="utf-8") as f:
            for line in f:
                doc_id, content = line.split("|", 1)  # Use '|' as the delimiter
                document_map[int(doc_id.strip())] = content.strip()
    except Exception as e:
        logging.error(f"Error loading document map: {e}")
    return document_map


def search_with_document_level(query, top_k=3):
    try:
        # Load FAISS index
        if not os.path.exists(faiss_index_path):
            raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}")
        index = faiss.read_index(faiss_index_path)

        # Load document mapping
        document_map_path = "faiss_data/document_map_fixed.txt"
        document_map = load_document_map(document_map_path)

        if len(document_map) != index.ntotal:
            logging.warning(f"Mismatch: FAISS index has {index.ntotal} vectors, but document map has {len(document_map)} entries!")

        # Generate query embedding
        query_embedding = model.encode([query], convert_to_numpy=True)

        # Perform FAISS search
        distances, indices = index.search(query_embedding, top_k)

        # Fetch and format results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < 0 or idx + 1 not in document_map:
                logging.warning(f"No document found for index {idx}. Skipping...")
                continue

            doc_id = idx + 1
            excerpt = document_map.get(doc_id, "No content available")
            results.append({"doc_id": doc_id, "excerpt": excerpt, "distance": float(distance)})

        return results
    except Exception as e:
        logging.error(f"Error in search_with_document_level: {e}")
        return []



# Usage example
if __name__ == "__main__":
    query = "Wat is de rol van DUS-I in subsidies?"
    results = search_with_document_level(query, top_k=3)
    for res in results:
        print(f"Document ID: {res['doc_id']}\nExcerpt: {res['excerpt']}\nDistance: {res['distance']}\n")
