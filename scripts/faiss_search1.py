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
    Load the document map from the text file, returning a dictionary of document IDs and excerpts.
    """
    document_map = {}
    try:
        with open(document_map_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|", 1)  # Split by the delimiter '|'
                if len(parts) == 2:
                    doc_id, excerpt = parts
                    document_map[int(doc_id)] = excerpt
    except Exception as e:
        logging.error(f"Error loading document map: {e}")
    return document_map


def search_with_document_level(query, top_k=5):
    """
    Perform document-level search using FAISS and return results.
    """
    try:
        # Ensure FAISS index is loaded
        if not os.path.exists(faiss_index_path):
            raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}")
        index = faiss.read_index(faiss_index_path)

        # Load document mapping
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
            if idx < 0 or idx >= len(document_map):
                logging.warning(f"No document found for index {idx}. Skipping...")
                continue

            doc_id = idx + 1
            excerpt = document_map.get(doc_id, "No excerpt available")
            results.append({
                "doc_id": doc_id,
                "excerpt": excerpt,
                "distance": float(distance)  # Ensure JSON serialization compatibility
            })

        return results
    except Exception as e:
        logging.error(f"Error in search_with_document_level: {e}")
        return []


# Usage example
if __name__ == "__main__":
    query = "Wat is de rol van DUS-I in subsidies?"
    results = search_with_document_level(query, top_k=5)
    for res in results:
        print(f"Document ID: {res['doc_id']}\nExcerpt: {res['excerpt']}\nDistance: {res['distance']}\n")
