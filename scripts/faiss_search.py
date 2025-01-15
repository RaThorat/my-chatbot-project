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
    return sorted(results, key=lambda x: x['distance'])

def search_faiss_with_content(query, top_k=5):
    try:
        # Load FAISS index
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}")
        index = faiss.read_index(FAISS_INDEX_PATH)

        # Load document chunks
        if not os.path.exists(DOC_CHUNKS_PATH):
            raise FileNotFoundError(f"doc_chunks.json not found at {DOC_CHUNKS_PATH}")

        with open(DOC_CHUNKS_PATH, "r", encoding="utf-8") as f:
            doc_chunks = json.load(f)  # Load the JSON array

        # Validate alignment
        if len(doc_chunks) != index.ntotal:
            logging.error(f"Error: Misalignment between FAISS index ({index.ntotal}) and doc_chunks.json ({len(doc_chunks)}).")
            logging.warning("Retrieval may be incomplete.")

        # Generate query embedding
        query_embedding = model.encode([query], convert_to_numpy=True)

        # Perform FAISS search
        distances, indices = index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            # Validate index bounds
            if idx >= len(doc_chunks) or idx < 0:
                logging.error(f"Invalid chunk index {idx} retrieved from FAISS. Skipping...")
                continue

            try:
                # Parse the document chunk
                chunk_entry = doc_chunks[idx]
                doc_id = chunk_entry.get("Doc ID")
                chunk_content = chunk_entry.get("Chunk")

                if not chunk_content:
                    logging.error(f"Missing content in chunk at index {idx}. Skipping...")
                    continue

                # Use chunk content directly for results
                results.append({"content": chunk_content, "distance": distances[0][i]})
            except Exception as e:
                logging.error(f"Error processing chunk index {idx}: {e}")

        return rerank_results(results, query)
    except Exception as e:
        logging.error(f"Error in search_faiss_with_content: {e}")
        return []

# Usage example
if __name__ == "__main__":
    query = "Wat is de rol van DUS-I in subsidies?"
    results = search_faiss_with_content(query, top_k=5)
    for res in results:
        print(f"Content: {res['content']}\nDistance: {res['distance']}\n")
