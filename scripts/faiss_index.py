from sentence_transformers import SentenceTransformer
import sqlite3
import faiss
import os
import numpy as np  # Import numpy
import logging

# Configure logging
logging.basicConfig(
    filename="document_level_index.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize the model
model = SentenceTransformer('all-mpnet-base-v2')  # Choose a high-quality model

# Load documents from the database
def load_documents(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT rowid, content FROM docs")  # Fetch document ID and content
        results = cursor.fetchall()
        conn.close()
        return results
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        return []

db_path = "documents.db"
documents = load_documents(db_path)

if not documents:
    logging.error("No documents found in the database. Exiting.")
    exit()

# Extract document IDs and content, skipping empty or invalid documents
doc_ids = []
doc_texts = []
for doc_id, content in documents:
    if content and content.strip():
        doc_ids.append(doc_id)
        doc_texts.append(content)
    else:
        logging.warning(f"Skipping document ID {doc_id} due to empty or invalid content.")

if not doc_ids:
    logging.error("No valid documents to process. Exiting.")
    exit()

# Generate document-level embeddings in batches
def generate_embeddings_in_batches(texts, model, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings.append(model.encode(batch, convert_to_numpy=True))
    return np.vstack(embeddings)

logging.info(f"Generating embeddings for {len(doc_texts)} valid documents...")
doc_embeddings = generate_embeddings_in_batches(doc_texts, model)
logging.info(f"Generated {doc_embeddings.shape[0]} embeddings.")

# Create a FAISS index
embedding_dim = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(doc_embeddings)

logging.info(f"FAISS index created with {index.ntotal} entries.")

# Sanity check for alignment
if len(doc_embeddings) != len(doc_ids):
    logging.error(f"Mismatch: {len(doc_embeddings)} embeddings but {len(doc_ids)} documents!")
    exit()
else:
    logging.info("FAISS embeddings align with document IDs.")

# Save the FAISS index
output_dir = "faiss_data"
os.makedirs(output_dir, exist_ok=True)

faiss_index_path = os.path.join(output_dir, "faiss_index_documents.bin")
faiss.write_index(index, faiss_index_path)

# Save the mapping of embeddings to document IDs and excerpts
mapping_path = os.path.join(output_dir, "document_map.txt")
with open(mapping_path, "w", encoding="utf-8") as f:
    for doc_id, content in zip(doc_ids, doc_texts):
        excerpt = content[:100].replace("\n", " ")  # Save the first 100 characters as a preview
        f.write(f"{doc_id} -> {excerpt}\n")

logging.info(f"Document-level index and mappings saved at {output_dir}.")
