import os
import faiss
import json
from sentence_transformers import SentenceTransformer

# Step 1: Define paths
input_file_path = "./faiss_data/doc_chunks.json"
output_dir = "faiss_data"
os.makedirs(output_dir, exist_ok=True)
faiss_index_path = os.path.join(output_dir, "faiss_index.bin")
doc_chunks_output_path = os.path.join(output_dir, "doc_chunks.json")
embedding_doc_map_path = os.path.join(output_dir, "embedding_doc_map.txt")

# Step 2: Load chunks from doc_chunks.json
def load_chunks_json(file_path):
    """Loads document IDs and chunks from a JSON file."""
    doc_ids = []
    chunks = []
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)  # Load the entire JSON array
        for entry in data:
            doc_ids.append(entry["Doc ID"])
            chunks.append(entry["Chunk"])
    return doc_ids, chunks

print("Loading chunks from doc_chunks.json...")
doc_ids, all_chunks = load_chunks_json(input_file_path)
print(f"Loaded {len(all_chunks)} chunks.")

# Step 3: Generate embeddings
print("Loading SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"Generating embeddings for {len(all_chunks)} chunks...")
embeddings = model.encode(all_chunks, convert_to_numpy=True)
print(f"Generated {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}.")

# Step 4: Create and populate FAISS index
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)
print(f"FAISS index created with {index.ntotal} vectors.")

# Step 5: Validate alignment
if len(all_chunks) != len(doc_ids):
    raise ValueError("Mismatch between number of chunks and document IDs.")
if len(all_chunks) != index.ntotal:
    raise ValueError("Mismatch between FAISS index size and number of chunks.")

# Step 6: Save FAISS index
print(f"Saving FAISS index to {faiss_index_path}...")
faiss.write_index(index, faiss_index_path)

# Step 7: Save chunks and mappings
print(f"Saving chunks to {doc_chunks_output_path}...")
with open(doc_chunks_output_path, "w", encoding="utf-8") as chunk_file:
    json.dump([{"Doc ID": doc_id, "Chunk": chunk} for doc_id, chunk in zip(doc_ids, all_chunks)], chunk_file, ensure_ascii=False, indent=4)

print(f"Saving embedding-to-document mapping to {embedding_doc_map_path}...")
with open(embedding_doc_map_path, "w", encoding="utf-8") as map_file:
    for idx, doc_id in enumerate(doc_ids):
        map_file.write(f"Embedding {idx} -> Document ID: {doc_id}\n")

print("All outputs saved successfully.")
