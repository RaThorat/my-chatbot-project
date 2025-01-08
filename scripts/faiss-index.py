import os
import sqlite3
import faiss
from sentence_transformers import SentenceTransformer

# Stap 1: Laad het embedding-model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Snel en efficiënt voor kleinere datasets

# Stap 2: Laad de data uit je database
db_path = "documents.db"

def load_documents(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT rowid, content FROM docs")  # Haal ID en tekstinhoud op
    results = cursor.fetchall()
    conn.close()
    return results  # [(id, content), ...]

documents = load_documents(db_path)
doc_ids = [doc[0] for doc in documents]
doc_texts = [doc[1] for doc in documents]

# Stap 3: Genereer embeddings
print("Genereren van embeddings...")
doc_embeddings = model.encode(doc_texts, convert_to_numpy=True)

# Stap 4: FAISS-index maken
embedding_dim = doc_embeddings.shape[1]  # Dimensie van de embeddings
index = faiss.IndexFlatL2(embedding_dim)  # L2 afstand (Euclidische afstand)
index.add(doc_embeddings)  # Voeg embeddings toe aan de FAISS-index

print(f"FAISS-index bevat {index.ntotal} vectoren.")

# Stap 5: Sla de FAISS-index en document-IDs op
faiss.write_index(index, "faiss_index.bin")  # Sla de index op
with open("doc_ids.txt", "w") as f:
    for doc_id in doc_ids:
        f.write(f"{doc_id}\n")

print("FAISS-index en document-ID's opgeslagen.")
