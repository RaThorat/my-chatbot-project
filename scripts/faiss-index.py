import os
import sqlite3
import faiss
from sentence_transformers import SentenceTransformer

# Stap 1: Laad het embedding-model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Model laden

# Stap 2: Laad de data uit je database
db_path = "documents.db"

def load_documents(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT rowid, content FROM docs")  # Haal ID en tekstinhoud op
        results = cursor.fetchall()
        conn.close()
        return results
    except sqlite3.Error as e:
        print(f"Databasefout: {e}")
        return []

documents = load_documents(db_path)
if not documents:
    print("Geen documenten gevonden in de database. Controleer uw databaseconfiguratie.")
    exit()

doc_ids = [doc[0] for doc in documents]
doc_texts = [doc[1] for doc in documents]

# Stap 3: Genereer embeddings
print(f"Genereren van embeddings voor {len(doc_texts)} documenten...")
doc_embeddings = model.encode(doc_texts, convert_to_numpy=True)

# Stap 4: FAISS-index maken
embedding_dim = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(doc_embeddings)

print(f"FAISS-index bevat {index.ntotal} vectoren.")

# Stap 5: Sla de FAISS-index en document-IDs op
output_dir = "faiss_data"
os.makedirs(output_dir, exist_ok=True)

faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
with open(os.path.join(output_dir, "doc_ids.txt"), "w") as f:
    for doc_id in doc_ids:
        f.write(f"{doc_id}\n")

print(f"FAISS-index opgeslagen in {output_dir}/faiss_index.bin")
print(f"Document-ID's opgeslagen in {output_dir}/doc_ids.txt")
