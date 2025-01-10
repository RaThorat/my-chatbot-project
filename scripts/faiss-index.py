import os
import sqlite3
import faiss
from sentence_transformers import SentenceTransformer
from convert_clean_combine import convert_to_markdown

def chunk_text_semantically(markdown_text, max_chunk_size=300):
    """
    Splits tekst op in semantische blokken, bijvoorbeeld op basis van Markdown-secties of paragrafen.
    - markdown_text: De volledige tekst in Markdown-indeling.
    - max_chunk_size: Maximale lengte van een chunk (in tokens of woorden).
    """
    chunks = []
    current_chunk = []

    for line in markdown_text.split("\n"):
        line = line.strip()
        if line.startswith("#"):  # Splits op Markdown-titels zoals H1, H2
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
        current_chunk.append(line)
        # Limiteer de grootte van chunks
        if sum(len(c.split()) for c in current_chunk) > max_chunk_size:
            chunks.append("\n".join(current_chunk))
            current_chunk = []

    # Voeg de laatste chunk toe
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks

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
doc_markdown = [convert_to_markdown(text) for text in doc_texts]  # Markdown genereren
all_chunks = []
doc_chunk_map = []  # Om te onthouden welke chunks bij welk document horen

for doc_id, markdown in zip(doc_ids, doc_markdown):
    if not markdown.strip():
        print(f"Document ID {doc_id} is leeg of ongeldig, overslaan.")
        continue
    chunks = chunk_text_semantically(markdown)
    all_chunks.extend(chunks)
    doc_chunk_map.extend([doc_id] * len(chunks))  # Map de chunks aan hun originele document-ID

print(f"Semantisch opgesplitst in {len(all_chunks)} chunks.")

# Opslaan van Markdown in de database (optioneel)
try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for doc_id, markdown in zip(doc_ids, doc_markdown):
        cursor.execute("UPDATE docs SET markdown = ? WHERE rowid = ?", (markdown, doc_id))
    conn.commit()
    conn.close()
    print("Markdown succesvol opgeslagen in de database.")
except sqlite3.Error as e:
    print(f"Databasefout bij opslaan van Markdown: {e}")

# Stap 3: Genereer embeddings
print(f"Genereren van embeddings voor {len(all_chunks)} chunks...")
doc_embeddings = model.encode(all_chunks, convert_to_numpy=True)

# Stap 4: FAISS-index maken
embedding_dim = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(doc_embeddings)

print(f"FAISS-index bevat {index.ntotal} vectoren.")

# Stap 5: Sla de FAISS-index en document-IDs op
output_dir = "faiss_data"
os.makedirs(output_dir, exist_ok=True)

# Verwijder oude bestanden indien nodig
if os.path.exists(os.path.join(output_dir, "faiss_index.bin")):
    os.remove(os.path.join(output_dir, "faiss_index.bin"))
if os.path.exists(os.path.join(output_dir, "doc_chunks.txt")):
    os.remove(os.path.join(output_dir, "doc_chunks.txt"))

faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
with open(os.path.join(output_dir, "doc_chunks.txt"), "w") as f:
    for doc_id, chunk in zip(doc_chunk_map, all_chunks):
        f.write(f"Doc ID: {doc_id}\nChunk: {chunk}\n\n")

print(f"FAISS-index opgeslagen in {output_dir}/faiss_index.bin")
print(f"Document-chunks opgeslagen in {output_dir}/doc_chunks.txt")
