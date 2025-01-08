from faiss_search import search_faiss

query = "Wat doet DUS-i?"
results = search_faiss(query)

print("Zoekresultaten:")
for doc_id, distance in results:
    print(f"Document ID: {doc_id}, Afstand: {distance}")
