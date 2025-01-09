from faiss_search import search_faiss_with_content

query = "Wat doet DUS-I?"
results = search_faiss_with_content(query, top_k=5)

print("Zoekresultaten:")
for title, distance in results:
    print(f"Titel: {title} (Afstand: {distance})")


