from faiss_search import search_faiss_with_content

query = "Wat doet DUS-i?"
results = search_faiss_with_content(query)

print("Zoekresultaten:")
for content, distance in results:
    print(f"Content: {content[:100]}..., Afstand: {distance}")
