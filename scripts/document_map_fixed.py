input_path = "faiss_data/document_map.txt"
output_path = "faiss_data/document_map_fixed.txt"

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        doc_id, content = line.split("->", 1)  # Split by the arrow
        outfile.write(f"{doc_id.strip()}|{content.strip()}\n")
