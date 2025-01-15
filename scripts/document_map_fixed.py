input_path = "faiss_data/document_map.txt"
output_path = "faiss_data/document_map_fixed.txt"

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    current_doc_id = None
    current_content = []

    for line in infile:
        # Check if the line contains the delimiter "->"
        if "->" in line:
            # If we are already collecting content for a previous document, save it
            if current_doc_id is not None:
                # Write the current document ID and content
                outfile.write(f"{current_doc_id.strip()}|{' '.join(current_content).strip()}\n")
            
            # Start a new document
            current_doc_id, content = line.split("->", 1)
            current_doc_id = current_doc_id.strip()
            current_content = [content.strip()]
        else:
            # Append the line to the current content
            current_content.append(line.strip())
    
    # Write the last document
    if current_doc_id is not None:
        outfile.write(f"{current_doc_id.strip()}|{' '.join(current_content).strip()}\n")

