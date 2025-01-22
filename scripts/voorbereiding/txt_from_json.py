import os
import json

# Paths
json_file_path = './faiss_data/doc_chunks.json'  # Path to the JSON file
txt_data_path = './Data/txt'  # Path to store text files

# Create the txt directory if it doesn't exist
os.makedirs(txt_data_path, exist_ok=True)

# Load the JSON data
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Iterate through the documents in the JSON
for document in data['Documents']:
    title = document['Title']
    chunks = document['Chunks']

    # Combine all chunks into a single string
    content = "\n".join(chunk['Chunk'] for chunk in chunks)

    # Generate a valid filename for the title
    safe_title = title.replace("+", "_").replace(" ", "_").replace("/", "_")
    txt_file_path = os.path.join(txt_data_path, f"{safe_title}.txt")

    # Write the content to the text file
    with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(content)

    print(f"Created: {txt_file_path}")

print("Text file generation complete.")
