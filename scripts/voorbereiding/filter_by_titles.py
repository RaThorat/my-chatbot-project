import os
import shutil
import json

# Paths
json_file_path = './faiss_data/doc_chunks.json'  # Path to the JSON file
raw_data_path = './Data/raw'  # Path to the raw files
filtered_data_path = './Data/filtered'  # Path to store filtered files

# Create the filtered directory if it doesn't exist
os.makedirs(filtered_data_path, exist_ok=True)

# Load the JSON data
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

titles = [doc['Title'] for doc in data['Documents']]  # Extract titles from JSON

# Iterate over files in the raw data directory
for file_name in os.listdir(raw_data_path):
    # Check if any title matches the file name
    if any(title in file_name for title in titles):
        source_path = os.path.join(raw_data_path, file_name)
        dest_path = os.path.join(filtered_data_path, file_name)
        
        # Copy the file to the filtered directory
        shutil.copy2(source_path, dest_path)
        print(f"Copied: {file_name}")

print("Filtering complete.")
