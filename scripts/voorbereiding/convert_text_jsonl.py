import os
import json
from odf.opendocument import load
from odf.text import P
from PyPDF2 import PdfReader

# Paths
raw_data_path = './Data/raw'
output_txt_path = './Data/txt'
input_jsonl_file = './Data/cleaned_labeled_data.jsonl'
filtered_jsonl_file = './Data/filtered_data.jsonl'
output_json_file = './Data/processed_data.json'

# Ensure output directory exists
os.makedirs(output_txt_path, exist_ok=True)

# Helper functions
def convert_odt_to_txt(file_path, output_path):
    doc = load(file_path)
    paragraphs = []
    for paragraph in doc.getElementsByType(P):
        # Extract text from child nodes of the paragraph
        text = ''.join([node.data for node in paragraph.childNodes if node.nodeType == 3])  # NodeType 3 is a text node
        paragraphs.append(text)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\\n".join(paragraphs))


def convert_pdf_to_txt(file_path, output_path):
    reader = PdfReader(file_path)
    text = "\n".join([page.extract_text() for page in reader.pages])
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

def process_files():
    for file_name in os.listdir(raw_data_path):
        file_path = os.path.join(raw_data_path, file_name)
        output_path = os.path.join(output_txt_path, f"{os.path.splitext(file_name)[0]}.txt")
        
        if file_name.endswith('.odt'):
            convert_odt_to_txt(file_path, output_path)
        elif file_name.endswith('.pdf'):
            convert_pdf_to_txt(file_path, output_path)
        elif file_name.endswith('.txt'):
            # Copy .txt files directly to the output directory
            with open(file_path, 'r', encoding='utf-8') as src, open(output_path, 'w', encoding='utf-8') as dest:
                dest.write(src.read())

def search_and_filter_jsonl():
    with open(input_jsonl_file, 'r', encoding='utf-8') as f:
        jsonl_data = [json.loads(line) for line in f]

    processed_documents = []
    filtered_data = []
    doc_id = 1

    for txt_file in os.listdir(output_txt_path):
        if not txt_file.endswith('.txt'):
            continue

        with open(os.path.join(output_txt_path, txt_file), 'r', encoding='utf-8') as f:
            content = f.read()

        matched_chunks = []
        for entry in jsonl_data[:]:  # Iterate over a copy to allow removal
            if entry['text'] in content:
                matched_chunks.append({"Chunk": entry['text']})
                filtered_data.append(entry)  # Add entry to filtered data
                jsonl_data.remove(entry)  # Remove matched entry from the original list

        if matched_chunks:
            processed_documents.append({
                "Doc ID": doc_id,
                "Title": os.path.splitext(txt_file)[0],
                "Chunks": matched_chunks
            })
            doc_id += 1

    # Save processed data to JSON
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump({"Documents": processed_documents}, f, ensure_ascii=False, indent=4)

    # Save filtered JSONL
    with open(filtered_jsonl_file, 'w', encoding='utf-8') as f:
        for entry in filtered_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    process_files()
    search_and_filter_jsonl()
    print("Processing completed.")
