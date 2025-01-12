import logging

def process_doc_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_chunks = f.readlines()

    processed_chunks = []
    current_doc_id = None
    current_chunk = None

    for idx, line in enumerate(raw_chunks):
        line = line.strip()
        if line.startswith("Doc ID: "):
            if current_doc_id and current_chunk:
                processed_chunks.append(f"Doc ID: {current_doc_id}\nChunk: {current_chunk}")
            current_doc_id = line.replace("Doc ID: ", "").strip()
            current_chunk = ""
        elif line.startswith("Chunk: "):
            current_chunk = line.replace("Chunk: ", "").strip()
        elif current_chunk is not None:
            current_chunk += f" {line}"
        else:
            logging.warning(f"Skipping malformed line at index {idx}: {line}")

    if current_doc_id and current_chunk:
        processed_chunks.append(f"Doc ID: {current_doc_id}\nChunk: {current_chunk}")

    return processed_chunks
