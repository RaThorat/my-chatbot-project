import json

# Define paths to the input and output files
input_file_path = './Data/raw_labeled_revised.jsonl'  # Replace with your input file path
output_file_path = './Data/cleaned_labeled_data.jsonl'  # Replace with your desired output file path

# Helper function to detect if a text contains a question
def contains_question(text):
    question_words = ["hoe", "wat", "waarom", "wie", "waar", "wanneer", "welke", "welk","?"]
    return any(word in text.lower() for word in question_words)

# Process the file
unique_entries = set()  # To track unique (text, label) pairs
cleaned_entries = []

with open(input_file_path, 'r', encoding='utf-8') as infile:
    for line in infile:
        # Parse the JSONL entry
        entry = json.loads(line.strip())
        text = entry.get("text", "").replace("\n", "").strip()
        label = entry.get("label", "")

        # 1. Remove entries with text length < 20 or > 400 words
        word_count = len(text.split())
        if word_count < 20 or word_count > 400:
            continue

        # 2. Remove entries where the text does not end with '.' or '?'
        if not text.endswith(('.', '?')):
            continue

        # 3. Change label to "VRAGENLIJST" if a question is detected
        if contains_question(text):
            label = "VRAGENLIJST"

        # 4. Remove duplicate (text, label) pairs
        if (text, label) in unique_entries:
            continue
        unique_entries.add((text, label))

        # Append cleaned entry
        cleaned_entries.append({"text": text, "label": label})

# Save the cleaned entries back to a JSONL file
with open(output_file_path, 'w', encoding='utf-8') as outfile:
    for entry in cleaned_entries:
        outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Cleaned data has been saved to: {output_file_path}")
