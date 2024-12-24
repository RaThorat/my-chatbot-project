def segment_text(input_file, output_file):
    """
    Segmenteer tekst in zinnen en sla ze op in een JSONL-bestand.

    Parameters:
    - input_file (str): Pad naar het invoerbestand.
    - output_file (str): Pad naar het uitvoerbestand (JSONL-formaat).
    """
    import json
    from nltk.tokenize import sent_tokenize

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line = line.strip()
            if line and not line.startswith("###"):  # Negeer kopregels
                sentences = sent_tokenize(line)
                for sentence in sentences:
                    json_line = {"text": sentence}
                    outfile.write(json.dumps(json_line) + "\n")
    print(f"De tekst is gesegmenteerd en opgeslagen in '{output_file}'.")

# Zorg dat NLTK geïnstalleerd is
# pip install nltk
import nltk
nltk.download('punkt')

# Voer de functie uit
segment_text("/home/RaThorat/my-chatbot-project/Data/cleaned_documents.txt", "/home/RaThorat/my-chatbot-project/Data/segmented_documents.jsonl")

