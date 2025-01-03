import os
import re
import json

def generate_labels_from_filenames(directory):
    """
    Genereer labels op basis van bestandsnamen.

    Parameters:
    - directory (str): Pad naar de map met bestanden.

    Returns:
    - list: Een lijst met tuples (bestandsnaam, label).
    """
    file_label_mapping = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            # Genereer een label op basis van de bestandsnaam
            label = generate_label_from_name(filename)
            file_label_mapping.append((filename, label))
    return file_label_mapping

def generate_label_from_name(filename):
    """
    Genereer een label op basis van de bestandsnaam.

    Parameters:
    - filename (str): Naam van het bestand.

    Returns:
    - str: Een gegenereerd label.
    """
    filename_lower = filename.lower()
    if "loting" in filename_lower:
        return "UITSLAG"
    elif "stageaanbieders" in filename_lower:
        return "STAGEAANBIEDERS"
    elif "subsidie" in filename_lower:
        return "SUBSIDIE_INFORMATIE"
    elif "project" in filename_lower:
        return "PROJECT_DETAILS"
    elif "activiteitenplan" in filename_lower:
        return "PLAN"
    elif "plan" in filename_lower:
        return "PLAN"
    elif "formulieren" in filename_lower:
        return "FORMULIEREN"
    elif "interview" in filename_lower:
        return "INTERN_DUSI"
    elif "stand van uitvoering 2023" in filename_lower:
        return "INTERN_DUSI"
    elif "terugblik 2023" in filename_lower:
        return "INTERN_DUSI"
    elif "samenvattingen" in filename_lower:
        return "SAMENVATTINGEN"
    elif "handleiding" in filename_lower:
        return "HANDLEIDINGEN"
    elif "toelichting" in filename_lower:
        return "HANDLEIDINGEN"
    elif "vragenlijst" in filename_lower:
        return "VRAGENLIJST"
    elif "handreiking" in filename_lower:
        return "HANDLEIDINGEN"
    else:
        return "INZICHT"

def split_text_into_segments(text):
    """
    Verdeel tekst in kleinere paragrafen of zinnen.

    Parameters:
    - text (str): De volledige tekst.

    Returns:
    - list: Een lijst met tekstsegmenten.
    """
    # Verdeel tekst op basis van lege regels of zinnen
    segments = re.split(r'\n\s*\n|(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return [segment.strip() for segment in segments if segment.strip()]

def create_jsonl_from_files(directory, output_file):
    """
    Maak een JSONL-bestand op basis van bestanden in een map.

    Parameters:
    - directory (str): Pad naar de map met bestanden.
    - output_file (str): Pad naar het uitvoerbestand.
    """
    file_label_mapping = generate_labels_from_filenames(directory)
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename, label in file_label_mapping:
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as infile:
                text = infile.read().strip()
                segments = split_text_into_segments(text)
                for segment in segments:
                    json_line = {"text": segment, "label": label}
                    outfile.write(json.dumps(json_line) + "\n")
    print(f"JSONL-bestand gemaakt: {output_file}")

# Gebruik het script
directory = "/home/gebruiker/Documenten/git_workspace/my-chatbot-project/Data/raw"
output_file = "/home/gebruiker/Documenten/git_workspace/my-chatbot-project/Data/raw_labeled.jsonl"
create_jsonl_from_files(directory, output_file)
