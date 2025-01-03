import os
import re
import string
from odf.opendocument import load
from odf.text import P
from pdfminer.high_level import extract_text

# Functie om .odt-bestanden naar tekst te converteren
def convert_odt_to_txt(input_path, output_path):
    try:
        odt_file = load(input_path)
        paragraphs = [str(p) for p in odt_file.getElementsByType(P)]
        with open(output_path, "w", encoding="utf-8") as txt_file:
            txt_file.write("\n".join(paragraphs))
        print(f"ODT-bestand geconverteerd: {output_path}")
    except Exception as e:
        print(f"Fout bij het converteren van ODT: {e}")

# Functie om .pdf-bestanden naar tekst te converteren
def convert_pdf_to_txt(input_path, output_path):
    try:
        text = extract_text(input_path)
        with open(output_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(text)
        print(f"PDF-bestand geconverteerd: {output_path}")
    except Exception as e:
        print(f"Fout bij het converteren van PDF: {e}")

# Functie om tekst schoon te maken
def clean_text(text):
    # Verwijder HTML-tags
    text = re.sub(r"<.*?>", " ", text)
    # Verwijder speciale karakters en cijfers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Verwijder meerdere spaties
    text = re.sub(r"\s+", " ", text)
    # Normaliseer naar kleine letters
    return text.lower().strip()

# Functie om tekstbestanden te normaliseren
def normalize_txt_file(input_path, output_path):
    try:
        with open(input_path, "r", encoding="utf-8") as infile:
            text = infile.read()
        cleaned_text = clean_text(text)
        with open(output_path, "w", encoding="utf-8") as outfile:
            outfile.write(cleaned_text)
        print(f"Bestand opgeschoond en genormaliseerd: {output_path}")
    except Exception as e:
        print(f"Fout bij het normaliseren van bestand: {e}")

# Functie om alle bestanden in een map te combineren
def combine_text_files(input_folder, output_file):
    try:
        with open(output_file, "w", encoding="utf-8") as outfile:
            for filename in sorted(os.listdir(input_folder)):
                if filename.endswith(".txt"):
                    file_path = os.path.join(input_folder, filename)
                    with open(file_path, "r", encoding="utf-8") as infile:
                        outfile.write(f"### Bestand: {filename} ###\n")
                        outfile.write(infile.read())
                        outfile.write("\n\n")
        print(f"Alle bestanden zijn samengevoegd in: {output_file}")
    except Exception as e:
        print(f"Fout bij het combineren van bestanden: {e}")

# Hoofdfunctie
def process_files(input_folder, temp_folder, output_file):
    os.makedirs(temp_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        temp_path = os.path.join(temp_folder, os.path.splitext(filename)[0] + ".txt")

        if filename.endswith(".odt"):
            convert_odt_to_txt(input_path, temp_path)
        elif filename.endswith(".pdf"):
            convert_pdf_to_txt(input_path, temp_path)
        elif filename.endswith(".txt"):
            normalize_txt_file(input_path, temp_path)

    combine_text_files(temp_folder, output_file)

# Paden instellen
input_folder = "Data/raw"  # Map met ruwe bestanden
temp_folder = "Data/processed"  # Map voor tussenresultaten
output_file = "Data/combined_documents.txt"  # Uitvoerbestand

# Verwerk bestanden
process_files(input_folder, temp_folder, output_file)
