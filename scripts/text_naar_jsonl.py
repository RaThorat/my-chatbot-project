import os
import re
import json

# Function to generate labels from filenames
def generate_labels_from_filenames(directory):
    """
    Generate labels based on filenames.
    """
    file_label_mapping = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            label = generate_label_from_name(filename)
            file_label_mapping.append((filename, label))
    return file_label_mapping

def generate_label_from_name(filename):
    """
    Generate a label based on the filename.
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
        return "ANDERS"
        
def is_valid_segment(segment):
    """
    Controleer of een segment geldig is en betekenisvolle inhoud heeft.
    """
    segment = segment.strip()
    # Minimaal aantal woorden en geen numerieke of speciale karakters
    return (
        len(segment.split()) > 2 and  # Meer dan 2 woorden
        not segment.isdigit() and     # Geen volledig numerieke segmenten
        not all(char in "|-_/." for char in segment)  # Geen segmenten met alleen speciale karakters
    )

def split_text_into_segments(text):
    """
    Split text into smaller paragraphs or sentences.
    """
    segments = re.split(r'\n\s*\n|(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return [segment.strip() for segment in segments if segment.strip()]

def assign_label_with_model(segment, textcat_pipeline):
    """
    Assign a label to a segment using a trained textcat model.
    """
    classification = textcat_pipeline(segment, truncation=True, max_length=512)
    if not classification or classification[0]["score"] < 0.7:
        return "UNKNOWN"
    return classification[0]["label"]

def create_jsonl_from_files_with_model(directory, output_file, textcat_pipeline=None):
    """
    Maak een JSONL-bestand met filtering en dynamische labels.
    """
    file_label_mapping = generate_labels_from_filenames(directory)
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename, label in file_label_mapping:
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as infile:
                text = infile.read().strip()
                segments = split_text_into_segments(text)
                for segment in segments:
                    if not is_valid_segment(segment):
                        continue  # Filter irrelevante segmenten
                    if textcat_pipeline:
                        label = assign_label_with_model(segment, textcat_pipeline)
                    json_line = {"text": segment, "label": label}
                    outfile.write(json.dumps(json_line) + "\n")
    print(f"JSONL-bestand gemaakt: {output_file}")


# Example Usage
directory = "/home/RaThorat/my-chatbot-project/Data/processed"
output_file = "/home/RaThorat/my-chatbot-project/Data/raw_labeled.jsonl"

# Optional: Use an existing textcat model
# Assuming `textcat_pipeline` is your Hugging Face pipeline for the textcat model
# Uncomment below if your model is available
from transformers import pipeline
textcat_pipeline = pipeline("text-classification", model="/home/RaThorat/my-chatbot-project/models/textcat_model")

# Use the script with or without the textcat model
create_jsonl_from_files_with_model(directory, output_file, textcat_pipeline=None)
