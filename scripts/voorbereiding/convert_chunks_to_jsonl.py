import json

def reassign_label(text_segment):
    """
    Wijs een label toe aan een tekstsegment op basis van aanwezige sleutelwoorden.
    """
    text_segment_lower = text_segment.lower()
    if any(word in text_segment_lower for word in ["loting", "uitslag", "resultaat"]):
        return "UITSLAG"
    elif any(word in text_segment_lower for word in ["stageaanbieders", "partner", "locatie"]):
        return "STAGEAANBIEDERS"
    elif any(word in text_segment_lower for word in ["subsidie", "financiering", "aanvraag"]):
        return "SUBSIDIE_INFORMATIE"
    elif any(word in text_segment_lower for word in ["project", "doelstellingen", "uitvoering"]):
        return "PROJECT_DETAILS"
    elif any(word in text_segment_lower for word in ["activiteitenplan", "planning", "strategie"]):
        return "PLAN"
    elif any(word in text_segment_lower for word in ["formulier", "bijlage", "toelichting"]):
        return "FORMULIEREN"
    elif any(word in text_segment_lower for word in ["interview", "intern", "notitie"]):
        return "INTERN_DUSI"
    elif any(word in text_segment_lower for word in ["handleiding", "gebruikersgids", "stappenplan"]):
        return "HANDLEIDINGEN"
    elif any(word in text_segment_lower for word in ["vragenlijst", "survey", "feedback"]):
        return "VRAGENLIJST"
    return "ANDERS"

def convert_json_to_jsonl(input_file, output_file):
    """
    Converteer een JSON-bestand met Chunks naar een JSONL-bestand met text en label.
    """
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    documents = data.get("Documents", [])
    jsonl_data = []

    for document in documents:
        chunks = document.get("Chunks", [])
        for chunk in chunks:
            text = chunk.get("Chunk", "").strip()
            label = reassign_label(text)
            jsonl_data.append({"text": text, "label": label})

    # Write to JSONL
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in jsonl_data:
            outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Conversie voltooid. JSONL-bestand opgeslagen als: {output_file}")

# Gebruik het script
input_file = "./faiss_data/doc_chunks.json"  # Vervang met jouw JSON-bestand
output_file = "./Data/output.jsonl"  # Vervang met de gewenste output-bestandsnaam
convert_json_to_jsonl(input_file, output_file)
