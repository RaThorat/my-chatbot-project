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


# Functie om het JSONL-bestand te herlabelen
def relabel_jsonl(input_file, output_file):
    """
    Lees een JSONL-bestand, herlabel "ANDERS"-segmenten en schrijf het naar een nieuw bestand.
    """
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            if data["label"] == "ANDERS":
                new_label = reassign_label(data["text"])
                data["label"] = new_label
            outfile.write(json.dumps(data) + "\n")
    print(f"Herlabeld bestand opgeslagen in: {output_file}")

# Gebruik het script
input_file = "/home/RaThorat/my-chatbot-project/Data/raw_labeled.jsonl"
output_file = "/home/RaThorat/my-chatbot-project/Data/raw_labeled_revised.jsonl"
relabel_jsonl(input_file, output_file)
