import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import logging

# Stel logging in
logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Laad SpaCy NER-model
spacy_ner_model_path = "./models/ner_model"
spacy_nlp = spacy.load(spacy_ner_model_path)

# Laad Hugging Face textcat-model
hf_model_path = "./models/textcat_model"
textcat_tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
textcat_model = AutoModelForSequenceClassification.from_pretrained(hf_model_path)
textcat_pipeline = pipeline("text-classification", model=textcat_model, tokenizer=textcat_tokenizer)

def process_user_input(user_input):
    try:
        # Verwerk gebruikersinvoer met SpaCy NER
        doc = spacy_nlp(user_input)
        entities = [{"entity": ent.text, "label": ent.label_} for ent in doc.ents]

        # Intentieherkenning met confidence score
        intent_result = textcat_pipeline(user_input)
        if intent_result and intent_result[0]["score"] >= 0.7:  # Controleer confidence score
            intent = intent_result[0]["label"]
        else:
            intent = "default"  # Fallback intentie

        # Retourneer intentie en entiteiten
        return {"intent": intent, "entities": entities}
    except Exception as e:
        # Log een fout als de verwerking mislukt
        logging.error(f"Fout bij NER of intentieherkenning: {e}")
        return {"intent": "default", "entities": []}
