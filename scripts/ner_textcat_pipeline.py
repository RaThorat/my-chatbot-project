import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Laad SpaCy NER-model
spacy_ner_model_path = "./models/best-model"
spacy_nlp = spacy.load(spacy_ner_model_path)

# Laad Hugging Face textcat-model
hf_model_path = "./models/textcat_model"
textcat_tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
textcat_model = AutoModelForSequenceClassification.from_pretrained(hf_model_path)
textcat_pipeline = pipeline("text-classification", model=textcat_model, tokenizer=textcat_tokenizer)


def process_user_input(user_input):
    # NER met SpaCy
    doc = spacy_nlp(user_input)
    entities = [{"entity": ent.text, "label": ent.label_} for ent in doc.ents]

    # Textcat met Hugging Face
    intent = textcat_pipeline(user_input)[0]

    return {"intent": intent["label"], "entities": entities}
