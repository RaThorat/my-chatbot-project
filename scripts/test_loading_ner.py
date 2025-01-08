import spacy

try:
    spacy_ner_model_path = "./models/best-model"
    spacy_nlp = spacy.load(spacy_ner_model_path)
    print("SpaCy NER-model geladen!")
    doc = spacy_nlp("Wat is het doel van DUS-I?")
    print("Entiteiten:", [(ent.text, ent.label_) for ent in doc.ents])
except Exception as e:
    print(f"Fout bij het laden van SpaCy NER-model: {e}")
