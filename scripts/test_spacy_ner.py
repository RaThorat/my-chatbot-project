import spacy

# Laad het SpaCy-model
model_path = "./models/ner_model"  # Update dit pad als je model ergens anders staat
nlp = spacy.load(model_path)

# Test een voorbeeldtekst
text = "Wat is de subsidie voor Groningen?"
doc = nlp(text)

# Print de entiteiten
print("Entiteiten:")
for ent in doc.ents:
    print(f" - {ent.text} ({ent.label_})")
