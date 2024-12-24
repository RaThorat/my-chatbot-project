from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

model_name = "./models/ner_model"  # Pad naar je NER-model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

text = "Wat is de subsidie voor Amsterdam?"
entities = nlp(text)
print(entities)
