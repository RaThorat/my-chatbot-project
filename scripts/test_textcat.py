from transformers import pipeline

classifier = pipeline("text-classification", model="./textcat_model", tokenizer="./textcat_model")

# Test een voorbeeld
result = classifier("De aanvraagperiode liep van 20 augustus 2024 tot en met 20 september 2024.")
print(result)

