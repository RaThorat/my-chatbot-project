from transformers import pipeline

classifier = pipeline("text-classification", model="/home/RaThorat/my-chatbot-project/models/textcat_model", tokenizer="/home/RaThorat/my-chatbot-project/models/textcat_model")

# Test een voorbeeld
result = classifier("De aanvraagperiode liep van 20 augustus 2024 tot en met 20 september 2024.")
print(result)

