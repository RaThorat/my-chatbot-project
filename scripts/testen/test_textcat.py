from transformers import pipeline

classifier = pipeline("text-classification", model="/home/RaThorat/my-chatbot-project/models/textcat_model", tokenizer="/home/RaThorat/my-chatbot-project/models/textcat_model")

# Test een voorbeeld
result = classifier("Wie is directeur van DUS-I?")
print(result)

