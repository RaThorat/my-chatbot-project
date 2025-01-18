from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

hf_model_path = "./models/textcat_model"
textcat_tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
textcat_model = AutoModelForSequenceClassification.from_pretrained(hf_model_path)
textcat_pipeline = pipeline("text-classification", model=textcat_model, tokenizer=textcat_tokenizer)

result = textcat_pipeline("Test deze tekst voor classificatie.")
print("Result:", result)
