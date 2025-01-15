from flask import Flask, request, jsonify, render_template
import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, pipeline
import logging
import os
from faiss_search import search_faiss_with_content
import webbrowser
import warnings
import traceback

from torchvision import disable_beta_transforms_warning

warnings.filterwarnings("ignore")
disable_beta_transforms_warning()

# Flask app initialization
app = Flask(__name__, template_folder=os.path.join(os.getcwd(), "templates"))

# Logging setup
logging.basicConfig(
    filename="chatbot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load NER and TEXTCAT models
spacy_ner_model_path = "./models/best-model"
spacy_nlp = spacy.load(spacy_ner_model_path)

hf_model_path = "./models/textcat_model"
textcat_tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
textcat_model = AutoModelForSequenceClassification.from_pretrained(hf_model_path)
textcat_pipeline = pipeline("text-classification", model=textcat_model, tokenizer=textcat_tokenizer)

# Load generative model
tokenizer_generative = AutoTokenizer.from_pretrained("google/flan-t5-large")
model_generative = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

def generate_answer_with_context(prompt, max_length=150, temperature=0.7):
    """
    Generate a response.
    """
    inputs = tokenizer_generative(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model_generative.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_length,
        temperature=temperature,
        do_sample=True,
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,
        pad_token_id=tokenizer_generative.pad_token_id
    )
    return tokenizer_generative.decode(outputs[0], skip_special_tokens=True)

def process_user_input(user_input):
    try:
        doc = spacy_nlp(user_input)
        entities = [{"entity": ent.text, "label": ent.label_} for ent in doc.ents]

        intent_result = textcat_pipeline(user_input)
        intent = intent_result[0]["label"] if intent_result and intent_result[0]["score"] >= 0.7 else "default"

        return {"intent": intent, "entities": entities}
    except Exception as e:
        logging.error(f"Error in process_user_input: {e}")
        return {"intent": "default", "entities": []}

def summarize_content(content, max_length=150):
    if not content:
        return "Geen relevante context beschikbaar."
    if len(content) > max_length:
        return content[:max_length].strip() + "..."
    return content

@app.route("/")
def index():
    try:
        return render_template("index.html")
    except Exception as e:
        logging.error(f"Error loading template: {e}")
        return "Error: index.html not found. Please check your templates folder.", 404

@app.route("/chat", methods=["GET"])
def chat():
    try:
        query = request.args.get("query")
        if not query:
            return jsonify({"error": "Query parameter is required"}), 400

        model_results = process_user_input(query)
        detected_entities = [entity["entity"] for entity in model_results["entities"]]
        raw_faiss_results = search_faiss_with_content(query, top_k=2)

        relevant_faiss_result = None
        for result in raw_faiss_results:
            if any(entity in result["content"] for entity in detected_entities):
                relevant_faiss_result = result["content"]
                break
        if not relevant_faiss_result and raw_faiss_results:
            relevant_faiss_result = raw_faiss_results[0]["content"]

        summarized_context = summarize_content(relevant_faiss_result)

        combined_prompt = (
            f"Vraag: {query}\n"
            f"Context: {relevant_faiss_result}\n\n"
            "Geef een bondig antwoord in 2-3 zinnen."
        )
        concise_answer = generate_answer_with_context(combined_prompt, max_length=150)

        return jsonify({
            "query": query,
            "intent": model_results["intent"],
            "entities": model_results["entities"],
            "context_document": summarized_context,
            "concise_answer": concise_answer
        })
    except Exception as e:
        logging.error(f"Error in chat endpoint: {e}")
        logging.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = 5000
    url = f"http://127.0.0.1:{port}"
    try:
        webbrowser.open(url)
    except Exception as e:
        logging.warning(f"Could not open browser: {e}")
    app.run(host="0.0.0.0", port=port, debug=False)
