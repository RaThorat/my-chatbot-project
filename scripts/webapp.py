from flask import Flask, request, jsonify, render_template
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import logging
import os
import sqlite3
import faiss
from faiss_search import search_faiss

import warnings
from torchvision import disable_beta_transforms_warning

warnings.filterwarnings("ignore")
disable_beta_transforms_warning()

from transformers import AutoModelForCausalLM, AutoTokenizer

# Laad DistilGPT-2
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

def generate_answer(prompt, max_length=150):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Logging instellen
logging.basicConfig(
    filename="chatbot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
print("Flask app initializing...")
# Print huidige werkdirectory
print("Current working directory:", os.getcwd())

# Database pad
db_path = "documents.db"

# Laad SpaCy NER-model
print("Loading NER model...")
spacy_ner_model_path = "./models/best-model"
spacy_nlp = spacy.load(spacy_ner_model_path)
print("NER model loaded successfully.")

# Laad Hugging Face textcat-model
print("Loading textcat model...")
hf_model_path = "./models/textcat_model"
textcat_tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
textcat_model = AutoModelForSequenceClassification.from_pretrained(hf_model_path)
textcat_pipeline = pipeline("text-classification", model=textcat_model, tokenizer=textcat_tokenizer)
print("Textcat model loaded successfully.")
# Flask app initialiseren
app = Flask(__name__, template_folder="/home/RaThorat/my-chatbot-project/templates")

def process_user_input(user_input):
    """
    Verwerk gebruikersinvoer met NER en Textcat-modellen.
    """
    try:
        # SpaCy NER
        doc = spacy_nlp(user_input)
        entities = [{"entity": ent.text, "label": ent.label_} for ent in doc.ents]

        # Textcat intentieherkenning
        intent_result = textcat_pipeline(user_input)
        if intent_result and intent_result[0]["score"] >= 0.7:  # Controleer confidence score
            intent = intent_result[0]["label"]
        else:
            intent = "default"  # Fallback intentie

        return {"intent": intent, "entities": entities}
    except Exception as e:
        logging.error(f"Error in process_user_input: {e}")
        return {"intent": "default", "entities": []}

def chatbot_query(query):
    """
    Zoek resultaten in SQLite en combineer met modellen.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        # Debug-log: controleer of query goed is opgebouwd
        logging.info(f"Executing database query: {query}")
        cursor.execute("SELECT filename, content FROM docs WHERE docs MATCH ?", (f'"{query}"',))
        results = cursor.fetchall()
        logging.info(f"Query returned {len(results)} results")
        
        # Check op dubbele documenten
        if len(results) > 0:
            unique_results = {result[1]: result for result in results}.values()
            results = list(unique_results)

        # Als geen resultaten, suggereer alternatieven
        if not results:
            suggestions = ["Geen relevante informatie gevonden. Probeer andere zoekwoorden."]
            return {"responses": [], "suggestions": suggestions}

        # Format resultaten
        formatted_results = [
            f"In bestand {filename}: {content[:200]}..." for filename, content in results
        ]
        return {"responses": formatted_results, "suggestions": []}
    except Exception as e:
        logging.error(f"Error querying database: {e}")
        return {"responses": [], "suggestions": ["Er is een fout opgetreden bij het zoeken."]}
    finally:
        conn.close()


@app.route("/")
def index():
    """
    Hoofdpagina serveren.
    """
    try:
        with open("templates/index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        logging.error("Index.html not found.")
        return "Index.html not found. Please check your templates directory.", 500

@app.route("/chat", methods=["GET"])
def chat():
    try:
        query = request.args.get("query")
        if not query:
            return jsonify({"error": "Query parameter is required"}), 400

        # Verwerk intentie en entiteiten
        model_results = process_user_input(query)

        # Database-query resultaten
        db_results = chatbot_query(query)

        # FAISS-zoekresultaten
        faiss_results = search_faiss(query)

        # Combineer FAISS-resultaten met de generatieve pipeline
        combined_prompt = f"Vraag: {query}\n"
        if faiss_results:
            combined_prompt += "Relevante documenten:\n" + "\n".join([doc for doc, _ in faiss_results]) + "\n"
        # Genereer een antwoord
        generative_response = generate_answer(combined_prompt)

        # Combineer alles in de JSON-response
        response = {
            "query": query,
            "intent": model_results["intent"],
            "entities": model_results["entities"],
            "faiss_results": [
                {"document": doc, "score": score} for doc, score in faiss_results
            ],
            "db_responses": db_results["responses"],
            "db_suggestions": db_results["suggestions"],
            "generative_response": generative_response
        }
        return jsonify(response)
    except Exception as e:
        logging.error(f"Error in /chat endpoint: {e}")
        return jsonify({"error": "Er is een fout opgetreden bij het verwerken van uw verzoek. Probeer het later opnieuw."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

