from flask import Flask, request, jsonify, render_template
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import logging
import os
import sqlite3

import warnings
from torchvision import disable_beta_transforms_warning

warnings.filterwarnings("ignore")
disable_beta_transforms_warning()

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
    """
    API voor chatqueries.
    """
    try:
        query = request.args.get("query")
        logging.info(f"Ontvangen query: {query}")
        if not query:
            return jsonify({"error": "Query parameter is required"}), 400

        # Verwerk query met de pipeline
        model_results = process_user_input(query)
        logging.info(f"Model resultaten: {model_results}")

        db_results = chatbot_query(query)
        logging.info(f"Database resultaten: {db_results}")

        # Combineer modellenresultaten met database
        response = {
            "intent": model_results["intent"], 
            "entities": model_results["entities"], 
            **db_results
        }
        logging.info(f"Finale response: {response}")
        return jsonify(response)
    except Exception as e:
        logging.error(f"Error in /chat endpoint: {e}", exc_info=True)  # exc_info=True voegt traceback toe
        return jsonify({"error": "Er is een fout opgetreden bij het verwerken van uw verzoek. Probeer het later opnieuw."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

