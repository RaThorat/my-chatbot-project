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
from apscheduler.schedulers.background import BackgroundScheduler  # Toegevoegd voor geplande opschoning
from datetime import datetime, timedelta  # Gebruikt voor sessietimestampbeheer
import atexit  # Voor nette afsluiting van de planner

warnings.filterwarnings("ignore")
disable_beta_transforms_warning()

# Flask-app initialisatie
app = Flask(__name__, template_folder=os.path.join(os.getcwd(), "templates"))

# Logging-instellingen
logging.basicConfig(
    filename="chatbot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Modellen laden
spacy_ner_model_path = "./models/best-model"
spacy_nlp = spacy.load(spacy_ner_model_path)

hf_model_path = "./models/textcat_model"
textcat_tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
textcat_model = AutoModelForSequenceClassification.from_pretrained(hf_model_path)
textcat_pipeline = pipeline("text-classification", model=textcat_model, tokenizer=textcat_tokenizer)

# Generatief model laden
tokenizer_generative = AutoTokenizer.from_pretrained("google/flan-t5-large")
model_generative = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

conversation_history = {}  # Sleutel: sessie-ID, Waarde: lijst van gespreksbeurten
session_timestamps = {}  # Sleutel: sessie-ID, Waarde: laatste actieve timestamp

@app.route("/opschonen", methods=["POST"])
def cleanup_sessions():
    """
    Oude sessies opruimen die langer dan 30 minuten inactief zijn.
    """
    nu = datetime.now()
    inactief_drempel = timedelta(minutes=30)
    te_verwijderen_sessies = [
        session_id for session_id, timestamp in session_timestamps.items()
        if nu - timestamp > inactief_drempel
    ]

    for session_id in te_verwijderen_sessies:
        conversation_history.pop(session_id, None)
        session_timestamps.pop(session_id, None)

    logging.info(f"Opschonen voltooid. {len(te_verwijderen_sessies)} sessies verwijderd.")
    return jsonify({"status": "Opschonen voltooid", "verwijderde_sessies": len(te_verwijderen_sessies)})

def generate_answer_with_context(prompt, max_length=150, temperature=0.7):
    try:
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
    except Exception as e:
        logging.error(f"Fout bij genereren van antwoord: {e}")
        return "Er is een fout opgetreden bij het genereren van het antwoord. Probeer het later opnieuw."


def process_user_input(user_input):
    try:
        doc = spacy_nlp(user_input)
        entities = [{"entity": ent.text, "label": ent.label_} for ent in doc.ents]

        intent_result = textcat_pipeline(user_input)
        intent = intent_result[0]["label"] if intent_result and intent_result[0]["score"] >= 0.25 else "default"

        return {"intent": intent, "entities": entities}
    except Exception as e:
        logging.error(f"Fout in process_user_input: {e}")
        return {"intent": "default", "entities": []}

def summarize_content(Content, max_length=150):
    if not Content:
        return "Geen relevante context beschikbaar."
    if len(Content) > max_length:
        return Content[:max_length].strip() + "..."
    return Content

@app.route("/")
def index():
    try:
        return render_template("index.html")
    except Exception as e:
        logging.error(f"Fout bij het laden van template: {e}")
        return "Fout: index.html niet gevonden. Controleer de templates-map.", 404

@app.route("/chat", methods=["GET"])
def chat():
    try:
        query = request.args.get("query")
        session_id = request.args.get("session_id")  # Veronderstel dat session_id door de frontend wordt meegegeven
        if not query or not session_id:
            return jsonify({"error": "Query en session_id parameters zijn verplicht"}), 400

        # Initialiseer gespreksgeschiedenis als deze nog niet bestaat
        if session_id not in conversation_history:
            conversation_history[session_id] = []

        # Update de laatste actieve timestamp voor de sessie
        session_timestamps[session_id] = datetime.now()

        # Verwerk gebruikersinvoer
        model_results = process_user_input(query)
        # Voeg dit toe in de chat-functie, vlak na het verkrijgen van `model_results`
        intent = model_results["intent"]  # Definieer intent expliciet
        detected_entities = [entity["entity"] for entity in model_results["entities"]]
        raw_faiss_results = search_faiss_with_content(query, top_k=5)

        # Relevante context ophalen
        relevant_faiss_result = None
        for result in raw_faiss_results:
            if any(entity in result["Content"] for entity in detected_entities):
                relevant_faiss_result = result
                break
        if not relevant_faiss_result and raw_faiss_results:
            relevant_faiss_result = raw_faiss_results[0]

        # Haal titel en samenvatting op
        summarized_context = summarize_content(relevant_faiss_result["Content"])
        title = relevant_faiss_result.get("Title", "Geen titel beschikbaar")

        # Stel de gespreksgeschiedenis samen (laatste 5 interacties)
        history = "\n".join(
            [f"Gebruiker: {item['user']}\nBot: {item['bot']}" for item in conversation_history[session_id][-5:]]
        )

        combined_prompt = (
            f"Gespreksgeschiedenis:\n{history}\n"
            f"Nieuwe vraag: {query}\n"
            f"Context: {relevant_faiss_result['Content']}\n\n"
            f"intent: {intent}\n\n"
            f"entities: {detected_entities}\n\n"
            "Geef een gedetailleerd antwoord op basis van de geschiedenis en context."
        )
        concise_answer = generate_answer_with_context(combined_prompt, max_length=150)

        # Update gespreksgeschiedenis
        conversation_history[session_id].append({"user": query, "bot": concise_answer})

        return jsonify({
            "query": query,
            "intent": model_results["intent"],
            "entities": model_results["entities"],
            "context_document": summarized_context,
            "document_title": title,
            "concise_answer": concise_answer
        })

    except Exception as e:
        logging.error(f"Fout in chat endpoint: {e}")
        logging.error(traceback.format_exc())
        return jsonify({"error": "Interne serverfout"}), 500


# Planner voor periodieke sessie-opschoning
def trigger_session_cleanup():
    """
    Roep periodiek het opschoningsendpoint aan.
    """
    with app.app_context():
        cleanup_sessions()

scheduler = BackgroundScheduler()
scheduler.add_job(trigger_session_cleanup, 'interval', minutes=30)  # Elke 30 minuten uitvoeren
scheduler.start()

# Zorg ervoor dat de planner stopt bij afsluiten
atexit.register(lambda: scheduler.shutdown())

if __name__ == "__main__":
    port = 5000
    url = f"http://127.0.0.1:{port}"
    try:
        webbrowser.open(url)
    except Exception as e:
        logging.warning(f"Kon browser niet openen: {e}")
    app.run(host="0.0.0.0", port=port, debug=False)
