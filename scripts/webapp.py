from flask import Flask, request, jsonify, render_template
import spacy
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, pipeline
import logging
import os
from faiss_search import search_faiss_with_content
import webbrowser
import warnings
import traceback
import re
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import atexit

# Zet omgevingsvariabelen
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Onderdruk waarschuwingen
warnings.filterwarnings("ignore")

# Initialiseer Flask-app
app = Flask(__name__, template_folder=os.path.join(os.getcwd(), "templates"))

# Stel logging in
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

# Laad fine tuned GPT-NL 1.3B generatief model
tokenizer_generative = AutoTokenizer.from_pretrained("./models/fine_tuned_gpt_nl")
model_generative = AutoModelForCausalLM.from_pretrained("./models/fine_tuned_gpt_nl")

# Voeg padding-token toe
if tokenizer_generative.pad_token is None:
    tokenizer_generative.pad_token = tokenizer_generative.eos_token
    model_generative.resize_token_embeddings(len(tokenizer_generative))

conversation_history = {}
session_timestamps = {}

@app.route("/opschonen", methods=["POST"])
def cleanup_sessions():
    """
    Verwijder inactieve sessies ouder dan 30 minuten.
    """
    now = datetime.now()
    inactive_threshold = timedelta(minutes=30)
    sessions_to_remove = [
        session_id for session_id, timestamp in session_timestamps.items()
        if now - timestamp > inactive_threshold
    ]

    for session_id in sessions_to_remove:
        conversation_history.pop(session_id, None)
        session_timestamps.pop(session_id, None)

    logging.info(f"Opschonen voltooid. {len(sessions_to_remove)} sessies verwijderd.")
    return jsonify({"status": "Opschonen voltooid", "verwijderde_sessies": len(sessions_to_remove)})

def generate_answer_with_context(prompt, max_length=100, temperature=0.7):
    try:
        inputs = tokenizer_generative(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model_generative.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_length,
            temperature=0.5,  # Meer deterministische antwoorden
            top_p=0.9,
            top_k=50,
            do_sample=True,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer_generative.pad_token_id
        )

        response = tokenizer_generative.decode(outputs[0], skip_special_tokens=True).strip()
        if not response:
            return "Ik kan daar op dit moment geen antwoord op geven."
        return response
    except Exception as e:
        logging.error(f"Fout bij het genereren van een antwoord: {e}")
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

def summarize_content(content, max_length=150):
    if not content:
        return "Geen relevante context beschikbaar."
    clean_content = " ".join(content.split()).strip()  # Verwijder overbodige witruimtes
    if len(clean_content) > max_length:
        return clean_content[:max_length].strip() + "..."
    return clean_content


def get_relevant_history(conversation_history, intent=None):
    if intent:
        return "\n".join(
            [f"Gebruiker: {item['user']}\nBot: {item['bot']}" 
             for item in conversation_history if "intent" in item and item["intent"] == intent]
        )
    return "\n".join(
        [f"Gebruiker: {item['user']}\nBot: {item['bot']}" for item in conversation_history]
    )

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
        session_id = request.args.get("session_id")
        if not query or not session_id:
            return jsonify({"error": "Query- en session_id-parameters zijn verplicht"}), 400

        if session_id not in conversation_history:
            conversation_history[session_id] = []

        session_timestamps[session_id] = datetime.now()

        # Verwerk de query
        model_results = process_user_input(query)
        intent = model_results["intent"]
        detected_entities = [entity["entity"] for entity in model_results["entities"]]
        if not detected_entities:
            detected_entities = ["Geen entiteiten gevonden."]

        # FAISS-zoekopdracht
        raw_faiss_results = search_faiss_with_content(query, top_k=5)

        relevant_faiss_result = None
        for result in raw_faiss_results:
            if any(entity in result["Content"] for entity in detected_entities):
                relevant_faiss_result = result
                break
        if not relevant_faiss_result:
            relevant_faiss_result = raw_faiss_results[0] if raw_faiss_results else {"Content": "Geen relevante context gevonden.", "Title": "Geen titel beschikbaar"}

        summarized_context = summarize_content(relevant_faiss_result["Content"])
        title = relevant_faiss_result.get("Title", "Geen titel beschikbaar")

        # Combineer de prompt voor generatieve modellen
        combined_prompt = (
            f"Je bent een chatbot die informatie geeft over subsidies en DUS-i. "
            f"Gebruik alleen de gegeven context. "
            f"Gespreksgeschiedenis:\n{get_relevant_history(conversation_history[session_id])}\n"
            f"Nieuwe vraag: {query}\n"
            f"Context: {summarized_context}\n\n"
            f"intent: {intent}\n\n"
            f"entities: {detected_entities}\n\n"
            f"Geef een bondig en feitelijk antwoord."
        )

        # Genereer het antwoord
        full_answer = generate_answer_with_context(combined_prompt, max_length=170)

        # Isolate the part after the prompt
        def extract_model_answer(prompt, response):
            # Remove the prompt from the response
            if response.startswith(prompt):
                return response[len(prompt):].strip()
            return response.strip()

        # Get the actual answer by removing the prompt
        actual_answer = extract_model_answer(combined_prompt, full_answer)

        # Truncate to two sentences
        def get_two_sentences(text):
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            return ' '.join(sentences[:2])

        concise_answer = get_two_sentences(actual_answer)

        # Sla de gespreksgeschiedenis op
        conversation_history[session_id].append({"user": query, "bot": concise_answer})

        # Return alleen relevante data, zonder de prompt
        return jsonify({
            "query": query,
            "intent": model_results["intent"],
            "entities": model_results["entities"],
            "context_document": summarized_context,
            "document_title": title,
            "concise_answer": concise_answer  # Alleen het antwoord
        })
    except Exception as e:
        logging.error(f"Fout in chat endpoint: {e}")
        logging.error(traceback.format_exc())
        return jsonify({"error": "Interne serverfout"}), 500


# Geplande sessie-opschoning
def trigger_session_cleanup():
    with app.app_context():
        cleanup_sessions()

scheduler = BackgroundScheduler()
scheduler.add_job(trigger_session_cleanup, 'interval', minutes=30)
scheduler.start()

atexit.register(lambda: scheduler.shutdown())

if __name__ == "__main__":
    port = 5000
    url = f"http://127.0.0.1:{port}"
    try:
        webbrowser.open(url)
    except Exception as e:
        logging.warning(f"Kon browser niet openen: {e}")
    app.run(host="0.0.0.0", port=port, debug=False)


