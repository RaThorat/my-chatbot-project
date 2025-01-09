from flask import Flask, request, jsonify, render_template
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import logging
import os
import sqlite3
import faiss
from faiss_search import search_faiss_with_content
import webbrowser

import warnings
from torchvision import disable_beta_transforms_warning

warnings.filterwarnings("ignore")
disable_beta_transforms_warning()

from transformers import AutoModelForCausalLM, AutoTokenizer

# Laad GPT-Neo 125M model en tokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_answer_with_context(prompt, max_length=150, temperature=0.7):
    """
    Genereer een antwoord met GPT-Neo 125M.
    - max_length: Maximale lengte van de gegenereerde output.
    - temperature: Controleert de mate van creativiteit van het model.
    """

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,  # Voeg expliciete attention_mask toe
        max_new_tokens=200,
        temperature=0.6,
        do_sample=True,
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,
        pad_token_id=tokenizer.pad_token_id
    )
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
            return {"responses": []}

        # Format resultaten
        formatted_results = [
            f"In bestand {filename}: {content[:200]}..." for filename, content in results
        ]
        return {"responses": formatted_results}
    except Exception as e:
        logging.error(f"Error querying database: {e}")
        return {"responses": [], "suggestions": ["Er is een fout opgetreden bij het zoeken."]}
    finally:
        conn.close()

def clean_response(response):
    """
    Verwijdert irrelevante delen uit de output van het model.
    """
    # Splits in zinnen en neem de eerste 2
    sentences = response.split('.')
    cleaned = '. '.join(sentences[:2]).strip() + '.'

    # Verwijder generieke opmerkingen zoals 'Ik weet het niet' of model-specifieke artefacten
    for phrase in ["Ik weet het niet", "Dit is een antwoord"]:
        cleaned = cleaned.replace(phrase, "")
    return cleaned.strip()

def convert_numerics_to_float(data):
    if isinstance(data, list):
        return [convert_numerics_to_float(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_numerics_to_float(value) for key, value in data.items()}
    elif isinstance(data, (float, int)):  # Detecteer numerieke waarden
        return float(data)  # Converteer naar standaard float
    return data  # Laat andere types ongemoeid

def truncate_prompt(prompt, max_tokens=2048):
    """
    Beperk de prompt tot het maximale aantal tokens door
    minder relevante FAISS-resultaten te verwijderen.
    """
    prompt_lines = prompt.split("\n")
    while len(" ".join(prompt_lines).split()) > max_tokens and len(prompt_lines) > 3:
        # Verwijder het minst relevante document (onderste regel van de resultaten)
        prompt_lines.pop(-2)  # Laatste -2 omdat de structuur "Antwoord: ..." onderaan heeft
    return "\n".join(prompt_lines)

def remove_duplicate_results(results):
    """
    Verwijder dubbele of soortgelijke resultaten op basis van de titel en inhoud.
    """
    unique_titles = set()
    filtered_results = []
    for title, score in results:
        if title not in unique_titles:
            unique_titles.add(title)
            filtered_results.append((title, score))
    return filtered_results

def summarize_content(content, max_length=300):
    """
    Beperk de lengte van de content tot max_length tekens.
    Voegt "..." toe als het ingekort wordt.
    """
    return (content[:max_length] + "...") if len(content) > max_length else content


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
        faiss_results = search_faiss_with_content(query)
        
        # Pas toe na FAISS-resultaten ophalen
        logging.info(f"FAISS Results: {faiss_results}")
        print(f"FAISS Results Debug: {faiss_results}")
        logging.info(f"FAISS resultaten: {len(faiss_results)} documenten gevonden.")
        for idx, (title, score) in enumerate(faiss_results):
            logging.info(f"Resultaat {idx + 1}: {title[:50]}... (score: {score:.2f})")

        # Verwerk FAISS-resultaten
        if faiss_results:
            faiss_results = remove_duplicate_results(faiss_results)
            formatted_faiss_results = [
            {
                "title": summarize_content(content) if content else f"Document zonder titel (ID: {idx})",
                "score": float(score)
            }
            for idx, (content, score) in enumerate(faiss_results)
        ]
        else:
            formatted_faiss_results = []


        # Controleer of er resultaten zijn uit zowel database als FAISS
        if not db_results["responses"] and not formatted_faiss_results:
            generative_response = (
                "Er zijn geen relevante documenten gevonden in de database of FAISS. "
                "Probeer uw vraag opnieuw te stellen met andere zoekwoorden."
            )
            return jsonify({
                "query": query,
                "faiss_results": [],
                "db_responses": [],
                "generative_response": generative_response
            })

        # Bouw de prompt op basis van FAISS-resultaten
        # Beperk het aantal FAISS-resultaten in de prompt
        # Combineer prompt met maximaal 3-5 relevante resultaten
        if formatted_faiss_results:
            combined_prompt = f"Vraag: {query}\nRelevante documenten:\n"
            for idx, (title, score) in enumerate(faiss_results):
                safe_title = title[:50] if title else "Geen titel"
                logging.info(f"Resultaat {idx + 1}: {safe_title}... (score: {score:.2f})")
            combined_prompt += "Antwoord kort en bondig op basis van bovenstaande documenten."
        else:
            combined_prompt = f"Vraag: {query}\nGeen relevante documenten gevonden. Antwoord zo goed mogelijk op basis van algemene kennis."

        logging.info(f"Combined prompt: {combined_prompt}")

        # Controleer op tokenlimiet en genereer een antwoord
        # Gebruik de functie
        if len(combined_prompt.split()) > 2048:
            combined_prompt = truncate_prompt(combined_prompt, max_tokens=2048)
            logging.warning("Prompt was too long and has been truncated intelligently.")

        # Genereer een antwoord met context
        generative_response = generate_answer_with_context(combined_prompt)
        generative_response = clean_response(generative_response)

        # Combineer alles in de JSON-response
        response = {
            "query": query,
            "faiss_results": formatted_faiss_results,
            "db_responses": db_results["responses"],
            "generative_response": generative_response
        }
        response = convert_numerics_to_float(response)

        # Voeg intenties en entiteiten toe indien aanwezig
        if model_results["intent"] != "default":
            response["intent"] = model_results["intent"]

        if model_results["entities"]:
            response["entities"] = model_results["entities"]
        else:
            logging.info("Geen entiteiten gevonden voor query.")
            response["entities"] = [{"entity": "geen specifieke entiteiten", "label": "N/A"}]


        return jsonify(response)
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logging.error(f"Error in /chat endpoint: {e}\n{error_trace}")
        return jsonify({"error": "Er is een fout opgetreden bij het verwerken van uw verzoek. Probeer het later opnieuw."}), 500

if __name__ == "__main__":
    port = 5000
    url = f"http://127.0.0.1:{port}"
    print(f"Opening browser at {url}...")
    webbrowser.open(url)
    app.run(host="0.0.0.0", port=port, debug=False)
