import sys
import os
import logging
from flask_cors import CORS


# Voeg de scripts-map toe aan het Python-pad
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ner_textcat_pipeline import process_user_input
from dialoogbeheer import chatbot_response
from flask import Flask, request, jsonify

# Logging instellen
logging.basicConfig(filename="chatbot.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Flask-app instellen
app = Flask(__name__)
CORS(app)
# Rootroute toevoegen
@app.route("/", methods=["GET"])
def home():
    return """
    <h1>Welkom bij de Chatbot API</h1>
    <p>Deze API heeft de volgende routes:</p>
    <ul>
        <li><b>GET /</b>: Toont dit bericht.</li>
        <li><b>POST /chat</b>: Verwerk berichten met een JSON-body {"message": "uw bericht"}.</li>
    </ul>
    """, 200

@app.route("/chat", methods=["POST"])
def chat():
    # Validatie van het verzoek
    if not request.json or "message" not in request.json:
        logging.warning("Ongeldig verzoek: geen 'message' gevonden.")
        return jsonify({"error": "Geen bericht ontvangen. Stuur een JSON met 'message'."}), 400

    user_input = request.json.get("message", "")

    if not user_input.strip():
        logging.warning("Ongeldig verzoek: leeg bericht ontvangen.")
        return jsonify({"error": "Het bericht mag niet leeg zijn."}), 400

    # Logging van de gebruikersinvoer
    logging.info(f"Gebruikersinvoer: {user_input}")

    try:
        # Gebruik NER- en textcat-modellen
        intent_and_entities = process_user_input(user_input)
        logging.info(f"Gevonden intentie: {intent_and_entities['intent']}, entiteiten: {intent_and_entities['entities']}")
        
        # Genereer chatbotantwoord
        response_text = chatbot_response(intent_and_entities)
        logging.info(f"Chatbotantwoord: {response_text}")
        return jsonify({"response": response_text})

    except Exception as e:
        # Fallback voor foutafhandeling
        logging.error(f"Fout bij het verwerken van de invoer: {e}")
        return jsonify({"error": "Er is een fout opgetreden bij het verwerken van uw bericht."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
