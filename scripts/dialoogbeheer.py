import logging
import random
import json

# Stel logging in
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Laad intentie-configuratie vanuit een JSON-bestand
with open('./scripts/intent_config.json', 'r') as config_file:
    intent_config = json.load(config_file)

def process_entities(entities):
    """Verwerk entiteiten in een dictionary."""
    entity_dict = {}
    for item in entities:
        label = item["label"]
        entity = item["entity"]
        if label not in entity_dict:
            entity_dict[label] = []
        entity_dict[label].append(entity)
    return entity_dict


def create_response(message):
    """Creëer een gestandaardiseerd antwoord."""
    return f"{message} Neem contact op voor meer informatie."

def validate_entities(required_entities, entity_dict):
    missing = [e for e in required_entities if e not in entity_dict]
    if missing:
        return f"Ik heb de volgende details nodig: {', '.join(missing)}."
    return None

def handle_project_details(entity_dict):
    if "SUB" in entity_dict:
        subsidie = entity_dict["SUB"][0]
        return create_response(f"De subsidie voor {subsidie} is beschikbaar.")
    if "ORG" in entity_dict:
        organisatie = entity_dict["ORG"][0]
        return create_response(f"De organisatie {organisatie} is betrokken bij dit project.")
    if "PROJECT" in entity_dict:
        project = entity_dict["PROJECT"][0]
        return create_response(f"Informatie over het project {project} is beschikbaar.")


def handle_formulieren(entity_dict):
    if "PERSOON" in entity_dict:
        persoon = entity_dict["PERSOON"][0]
        return create_response(f"Het formulier voor {persoon} kan worden gedownload van onze website.")
    if "LOC" in entity_dict:
        locatie = entity_dict["LOC"][0]
        return create_response(f"De formulieren voor {locatie} zijn beschikbaar.")
    return "Kun je specifieker zijn over het formulier dat je zoekt?"

def handle_stageaanbieders(entity_dict):
    if "LOC" in entity_dict:
        locatie = entity_dict["LOC"][0]
        return create_response(f"De stageaanbieders in {locatie} zijn beschikbaar.")
    return "Kun je de locatie specificeren voor de stageaanbieders die je zoekt?"

def handle_inzicht(entity_dict):
    if "BEDRAG" in entity_dict:
        bedrag = entity_dict["BEDRAG"][0]
        return create_response(f"Het inzicht over het bedrag {bedrag} is beschikbaar.")
    return "Kun je specifieker zijn over het inzicht dat je zoekt?"

def handle_plan(entity_dict):
    if "TIJD" in entity_dict:
        tijd = entity_dict["TIJD"][0]
        return create_response(f"Het plan voor de periode {tijd} is beschikbaar.")
    return "Kun je meer informatie geven over het plan dat je zoekt?"

def handle_uitslag(entity_dict):
    if "PERSOON" in entity_dict:
        persoon = entity_dict["PERSOON"][0]
        return create_response(f"De uitslag voor {persoon} is beschikbaar.")
    if "ORG" in entity_dict:
        organisatie = entity_dict["ORG"][0]
        return create_response(f"De uitslag voor de organisatie {organisatie} is beschikbaar.")
    return "Kun je specifieker zijn over de uitslag die je zoekt?"

def dynamic_response(base_message, variations):
    return random.choice([base_message] + variations)

def intent_dispatcher(intent, entity_dict):
    try:
        handler_name = intent_config.get("intent_handlers", {}).get(intent, None)
        logging.info(f"Handler gevonden: {handler_name} voor intentie: {intent}")
        if handler_name:
            handler_function = globals().get(handler_name, None)
            if handler_function:
                return handler_function(entity_dict)
        logging.warning(f"Geen geldige handler gevonden voor intentie: {intent}")
    except Exception as e:
        logging.error(f"Fout bij het afhandelen van intent '{intent}': {e}")
    return fallback_response(intent)



def fallback_response(intent):
    fallback_messages = {
        "PROJECT_DETAILS": "Kun je meer details geven over het project dat je zoekt?",
        "FORMULIEREN": "Kun je specifieker zijn over het formulier dat je zoekt?",
        "INZICHT": "Kun je meer informatie geven over het inzicht dat je zoekt?",
        "HANDLEIDINGEN": "Kun je meer details geven over de handleidingen die je zoekt?",
        "STAGEAANBIEDERS": "Kun je specifieker zijn over de stageaanbieder die je zoekt?",
        "Uitslag": "Kun je meer informatie geven over uitslagen die je zoekt?",
        "PLAN": "Kun je specifieker zijn over de plannen die je zoekt?",
        "INTERN_DUSI": "Kun je meer informatie geven wat je binnen DUS-i zoekt?",
        "default": "Ik begrijp je vraag niet helemaal. Kun je het herformuleren?"
    }
    return fallback_messages.get(intent, fallback_messages["default"])

def chatbot_response(result):
    """Genereer een reactie gebaseerd op intenties en entiteiten."""
    try:
        intent = result.get("intent", "default")  # Zorg voor een fallback-intentie
        entities = result.get("entities", [])  # Zorg voor een lege lijst als entities ontbreekt
        logging.info(f"Intent: {intent}, Entities: {entities}")

        # Controleer of entities correct zijn
        if not isinstance(entities, list):
            raise ValueError("Entiteiten zijn niet correct geformatteerd.")

        # Verwerk entiteiten
        entity_dict = process_entities(entities)
        logging.info(f"Entity Dict: {entity_dict}")

        # Gebruik intent dispatcher
        response = intent_dispatcher(intent, entity_dict)
        logging.info(f"Response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error in chatbot_response: {e}")
        return "Er is een fout opgetreden. Probeer het opnieuw."

