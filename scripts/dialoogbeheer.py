import logging
import random

# Stel logging in
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_entities(entities):
    """Verwerk entiteiten in een dictionary."""
    entity_dict = {}
    for text, label in entities:
        if label not in entity_dict:
            entity_dict[label] = []
        entity_dict[label].append(text)
    return entity_dict

def create_response(message):
    """Creëer een gestandaardiseerd antwoord."""
    return f"{message} Neem contact op voor meer informatie."

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
    return "Kun je meer details geven over het project dat je zoekt?"

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

def chatbot_response(result):
    """Genereer een reactie gebaseerd op intenties en entiteiten."""
    intent = result["intent"]
    entities = result["entities"]

    # Log intent en entiteiten
    logging.info(f"Intent: {intent}, Entities: {entities}")

    # Verwerk entiteiten
    entity_dict = process_entities(entities)

    # Intent-logica
    if intent == "HANDLEIDINGEN" and "LOC" in entity_dict:
        locatie = entity_dict["LOC"][0]
        return create_response(f"De handleiding voor {locatie} is beschikbaar.")

    if intent == "PROJECT_DETAILS":
        return handle_project_details(entity_dict)

    if intent == "FORMULIEREN":
        return handle_formulieren(entity_dict)

    if intent == "STAGEAANBIEDERS":
        return handle_stageaanbieders(entity_dict)

    if intent == "INTERN_DUSI":
        return "Dit is een interne vraag. Neem contact op met de verantwoordelijke afdeling."

    if intent == "INZICHT":
        return handle_inzicht(entity_dict)

    if intent == "PLAN":
        return handle_plan(entity_dict)

    if intent == "UITSLAG":
        return handle_uitslag(entity_dict)

    # Fallback-antwoorden
    fallback_responses = [
        "Ik begrijp je vraag niet helemaal. Kun je het herformuleren?",
        "Sorry, ik ben hier niet zeker van. Kun je meer details geven?",
        "Kun je je vraag anders stellen? Ik probeer je te helpen!"
    ]
    return random.choice(fallback_responses)

