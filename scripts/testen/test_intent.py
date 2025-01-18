from ner_textcat_pipeline import process_user_input

user_input = "Wat doet DUS-i?"
intent_and_entities = process_user_input(user_input)
print(f"Intent: {intent_and_entities['intent']}, Entities: {intent_and_entities['entities']}")

