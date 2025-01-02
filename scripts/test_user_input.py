from ner_textcat_pipeline import process_user_input
from dialoogbeheer import chatbot_response

user_input = "Wat doet DUS-i?"
intent_and_entities = process_user_input(user_input)
response = chatbot_response(intent_and_entities)
print(response)
