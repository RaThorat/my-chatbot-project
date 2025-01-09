from webapp import generate_answer_with_context

prompt = "Vraag: Wat is DUS-i?\nRelevante documenten:\n1. DUS-i staat voor Dienst Uitvoering Subsidies aan Instellingen."
response = generate_answer_with_context(prompt)
print(response)
