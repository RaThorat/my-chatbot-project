from transformers import AutoTokenizer, AutoModelForCausalLM

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
        max_length=max_length,
        temperature=0.6,
        do_sample=True,
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,
        pad_token_id=tokenizer.pad_token_id
    )


    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Test de functie
prompt = (
    "Vraag: Wat is DUS-i?\n"
    "Relevante documenten:\n"
    "1. DUS-i staat voor Dienst Uitvoering Subsidies aan Instellingen.\n"
    "2. Het is een afdeling van het ministerie van Volksgezondheid, Welzijn en Sport.\n"
    "Antwoord kort en bondig in één zin."
)

def clean_response(response):
    """
    Verwijdert irrelevante delen uit de output van het model.
    """
    # Neem alleen de eerste zin als output te lang is
    sentences = response.split('.')
    return '. '.join(sentences[:2]) + '.' if len(sentences) > 2 else response

response = generate_answer_with_context(prompt)
response = clean_response(response)
print("Generatief antwoord:", response)

