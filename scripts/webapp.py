from flask import Flask, request, jsonify, render_template, send_from_directory
import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, pipeline
import logging
import os
import sqlite3
from faiss_search import search_with_document_level
import webbrowser

import warnings
from torchvision import disable_beta_transforms_warning

warnings.filterwarnings("ignore")
disable_beta_transforms_warning()

# Flask app initialization
app = Flask(__name__, template_folder="/home/RaThorat/my-chatbot-project/templates")

# Logging setup
logging.basicConfig(
    filename="chatbot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Database path
db_path = "documents.db"

# Load NER and TEXTCAT models
spacy_ner_model_path = "./models/best-model"
spacy_nlp = spacy.load(spacy_ner_model_path)

hf_model_path = "./models/textcat_model"
textcat_tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
textcat_model = AutoModelForSequenceClassification.from_pretrained(hf_model_path)
textcat_pipeline = pipeline("text-classification", model=textcat_model, tokenizer=textcat_tokenizer)

# Load generative model


tokenizer_generative = AutoTokenizer.from_pretrained("google/flan-t5-small")
model_generative = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

def process_user_input(user_input):
    """
    Process user input with NER and TEXTCAT models.
    """
    try:
        doc = spacy_nlp(user_input)
        entities = [{"entity": ent.text, "label": ent.label_} for ent in doc.ents]

        intent_result = textcat_pipeline(user_input)
        intent = intent_result[0]["label"] if intent_result and intent_result[0]["score"] >= 0.7 else "default"

        return {"intent": intent, "entities": entities}
    except Exception as e:
        logging.error(f"Error in process_user_input: {e}")
        return {"intent": "default", "entities": []}



def generate_answer_with_context(prompt, max_length=100):
    inputs = tokenizer_generative(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model_generative.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_length,
        temperature=0.5,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.5,
    )
    return tokenizer_generative.decode(outputs[0], skip_special_tokens=True)



def truncate_prompt(prompt, max_tokens=2048):
    """
    Truncate the prompt to fit within the model's token limit.
    """
    prompt_lines = prompt.split("\n")
    while len(" ".join(prompt_lines).split()) > max_tokens and len(prompt_lines) > 3:
        prompt_lines.pop(-2)
    return "\n".join(prompt_lines)

def chatbot_query(query):
    """
    Search the SQLite database for matching documents.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        logging.info(f"Executing database query: {query}")
        cursor.execute("SELECT rowid, content FROM docs WHERE content LIKE ?", (f"%{query}%",))
        results = cursor.fetchall()
        logging.info(f"Query returned {len(results)} results")

        formatted_results = [
            {"rowid": int(rowid), "excerpt": content[:200]}
            for rowid, content in results
        ]
        return {"responses": formatted_results}
    except Exception as e:
        logging.error(f"Error querying database: {e}")
        return {"responses": []}
    finally:
        conn.close()

def clean_generative_response(response):
    """
    Extract the generated answer by removing prompt echoes or irrelevant text.
    """
    # Extract the part after "Antwoord bondig"
    if "Antwoord bondig" in response:
        response = response.split("Antwoord bondig", 1)[-1]
    return response.strip()

@app.route("/")
def index():
    try:
        return render_template("index.html")
    except FileNotFoundError:
        logging.error("index.html not found in the templates directory.")
        return "Error: index.html not found. Please check your templates folder.", 404

# Helper function to summarize excerpts
def summarize_excerpt(excerpt, max_length=150):
    if len(excerpt) > max_length:
        return excerpt[:max_length].strip() + "..."
    return excerpt

@app.route("/chat", methods=["GET"])
def chat():
    try:
        query = request.args.get("query")
        if not query:
            return jsonify({"error": "Query parameter is required"}), 400

        # Process user input
        model_results = process_user_input(query)
        detected_entities = [entity["entity"] for entity in model_results["entities"]]
        faiss_results = search_with_document_level(query, top_k=3)

        # Extract relevant FAISS context
        relevant_faiss_result = None
        if detected_entities:
            for result in faiss_results:
                if any(entity in result["excerpt"] for entity in detected_entities):
                    relevant_faiss_result = result["excerpt"]
                    break
        if not relevant_faiss_result and faiss_results:
            relevant_faiss_result = faiss_results[0]["excerpt"]
        summarized_context = summarize_excerpt(relevant_faiss_result, max_length=150) if relevant_faiss_result else "Geen relevante context beschikbaar."

        # Create concise prompt
        combined_prompt = (
            f"Vraag: {query}\n"
            f"Context: {relevant_faiss_result}\n\n"
            "Geef een bondig antwoord in 2-3 zinnen."
        )
        combined_prompt = truncate_prompt(combined_prompt)

        # Generate and clean the response
        generative_response = generate_answer_with_context(combined_prompt, max_length=100)
        concise_answer = clean_generative_response(generative_response)

        return jsonify({
            "query": query,
            "intent": model_results["intent"],
            "entities": model_results["entities"],
            "context_document": summarized_context,
            "concise_answer": concise_answer
        })
    except Exception as e:
        logging.error(f"Error in chat endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = 5000
    url = f"http://127.0.0.1:{port}"
    print(f"Opening browser at {url}...")
    webbrowser.open(url)
    app.run(host="0.0.0.0", port=port, debug=False)