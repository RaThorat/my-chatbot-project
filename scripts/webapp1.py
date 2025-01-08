from flask import Flask, request, jsonify, render_template
import sqlite3
import os

print("Current working directory:", os.getcwd())

db_path = "documents.db"

def chatbot_query(query):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        # Use full-text search with the MATCH clause
        cursor.execute("SELECT filename, content FROM docs WHERE docs MATCH ?", (f'"{query}"',))
        results = cursor.fetchall()
    except sqlite3.OperationalError as oe:
        print(f"OperationalError querying the database: {oe}")
        return ["An error occurred while querying the database. Please try again later."]
    except Exception as e:
        print(f"Unexpected error querying the database: {e}")
        return ["An unexpected error occurred. Please contact support."]
    finally:
        conn.close()

    if results:
        formatted_results = []
        for filename, content in results:
            snippet = content[:200]  # First 200 characters
            formatted_results.append(f"In file {filename}: {snippet}...")
        return formatted_results
    else:
        return ["No relevant information found for your query."]


app = Flask(__name__, template_folder="/home/RaThorat/my-chatbot-project/templates")

@app.route("/")
def index():
    try:
        # Serve the main HTML page
        with open("templates/index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "Index.html not found. Please check your templates directory.", 500

@app.route("/favicon.ico")
def favicon():
    return "", 204  # Return no content for favicon requests

@app.route("/chat", methods=["GET"])
def chat():
    try:
        query = request.args.get("query")
        if not query:
            return jsonify({"error": "Query parameter is required"}), 400

        # Call the chatbot_query function
        responses = chatbot_query(query)
        return jsonify({"responses": responses})
    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        return jsonify({"error": "An error occurred while processing your request. Please try again later."}), 500


if __name__ == "__main__":
    app.run(debug=True)
