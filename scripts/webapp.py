from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    result = process_user_input(user_input)
    response = chatbot_response(result)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
