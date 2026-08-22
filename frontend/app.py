from flask import Flask, render_template, request, jsonify
import sys
from pathlib import Path

# Allow the frontend to use the Part 3 answer generator
PART3_DIR = Path(__file__).resolve().parent.parent / "part3"
sys.path.insert(0, str(PART3_DIR))

from answer_generator import generate_answer

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({
            "answer": "Please enter a question."
        })

    try:
        answer = generate_answer(question)

        return jsonify({
            "answer": answer
        })

    except Exception as error:
        print("ERROR:", error)

        return jsonify({
            "answer": "Unable to generate an answer. Please check the backend logic."
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )