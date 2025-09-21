import matplotlib
matplotlib.use("Agg")  # For headless servers
import matplotlib.pyplot as plt
import io, base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from mathbot import (
    ask_math_bot,
    is_memory_query,
    handle_memory_query,
    is_plot_request,
    handle_plot_request,
)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")

    if is_memory_query(user_input):
        reply = handle_memory_query(user_input)
        return jsonify({"reply": reply})

    elif is_plot_request(user_input):
        img_base64 = handle_plot_request(user_input)
        if isinstance(img_base64, str) and img_base64.startswith("⚠️"):
            return jsonify({"reply": img_base64})
        return jsonify({"reply": "Here’s your graph 📈", "image": img_base64})

    else:
        reply = ask_math_bot(user_input)
        return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)
