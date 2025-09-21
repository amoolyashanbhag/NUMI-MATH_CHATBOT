import os
import re
import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from groq import Groq

# === CONFIG ===
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("❌ API key not found. Please set GROQ_API_KEY environment variable.")

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

client = Groq(api_key=API_KEY)

# === MEMORY ===
qa_history = []

# === SYSTEM PROMPT ===
system_prompt = {
    "role": "system",
    "content": (
        "You are a strict math tutor bot. "
        "You must ONLY answer math-related questions: arithmetic, algebra, geometry, calculus, statistics, etc. "
        "Special rule: If the user asks for a multiplication table, always generate it clearly, row by row, up to at least 10 multiples. "
        "For example, '12 table' should produce:\n"
        "12 × 1 = 12\n"
        "12 × 2 = 24\n"
        "... up to 12 × 10 = 120\n\n"
        "If the user asks about anything unrelated to math, reply: "
        "'I can only help with math questions. Please ask me a math problem.'"
    )
}

messages = [system_prompt]

# === MEMORY HANDLER ===
def is_memory_query(text: str):
    return bool(re.search(r"(first|last|\d+(st|nd|rd|th)?\s+question)", text.lower()))

def handle_memory_query(text: str):
    if not qa_history:
        return "⚠️ No questions in memory yet."

    text = text.lower().strip()

    if "first" in text:
        q, a = qa_history[0]
        return f"📝 First Question: {q}\n➡ Answer: {a}"

    if "last" in text:
        q, a = qa_history[-1]
        return f"📝 Last Question: {q}\n➡ Answer: {a}"

    match = re.search(r"(\d+)", text)
    if match:
        n = int(match.group(1))
        if 1 <= n <= len(qa_history):
            q, a = qa_history[n-1]
            return f"📝 {n}ᵗʰ Question: {q}\n➡ Answer: {a}"
        else:
            return f"⚠️ You only asked {len(qa_history)} question(s) so far."

    return "⚠️ I couldn’t figure out which past question you mean."

# === PLOT HANDLER ===
def is_plot_request(text: str):
    return bool(re.search(r"(plot|graph)\s+y\s*=", text.lower()))

def handle_plot_request(text: str):
    # Split by 'and' or ',' to support multiple equations
    raw_parts = re.split(r"\band\b|,", text.lower())
    equations = []

    for part in raw_parts:
        match = re.search(r"y\s*=\s*([a-z0-9np_+\-*/^\.\(\)]+)", part)
        if match:
            equations.append(match.group(1))

    if not equations:
        return "⚠️ Could not understand the equation(s) to plot."

    x = np.linspace(-10, 10, 400)
    plt.figure()

    for expr in equations:
        expr = expr.replace("^", "**")
        try:
            # Safe evaluation with numpy only
            y = eval(expr, {"x": x, "np": np, "__builtins__": {}})

            # Broadcast scalar constants (e.g., y = 5)
            if np.isscalar(y):
                y = np.full_like(x, y, dtype=float)

            plt.plot(x, y, label=f"y = {expr}")
        except Exception as e:
            return f"⚠️ Could not plot expression '{expr}': {e}"

    plt.legend()
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return img_base64

# === MATH BOT FUNCTION ===
def ask_math_bot(question: str):
    messages.append({"role": "user", "content": question})
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
    )
    reply = response.choices[0].message.content

    qa_history.append((question, reply))
    messages.append({"role": "assistant", "content": reply})
    return reply

