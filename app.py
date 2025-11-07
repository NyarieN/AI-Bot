import os
import json
import PyPDF2
import numpy as np
from openai import OpenAI
from flask import Flask, render_template, request, jsonify, redirect, flash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecret123")

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------
# Helpers
# -----------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# -----------------------
# PDF-only query
# -----------------------
@app.route("/query", methods=["POST"])
def query_pdf():
    data = request.json
    business_id = data.get("business_id")
    question = data.get("question", "").lower()

    text_file = os.path.join(UPLOAD_FOLDER, business_id, "all_text.txt")
    if not os.path.exists(text_file):
        return jsonify({"answer": "No documents uploaded for this business."})

    with open(text_file, "r", encoding="utf-8") as f:
        all_text = f.read()

    import re
    sentences = re.split(r'(?<=[.!?]) +', all_text)
    best_sentence = max(
        sentences,
        key=lambda s: sum(word in s.lower() for word in question.split()),
        default="I could not find an answer in the documents."
    )
    return jsonify({"answer": best_sentence})

# -----------------------
# AI / OpenAI query
# -----------------------
@app.route("/ask", methods=["POST"])
def ask_ai():
    data = request.json
    business_id = data.get("business_id")
    user_message = data.get("message")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": f"You are Kai, assistant for {business_id}."},
                      {"role": "user", "content": user_message}]
        )
        reply = response.choices[0].message.content
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"Error: {e}"})

# -----------------------
# Admin portal
# -----------------------
@app.route("/admin", methods=["GET", "POST"])
def admin_upload():
    key = request.args.get("key")
    if key != os.getenv("ADMIN_KEY", "supersecret123"):
        return "Unauthorized", 403

    if request.method == "POST":
        business_id = request.form.get("business_id").lower()
        files = request.files.getlist("files")
        business_folder = os.path.join(UPLOAD_FOLDER, business_id)
        os.makedirs(business_folder, exist_ok=True)

        all_texts = []
        for file in files:
            if allowed_file(file.filename):
                filename = secure_filename(file.filename)
                path = os.path.join(business_folder, filename)
                file.save(path)
                all_texts.append(extract_text(path))

        # Save combined text
        with open(os.path.join(business_folder, "all_text.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(all_texts))

        flash("Files uploaded successfully!")
        return redirect(request.url)

    return render_template("admin.html")

# -----------------------
# Home page
# -----------------------
@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
