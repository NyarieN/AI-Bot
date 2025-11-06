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
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecret")

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Folders
UPLOAD_FOLDER = "uploads"
EMBED_FOLDER = "embeddings"
ALLOWED_EXTENSIONS = {"pdf"}

for folder in [UPLOAD_FOLDER, EMBED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# --- Helpers ---
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_text(text_list):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text_list
    )
    return [item.embedding for item in response.data]

def build_embeddings(business_id):
    embeddings_data = []
    business_folder = os.path.join(UPLOAD_FOLDER, business_id)
    if not os.path.exists(business_folder):
        os.makedirs(business_folder)
    for pdf_file in os.listdir(business_folder):
        if pdf_file.endswith(".pdf"):
            text = extract_text_from_pdf(os.path.join(business_folder, pdf_file))
            chunks = chunk_text(text)
            chunk_embeddings = embed_text(chunks)
            for i, chunk in enumerate(chunks):
                embeddings_data.append({
                    "text": chunk,
                    "embedding": chunk_embeddings[i]
                })
    with open(os.path.join(EMBED_FOLDER, f"{business_id}.json"), "w", encoding="utf-8") as f:
        json.dump(embeddings_data, f)
    print(f"Embeddings updated for {business_id}")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_embeddings(query, business_id, top_k=3):
    embed_file = os.path.join(EMBED_FOLDER, f"{business_id}.json")
    if not os.path.exists(embed_file):
        return []
    with open(embed_file, "r", encoding="utf-8") as f:
        embeddings_data = json.load(f)
    query_emb = embed_text([query])[0]
    for item in embeddings_data:
        item["score"] = cosine_similarity(query_emb, item["embedding"])
    embeddings_data.sort(key=lambda x: x["score"], reverse=True)
    return [item["text"] for item in embeddings_data[:top_k]]

# --- Routes ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    business_id = data.get("business_id", "")
    user_message = data.get("message", "")
    relevant_chunks = search_embeddings(user_message, business_id)
    context = "\n".join(relevant_chunks)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are Kai, a professional assistant for {business_id}."},
                {"role": "user", "content": f"{user_message}\n\nReference knowledge:\n{context}"}
            ],
        )
        reply = response.choices[0].message.content
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"Error: {e}"})

# --- Admin Upload ---
@app.route("/admin", methods=["GET", "POST"])
def admin_upload():
    secret_key = request.args.get("key")
    if secret_key != os.getenv("ADMIN_KEY"):
        return "Unauthorized", 403

    if request.method == "POST":
        business_id = request.form.get("business_id")
        files = request.files.getlist("files")
        if not files:
            flash("No files selected!", "error")
            return redirect(request.url)

        business_folder = os.path.join(UPLOAD_FOLDER, business_id)
        os.makedirs(business_folder, exist_ok=True)

        for file in files:
            if file.filename.endswith(".pdf"):
                file.save(os.path.join(business_folder, file.filename))
                flash(f"Uploaded: {file.filename}", "success")
            else:
                flash(f"Skipped (not PDF): {file.filename}", "error")

        # Automatically generate embeddings after upload
        build_embeddings(business_id)
        flash("Embeddings generated successfully!", "success")

        return redirect(request.url)

    return render_template("admin.html")
