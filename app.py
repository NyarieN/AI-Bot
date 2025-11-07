import os
import json
import re
from flask import Flask, render_template, request, jsonify, redirect, flash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from cryptography.fernet import Fernet

# Optional libraries for multiple file types
import PyPDF2
import openpyxl
from docx import Document
from pptx import Presentation
import csv
from io import BytesIO, StringIO

# OpenAI
from openai import OpenAI

load_dotenv()

# -----------------------
# Config
# -----------------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecret123")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf", "txt", "docx", "doc", "xlsx", "xls", "csv", "pptx", "ppt"}

# Encryption key for files (store safely in .env)
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if not ENCRYPTION_KEY:
    ENCRYPTION_KEY = Fernet.generate_key().decode()
    print(f"Generated ENCRYPTION_KEY: {ENCRYPTION_KEY}")
fernet = Fernet(ENCRYPTION_KEY.encode())

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------
# Helpers
# -----------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def encrypt_file(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    encrypted = fernet.encrypt(data)
    with open(file_path, "wb") as f:
        f.write(encrypted)

def decrypt_file(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return fernet.decrypt(data)

def extract_text_from_file(file_path):
    ext = file_path.rsplit(".", 1)[1].lower()
    try:
        decrypted_data = decrypt_file(file_path)
        if ext == "pdf":
            f = BytesIO(decrypted_data)
            reader = PyPDF2.PdfReader(f)
            return "\n".join([page.extract_text() or "" for page in reader.pages])
        elif ext == "txt":
            return decrypted_data.decode()
        elif ext in ("docx", "doc"):
            f = BytesIO(decrypted_data)
            doc = Document(f)
            return "\n".join([p.text for p in doc.paragraphs])
        elif ext in ("xlsx", "xls"):
            f = BytesIO(decrypted_data)
            wb = openpyxl.load_workbook(f, data_only=True)
            text = ""
            for sheet in wb.sheetnames:
                ws = wb[sheet]
                for row in ws.iter_rows(values_only=True):
                    text += " ".join([str(cell) for cell in row if cell]) + "\n"
            return text
        elif ext == "csv":
            f = StringIO(decrypted_data.decode())
            reader = csv.reader(f)
            text = ""
            for row in reader:
                text += " ".join(row) + "\n"
            return text
        elif ext in ("pptx", "ppt"):
            f = BytesIO(decrypted_data)
            prs = Presentation(f)
            text = ""
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text += shape.text + "\n"
            return text
        else:
            return ""
    except Exception as e:
        print(f"Failed to extract {file_path}: {e}")
        return ""

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
        uploaded_files = []
        skipped_files = []

        for file in files:
            if allowed_file(file.filename):
                filename = secure_filename(file.filename)
                path = os.path.join(business_folder, filename)
                file.save(path)

                # Encrypt immediately
                encrypt_file(path)

                # Extract text for bot
                text = extract_text_from_file(path)
                if text:
                    all_texts.append(text)
                uploaded_files.append(filename)
            else:
                skipped_files.append(file.filename)

        # Save combined text for querying
        combined_text_path = os.path.join(business_folder, "all_text.txt")
        with open(combined_text_path, "w", encoding="utf-8") as f:
            f.write("\n".join(all_texts))
        encrypt_file(combined_text_path)

        if uploaded_files:
            flash(f"Uploaded: {', '.join(uploaded_files)}", "success")
        if skipped_files:
            flash(f"Skipped (unsupported): {', '.join(skipped_files)}", "error")

        return redirect(request.url)

    return render_template("admin.html")

# -----------------------
# PDF / text query mode
# -----------------------
@app.route("/query", methods=["POST"])
def query_pdf():
    data = request.json
    business_id = data.get("business_id")
    question = data.get("question", "").lower()

    text_file = os.path.join(UPLOAD_FOLDER, business_id, "all_text.txt")
    if not os.path.exists(text_file):
        return jsonify({"answer": "No documents uploaded for this business."})

    all_text = decrypt_file(text_file).decode()

    sentences = re.split(r'(?<=[.!?]) +', all_text)
    best_sentence = max(
        sentences,
        key=lambda s: sum(word in s.lower() for word in question.split()),
        default="I could not find an answer in the documents."
    )
    return jsonify({"answer": best_sentence})

# -----------------------
# AI / OpenAI query mode
# -----------------------
@app.route("/ask", methods=["POST"])
def ask_ai():
    data = request.json
    business_id = data.get("business_id")
    user_message = data.get("message")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are Kai, assistant for {business_id}."},
                {"role": "user", "content": user_message}
            ]
        )
        reply = response.choices[0].message.content
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"Error: {e}"})

# -----------------------
# Home
# -----------------------
@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
