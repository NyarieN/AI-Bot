import os
import re
from flask import Flask, request, redirect, flash, render_template, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import PyPDF2

load_dotenv()

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecret123")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------
# Helper function
# -----------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

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

        if not files:
            flash("No files selected!", "error")
            return redirect(request.url)

        business_folder = os.path.join(UPLOAD_FOLDER, business_id)
        os.makedirs(business_folder, exist_ok=True)

        all_texts = []
        uploaded_files = []
        skipped_files = []

        for file in files:
            if allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(business_folder, filename)
                file.save(file_path)
                uploaded_files.append(filename)

                # Extract text from PDF
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    all_texts.append(text)
            else:
                skipped_files.append(file.filename)

        # Save combined text for keyword search
        with open(os.path.join(business_folder, "all_text.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(all_texts))

        if uploaded_files:
            flash(f"Uploaded: {', '.join(uploaded_files)}", "success")
        if skipped_files:
            flash(f"Skipped (not PDF): {', '.join(skipped_files)}", "error")

        return redirect(request.url)

    return render_template("admin.html")

# -----------------------
# Query Kai based on PDF text
# -----------------------
@app.route("/query", methods=["POST"])
def query_kai():
    data = request.json
    business_id = data.get("business_id")
    question = data.get("question", "").lower()

    text_file = os.path.join(UPLOAD_FOLDER, business_id, "all_text.txt")
    if not os.path.exists(text_file):
        return jsonify({"answer": "No documents uploaded for this business."})

    with open(text_file, "r", encoding="utf-8") as f:
        all_text = f.read()

    # Simple keyword matching: return the most relevant sentence
    sentences = re.split(r'(?<=[.!?]) +', all_text)
    best_sentence = max(
        sentences,
        key=lambda s: sum(word in s.lower() for word in question.split()),
        default="I could not find an answer in the documents."
    )

    return jsonify({"answer": best_sentence})

# -----------------------
# Home route
# -----------------------
@app.route("/test")
def test_widget():
    return render_template("test.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
