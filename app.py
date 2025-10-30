from flask import Flask, request, jsonify
import os
import easyocr
import re
import google.generativeai as genai
from pdf2image import convert_from_path
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from flask_cors import CORS
from spellchecker import SpellChecker

# -------------------------
# LOAD ENVIRONMENT VARIABLES
# -------------------------
load_dotenv()

app = Flask(__name__)
CORS(app, origins=["http://localhost:8080", "http://127.0.0.1:8080"])

# Load keys
API_KEY = os.getenv("API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Uploads folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=False)


# -------------------------
# API KEY AUTH DECORATOR
# -------------------------
def require_api_key(f):
    def decorated_function(*args, **kwargs):
        client_key = request.headers.get("X-API-Key")
        if client_key != API_KEY:
            print("❌ Invalid or missing API key!")
            return jsonify({"error": "Unauthorized - Invalid API key"}), 401
        print("✅ API key authenticated successfully.")
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function


# -------------------------
# CLEAN TEXT FUNCTION
# -------------------------
def clean_text(raw_text):
    print("🧹 Cleaning extracted text...")
    text = re.sub(r'\s+', ' ', raw_text)
    text = re.sub(r'[^A-Za-z0-9%:.,/\\\- ()]+', '', text)
    return text.strip()


# -------------------------
# SPELLCHECK FUNCTION
# -------------------------
def correct_spelling(text):
    print("🔠 Correcting OCR spelling mistakes...")
    spell = SpellChecker()
    words = text.split()
    corrected = []
    for w in words:
        if w.isalpha():
            corrected.append(spell.correction(w) or w)
        else:
            corrected.append(w)
    return " ".join(corrected)


# -------------------------
# OCR EXTRACTION FUNCTION
# -------------------------
def extract_text_from_file(file_path):
    print(f"📄 Extracting text from: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    if ext == ".pdf":
        pages = convert_from_path(file_path)
        for i, page in enumerate(pages):
            temp_img = os.path.join(UPLOAD_FOLDER, f"page_{i}.jpg")
            page.save(temp_img, "JPEG")
            result = reader.readtext(temp_img, detail=0)
            text += "\n".join(result)
    else:
        result = reader.readtext(file_path, detail=0)
        text = "\n".join(result)
    print("✅ OCR extraction completed.")
    return text


# -------------------------
# SMART NUMERIC HIGHLIGHT
# -------------------------
def highlight_abnormal_values(text):
    print("📊 Checking numeric reference values...")
    pattern = r"([A-Za-z\s]+)\s+([\d.]+)\s*[a-zA-Z/%^]*\s*\(([\d.]+)\s*-\s*([\d.]+)\)"
    highlighted_text = text

    for match in re.finditer(pattern, text):
        test_name, value, ref_min, ref_max = match.groups()
        try:
            value = float(value)
            ref_min = float(ref_min)
            ref_max = float(ref_max)

            if value < ref_min or value > ref_max:
                replacement = f"<span style='color:red; font-weight:bold;'>{match.group(0)}</span>"
            else:
                replacement = f"<span style='color:green; font-weight:bold;'>{match.group(0)}</span>"

            highlighted_text = highlighted_text.replace(match.group(0), replacement)
        except ValueError:
            continue

    return highlighted_text


# -------------------------
# GEMINI SEVERITY CLASSIFIER
# -------------------------
def classify_severity(summary_text):
    print("🩺 Classifying severity level using Gemini...")
    prompt = f"""
    You are a medical text reviewer.
    Read the following medical summary carefully and determine the case severity.

    Choose only one of:
    A - Abnormal (serious findings or abnormal values)
    B - Mild (minor variations or mild findings)
    C - Normal (everything within healthy limits)

    Respond with only the letter (A, B, or C).

    SUMMARY:
    {summary_text}
    """

    model = genai.GenerativeModel("gemini-2.5-flash")
    try:
        response = model.generate_content(prompt)
        classification = getattr(response, "text", "").strip().upper()
        print(f"🔍 Gemini classification result: {classification}")
        if "A" in classification:
            return "A"
        elif "B" in classification:
            return "B"
        else:
            return "C"
    except Exception as e:
        print(f"⚠️ Gemini classification error: {e}")
        return "C"


# -------------------------
# GEMINI SUMMARY FUNCTION
# -------------------------
def summarize_medical_report(cleaned_text, doc_name=None):
    print("🧠 Generating medical summary dynamically using Gemini...")

    greeting = "Hello,"

    prompt = f"""
    You are an intelligent, empathetic medical summarizer.
    Read the medical report below carefully and summarize it in 5–10 lines.

    TASK:
    - Identify serious or abnormal findings and wrap them with:
      <span style='color:red; font-weight:bold;'>...</span>
    - Identify reassuring or normal findings and wrap them with:
      <span style='color:green; font-weight:bold;'>...</span>
    - Preserve medical accuracy, do NOT guess data.
    - Respect numeric reference ranges shown in the text.
    - Keep tone professional and clear.

    Structure:

    {greeting}

    [Summary of findings and interpretation]

    Doctor’s Advice:
    1. [Lifestyle or diet advice]
    2. [Rest or self-care suggestion]
    3. [Follow-up or next steps]

    MEDICAL REPORT:
    {cleaned_text}
    """

    model = genai.GenerativeModel("gemini-2.5-flash")
    try:
        response = model.generate_content(prompt)
        summary = getattr(response, "text", "").strip()
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        summary = "⚠️ Could not generate summary."

    # Highlight numeric abnormalities
    summary = highlight_abnormal_values(summary)

    # Classify the report
    severity = classify_severity(summary)

    # Append human-readable message with explanation
    if severity == "A":
        summary += (
            "<br><br>⚠️ <b>Some findings require follow-up with your doctor.</b> "
            "These results indicate notable abnormalities or tissue pattern changes "
            "that may need further medical evaluation to rule out potential risks."
        )
    elif severity == "B":
        summary += (
            "<br><br>🟡 <b>Mild variations observed.</b> "
            "Some readings are slightly outside normal limits, "
            "but they typically do not indicate serious problems. "
            "Monitoring and lifestyle adjustments may be advised."
        )
    else:
        summary += (
            "<br><br>✅ <b>All parameters appear within normal limits.</b> "
            "No major concerns detected in this report."
        )

    print(f"✅ Summary generated ({severity} case).")
    return summary


# -------------------------
# MAIN API ROUTE
# -------------------------
@app.route("/api/testing", methods=["POST"])
@require_api_key
def process_documents():
    print("📩 Received document processing request...")

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded!"}), 400

    files = request.files.getlist("file")
    results = []

    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        print(f"⚙️ Processing file: {filename}")

        raw_text = extract_text_from_file(file_path)
        cleaned = clean_text(raw_text)
        cleaned = correct_spelling(cleaned)

        lowered = cleaned.lower()
        if "histopathology" in lowered or "endometrial polyp" in lowered:
            doc_name = "Histopathology Report"
        elif "cytology" in lowered or "pap" in lowered:
            doc_name = "PAP Test Report"
        elif "haematology" in lowered or "blood count" in lowered:
            doc_name = "Blood Test Report"
        else:
            doc_name = "General Medical Report"

        summary = summarize_medical_report(cleaned, doc_name)

        results.append({
            "filename": filename,
            "document_type": doc_name,
            "summary": summary
        })

    print("✅ All files processed successfully.")
    combined_output = "\n\n=========================\n\n".join(
        [f"📄 {r['filename']} ({r['document_type']})\n\n{r['summary']}" for r in results]
    )

    return combined_output, 200, {"Content-Type": "text/html; charset=utf-8"}


# -------------------------
# RUN SERVER
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
