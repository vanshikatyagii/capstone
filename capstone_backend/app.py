import os
import uuid
import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient

# 🔥 IMPORT PIPELINES
from v1_pipeline import run_pipeline_on_text as run_v1
from v2_pipeline import run_pipeline_on_text as run_v2
from v3_pipeline import run_pipeline_on_text as run_v3

from v1_pipeline import run_pipeline as run_v1_pdf
from v2_pipeline  import run_pipeline as run_v2_pdf
from v3_pipeline import run_pipeline as run_v3_pdf

app = Flask(__name__)
CORS(app)

# ================= DB =================
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://legaleaseadmin:2405@legalease-cluster.n4smonr.mongodb.net/legalease?retryWrites=true&w=majority"
)

client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=5000,
    tls=True,
    tlsAllowInvalidCertificates=True
)

try:
    client.admin.command("ping")
    print("✅ MongoDB connected")
    
except Exception as e:
    print("❌ MongoDB connection failed:", e)

db = client["legalease"]
results = db["results"]
print("DB:", db.name)

# ================= PIPELINE MAP =================
PIPELINE_MAP = {
    "v1": run_v1,
    "v2": run_v2,
    "v3": run_v3
}

PIPELINE_PDF_MAP = {
    "v1": run_v1_pdf,
    "v2": run_v2_pdf,
    "v3": run_v3_pdf
}

# ================= TEXT =================
@app.route("/analyze-text", methods=["POST"])
def analyze_text():
    try:
        data = request.json
        text = data.get("text")
        pipeline = data.get("pipeline", "v3")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        if pipeline not in PIPELINE_MAP:
            return jsonify({"error": "Invalid pipeline selected"}), 400

        # 🔥 Direct function call (NO HTTP)
        result = PIPELINE_MAP[pipeline](text, "Text Input")

        if "error" in result:
            return jsonify(result), 400

        clauses = result.get("clauses", {})

        doc = {
            "docId": str(uuid.uuid4()),
            "filename": "Text Input",
            "pipeline": pipeline,
            "summary": result.get("summary", ""),
            "clauses": clauses,
            "clause_count": len(clauses),
            "timestamp": datetime.datetime.utcnow()
        }

        inserted = results.insert_one(doc)
        doc["_id"] = str(inserted.inserted_id)
        return jsonify(doc)

    except Exception as e:
        print("TEXT ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ================= PDF =================
@app.route("/analyze-pdf", methods=["POST"])
def analyze_pdf():
    try:
        file = request.files.get("pdf")
        pipeline = request.form.get("pipeline", "v3")

        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        if pipeline not in PIPELINE_PDF_MAP:
            return jsonify({"error": "Invalid pipeline selected"}), 400

        pdf_bytes = file.read()

        # 🔥 Direct function call
        result = PIPELINE_PDF_MAP[pipeline](pdf_bytes, file.filename)

        if "error" in result:
            return jsonify(result), 400

        clauses = result.get("clauses", {})

        doc = {
            "docId": str(uuid.uuid4()),
            "filename": file.filename,
            "pipeline": pipeline,
            "summary": result.get("summary", ""),
            "clauses": clauses,
            "clause_count": len(clauses),
            "timestamp": datetime.datetime.utcnow()
        }
        inserted = results.insert_one(doc)
        doc["_id"] = str(inserted.inserted_id)

        return jsonify(doc)

    except Exception as e:
        print("PDF ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ================= GET ALL =================
@app.route("/results", methods=["GET"])
def get_results():
    docs = list(results.find().sort("timestamp", -1))

    return jsonify([
        {
            "docId": d["docId"],
            "filename": d["filename"],
            "summary": d.get("summary", "")
        }
        for d in docs
    ])


# ================= GET ONE =================
@app.route("/results/<doc_id>", methods=["GET"])
def get_one(doc_id):
    doc = results.find_one({"docId": doc_id})

    if not doc:
        return jsonify({"error": "Not found"}), 404

    doc["_id"] = str(doc["_id"])
    return jsonify(doc)


# ================= RUN =================
if __name__ == "__main__":
    app.run(port=5001, debug=True)