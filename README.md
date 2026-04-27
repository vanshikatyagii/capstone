
# ⚖️ LegalEase — Contract Analysis System

LegalEase is an AI-powered contract analysis platform that extracts key clauses, identifies entities, and generates summaries from legal documents.

The system supports **three pipelines (v1, v2, v3)** with increasing levels of sophistication, allowing users to compare outputs and performance.

---

## 🚀 Tech Stack

- **Frontend:** React (Vite)
- **Backend:** Flask (Python)
- **Database:** MongoDB Atlas
- **ML Models:**
  - DeBERTa (QA)
  - LegalBERT (NER)
  - Sentence Transformers (semantic scoring)
  - OCR (Tesseract + pdf2image)

---

## 🧠 Pipelines Overview

### 🔹 v1 — Baseline (Rule-based)

- Simple keyword matching  
- Randomized clause extraction  
- Minimal scoring  
- Basic summary generation  
- Includes **basic non-legal text detection**

**Purpose:**
- Baseline comparison  
- Demonstrates naive extraction  

---

### 🔹 v2 — ML-enhanced Pipeline

- DeBERTa QA for clause extraction  
- LegalBERT for NER  
- Semantic scoring (SentenceTransformer)  
- Clause ranking + filtering  
- Improved summary generation  
- Stronger non-legal text classifier  

**Improvements over v1:**
- Better clause accuracy  
- Context-aware extraction  
- Entity-level insights  

---

### 🔹 v3 — Production-grade Pipeline

- Chunked processing for large documents  
- Hybrid QA + keyword fallback  
- Multi-factor scoring:
  - QA confidence  
  - Semantic similarity  
  - Keyword density  
  - Length heuristics  
- Clause deduplication  
- Entity-aware refinement  
- Hallucination-controlled summarization  
- Strong non-legal classification  

**Improvements over v2:**
- Higher precision  
- Reduced noise  
- Better filtering of irrelevant clauses  
- More stable summaries  

## 🛠️ Setup Instructions

### 1️⃣ Clone the Repository

    git clone <https:github.com/vanshikatyagii/capstone.git>
    cd capstone
### 🧪 Backend Setup (Flask)

### 📌 Navigate to backend

    cd capstone_backend

### 📌 Create virtual environment:

    python -m venv venv

    source venv/bin/activate   (Mac)

    venv\Scripts\activate      (Windows)

### 📌 Install dependencies

    pip install -r requirements.txt

### 📌 Run Flask server

    python app.py

#### Server runs on: http://localhost:5001


## 💻 Frontend Setup (React + Vite)

### 📌 Navigate to frontend
    cd capstone_frontend

### 📌 Install dependencies
    npm install
### 📌 Run frontend
    npm run dev

#### Frontend runs on: http://localhost:5173
# 🔄 How It Works
#### User inputs: Text OR PDF

#### Pipeline processes:

    1. Clause extraction
    2. Entity recognition
    3. Summary generation
    4. Results stored in MongoDB
    5. Response returned to frontend

