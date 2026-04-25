import time
import re
import random

# 🔥 simple keyword rules (baseline logic)
CLAUSE_KEYWORDS = {
    "Parties": ["party", "parties", "between"],
    "Effective Date": ["effective", "commence"],
    "Expiration Date": ["expire", "termination date"],
    "Payment Terms": ["payment", "fee", "amount"],
    "Confidentiality": ["confidential"],
    "Termination for Convenience": ["terminate"],
    "Governing Law": ["governed by", "law"],
    "Liability": ["liability", "damages"],
}


# ─────────────────────────────────────────────
# BASIC TEXT NORMALIZATION
# ─────────────────────────────────────────────

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


# ─────────────────────────────────────────────
# 🔥 BASIC NON-LEGAL CLASSIFIER (ADDED)
# ─────────────────────────────────────────────

def is_legal_text(text):
    text = text.lower()

    legal_keywords = [
        "agreement", "contract", "party", "shall",
        "liability", "termination", "confidential", "law"
    ]

    hits = sum(1 for kw in legal_keywords if kw in text)

    return hits >= 2 and len(text.split()) > 50


# ─────────────────────────────────────────────
# WEAKENED CLAUSE EXTRACTION
# ─────────────────────────────────────────────

def extract_clauses(text):
    sentences = text.split(".")
    clauses = {}

    for clause, keywords in CLAUSE_KEYWORDS.items():
        for sent in sentences:

            # 🔥 only use first keyword + random drop
            if keywords and keywords[0] in sent.lower() and random.random() > 0.5:

                clauses[clause] = {
                    "span": sent.strip(),
                    "score": 0.3
                }

                # 🔥 limit clauses early
                if len(clauses) >= 3:
                    return clauses

    return clauses


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

# PDF WRAPPER

def run_pipeline(pdf_bytes: bytes, filename=None):
    try:
        import pytesseract
        from pdf2image import convert_from_bytes

        # 🔥 OCR extraction
        images = convert_from_bytes(pdf_bytes, dpi=200)
        pages = [pytesseract.image_to_string(img, lang="eng") for img in images]
        text = "\n".join(pages)

        # 🔥 reuse existing text pipeline (no duplication)
        return run_pipeline_on_text(text, filename)

    except Exception as e:
        return {"error": f"PDF processing failed: {str(e)}"}

def run_pipeline_on_text(text: str, filename=None, gold_spans=None):
    start = time.time()

    if not text or len(text.strip()) < 50:
        return {"error": "Text too short"}

    # 🔥 ADDED CHECK (non-legal detection)
    if not is_legal_text(text):
        return {"error": "This does not look like a legal document"}

    # 🔥 Step 1: weakened extraction
    clauses = extract_clauses(text)

    # 🔥 Step 2: weakened summary
    clause_list = list(clauses.values())[:2]

    if clause_list:
        summary = " ".join(v["span"][:30] for v in clause_list)
    else:
        summary = text[:150]

    # 🔥 remove strong semantic anchors
    summary = summary.replace("agreement", "").replace("party", "")

    return {
        "filename": filename,
        "clauses": clauses,
        "summary": summary,
        "pipeline": "v1"
    }