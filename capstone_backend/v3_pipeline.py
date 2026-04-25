"""
CLNEA Pipeline Orchestrator
"""

import os, uuid, logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
# from rouge_score import rouge_scorer
# from bert_score import score as bertscore_fn
import pytesseract
from pdf2image import convert_from_bytes
from sentence_transformers import SentenceTransformer, util

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

logger = logging.getLogger(__name__)

# ──  V3 Non-Legal Classifier (Hybrid + Semantic) ───────────

LEGAL_PROTOTYPES = [
    "This agreement is made between two parties and defines obligations.",
    "The contract specifies terms including payment, liability, and termination.",
    "This document outlines legal obligations, rights, and governing law.",
]

NON_LEGAL_PROTOTYPES = [
    "Hey how are you doing today",
    "This is an email regarding project updates",
    "Let's catch up tomorrow",
    "Random notes and thoughts written casually",
]

# 🔥 cache embeddings (IMPORTANT)
_legal_embs = None
_non_legal_embs = None


def is_legal_text_v3(text: str) -> bool:
    global _semantic_model, _legal_embs, _non_legal_embs

    text_lower = text.lower()
    word_count = len(text.split())

    # 🔹 keyword signals
    legal_keywords = [
        "agreement", "contract", "party", "shall", "liability",
        "termination", "confidential", "governing law",
        "indemnify", "jurisdiction", "obligation"
    ]

    non_legal_keywords = [
        "hey", "hello", "thanks", "regards", "lol",
        "hi team", "good morning", "please find attached"
    ]

    legal_hits = sum(1 for kw in legal_keywords if kw in text_lower)
    non_legal_hits = sum(1 for kw in non_legal_keywords if kw in text_lower)

    # 🔹 structural signals
    has_sections = bool(re.search(r"\n\s*\d+[\.\)]", text))
    has_caps_headers = bool(re.search(r"\n[A-Z\s]{5,}\n", text))

    long_sentences = sum(
        1 for s in re.split(r"[.!?]", text)
        if len(s.split()) > 20
    )

    # 🔹 semantic similarity (SAFE)
    legal_sim = 0.0
    non_legal_sim = 0.0

    if _semantic_model is not None:
        try:
            text_emb = _semantic_model.encode(text[:1000], convert_to_tensor=True)

            if _legal_embs is None:
                _legal_embs = _semantic_model.encode(LEGAL_PROTOTYPES, convert_to_tensor=True)

            if _non_legal_embs is None:
                _non_legal_embs = _semantic_model.encode(NON_LEGAL_PROTOTYPES, convert_to_tensor=True)

            legal_sim = float(util.cos_sim(text_emb, _legal_embs).max().item())
            non_legal_sim = float(util.cos_sim(text_emb, _non_legal_embs).max().item())

        except:
            pass  # don't break classifier

    # 🔹 scoring
    score = 0
    score += legal_hits * 2
    score += 2 if has_sections else 0
    score += 2 if has_caps_headers else 0
    score += min(long_sentences, 3)
    score += legal_sim * 3
    score -= non_legal_sim * 2
    score -= non_legal_hits * 2

    # 🔥 relaxed threshold (IMPORTANT)
    return score >= 4 and word_count > 50

# Helpers

def split_text(text, chunk_size=1200, overlap=200):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]


def normalize_scores(spans):
    if not spans:
        return spans

    max_score = max((v["score"] for v in spans.values()), default=1.0) or 1.0
    for k in spans:
        spans[k]["score"] = round(spans[k]["score"] / max_score, 4)

    return spans


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)

    corrections = {
        "Thls": "This",
        "agreernent": "agreement",
        "clausee": "clause",
        "partles": "parties",
        "llability": "liability",
    }

    for k, v in corrections.items():
        text = text.replace(k, v)

    return text.strip()



# FINAL FILTERING LOGIC 

GENERIC_BAD = [
    "to the extent",
    "in accordance",
    "without limitation",
    "subject to",
    "as described",
    "herein",
]

# Model Cache

MODEL_CACHE_DIR = Path(os.getenv("CLNEA_MODEL_CACHE", ".hf-cache")).resolve()
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODEL_CACHE_DIR / "hub"))

_qa_pipeline  = None
_ner_pipeline = None
_tokenizer    = None
_model        = None
_semantic_model = None


# Model Loading

def preload_models():
    global _qa_pipeline, _ner_pipeline, _tokenizer, _model, _semantic_model

    from transformers import (
        pipeline,
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        AutoModelForTokenClassification,
        AutoTokenizer,
    )
    import torch

    # 🔥 Load QA / NER / Pegasus ONLY if not loaded
    if _qa_pipeline is None:
        logger.info("Loading QA model...")
        qa_tokenizer = AutoTokenizer.from_pretrained(
            "deepset/deberta-v3-base-squad2",
            cache_dir=str(MODEL_CACHE_DIR),
        )
        qa_model = AutoModelForQuestionAnswering.from_pretrained(
            "deepset/deberta-v3-base-squad2",
            cache_dir=str(MODEL_CACHE_DIR),
        )
        _qa_pipeline = pipeline(
            "question-answering",
            model=qa_model,
            tokenizer=qa_tokenizer,
            device=-1
        )

        logger.info("Loading NER model...")
        ner_tokenizer = AutoTokenizer.from_pretrained(
            "nlpaueb/legal-bert-base-uncased",
            cache_dir=str(MODEL_CACHE_DIR),
        )
        ner_model = AutoModelForTokenClassification.from_pretrained(
            "nlpaueb/legal-bert-base-uncased",
            cache_dir=str(MODEL_CACHE_DIR),
        )
        _ner_pipeline = pipeline(
            "ner",
            model=ner_model,
            tokenizer=ner_tokenizer,
            aggregation_strategy="simple",
            device=-1
        )

    # 🔥 ALWAYS ensure semantic model loads
    if _semantic_model is None:
        logger.info("Loading semantic reranker model...")
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    
# Stage 1 OCR

import pytesseract
from pdf2image import convert_from_bytes

def stage1_ocr(pdf_bytes: bytes) -> str:
    logger.info("Stage 1: OCR...")

    images = convert_from_bytes(pdf_bytes, dpi=300)

    pages = []
    for img in images:
        text = pytesseract.image_to_string(
            img,
            lang="eng",
            config="--oem 3 --psm 6"
        )
        pages.append(text)

    full_text = "\n".join(pages)

    print("OCR SAMPLE:", full_text[:500])  # 🔍 debug
    logger.info(f"OCR done: {len(full_text)} chars")

    return full_text


# Stage 2 (FINAL IMPROVED)


MIN_WORDS = 3

GENERIC_PATTERNS = [
    "to the extent permitted",
    "in accordance with",
    "subject to",
    "as applicable",
    "herein",
    "thereof",
]

CUAD_QUESTIONS = {
    "Parties": "This agreement is between which parties?",
    "Effective Date": "This agreement is effective on what date?",
    "Expiration Date": "This agreement ends on what date?",
    "Payment Terms": "What payment amount and payment terms are specified?",
    "Confidentiality": "What does the contract say about confidentiality?",
    "Termination for Convenience": "How can either party terminate this agreement?",
    "Limitation of Liability": "What is the liability limit or cap mentioned?",
    "Governing Law": "This agreement is governed by which law?",
    "Notice": "What notice period is required?",
    "IP Ownership Assignment": "Who owns the intellectual property or work product?",
    "License Grant": "What rights are granted under the license?",
    "Dispute Resolution": "How are disputes resolved?",
    "Arbitration": "Is arbitration mentioned in the agreement?",
    "Force Majeure": "What happens in case of events beyond control?",
    "Indemnification": "Who indemnifies whom under this agreement?", 
    "Non-Compete": "Is there any restriction on competing activities?",
    "Non-Disparagement": "Is there any restriction on negative statements?",
    "Anti-Assignment": "Can this agreement be assigned to another party?",
    "Change of Control": "What happens if ownership of a party changes?",
    "Third Party Beneficiary": "Does this agreement mention third party beneficiaries?",
    "Audit Rights": "Are there any audit rights mentioned?",
    "Warranty Duration": "What is the duration of the warranty?",
    "Insurance": "What insurance requirements are specified?",
    "Amendment": "How can this agreement be changed or amended?",
    "Renewal Term": "Does this agreement automatically renew?",
    "Severability": "Is there a clause about invalid provisions?", 
    "Entire Agreement": "Does this agreement represent the entire understanding?",
    "Waiver": "Is there a waiver clause mentioned?",
}

CLAUSE_KEYWORDS = {
    "Parties": ["party", "parties", "between", "entered into"],
    "Effective Date": ["effective date", "commencement", "effective from", "shall commence"],
    "Expiration Date": ["expire", "expiration", "termination date", "end date", "term ends"],
    "Payment Terms": ["payment", "fee", "amount", "compensation", "invoice"],
    "Confidentiality": ["confidential", "confidentiality", "disclose", "non-disclosure", "proprietary"],
    "Non-Compete": ["compete", "competition", "non-compete", "restrict competition"],
    "Non-Disparagement": ["disparage", "non-disparagement", "negative statements"],
    "Indemnification": ["indemnify", "indemnification", "hold harmless", "defend"],
    "Limitation of Liability": ["liability", "damages", "limit", "cap", "maximum liability"],
    "Cap on Liability": ["liability cap", "maximum amount", "limit liability"],
    "IP Ownership Assignment": ["intellectual property", "ownership", "assign", "ip", "work product"],
    "License Grant": ["license", "grant", "rights", "use rights"],
    "License Scope": ["scope", "use", "rights", "permitted use"],
    "License Exclusivity": ["exclusive", "non-exclusive", "sole license"],
    "Audit Rights": ["audit", "inspect", "records", "books", "examination", "review records"],
    "Insurance": ["insurance", "coverage", "insured"],
    "Anti-Assignment": ["assign", "assignment", "transfer", "delegation"],
    "Change of Control": ["change of control", "acquisition", "merger", "ownership change"],
    "Force Majeure": ["force majeure", "acts of god", "beyond control", "unforeseen events"],
    "Dispute Resolution": ["dispute", "resolve", "resolution", "settlement"],
    "Arbitration": ["arbitration", "arbitrate", "arbitrator"],
    "Governing Law": ["governed by", "laws of", "applicable law"],
    "Governing Jurisdiction": ["jurisdiction", "court", "venue"],
    "Notice": ["notice", "notify", "written notice", "deliver notice"],
    "Notice Period to Terminate Renewal": ["notice period", "prior notice", "advance notice"],
    "Amendment": ["amend", "amendment", "modify", "modification"],
    "Renewal Term": ["renew", "renewal", "automatically renew", "auto-renew"],
    "Revenue or Profit Sharing": ["revenue", "profit", "share", "profit sharing"],
    "Minimum Commitment": ["minimum", "commitment", "minimum purchase", "obligation"],  
    "Volume Restriction": ["volume", "limit", "restriction", "quota"],
    "Third Party Beneficiary": ["third party", "beneficiary"],
    "Source Code Escrow": ["escrow", "source code", "code escrow"],
    "Covenant Not to Sue": ["not to sue", "covenant", "waive claims"],
    "Most Favored Nation": ["most favored nation", "mfn", "best terms"],
    "Liquidated Damages": ["liquidated damages", "penalty", "pre-determined damages"],
    "Price Restrictions": ["price", "pricing", "rate", "fixed price"],
    "Severability": ["severability", "invalid provision", "unenforceable provision"],
    "Entire Agreement": ["entire agreement", "whole agreement", "complete agreement"],
    "Waiver": ["waiver", "waive", "failure to enforce"],
    "Termination for Convenience": ["terminate", "termination", "without cause", "at any time"],
    "Warranty Duration": ["warranty", "guarantee", "duration", "warranty period"],
}

#stage-2 helpers
def _clean_stage2_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _keyword_score(answer: str, clause_type: str) -> float:
    keywords = CLAUSE_KEYWORDS.get(clause_type, [])
    if not keywords:
        return 0.0
    hits = sum(1 for kw in keywords if kw in answer.lower())
    return hits / len(keywords)


def _length_score(answer: str) -> float:
    return min(len(answer.split()) / 40, 1.0)


def _score_candidate(question, answer, qa_score, clause_type):
    keyword = _keyword_score(answer, clause_type)
    length = _length_score(answer)
    semantic = _semantic_score(question, answer)

    return (
        0.45 * qa_score +
        0.25 * semantic +
        0.20 * keyword +
        0.10 * length
    )

def _semantic_score(question: str, answer: str) -> float:
    try:
        q_emb = _semantic_model.encode(question, convert_to_tensor=True)
        a_emb = _semantic_model.encode(answer, convert_to_tensor=True)
        return float(util.cos_sim(q_emb, a_emb).item())
    except:
        return 0.0

def stage2_clause_segmentation(text: str) -> dict:
    logger.info("Stage 2: Clause segmentation (improved stable)...")

    if _qa_pipeline is None:
        raise RuntimeError("QA pipeline not initialized.")

    text = _clean_stage2_text(text)
    chunks = split_text(text)

    spans = {}

    for clause_type, question in CUAD_QUESTIONS.items():
        candidates = []

        for chunk in chunks:
            fallback_ans = None
            try:
                res = _qa_pipeline(
                    question=question,
                    context=chunk,
                    max_answer_len=200,
                    handle_impossible_answer=True
                )

                ans = res.get("answer", "").strip()
                qa_score = res.get("score", 0.0)
                if not ans or len(ans.split()) < 3 or ans.lower().startswith(("both", "either")):
                    fallback_ans = None

                for sent in chunk.split("."):
                    if any(k in sent.lower() for k in CLAUSE_KEYWORDS.get(clause_type, [])):
                        fallback_ans = sent.strip()
                        break

                if fallback_ans:
                    ans = fallback_ans
                    qa_score = 0.25  # low confidence fallback
                else:
                    continue

                # ❌ remove generic legal junk
                if any(p in ans.lower() for p in GENERIC_PATTERNS):
                    continue

                # 🔥 keyword relevance (CRITICAL FIX)
                keywords = CLAUSE_KEYWORDS.get(clause_type, [])
                if keywords and not any(k in ans.lower() for k in keywords):
                    if qa_score < 0.3:
                        continue

                if clause_type == "Payment Terms":
                    if "tax" in ans.lower():
                        continue
                # ✅ use improved scoring (you already defined it)
                final_score = _score_candidate(question, ans, qa_score, clause_type)

                candidates.append((ans, final_score))

            except Exception as e:
                print("QA ERROR:", e)

        # ✅ pick best candidate
        if candidates:
            best_ans, raw_score = max(candidates, key=lambda x: x[1])

            confidence = raw_score

            # keyword penalty
            keyword_strength = _keyword_score(best_ans, clause_type)
            if keyword_strength < 0.2:
                confidence *= 0.7

            # generic penalty
            if any(p in best_ans.lower() for p in GENERIC_PATTERNS):
                confidence *= 0.6

            # short answer penalty
            if len(best_ans.split()) < 4:
                confidence *= 0.75

            confidence = round(min(confidence, 1.0), 4)

            # 🔥 prevent duplicate spans across clauses
            if best_ans in [v["span"] for v in spans.values()]:
                continue

            spans[clause_type] = {
                "span": best_ans,
                "score": confidence,
            }
    logger.info(f"Stage 2 done: {len(spans)} clauses found")
    return spans


# Stage 3 (Improved + stable)

CLAUSE_SCHEMA = {
    "Governing Law":               ["GPE", "LOC", "DATE", "ORG"],
    "Parties":                     ["ORG", "PERSON"],
    "Effective Date":              ["DATE", "TIME"],
    "Expiration Date":             ["DATE", "TIME"],
    "Termination for Convenience": ["DATE", "TIME", "CARDINAL", "QUANTITY"],
    "Confidentiality":             ["DATE", "TIME", "CARDINAL", "QUANTITY", "ORG"],
    "Non-Compete":                 ["DATE", "TIME", "GPE", "LOC", "ORG"],
    "Non-Disparagement":           ["ORG", "PERSON"],
    "Indemnification":             ["ORG", "PERSON"],
    "Limitation of Liability":     ["MONEY", "PERCENT", "CARDINAL"],
    "IP Ownership Assignment":     ["ORG", "PERSON"],
    "License Grant":               ["ORG", "PERSON"],
    "License Scope":               ["ORG", "PERSON", "GPE"],
    "License Exclusivity":         ["ORG", "PERSON"],
    "Payment Terms":               ["MONEY", "DATE", "CARDINAL", "PERCENT"],
    "Audit Rights":                ["DATE", "TIME", "CARDINAL", "ORG"],
    "Warranty Duration":           ["DATE", "TIME", "CARDINAL", "QUANTITY"],
    "Price Restrictions":          ["MONEY", "PERCENT", "CARDINAL", "ORG"],
    "Minimum Commitment":          ["MONEY", "CARDINAL", "QUANTITY", "PERCENT"],
    "Volume Restriction":          ["CARDINAL", "QUANTITY", "PERCENT", "MONEY"],
    "Insurance":                   ["MONEY", "CARDINAL", "ORG"],
    "Anti-Assignment":             ["ORG", "PERSON"],
    "Change of Control":           ["ORG", "PERSON"],
    "Revenue or Profit Sharing":   ["PERCENT", "MONEY", "CARDINAL"],
    "Source Code Escrow":          ["ORG"],
    "Covenant Not to Sue":         ["ORG", "PERSON", "DATE"],
    "Third Party Beneficiary":     ["ORG", "PERSON"],
    "Most Favored Nation":         ["ORG", "PERSON"],
    "Cap on Liability":            ["MONEY", "PERCENT", "CARDINAL"],
    "Liquidated Damages":          ["MONEY", "CARDINAL", "DATE"],
    "Force Majeure":               ["DATE", "TIME", "GPE", "CARDINAL"],
    "Dispute Resolution":          ["GPE", "LOC", "ORG"],
    "Arbitration":                 ["GPE", "LOC", "ORG"],
    "Governing Jurisdiction":      ["GPE", "LOC", "ORG"],
    "Notice":                      ["DATE", "TIME", "CARDINAL", "ORG"],
    "Amendment":                   ["DATE", "CARDINAL", "ORG"],
    "Entire Agreement":            ["DATE", "ORG"],
    "Severability":                ["ORG"],
    "Waiver":                      ["DATE", "ORG"],
    "Renewal Term":                ["DATE", "TIME", "CARDINAL"],
    "Notice Period to Terminate Renewal": ["DATE", "TIME", "CARDINAL"],
}

DEFAULT_SCHEMA = ["ORG","PERSON","DATE","GPE","MONEY","CARDINAL"]
STOPWORDS = {"the","this","agreement","party","section"}


def _clean_entity(text: str) -> str:
    text = text.replace("##", "").replace(" ##", "")
    return text.strip()


def stage3_ner(clause_spans: dict) -> dict:
    logger.info("Stage 3: Clause-aware NER (refined)...")

    if _ner_pipeline is None:
        raise RuntimeError("NER pipeline not initialized.")

    results = {}

    for clause_type, span_data in clause_spans.items():
        span = span_data.get("span", "").strip()
        valid = CLAUSE_SCHEMA.get(clause_type, DEFAULT_SCHEMA)

        try:
            ents = _ner_pipeline(span) or []
        except Exception as e:
            print("NER ERROR:", e)
            results[clause_type] = []
            continue

        filtered = []
        seen = set()

        for e in ents:
            text = _clean_entity(e.get("word", ""))
            label = e.get("entity_group", "")
            score = e.get("score", 0.0)

            key = (text.lower(), label)

            # ✅ basic cleaning
            if not text or len(text) < 3:
                continue

            if text.lower() in STOPWORDS:
                continue

            # ✅ schema filtering
            if label not in valid:
                continue

            # ✅ dedup (stronger)
            if key in seen:
                continue

            # ✅ refined thresholds (less noise)
            if label in ["MONEY", "DATE"]:
                threshold = 0.6
            elif label in ["ORG", "PERSON"]:
                threshold = 0.72
            else:
                threshold = 0.8

            if score < threshold:
                continue

            seen.add(key)

            filtered.append({
                "text": text,
                "label": label,
                "score": round(score, 4)
            })

        # 🔥 smarter boost (prevents overconfidence)
        if filtered:
            boost = min(0.12, len(filtered) * 0.02)
            clause_spans[clause_type]["score"] = min(
                1.0,
                clause_spans[clause_type]["score"] + boost
            )

        results[clause_type] = filtered

    logger.info(f"Stage 3 done: {sum(len(v) for v in results.values())} entities")
    return results

# ── Stage 4: Hallucination-safe summarization ─────────────────────

PRIORITY_CLAUSES = [
    "Parties", "Effective Date", "Governing Law",
    "Termination for Convenience", "Confidentiality",
    "Payment Terms", "Limitation of Liability",
    "Indemnification", "IP Ownership Assignment",
]

MIN_CONFIDENCE_DISPLAY = 0.35


def stage4_summarize(clause_spans: dict, entities_per_clause: dict) -> dict:
    logger.info("Stage 4: Coherent summary (fixed)...")

    if not clause_spans:
        return {
            "summary": "No clauses extracted.",
            "prompt": "",
            "prompt_tokens": 0
        }

    IMPORTANT = [
        "Parties",
        "Effective Date",
        "Payment Terms",
        "Confidentiality",
        "Termination for Convenience",
        "Limitation of Liability",
        "Governing Law"
    ]

    # 🔹 STEP 1: select best clauses
    selected = []

    for clause in IMPORTANT:
        if clause in clause_spans:
            data = clause_spans[clause]
            span = data.get("span", "").strip()
            score = data.get("score", 0.0)

            if span and score >= 0.5:
                selected.append((clause, span))

    # fallback if too few
    if len(selected) < 3:
        others = sorted(
            clause_spans.items(),
            key=lambda x: -x[1].get("score", 0)
        )
        for clause, data in others:
            if clause not in [c for c, _ in selected]:
                span = data.get("span", "")
                if span:
                    selected.append((clause, span))
            if len(selected) >= 5:
                break

    # 🔹 STEP 2: build coherent sentences
    sentences = []

    for clause, span in selected:
        span = re.sub(r"\s+", " ", span).strip()

        # clean clause prefix
        span = re.sub(rf"^{clause}\s*", "", span, flags=re.IGNORECASE)

        # 🔥 make sentence readable
        if not span.endswith("."):
            span += "."

        # 🔥 convert to natural phrasing
        if clause == "Parties":
            sentences.append(f"The agreement is between {span}")
        elif clause == "Effective Date":
            sentences.append(f"It becomes effective {span}")
        elif clause == "Payment Terms":
            sentences.append(f"Payment terms specify that {span}")
        elif clause == "Confidentiality":
            sentences.append(f"The agreement includes confidentiality obligations where {span}")
        elif clause == "Termination for Convenience":
            sentences.append(f"The agreement may be terminated as follows: {span}")
        elif clause == "Limitation of Liability":
            sentences.append(f"Liability is limited such that {span}")
        elif clause == "Governing Law":
            sentences.append(f"The agreement is governed by {span}")
        else:
            sentences.append(span)

    # 🔹 STEP 3: final summary
    summary = " ".join(sentences[:5])

    return {
        "summary": summary,
        "prompt": "coherent-summary-v2",
        "prompt_tokens": len(summary.split())
    }

#Evaluation Metrics

# import string

# # TEXT NORMALIZATION

# def _normalize(text):
#     text = text.lower()
#     text = text.translate(str.maketrans("", "", string.punctuation))
#     return text.split()

# # SPAN F1 (Stage 2)

# def compute_span_f1(pred: str, gold: str) -> float:
#     p = set(_normalize(pred))
#     g = set(_normalize(gold))

#     if not p or not g:
#         return 0.0

#     common = len(p & g)
#     precision = common / len(p)
#     recall = common / len(g)

#     if precision + recall == 0:
#         return 0.0

#     return round(2 * precision * recall / (precision + recall), 4)

# # EXACT MATCH

# def compute_exact_match(pred: str, gold: str) -> float:
#     return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0


# # MACRO F1 (across clauses)
# def compute_macro_f1(pred_spans: dict, gold_spans: dict) -> dict:
#     f1_scores = {}
#     em_scores = {}

#     for clause_type, gold_text in gold_spans.items():
#         if clause_type not in pred_spans:
#             continue

#         pred_data = pred_spans.get(clause_type, {})
#         pred_text = pred_data.get("span", "")

#         f1 = compute_span_f1(pred_text, gold_text)
#         em = compute_exact_match(pred_text, gold_text)

#         f1_scores[clause_type] = f1
#         em_scores[clause_type] = em

#     macro_f1 = round(sum(f1_scores.values()) / len(f1_scores), 4) if f1_scores else 0.0
#     em_avg = round(sum(em_scores.values()) / len(em_scores), 4) if em_scores else 0.0

#     return {
#         "macro_f1": macro_f1,
#         "exact_match": em_avg,
#         "per_clause_f1": f1_scores,
#     }

# # ROUGE-L (Stage 4)
# def compute_rouge_l(summary: str, gold_text: str) -> float:
#     try:
#         scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
#         score = scorer.score(gold_text, summary)
#         return round(score["rougeL"].fmeasure, 4)
#     except:
#         return 0.0


# # BERTScore (Stage 4 semantic)

# def compute_bertscore(summary: str, gold_text: str) -> float:
#     try:
#         P, R, F1 = bertscore_fn([summary], [gold_text], lang="en")
#         return round(F1.mean().item(), 4)
#     except:
#         return 0.0

# # FINAL METRICS WRAPPER

# def compute_all_metrics(pred_spans, gold_spans, summary=""):
#     metrics = compute_macro_f1(pred_spans, gold_spans)

#     if gold_spans and summary:
#         gold_text = " ".join(gold_spans.values())

#         metrics["rougeL"] = compute_rouge_l(summary, gold_text)
#         metrics["bert_score"] = compute_bertscore(summary, gold_text)

#     metrics["n_clauses_found"] = len(pred_spans)
#     metrics["n_gold_clauses"] = len(gold_spans)

#     return metrics

# Pipeline

def run_pipeline(pdf_bytes: bytes, filename: str):
    text = stage1_ocr(pdf_bytes)
    return run_pipeline_on_text(text, filename)


def run_pipeline_on_text(text: str, filename: str, gold_spans: Optional[dict] = None):
    preload_models()

    if _semantic_model is None:
        logger.warning("Semantic model not loaded, classifier fallback mode")

    if not text or len(text.strip()) < 50:
        return {"error": "Text too short"}
    
    if not is_legal_text_v3(text):
        return {"error": "Uploaded content does not appear to be a legal document"}

    text = clean_text(text)
    clauses = stage2_clause_segmentation(text)
    clauses = normalize_scores(clauses)
    entities = stage3_ner(clauses)
    summary = stage4_summarize(clauses, entities)
    # metrics = {}
    # if gold_spans:
    #     metrics = compute_all_metrics(
    #         clauses,                    
    #         gold_spans,
    #         summary=summary["summary"] 
    #     )
    clauses_clean = {}
    for clause_type in set(list(clauses.keys()) + list(entities.keys())):
        span_data = clauses.get(clause_type, {})
        clauses_clean[clause_type] = {
            "span": span_data.get("span", ""),
            "score": span_data.get("score", 0.0),
            "entities": entities.get(clause_type, [])
        }

    return {
        "id": str(uuid.uuid4())[:8],
        "filename": filename,
        "timestamp": datetime.utcnow().isoformat(),
        "clauses": clauses_clean,
        "summary": summary["summary"],
        "prompt": summary["prompt"],
        "pipeline": "v3"
    }