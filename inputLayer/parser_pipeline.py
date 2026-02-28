"""
parser_pipeline.py
===================
FULL EXTRACTION PIPELINE ORCHESTRATOR

Flow:
  PDF Upload
      │
      ▼
  Raw Text Extraction (pdfplumber)
      │
      ▼
  LLM Extraction (Groq llama-3.3-70b-versatile)
      │
      ▼
  Deterministic Validation (hard rules, zero AI)
      │
      ├── PASS (confidence ≥ 0.70, no errors) ──► Proceed to Assessment
      │
      └── FAIL ──► Enqueue for Human Review
                        │
                        └── Human approves/corrects/rejects
                                    │
                                    └── If approved ──► Proceed to Assessment
"""

import io
import logging
from typing import Dict, Any, Optional, Tuple

import pdfplumber

from llm_extractor import extract_with_llm
from deterministic_validator import validate_extracted_data, format_validation_report
from review_queue import enqueue_for_review, get_approved_data

logger = logging.getLogger(__name__)


# =========================================================
# STEP 1: PDF → RAW TEXT
# =========================================================

def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract raw text from PDF using pdfplumber.
    Works with both file paths and file-like objects (Streamlit uploads).
    """
    raw_text = ""

    with pdfplumber.open(uploaded_file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                raw_text += f"\n--- PAGE {page_num} ---\n"
                raw_text += text

    if not raw_text.strip():
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It may be a scanned/image-only PDF. "
            "Consider using OCR preprocessing."
        )

    return raw_text.strip()


# =========================================================
# STEP 2: LLM EXTRACTION + DETERMINISTIC VALIDATION
# =========================================================

def extract_and_validate(
    raw_text: str,
    groq_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run LLM extraction then immediately validate with deterministic rules.

    Returns a result dict:
    {
        "status": "ok" | "needs_review",
        "data": {...},                    # extracted fields
        "validation_report": ...,         # ValidationReport object
        "validation_text": "...",         # formatted text report
        "queue_id": "..." | None,         # set if sent to review queue
        "llm_metadata": {...},            # model, confidence, warnings
    }
    """

    # --- LLM EXTRACTION ---
    logger.info("Starting LLM extraction with Groq llama-3.3-70b-versatile...")
    llm_result = extract_with_llm(raw_text, api_key=groq_api_key)

    extracted_data = llm_result["data"]
    llm_metadata = {
        "model_used":        llm_result["model_used"],
        "extraction_method": llm_result["extraction_method"],
        "llm_confidence":    llm_result["llm_confidence"],
        "llm_warnings":      llm_result["llm_warnings"],
    }

    logger.info(f"LLM extraction complete. LLM self-reported confidence: {llm_result['llm_confidence']}")

    # --- DETERMINISTIC VALIDATION ---
    logger.info("Running deterministic validation rules...")
    validation_report = validate_extracted_data(extracted_data)
    validation_text   = format_validation_report(validation_report)

    logger.info(
        f"Validation complete. "
        f"Confidence: {validation_report.overall_confidence:.1%} | "
        f"Errors: {len(validation_report.errors)} | "
        f"Warnings: {len(validation_report.warnings)} | "
        f"Needs review: {validation_report.needs_human_review}"
    )

    # --- ROUTING DECISION ---
    if validation_report.needs_human_review:
        queue_id = enqueue_for_review(extracted_data, validation_report)
        logger.warning(
            f"Application queued for human review. "
            f"Queue ID: {queue_id} | Reasons: {validation_report.review_reasons}"
        )

        return {
            "status": "needs_review",
            "data": extracted_data,
            "validation_report": validation_report,
            "validation_text": validation_text,
            "queue_id": queue_id,
            "llm_metadata": llm_metadata,
        }

    return {
        "status": "ok",
        "data": extracted_data,
        "validation_report": validation_report,
        "validation_text": validation_text,
        "queue_id": None,
        "llm_metadata": llm_metadata,
    }


# =========================================================
# STEP 3: FULL PDF PIPELINE (entry point)
# =========================================================

def run_extraction_pipeline(
    uploaded_file,
    groq_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Complete end-to-end pipeline from PDF to validated structured data.

    Args:
        uploaded_file: File path string or file-like object (Streamlit UploadedFile)
        groq_api_key:  Groq API key (or reads from GROQ_API_KEY env var)

    Returns:
        Pipeline result dict (see extract_and_validate for schema)
    """
    try:
        raw_text = extract_text_from_pdf(uploaded_file)
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise RuntimeError(f"PDF text extraction failed: {e}")

    logger.info(f"Extracted {len(raw_text)} characters from PDF.")

    result = extract_and_validate(raw_text, groq_api_key=groq_api_key)
    result["raw_text_length"] = len(raw_text)

    return result


# =========================================================
# RESUME AFTER HUMAN REVIEW
# =========================================================

def get_data_after_review(queue_id: str) -> Optional[Dict[str, Any]]:
    """
    After a human has reviewed/corrected an item, retrieve the
    approved data to pass downstream to the assessment engine.

    Returns None if not yet reviewed or if rejected.
    """
    return get_approved_data(queue_id)
