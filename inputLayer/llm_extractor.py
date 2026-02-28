"""
llm_extractor.py
=================
LLM-BASED FIELD EXTRACTION
Uses Groq's free llama-3.3-70b-versatile model to extract trademark
fields from raw PDF text. Outputs a structured JSON dict that matches
the downstream pipeline schema.

WHY GROQ + LLAMA-3.3-70b-versatile?
  - 100% free tier (no credit card required for basic use)
  - Fastest inference available (200+ tokens/sec)
  - llama-3.3-70b handles structured JSON reliably
  - Groq free tier: 14,400 requests/day, 500,000 tokens/min
"""

import os
import json
import re
import time
import logging
from typing import Dict, Any, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# =========================================================
# CONFIG
# =========================================================

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"   # Best free model on Groq

MAX_RETRIES     = 3
RETRY_DELAY_SEC = 2.0
REQUEST_TIMEOUT = 30  # seconds


# =========================================================
# SYSTEM PROMPT — STRICT EXTRACTION INSTRUCTIONS
# =========================================================

SYSTEM_PROMPT = """You are a precision document parser for USPTO trademark applications.

Your job: extract structured fields from raw trademark application text.

RULES (follow exactly):
1. Extract ONLY what is present in the text. Do NOT invent, infer, or guess.
2. If a field is missing or unclear, use an empty string "".
3. For identification of goods/services: copy the text VERBATIM. Do not paraphrase.
4. For filing_basis: normalize to one of: "1a", "1b", "44d", "44e", "66a". 
   If you see "Section 1(a)" output "1a". If "Section 1(b)" output "1b". 
   If unclear use "".
5. For mark_type: use exactly one of:
   "standard_character", "design_plus_words", "design_only", "sound", "color", "trade_dress"
6. For dates: preserve the exact date string as it appears in the document.
7. Return ONLY valid JSON. No markdown. No explanation. No backticks.
8. For classes array: create one entry per International Class found.
9. Confidence: for each field, estimate 0.0-1.0 how confident you are in the extraction.

OUTPUT FORMAT (return exactly this structure):
{
  "applicant_name": "",
  "mark_text": "",
  "mark_type": "",
  "filing_date": "",
  "nice_edition_claimed": "12th",
  "application_serial": "",
  "filing_type": "TEAS_PLUS",
  "fees_paid_count": 0,
  "total_fee_paid": 0.0,
  "notes": "",
  "classes": [
    {
      "class_number": 0,
      "identification": "",
      "specimen_type": "",
      "specimen_description": "",
      "fee_paid": true,
      "filing_basis": "",
      "date_of_first_use": "",
      "date_of_first_use_commerce": ""
    }
  ],
  "_extraction_confidence": {
    "applicant_name": 0.0,
    "mark_text": 0.0,
    "mark_type": 0.0,
    "filing_date": 0.0,
    "application_serial": 0.0,
    "classes_overall": 0.0
  },
  "_extraction_warnings": []
}"""


USER_PROMPT_TEMPLATE = """Extract all trademark fields from the following USPTO application text.

TEXT:
---
{text}
---

Remember: Return ONLY valid JSON. No extra text."""


# =========================================================
# GROQ API CLIENT
# =========================================================

class GroqClient:

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. "
                "Set it in your .env file or as environment variable.\n"
                "Get a free key at: https://console.groq.com"
            )

    def complete(self, user_text: str) -> str:
        """Call Groq API with retry logic. Returns raw response string."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=user_text)},
            ],
            "temperature": 0.0,        # deterministic — we want extraction not creativity
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},  # force JSON output
        }

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(
                    GROQ_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )

                if response.status_code == 429:
                    # Rate limited — wait and retry
                    wait = RETRY_DELAY_SEC * attempt
                    logger.warning(f"Rate limited by Groq. Waiting {wait}s before retry {attempt}/{MAX_RETRIES}.")
                    time.sleep(wait)
                    continue

                if response.status_code == 401:
                    raise PermissionError("Invalid Groq API key. Check GROQ_API_KEY in .env")

                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]

            except requests.exceptions.Timeout:
                last_error = f"Request timed out (attempt {attempt}/{MAX_RETRIES})"
                logger.warning(last_error)
                time.sleep(RETRY_DELAY_SEC)

            except requests.exceptions.ConnectionError:
                last_error = f"Connection error to Groq API (attempt {attempt}/{MAX_RETRIES})"
                logger.warning(last_error)
                time.sleep(RETRY_DELAY_SEC)

            except Exception as e:
                last_error = str(e)
                logger.error(f"Groq API error: {e}")
                break

        raise RuntimeError(f"Groq API failed after {MAX_RETRIES} attempts. Last error: {last_error}")


# =========================================================
# JSON PARSER — SAFE
# =========================================================

def safe_parse_json(raw: str) -> Dict[str, Any]:
    """Parse LLM JSON response. Strip markdown fences if present."""
    # Remove markdown code blocks if LLM ignored the instruction
    cleaned = re.sub(r"```json\s*", "", raw)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Try to extract JSON object from within the text
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"LLM returned unparseable JSON: {e}\nRaw: {raw[:300]}")


# =========================================================
# POST-PROCESSING — COERCE TYPES
# =========================================================

def coerce_types(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Force correct Python types on extracted fields.
    LLMs sometimes return numbers as strings, etc.
    """
    def to_int(v, default=0):
        try:
            return int(str(v).strip())
        except (ValueError, TypeError):
            return default

    def to_float(v, default=0.0):
        try:
            return float(str(v).replace("$", "").replace(",", "").strip())
        except (ValueError, TypeError):
            return default

    def to_str(v):
        if v is None:
            return ""
        return str(v).strip()

    data["applicant_name"]     = to_str(data.get("applicant_name"))
    data["mark_text"]          = to_str(data.get("mark_text"))
    data["mark_type"]          = to_str(data.get("mark_type")).lower()
    data["filing_date"]        = to_str(data.get("filing_date"))
    data["application_serial"] = to_str(data.get("application_serial"))
    data["filing_type"]        = to_str(data.get("filing_type")) or "TEAS_PLUS"
    data["nice_edition_claimed"] = to_str(data.get("nice_edition_claimed")) or "12th"
    data["fees_paid_count"]    = to_int(data.get("fees_paid_count"))
    data["total_fee_paid"]     = to_float(data.get("total_fee_paid"))
    data["notes"]              = to_str(data.get("notes"))

    coerced_classes = []
    for cls in data.get("classes", []):
        coerced_classes.append({
            "class_number":            to_int(cls.get("class_number")),
            "identification":          to_str(cls.get("identification")),
            "specimen_type":           to_str(cls.get("specimen_type")),
            "specimen_description":    to_str(cls.get("specimen_description")),
            "fee_paid":                bool(cls.get("fee_paid", True)),
            "filing_basis":            to_str(cls.get("filing_basis")).lower(),
            "date_of_first_use":       to_str(cls.get("date_of_first_use")),
            "date_of_first_use_commerce": to_str(cls.get("date_of_first_use_commerce")),
        })
    data["classes"] = coerced_classes

    return data


# =========================================================
# MAIN EXTRACTION FUNCTION
# =========================================================

def extract_with_llm(raw_text: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Full pipeline:
      1. Send raw PDF text to Groq LLM
      2. Parse JSON response
      3. Coerce types
      4. Return structured dict + extraction metadata

    Returns:
        {
            "data": {...},                    # the extracted trademark fields
            "llm_confidence": {...},          # per-field confidence from LLM
            "llm_warnings": [...],            # any warnings LLM flagged
            "extraction_method": "llm_groq",
            "model_used": "llama-3.3-70b-versatile"
        }
    """
    client = GroqClient(api_key=api_key)

    # Truncate text if extremely long (Groq context limit is 128k tokens)
    # 1 token ≈ 4 chars, keep 80k tokens ≈ 320k chars
    max_chars = 320_000
    if len(raw_text) > max_chars:
        logger.warning(f"Text truncated from {len(raw_text)} to {max_chars} chars.")
        raw_text = raw_text[:max_chars]

    raw_response = client.complete(raw_text)
    parsed = safe_parse_json(raw_response)

    # Extract metadata before coercion
    llm_confidence = parsed.pop("_extraction_confidence", {})
    llm_warnings   = parsed.pop("_extraction_warnings", [])

    coerced = coerce_types(parsed)

    return {
        "data": coerced,
        "llm_confidence": llm_confidence,
        "llm_warnings": llm_warnings,
        "extraction_method": "llm_groq",
        "model_used": GROQ_MODEL,
    }
