"""
review_queue.py
================
HUMAN REVIEW QUEUE
Stores and manages applications that failed deterministic validation
and require manual human inspection before proceeding to assessment.

Uses a local SQLite database (zero dependency, zero cost, portable).
"""

import sqlite3
import json
import os
import uuid
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from enum import Enum


# =========================================================
# CONFIG
# =========================================================

DB_PATH = os.getenv("REVIEW_QUEUE_DB", "trademark_review_queue.db")


# =========================================================
# MODELS
# =========================================================

class ReviewStatus(str, Enum):
    PENDING   = "pending"
    APPROVED  = "approved"   # Human reviewed — proceed
    REJECTED  = "rejected"   # Human reviewed — discard
    CORRECTED = "corrected"  # Human corrected data — use corrected version


@dataclass
class ReviewItem:
    queue_id: str
    submitted_at: str
    status: str
    application_serial: str
    applicant_name: str
    review_reasons: List[str]
    overall_confidence: float
    original_data: Dict[str, Any]          # what LLM extracted
    corrected_data: Optional[Dict[str, Any]]  # what human corrected to
    validation_errors: List[Dict]
    validation_warnings: List[Dict]
    reviewer_notes: str = ""
    reviewed_at: Optional[str] = None
    reviewed_by: str = ""


# =========================================================
# DATABASE SETUP
# =========================================================

def _get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db(db_path: str = DB_PATH):
    """Create tables if they don't exist."""
    conn = _get_connection(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS review_queue (
            queue_id            TEXT PRIMARY KEY,
            submitted_at        TEXT NOT NULL,
            status              TEXT NOT NULL DEFAULT 'pending',
            application_serial  TEXT,
            applicant_name      TEXT,
            review_reasons      TEXT,   -- JSON array
            overall_confidence  REAL,
            original_data       TEXT,   -- JSON
            corrected_data      TEXT,   -- JSON or NULL
            validation_errors   TEXT,   -- JSON array
            validation_warnings TEXT,   -- JSON array
            reviewer_notes      TEXT DEFAULT '',
            reviewed_at         TEXT,
            reviewed_by         TEXT DEFAULT ''
        )
    """)
    conn.commit()
    conn.close()


# =========================================================
# QUEUE OPERATIONS
# =========================================================

def enqueue_for_review(
    original_data: Dict[str, Any],
    validation_report,          # ValidationReport from deterministic_validator
    db_path: str = DB_PATH,
) -> str:
    """
    Add an application to the human review queue.
    Returns the queue_id for tracking.
    """
    initialize_db(db_path)

    queue_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    errors = [
        {
            "field": r.field,
            "rule": r.rule,
            "message": r.message,
            "value": str(r.extracted_value)[:200],
        }
        for r in validation_report.results
        if r.severity == "ERROR" and not r.passed
    ]

    warnings = [
        {
            "field": r.field,
            "rule": r.rule,
            "message": r.message,
            "value": str(r.extracted_value)[:200],
        }
        for r in validation_report.results
        if r.severity == "WARNING" and not r.passed
    ]

    conn = _get_connection(db_path)
    conn.execute("""
        INSERT INTO review_queue (
            queue_id, submitted_at, status, application_serial,
            applicant_name, review_reasons, overall_confidence,
            original_data, corrected_data, validation_errors, validation_warnings
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        queue_id,
        now,
        ReviewStatus.PENDING.value,
        str(original_data.get("application_serial", ""))[:50],
        str(original_data.get("applicant_name", ""))[:200],
        json.dumps(validation_report.review_reasons),
        validation_report.overall_confidence,
        json.dumps(original_data),
        None,
        json.dumps(errors),
        json.dumps(warnings),
    ))
    conn.commit()
    conn.close()

    return queue_id


def get_pending_items(db_path: str = DB_PATH) -> List[ReviewItem]:
    """Fetch all items awaiting review."""
    initialize_db(db_path)
    conn = _get_connection(db_path)
    rows = conn.execute("""
        SELECT * FROM review_queue
        WHERE status = 'pending'
        ORDER BY submitted_at ASC
    """).fetchall()
    conn.close()
    return [_row_to_item(row) for row in rows]


def get_item(queue_id: str, db_path: str = DB_PATH) -> Optional[ReviewItem]:
    """Fetch a single item by queue_id."""
    initialize_db(db_path)
    conn = _get_connection(db_path)
    row = conn.execute(
        "SELECT * FROM review_queue WHERE queue_id = ?", (queue_id,)
    ).fetchone()
    conn.close()
    return _row_to_item(row) if row else None


def approve_item(
    queue_id: str,
    reviewer: str = "human",
    notes: str = "",
    db_path: str = DB_PATH,
) -> bool:
    """Mark an item as approved — allows pipeline to proceed."""
    return _update_status(queue_id, ReviewStatus.APPROVED, reviewer, notes, None, db_path)


def reject_item(
    queue_id: str,
    reviewer: str = "human",
    notes: str = "",
    db_path: str = DB_PATH,
) -> bool:
    """Mark an item as rejected — discard from pipeline."""
    return _update_status(queue_id, ReviewStatus.REJECTED, reviewer, notes, None, db_path)


def correct_and_approve_item(
    queue_id: str,
    corrected_data: Dict[str, Any],
    reviewer: str = "human",
    notes: str = "",
    db_path: str = DB_PATH,
) -> bool:
    """
    Human provides corrected data and approves.
    The corrected_data will be used downstream instead of the LLM extraction.
    """
    return _update_status(
        queue_id, ReviewStatus.CORRECTED, reviewer, notes, corrected_data, db_path
    )


def get_approved_data(queue_id: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """
    Returns the data to use after review:
    - If CORRECTED: returns corrected_data
    - If APPROVED: returns original_data
    - Otherwise: returns None
    """
    item = get_item(queue_id, db_path)
    if not item:
        return None
    if item.status == ReviewStatus.CORRECTED and item.corrected_data:
        return item.corrected_data
    if item.status == ReviewStatus.APPROVED:
        return item.original_data
    return None


def get_queue_stats(db_path: str = DB_PATH) -> Dict[str, int]:
    """Summary counts for the dashboard."""
    initialize_db(db_path)
    conn = _get_connection(db_path)
    rows = conn.execute("""
        SELECT status, COUNT(*) as cnt
        FROM review_queue
        GROUP BY status
    """).fetchall()
    conn.close()
    return {row["status"]: row["cnt"] for row in rows}


# =========================================================
# INTERNAL HELPERS
# =========================================================

def _update_status(
    queue_id: str,
    status: ReviewStatus,
    reviewer: str,
    notes: str,
    corrected_data: Optional[Dict],
    db_path: str,
) -> bool:
    initialize_db(db_path)
    conn = _get_connection(db_path)
    cursor = conn.execute("""
        UPDATE review_queue
        SET status = ?, reviewed_at = ?, reviewed_by = ?,
            reviewer_notes = ?, corrected_data = ?
        WHERE queue_id = ?
    """, (
        status.value,
        datetime.utcnow().isoformat(),
        reviewer,
        notes,
        json.dumps(corrected_data) if corrected_data else None,
        queue_id,
    ))
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


def _row_to_item(row: sqlite3.Row) -> ReviewItem:
    return ReviewItem(
        queue_id            = row["queue_id"],
        submitted_at        = row["submitted_at"],
        status              = row["status"],
        application_serial  = row["application_serial"] or "",
        applicant_name      = row["applicant_name"] or "",
        review_reasons      = json.loads(row["review_reasons"] or "[]"),
        overall_confidence  = row["overall_confidence"] or 0.0,
        original_data       = json.loads(row["original_data"] or "{}"),
        corrected_data      = json.loads(row["corrected_data"]) if row["corrected_data"] else None,
        validation_errors   = json.loads(row["validation_errors"] or "[]"),
        validation_warnings = json.loads(row["validation_warnings"] or "[]"),
        reviewer_notes      = row["reviewer_notes"] or "",
        reviewed_at         = row["reviewed_at"],
        reviewed_by         = row["reviewed_by"] or "",
    )
