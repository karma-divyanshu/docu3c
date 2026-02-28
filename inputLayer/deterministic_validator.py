"""
deterministic_validator.py
===========================
DETERMINISTIC VALIDATION ENGINE
Validates LLM-extracted trademark fields using hard rules.
These rules NEVER rely on AI — they are mathematically certain.
"""

import re
from datetime import datetime, date
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


# =========================================================
# RESULT MODELS
# =========================================================

@dataclass
class ValidationResult:
    field: str
    rule: str
    passed: bool
    severity: str          # ERROR | WARNING | INFO
    message: str
    extracted_value: Any
    confidence_penalty: float = 0.0  # 0.0 to 1.0, how much this reduces field confidence


@dataclass
class ValidationReport:
    results: List[ValidationResult] = field(default_factory=list)
    needs_human_review: bool = False
    review_reasons: List[str] = field(default_factory=list)
    overall_confidence: float = 1.0

    @property
    def errors(self):
        return [r for r in self.results if r.severity == "ERROR" and not r.passed]

    @property
    def warnings(self):
        return [r for r in self.results if r.severity == "WARNING" and not r.passed]

    @property
    def passed_count(self):
        return sum(1 for r in self.results if r.passed)

    @property
    def total_count(self):
        return len(self.results)


# =========================================================
# CONSTANTS — USPTO RULES
# =========================================================

VALID_CLASS_RANGE = range(1, 46)         # Nice Classification: 1 to 45

VALID_FILING_BASES = {
    "1a",    # Use in commerce
    "1b",    # Intent to use
    "44d",   # Foreign application
    "44e",   # Foreign registration
    "66a",   # Madrid Protocol
}

VALID_MARK_TYPES = {
    "standard_character",
    "design_plus_words",
    "design_only",
    "sound",
    "color",
    "trade_dress",
}

VALID_FILING_TYPES = {
    "TEAS_PLUS",
    "TEAS_STANDARD",
    "TEAS_RF",
}

# USPTO TEAS Plus filing fee per class (2024)
TEAS_PLUS_FEE_PER_CLASS = 250.0
TEAS_STANDARD_FEE_PER_CLASS = 350.0

# Date format patterns USPTO uses
DATE_FORMATS = [
    "%m/%d/%Y",    # 01/15/2024
    "%B %d, %Y",   # January 15, 2024
    "%Y-%m-%d",    # 2024-01-15
    "%m-%d-%Y",    # 01-15-2024
]

# Max reasonable description length for goods/services
MAX_IDENTIFICATION_LENGTH = 3000
MIN_IDENTIFICATION_LENGTH = 10

# Serial number: exactly 8 digits
SERIAL_NUMBER_PATTERN = re.compile(r"^\d{8}$")

# URL pattern (basic)
URL_PATTERN = re.compile(
    r"^(https?://)?"
    r"[\w\-]+(\.[\w\-]+)+"
    r"(/[\w\-._~:/?#\[\]@!$&'()*+,;=%]*)?$",
    re.IGNORECASE
)

# Name sanity — must have at least 2 chars, no pure numbers
NAME_PATTERN = re.compile(r"^(?!\d+$).{2,}", re.UNICODE)


# =========================================================
# DATE PARSING HELPER
# =========================================================

def parse_date_flexible(date_str: str) -> Optional[date]:
    """Try all known USPTO date formats. Return None if unparseable."""
    if not date_str or not date_str.strip():
        return None
    cleaned = date_str.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


# =========================================================
# INDIVIDUAL RULE FUNCTIONS
# Each returns a ValidationResult
# =========================================================

def rule_serial_number_format(serial: str) -> ValidationResult:
    passed = bool(SERIAL_NUMBER_PATTERN.match(str(serial).strip()))
    return ValidationResult(
        field="application_serial",
        rule="serial_number_format",
        passed=passed,
        severity="ERROR",
        message="Serial number must be exactly 8 digits." if not passed
                else "Serial number format valid.",
        extracted_value=serial,
        confidence_penalty=0.9 if not passed else 0.0,
    )


def rule_filing_date_valid(filing_date: str) -> ValidationResult:
    parsed = parse_date_flexible(filing_date)
    today = date.today()

    if parsed is None:
        return ValidationResult(
            field="filing_date",
            rule="filing_date_parseable",
            passed=False,
            severity="ERROR",
            message=f"Filing date '{filing_date}' cannot be parsed into a valid date.",
            extracted_value=filing_date,
            confidence_penalty=0.8,
        )

    if parsed > today:
        return ValidationResult(
            field="filing_date",
            rule="filing_date_not_future",
            passed=False,
            severity="ERROR",
            message=f"Filing date {filing_date} is in the future — impossible.",
            extracted_value=filing_date,
            confidence_penalty=1.0,
        )

    if parsed.year < 1870:
        return ValidationResult(
            field="filing_date",
            rule="filing_date_reasonable_year",
            passed=False,
            severity="ERROR",
            message=f"Filing date year {parsed.year} predates USPTO existence.",
            extracted_value=filing_date,
            confidence_penalty=1.0,
        )

    return ValidationResult(
        field="filing_date",
        rule="filing_date_valid",
        passed=True,
        severity="INFO",
        message=f"Filing date {filing_date} is valid.",
        extracted_value=filing_date,
    )


def rule_class_number_valid(class_number: Any, index: int) -> ValidationResult:
    try:
        cn = int(class_number)
        passed = cn in VALID_CLASS_RANGE
    except (TypeError, ValueError):
        cn = class_number
        passed = False

    return ValidationResult(
        field=f"classes[{index}].class_number",
        rule="class_number_in_nice_range",
        passed=passed,
        severity="ERROR",
        message=f"Class {cn} is not a valid Nice Classification (1–45)." if not passed
                else f"Class {cn} is valid.",
        extracted_value=class_number,
        confidence_penalty=1.0 if not passed else 0.0,
    )


def rule_filing_basis_valid(filing_basis: str, index: int) -> ValidationResult:
    normalized = str(filing_basis).strip().lower().replace("section ", "")
    passed = normalized in VALID_FILING_BASES

    return ValidationResult(
        field=f"classes[{index}].filing_basis",
        rule="filing_basis_recognized",
        passed=passed,
        severity="ERROR",
        message=f"Filing basis '{filing_basis}' is not recognized. "
                f"Valid values: {', '.join(sorted(VALID_FILING_BASES))}." if not passed
                else f"Filing basis '{filing_basis}' recognized.",
        extracted_value=filing_basis,
        confidence_penalty=0.7 if not passed else 0.0,
    )


def rule_identification_length(identification: str, index: int) -> List[ValidationResult]:
    results = []
    length = len(str(identification).strip())

    if length < MIN_IDENTIFICATION_LENGTH:
        results.append(ValidationResult(
            field=f"classes[{index}].identification",
            rule="identification_min_length",
            passed=False,
            severity="ERROR",
            message=f"Identification text is too short ({length} chars). "
                    f"Minimum is {MIN_IDENTIFICATION_LENGTH}.",
            extracted_value=identification,
            confidence_penalty=0.8,
        ))
    elif length > MAX_IDENTIFICATION_LENGTH:
        results.append(ValidationResult(
            field=f"classes[{index}].identification",
            rule="identification_max_length",
            passed=False,
            severity="WARNING",
            message=f"Identification text is unusually long ({length} chars). "
                    f"May indicate extraction error.",
            extracted_value=f"{identification[:80]}...",
            confidence_penalty=0.3,
        ))
    else:
        results.append(ValidationResult(
            field=f"classes[{index}].identification",
            rule="identification_length_ok",
            passed=True,
            severity="INFO",
            message=f"Identification length ({length} chars) is acceptable.",
            extracted_value=f"{identification[:80]}...",
        ))

    return results


def rule_identification_no_class_placeholder(identification: str, index: int) -> ValidationResult:
    """LLMs sometimes put class number or 'N/A' as identification."""
    suspicious_patterns = [
        r"^\d{1,2}$",                      # just a number like "9" or "42"
        r"^(n/?a|none|null|empty|unknown)$",
        r"^class\s*\d+$",
        r"^\[.*\]$",                        # [field not found]
        r"^<.*>$",                          # <not extracted>
    ]
    text = str(identification).strip().lower()
    flagged = any(re.match(p, text, re.I) for p in suspicious_patterns)

    return ValidationResult(
        field=f"classes[{index}].identification",
        rule="identification_not_placeholder",
        passed=not flagged,
        severity="ERROR",
        message=f"Identification '{identification}' looks like a placeholder or LLM hallucination."
                if flagged else "Identification text appears substantive.",
        extracted_value=identification,
        confidence_penalty=1.0 if flagged else 0.0,
    )


def rule_use_based_dates_present(
    filing_basis: str,
    date_of_first_use: str,
    date_of_first_use_commerce: str,
    index: int
) -> List[ValidationResult]:
    """
    If filing basis is 1(a) (use in commerce), both dates MUST be present.
    If filing basis is 1(b) (intent to use), dates must NOT be present.
    """
    results = []
    basis = str(filing_basis).strip().lower().replace("section ", "")

    if basis == "1a":
        # Both dates required
        if not date_of_first_use or not str(date_of_first_use).strip():
            results.append(ValidationResult(
                field=f"classes[{index}].date_of_first_use",
                rule="use_basis_requires_first_use_date",
                passed=False,
                severity="ERROR",
                message="Filing basis 1(a) requires a date of first use anywhere.",
                extracted_value=date_of_first_use,
                confidence_penalty=0.6,
            ))

        if not date_of_first_use_commerce or not str(date_of_first_use_commerce).strip():
            results.append(ValidationResult(
                field=f"classes[{index}].date_of_first_use_commerce",
                rule="use_basis_requires_commerce_date",
                passed=False,
                severity="ERROR",
                message="Filing basis 1(a) requires a date of first use in commerce.",
                extracted_value=date_of_first_use_commerce,
                confidence_penalty=0.6,
            ))

    elif basis == "1b":
        # Dates should NOT be present for intent-to-use
        if date_of_first_use and str(date_of_first_use).strip():
            results.append(ValidationResult(
                field=f"classes[{index}].date_of_first_use",
                rule="intent_to_use_no_dates_expected",
                passed=False,
                severity="WARNING",
                message="Filing basis 1(b) (intent to use) should not have a first use date. "
                        "Possible extraction error.",
                extracted_value=date_of_first_use,
                confidence_penalty=0.4,
            ))

    return results


def rule_date_chronological_order(
    date_of_first_use: str,
    date_of_first_use_commerce: str,
    index: int
) -> ValidationResult:
    """First use in commerce CANNOT be before first use anywhere."""
    d1 = parse_date_flexible(str(date_of_first_use))
    d2 = parse_date_flexible(str(date_of_first_use_commerce))

    if d1 is None or d2 is None:
        return ValidationResult(
            field=f"classes[{index}].dates",
            rule="dates_chronological_order",
            passed=True,   # can't check if not both present
            severity="INFO",
            message="Date chronology check skipped — one or both dates missing.",
            extracted_value={"first_use": date_of_first_use, "first_use_commerce": date_of_first_use_commerce},
        )

    passed = d2 >= d1
    return ValidationResult(
        field=f"classes[{index}].dates",
        rule="dates_chronological_order",
        passed=passed,
        severity="ERROR",
        message=f"Date of first use in commerce ({date_of_first_use_commerce}) "
                f"cannot be before date of first use ({date_of_first_use})."
                if not passed else "Date chronological order is valid.",
        extracted_value={"first_use": date_of_first_use, "first_use_commerce": date_of_first_use_commerce},
        confidence_penalty=0.9 if not passed else 0.0,
    )


def rule_first_use_not_future(date_str: str, field_name: str, index: int) -> ValidationResult:
    parsed = parse_date_flexible(str(date_str))
    today = date.today()

    if parsed is None:
        return ValidationResult(
            field=f"classes[{index}].{field_name}",
            rule="first_use_not_future",
            passed=True,
            severity="INFO",
            message="Date not present — future check skipped.",
            extracted_value=date_str,
        )

    passed = parsed <= today
    return ValidationResult(
        field=f"classes[{index}].{field_name}",
        rule="first_use_not_future",
        passed=passed,
        severity="ERROR",
        message=f"Date {date_str} is in the future — cannot be a use date.",
        extracted_value=date_str,
        confidence_penalty=1.0 if not passed else 0.0,
    )


def rule_fee_amount_reasonable(
    total_fee: float,
    filing_type: str,
    class_count: int
) -> ValidationResult:
    """Fee must align with class count and filing type."""
    if class_count == 0:
        return ValidationResult(
            field="total_fee_paid",
            rule="fee_amount_reasonable",
            passed=False,
            severity="ERROR",
            message="Class count is 0 — cannot validate fee.",
            extracted_value=total_fee,
            confidence_penalty=0.5,
        )

    if filing_type == "TEAS_PLUS":
        expected = TEAS_PLUS_FEE_PER_CLASS * class_count
    elif filing_type in ("TEAS_STANDARD", "TEAS_RF"):
        expected = TEAS_STANDARD_FEE_PER_CLASS * class_count
    else:
        expected = None

    if expected is None:
        return ValidationResult(
            field="total_fee_paid",
            rule="fee_amount_reasonable",
            passed=True,
            severity="INFO",
            message=f"Unknown filing type '{filing_type}' — fee check skipped.",
            extracted_value=total_fee,
        )

    # Allow 10% tolerance for fee changes or multi-class discounts
    tolerance = 0.10
    lower = expected * (1 - tolerance)
    upper = expected * (1 + tolerance)
    passed = lower <= float(total_fee) <= upper

    return ValidationResult(
        field="total_fee_paid",
        rule="fee_amount_reasonable",
        passed=passed,
        severity="WARNING",
        message=f"Fee ${total_fee:.2f} does not match expected ${expected:.2f} "
                f"for {class_count} class(es) at {filing_type}." if not passed
                else f"Fee ${total_fee:.2f} matches expected ${expected:.2f}.",
        extracted_value=total_fee,
        confidence_penalty=0.5 if not passed else 0.0,
    )


def rule_fee_count_matches_classes(fees_paid_count: int, class_count: int) -> ValidationResult:
    passed = fees_paid_count == class_count
    return ValidationResult(
        field="fees_paid_count",
        rule="fee_count_matches_class_count",
        passed=passed,
        severity="ERROR",
        message=f"fees_paid_count ({fees_paid_count}) does not match "
                f"number of classes ({class_count})." if not passed
                else "Fee count matches class count.",
        extracted_value={"fees_paid_count": fees_paid_count, "class_count": class_count},
        confidence_penalty=0.8 if not passed else 0.0,
    )


def rule_applicant_name_valid(name: str) -> ValidationResult:
    cleaned = str(name).strip()
    passed = bool(NAME_PATTERN.match(cleaned)) and cleaned.lower() not in {
        "n/a", "none", "null", "unknown", "applicant", "[not found]"
    }

    return ValidationResult(
        field="applicant_name",
        rule="applicant_name_valid",
        passed=passed,
        severity="ERROR",
        message=f"Applicant name '{name}' appears invalid or is a placeholder." if not passed
                else "Applicant name appears valid.",
        extracted_value=name,
        confidence_penalty=0.8 if not passed else 0.0,
    )


def rule_mark_text_present_if_word_mark(mark_text: str, mark_type: str) -> ValidationResult:
    word_mark_types = {"standard_character", "design_plus_words"}

    if mark_type not in word_mark_types:
        return ValidationResult(
            field="mark_text",
            rule="mark_text_present_if_word_mark",
            passed=True,
            severity="INFO",
            message=f"Mark type '{mark_type}' does not require literal text.",
            extracted_value=mark_text,
        )

    cleaned = str(mark_text).strip()
    passed = len(cleaned) >= 1 and cleaned.lower() not in {
        "n/a", "none", "null", "unknown", "[not found]", ""
    }

    return ValidationResult(
        field="mark_text",
        rule="mark_text_present_if_word_mark",
        passed=passed,
        severity="ERROR",
        message=f"Mark type is '{mark_type}' but mark text is missing or placeholder." if not passed
                else f"Mark text '{mark_text}' present for word mark.",
        extracted_value=mark_text,
        confidence_penalty=0.7 if not passed else 0.0,
    )


def rule_mark_type_recognized(mark_type: str) -> ValidationResult:
    passed = str(mark_type).strip().lower() in VALID_MARK_TYPES
    return ValidationResult(
        field="mark_type",
        rule="mark_type_recognized",
        passed=passed,
        severity="WARNING",
        message=f"Mark type '{mark_type}' is not in recognized set: "
                f"{', '.join(VALID_MARK_TYPES)}." if not passed
                else f"Mark type '{mark_type}' recognized.",
        extracted_value=mark_type,
        confidence_penalty=0.3 if not passed else 0.0,
    )


def rule_duplicate_classes(classes: List[Dict]) -> ValidationResult:
    """Two entries with the same class number — likely extraction error."""
    seen = []
    duplicates = []
    for cls in classes:
        cn = cls.get("class_number")
        if cn in seen:
            duplicates.append(cn)
        seen.append(cn)

    passed = len(duplicates) == 0
    return ValidationResult(
        field="classes",
        rule="no_duplicate_class_numbers",
        passed=passed,
        severity="ERROR",
        message=f"Duplicate class numbers found: {duplicates}. "
                f"Likely LLM extraction error." if not passed
                else "No duplicate class numbers.",
        extracted_value=duplicates,
        confidence_penalty=0.9 if not passed else 0.0,
    )


def rule_nice_edition_format(edition: str) -> ValidationResult:
    """Nice edition should be something like '12th', '11th', etc."""
    pattern = re.compile(r"^\d{1,2}(st|nd|rd|th)$", re.IGNORECASE)
    passed = bool(pattern.match(str(edition).strip()))
    return ValidationResult(
        field="nice_edition_claimed",
        rule="nice_edition_format",
        passed=passed,
        severity="WARNING",
        message=f"Nice edition '{edition}' format unexpected." if not passed
                else f"Nice edition '{edition}' format valid.",
        extracted_value=edition,
        confidence_penalty=0.1 if not passed else 0.0,
    )


# =========================================================
# CONFIDENCE SCORER
# =========================================================

def compute_confidence(results: List[ValidationResult]) -> float:
    """
    Compute overall field extraction confidence score (0.0 to 1.0).
    Each failed rule subtracts its confidence_penalty proportionally.
    """
    if not results:
        return 1.0

    total_penalty = sum(
        r.confidence_penalty
        for r in results
        if not r.passed and r.confidence_penalty > 0
    )
    # Cap at 1.0 — fully untrustworthy
    normalized = min(total_penalty / max(len(results), 1), 1.0)
    return round(1.0 - normalized, 3)


# =========================================================
# HUMAN REVIEW TRIGGER RULES
# =========================================================

HUMAN_REVIEW_TRIGGERS = [
    # If confidence is below this threshold — always flag
    ("low_confidence", lambda conf, _: conf < 0.70),

    # If ANY error-severity rule failed
    ("has_errors", lambda _, results: any(
        r.severity == "ERROR" and not r.passed for r in results
    )),

    # If fee doesn't match classes
    ("fee_mismatch", lambda _, results: any(
        r.rule == "fee_count_matches_class_count" and not r.passed for r in results
    )),

    # If serial number is bad
    ("bad_serial", lambda _, results: any(
        r.rule == "serial_number_format" and not r.passed for r in results
    )),

    # If any identification is a placeholder
    ("placeholder_identification", lambda _, results: any(
        r.rule == "identification_not_placeholder" and not r.passed for r in results
    )),
]


# =========================================================
# MASTER VALIDATOR
# =========================================================

def validate_extracted_data(data: Dict[str, Any]) -> ValidationReport:
    """
    Run all deterministic rules against LLM-extracted data.
    Returns a full ValidationReport with pass/fail for every rule.
    """
    all_results: List[ValidationResult] = []

    # --- TOP-LEVEL FIELDS ---
    all_results.append(rule_serial_number_format(data.get("application_serial", "")))
    all_results.append(rule_filing_date_valid(data.get("filing_date", "")))
    all_results.append(rule_applicant_name_valid(data.get("applicant_name", "")))
    all_results.append(rule_mark_text_present_if_word_mark(
        data.get("mark_text", ""),
        data.get("mark_type", "")
    ))
    all_results.append(rule_mark_type_recognized(data.get("mark_type", "")))
    all_results.append(rule_nice_edition_format(data.get("nice_edition_claimed", "12th")))

    classes = data.get("classes", [])
    all_results.append(rule_duplicate_classes(classes))
    all_results.append(rule_fee_count_matches_classes(
        data.get("fees_paid_count", 0),
        len(classes)
    ))
    all_results.append(rule_fee_amount_reasonable(
        data.get("total_fee_paid", 0.0),
        data.get("filing_type", "TEAS_PLUS"),
        len(classes)
    ))

    # --- PER-CLASS FIELDS ---
    for i, cls in enumerate(classes):
        all_results.append(rule_class_number_valid(cls.get("class_number"), i))
        all_results.append(rule_filing_basis_valid(cls.get("filing_basis", ""), i))

        identification = cls.get("identification", "")
        all_results.extend(rule_identification_length(identification, i))
        all_results.append(rule_identification_not_placeholder := rule_identification_no_class_placeholder(identification, i))

        # Date rules
        dfu = cls.get("date_of_first_use", "")
        dfuc = cls.get("date_of_first_use_commerce", "")
        basis = cls.get("filing_basis", "")

        all_results.extend(rule_use_based_dates_present(basis, dfu, dfuc, i))
        all_results.append(rule_date_chronological_order(dfu, dfuc, i))

        if dfu:
            all_results.append(rule_first_use_not_future(dfu, "date_of_first_use", i))
        if dfuc:
            all_results.append(rule_first_use_not_future(dfuc, "date_of_first_use_commerce", i))

    # --- CONFIDENCE & REVIEW ---
    confidence = compute_confidence(all_results)
    review_reasons = []

    for trigger_name, trigger_fn in HUMAN_REVIEW_TRIGGERS:
        if trigger_fn(confidence, all_results):
            review_reasons.append(trigger_name)

    report = ValidationReport(
        results=all_results,
        needs_human_review=len(review_reasons) > 0,
        review_reasons=review_reasons,
        overall_confidence=confidence,
    )

    return report


# =========================================================
# PRETTY PRINT HELPER
# =========================================================

def format_validation_report(report: ValidationReport) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("DETERMINISTIC VALIDATION REPORT")
    lines.append("=" * 70)
    lines.append(f"Overall Confidence : {report.overall_confidence:.1%}")
    lines.append(f"Human Review Needed: {'YES ⚠️' if report.needs_human_review else 'NO ✅'}")
    if report.review_reasons:
        lines.append(f"Review Reasons     : {', '.join(report.review_reasons)}")
    lines.append(f"Rules Passed       : {report.passed_count}/{report.total_count}")
    lines.append("")

    if report.errors:
        lines.append("─── ERRORS ──────────────────────────────────────────────────────────")
        for r in report.errors:
            lines.append(f"  ✗ [{r.field}] {r.message}")

    if report.warnings:
        lines.append("")
        lines.append("─── WARNINGS ────────────────────────────────────────────────────────")
        for r in report.warnings:
            lines.append(f"  ⚠ [{r.field}] {r.message}")

    passed = [r for r in report.results if r.passed]
    if passed:
        lines.append("")
        lines.append("─── PASSED ──────────────────────────────────────────────────────────")
        for r in passed:
            lines.append(f"  ✓ [{r.field}] {r.message}")

    lines.append("=" * 70)
    return "\n".join(lines)
