"""
app.py
=======
TRADEMARK PDF ADAPTIVE PARSER — FULL UI
Integrates: LLM Extraction + Deterministic Validation + Human Review Queue
"""

import os
import json
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from parser_pipeline import run_extraction_pipeline, get_data_after_review
from deterministic_validator import format_validation_report
from review_queue import (
    get_pending_items,
    get_queue_stats,
    approve_item,
    reject_item,
    correct_and_approve_item,
    initialize_db,
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Trademark Adaptive Parser",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_db()

# =========================================================
# SIDEBAR — CONFIG & NAVIGATION
# =========================================================

st.sidebar.title("⚖️ Trademark Parser")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📄 Extract & Validate", "👁️ Review Queue", "📊 Queue Stats"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔑 API Configuration")

groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    value=os.getenv("GROQ_API_KEY", ""),
    type="password",
    help="Get a free key at https://console.groq.com — no credit card needed.",
)

st.sidebar.markdown("""
**Model:** `llama-3.3-70b-versatile`  
**Why Groq?**
- ✅ Free tier (14,400 req/day)
- ✅ Fastest inference (200+ tok/sec)
- ✅ Best free JSON extraction model
- ✅ No credit card required
""")

st.sidebar.markdown("---")
st.sidebar.caption("Validation: deterministic rules only — zero AI for validation decisions.")


# =========================================================
# HELPERS
# =========================================================

def confidence_color(conf: float) -> str:
    if conf >= 0.85:
        return "🟢"
    if conf >= 0.70:
        return "🟡"
    return "🔴"


def severity_badge(severity: str) -> str:
    return {"ERROR": "🔴", "WARNING": "🟡", "INFO": "🟢", "OK": "✅"}.get(severity, "⚪")


# =========================================================
# PAGE 1: EXTRACT & VALIDATE
# =========================================================

if page == "📄 Extract & Validate":

    st.title("TARA AI For Trademark ")

    st.info("""
    **How this works:**
    1. Upload your USPTO trademark application PDF
    2. Groq LLM (`llama-3.3-70b-versatile`) extracts all fields
    3. Deterministic rules validate every field mathematically
    4. High-confidence results proceed to assessment
    5. Low-confidence results go to the Human Review Queue
    """)

    uploaded_file = st.file_uploader(
        "Upload Trademark Application PDF",
        type=["pdf"],
        help="Supports TEAS Plus, TEAS Standard, TEAS RF applications."
    )

    if uploaded_file:
        if not groq_api_key:
            st.error("⚠️ Please enter your Groq API key in the sidebar. Get one free at https://console.groq.com")
            st.stop()

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("🚀 Extract & Validate", type="primary", use_container_width=True):
                with st.spinner("Extracting with LLM + running deterministic validation..."):
                    try:
                        result = run_extraction_pipeline(
                            uploaded_file,
                            groq_api_key=groq_api_key,
                        )
                        st.session_state["pipeline_result"] = result
                    except Exception as e:
                        st.error(f"Pipeline failed: {e}")
                        st.stop()

    # ─── RESULTS ─────────────────────────────────────────

    if "pipeline_result" in st.session_state:
        result = st.session_state["pipeline_result"]
        report = result["validation_report"]
        data   = result["data"]

        st.markdown("---")

        # Status banner
        if result["status"] == "ok":
            st.success(f"✅ **Extraction validated** — Confidence: {report.overall_confidence:.1%} — Proceeding to assessment")
        else:
            st.warning(
                f"⚠️ **Sent to Human Review Queue** — "
                f"Confidence: {report.overall_confidence:.1%} — "
                f"Queue ID: `{result['queue_id']}` — "
                f"Reasons: {', '.join(result['validation_report'].review_reasons)}"
            )

        # Metrics row
        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        mcol1.metric("Confidence", f"{report.overall_confidence:.1%}")
        mcol2.metric("Rules Passed", f"{report.passed_count}/{report.total_count}")
        mcol3.metric("Errors", len(report.errors))
        mcol4.metric("Warnings", len(report.warnings))
        mcol5.metric("Classes Found", len(data.get("classes", [])))

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🧾 Extracted Data", "🔍 Validation Detail", "🤖 LLM Metadata", "📋 Raw JSON"
        ])

        with tab1:
            st.subheader("Extracted Fields")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Applicant Name:** {data.get('applicant_name', '—')}")
                st.markdown(f"**Mark Text:** {data.get('mark_text', '—')}")
                st.markdown(f"**Mark Type:** {data.get('mark_type', '—')}")
                st.markdown(f"**Filing Date:** {data.get('filing_date', '—')}")
                st.markdown(f"**Serial Number:** {data.get('application_serial', '—')}")
            with col2:
                st.markdown(f"**Filing Type:** {data.get('filing_type', '—')}")
                st.markdown(f"**Total Fee Paid:** ${data.get('total_fee_paid', 0):.2f}")
                st.markdown(f"**Fees Paid Count:** {data.get('fees_paid_count', 0)}")
                st.markdown(f"**Nice Edition:** {data.get('nice_edition_claimed', '—')}")

            st.subheader("Classes")
            for i, cls in enumerate(data.get("classes", [])):
                with st.expander(f"Class {cls.get('class_number', '?')} — {cls.get('filing_basis', '')}"):
                    st.markdown(f"**Identification:** {cls.get('identification', '—')}")
                    st.markdown(f"**Filing Basis:** {cls.get('filing_basis', '—')}")
                    st.markdown(f"**First Use:** {cls.get('date_of_first_use', '—')}")
                    st.markdown(f"**First Use Commerce:** {cls.get('date_of_first_use_commerce', '—')}")
                    st.markdown(f"**Specimen Type:** {cls.get('specimen_type', '—')}")
                    st.markdown(f"**Specimen Description:** {cls.get('specimen_description', '—')}")

        with tab2:
            st.subheader("Deterministic Validation Results")
            st.text(format_validation_report(report))

            if report.errors:
                st.subheader("❌ Errors")
                for r in report.errors:
                    st.error(f"**{r.field}** [{r.rule}]\n{r.message}")

            if report.warnings:
                st.subheader("⚠️ Warnings")
                for r in report.warnings:
                    st.warning(f"**{r.field}** [{r.rule}]\n{r.message}")

        with tab3:
            meta = result.get("llm_metadata", {})
            st.markdown(f"**Model Used:** `{meta.get('model_used', '—')}`")
            st.markdown(f"**Method:** `{meta.get('extraction_method', '—')}`")

            st.subheader("LLM Self-Reported Confidence")
            llm_conf = meta.get("llm_confidence", {})
            for field_name, conf_val in llm_conf.items():
                icon = confidence_color(float(conf_val or 0))
                st.markdown(f"{icon} **{field_name}:** {float(conf_val or 0):.1%}")

            if meta.get("llm_warnings"):
                st.subheader("LLM Extraction Warnings")
                for w in meta["llm_warnings"]:
                    st.warning(w)

        with tab4:
            st.json(data)

        # ── RUN ASSESSMENT (only if clean) ──────────────
        # if result["status"] == "ok":
        #     st.markdown("---")
        #     if st.button("⚖️ Run Pillar 1 Assessment", type="primary"):
        #         try:
        #             from main import assess_trademark_application
        #             assessment = assess_trademark_application(data)

        #             st.subheader("Pillar 1 Report (§1401 Classification)")
        #             st.text_area("Report", assessment["report"], height=500)
        #             st.subheader("Summary")
        #             st.json(assessment["summary"])
        #         except ImportError:
        #             st.error("main.py / assessment modules not found. Ensure they are in the same directory.")
        #         except Exception as e:
        #             st.error(f"Assessment error: {e}")
        if result["status"] == "ok":
            st.markdown("---")
            if st.button("⚖️ Run Full Examination (Pillar 1–3)", type="primary"):
                try:
                    from run_pipeline import run_full_pipeline

                    state = run_full_pipeline(data)

                    st.subheader("Pipeline Structural Summary")

                    st.write("Structurally Clean:", state.is_structurally_clean())
                    st.write("Partial Refusal Classes:", state.get_partial_refusal_classes())
                    st.write("Division Candidates:", state.get_division_candidates())

                    st.subheader("Pillar 1 Summary")
                    st.json(state.pillar1_output.get("summary", {}))

                    st.subheader("Pillar 3 Errors")
                    st.write(state.pillar3_output.total_errors)

                except ImportError:
                    st.error("run_pipeline.py not found in project root.")
                except Exception as e:
                    st.error(f"Pipeline error: {e}")

# =========================================================
# PAGE 2: REVIEW QUEUE
# =========================================================

elif page == "👁️ Review Queue":

    st.title("👁️ Human Review Queue")
    st.caption("Applications flagged by deterministic validation for human inspection.")

    pending = get_pending_items()

    if not pending:
        st.success("✅ No items pending review.")
    else:
        st.info(f"**{len(pending)} application(s)** awaiting review.")

        for item in pending:
            with st.expander(
                f"🔴 [{item.queue_id[:8]}...] "
                f"Serial: {item.application_serial or 'unknown'} | "
                f"Applicant: {item.applicant_name or 'unknown'} | "
                f"Confidence: {item.overall_confidence:.1%} | "
                f"Submitted: {item.submitted_at[:19]}"
            ):
                st.markdown(f"**Queue ID:** `{item.queue_id}`")
                st.markdown(f"**Review Reasons:** {', '.join(item.review_reasons)}")

                # Validation errors
                if item.validation_errors:
                    st.subheader("Validation Errors")
                    for err in item.validation_errors:
                        st.error(f"**{err['field']}** [{err['rule']}]: {err['message']}")

                # Validation warnings
                if item.validation_warnings:
                    st.subheader("Validation Warnings")
                    for warn in item.validation_warnings:
                        st.warning(f"**{warn['field']}** [{warn['rule']}]: {warn['message']}")

                # Original extracted data
                st.subheader("LLM Extracted Data")
                st.json(item.original_data)

                # Actions
                st.subheader("Review Actions")
                acol1, acol2, acol3 = st.columns(3)

                reviewer = st.text_input(
                    "Your name/ID",
                    value="reviewer",
                    key=f"reviewer_{item.queue_id}",
                )
                notes = st.text_area(
                    "Notes",
                    key=f"notes_{item.queue_id}",
                    placeholder="Add review notes here..."
                )

                with acol1:
                    if st.button(
                        "✅ Approve as-is",
                        key=f"approve_{item.queue_id}",
                        use_container_width=True,
                    ):
                        approve_item(item.queue_id, reviewer=reviewer, notes=notes)
                        st.success("Approved! Application can proceed to assessment.")
                        st.rerun()

                with acol2:
                    if st.button(
                        "❌ Reject",
                        key=f"reject_{item.queue_id}",
                        use_container_width=True,
                    ):
                        reject_item(item.queue_id, reviewer=reviewer, notes=notes)
                        st.error("Rejected and removed from pipeline.")
                        st.rerun()

                with acol3:
                    st.markdown("**Correct & Approve**")
                    corrected_json = st.text_area(
                        "Paste corrected JSON",
                        key=f"corrected_{item.queue_id}",
                        placeholder='{"applicant_name": "...", ...}',
                        height=150,
                    )
                    if st.button(
                        "💾 Save Correction & Approve",
                        key=f"correct_{item.queue_id}",
                        use_container_width=True,
                    ):
                        try:
                            corrected = json.loads(corrected_json)
                            correct_and_approve_item(
                                item.queue_id,
                                corrected_data=corrected,
                                reviewer=reviewer,
                                notes=notes,
                            )
                            st.success("Corrected and approved!")
                            st.rerun()
                        except json.JSONDecodeError:
                            st.error("Invalid JSON in correction field.")

                # ── Run assessment on approved items ──
                if item.status in ("approved", "corrected"):
                    if st.button(
                        "⚖️ Run Assessment on This Item",
                        key=f"assess_{item.queue_id}",
                    ):
                        approved_data = get_data_after_review(item.queue_id)
                        if approved_data:
                            try:
                                from main import assess_trademark_application
                                assessment = assess_trademark_application(approved_data)
                                st.subheader("Assessment Report")
                                st.text_area("Report", assessment["report"], height=400)
                                st.json(assessment["summary"])
                            except ImportError:
                                st.error("main.py not found.")
                        else:
                            st.warning("Item not yet approved or data unavailable.")


# =========================================================
# PAGE 3: QUEUE STATS
# =========================================================

elif page == "📊 Queue Stats":

    st.title("📊 Review Queue Statistics")

    stats = get_queue_stats()

    if not stats:
        st.info("No applications processed yet.")
    else:
        total = sum(stats.values())

        scol1, scol2, scol3, scol4, scol5 = st.columns(5)
        scol1.metric("Total", total)
        scol2.metric("⏳ Pending",   stats.get("pending", 0))
        scol3.metric("✅ Approved",  stats.get("approved", 0))
        scol4.metric("💾 Corrected", stats.get("corrected", 0))
        scol5.metric("❌ Rejected",  stats.get("rejected", 0))

        st.markdown("---")

        all_items = get_pending_items()
        if all_items:
            st.subheader("Pending Items")
            table_data = [
                {
                    "Queue ID":    item.queue_id[:12] + "...",
                    "Serial":      item.application_serial or "—",
                    "Applicant":   (item.applicant_name or "—")[:30],
                    "Confidence":  f"{item.overall_confidence:.1%}",
                    "Reasons":     ", ".join(item.review_reasons),
                    "Submitted":   item.submitted_at[:19],
                }
                for item in all_items
            ]
            st.dataframe(table_data, use_container_width=True)

        st.markdown("---")
        st.subheader("Validation Rule Reference")
        st.markdown("""
        | Rule | Type | What it checks |
        |------|------|----------------|
        | `serial_number_format` | ERROR | Must be exactly 8 digits |
        | `filing_date_parseable` | ERROR | Must be a recognizable date format |
        | `filing_date_not_future` | ERROR | Cannot be a future date |
        | `class_number_in_nice_range` | ERROR | Must be integer 1–45 |
        | `filing_basis_recognized` | ERROR | Must be 1a / 1b / 44d / 44e / 66a |
        | `identification_min_length` | ERROR | Min 10 chars |
        | `identification_not_placeholder` | ERROR | Not N/A, null, [field], etc. |
        | `use_basis_requires_first_use_date` | ERROR | 1(a) must have use dates |
        | `dates_chronological_order` | ERROR | Commerce date ≥ first use date |
        | `first_use_not_future` | ERROR | Use dates cannot be in future |
        | `fee_count_matches_class_count` | ERROR | fees_paid_count == len(classes) |
        | `applicant_name_valid` | ERROR | Not empty or placeholder |
        | `mark_text_present_if_word_mark` | ERROR | Word marks need literal text |
        | `fee_amount_reasonable` | WARNING | Fee within 10% of expected |
        | `identification_max_length` | WARNING | Max 3000 chars |
        | `intent_to_use_no_dates_expected` | WARNING | 1(b) should not have use dates |
        | `mark_type_recognized` | WARNING | Must be known mark type |
        | `nice_edition_format` | WARNING | Format like "12th" |
        | `no_duplicate_class_numbers` | ERROR | No two classes with same number |
        """)
