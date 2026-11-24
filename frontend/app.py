"""Streamlit client for the Indian Legal Judgment Summarizer API."""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
BACKEND_TIMEOUT_SECONDS = float(os.getenv("BACKEND_TIMEOUT_SECONDS", "1800"))
SUMMARY_ENDPOINT = f"{BACKEND_URL}/summarize"

SECTION_LABELS = {
    "case_overview": "Case overview",
    "facts_and_procedural_history": "Facts and procedural history",
    "issues_considered": "Legal issues",
    "petitioner's_arguments": "Petitioner's arguments",
    "respondent's_arguments": "Respondent's arguments",
    "court_reasoning": "Court's reasoning and findings",
    "final_decision_and_directions": "Final decision and directions",
    "important_precedents": "Important precedents",
}

STATUS_ICONS = {
    "supported_by_identified_evidence": "✅",
    "possible_contradiction": "🚩",
    "potential_role_attribution_problem": "⚠️",
    "insufficient_evidence": "❔",
    "requires_human_review": "👀",
}


class BackendResponseError(Exception):
    """A meaningful error returned by or while communicating with the API."""


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{size} B"
        value /= 1024
    return f"{value:.1f} GB"


def _response_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or response.reason or "No error details were returned."

    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        message = detail.get("message")
        formatted = json.dumps(detail, indent=2, ensure_ascii=False)
        return f"{message}\n\n{formatted}" if message else formatted
    if isinstance(detail, list):
        messages = [
            f"{'.'.join(map(str, item.get('loc', [])))}: {item.get('msg', 'Invalid input')}"
            for item in detail
            if isinstance(item, dict)
        ]
        return "\n".join(messages) or json.dumps(detail, indent=2, ensure_ascii=False)
    if detail is not None:
        return str(detail)
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _request_summary(filename: str, pdf_bytes: bytes) -> dict[str, Any]:
    try:
        response = requests.post(
            SUMMARY_ENDPOINT,
            files={"file": (filename, pdf_bytes, "application/pdf")},
            timeout=(10, BACKEND_TIMEOUT_SECONDS),
        )
    except requests.exceptions.Timeout as error:
        raise BackendResponseError(
            f"The backend did not finish within {BACKEND_TIMEOUT_SECONDS:g} seconds."
        ) from error
    except requests.RequestException as error:
        raise BackendResponseError(
            f"Could not reach the backend at {BACKEND_URL}: {error}"
        ) from error

    if not response.ok:
        raise BackendResponseError(
            f"Backend returned HTTP {response.status_code}:\n"
            f"{_response_error_message(response)}"
        )

    try:
        payload = response.json()
    except ValueError as error:
        raise BackendResponseError(
            "The backend returned a successful status but its response was not valid JSON."
        ) from error

    if not isinstance(payload, dict) or not isinstance(payload.get("summary"), str):
        raise BackendResponseError(
            "The backend response did not contain the expected string 'summary' field."
        )
    if not isinstance(payload.get("metadata"), dict):
        raise BackendResponseError(
            "The backend response did not contain the expected 'metadata' object."
        )
    return payload


def _section_label(key: str) -> str:
    return SECTION_LABELS.get(key, key.replace("_", " ").strip().capitalize())


def _render_source_reference(reference: Any) -> None:
    if not isinstance(reference, dict):
        st.write(reference)
        return

    details = []
    for field, label in (
        ("source_document_id", "Document"),
        ("source_passage_id", "Passage ID"),
        ("page_numbers", "Page(s)"),
        ("start_char", "Start offset"),
        ("end_char", "End offset"),
        ("paragraph_reference", "Paragraph"),
    ):
        value = reference.get(field)
        if value is not None:
            details.append(f"**{label}:** {value}")
    if details:
        st.markdown(" · ".join(details))
    unavailable = reference.get("unavailable_metadata")
    if unavailable:
        st.caption("Unavailable in source reference: " + ", ".join(map(str, unavailable)))


def _render_structured_summary(metadata: dict[str, Any]) -> None:
    sections = metadata.get("structured_summary")
    st.subheader("Structured summary")
    if not isinstance(sections, dict) or not sections:
        if metadata.get("summary_status") == "unverified_draft":
            st.info(
                "No claims passed the role checks for structured sections. "
                "Review the unverified draft above and its withheld-claim notes."
            )
        elif metadata.get("summary_status") == "no_usable_summary":
            st.info("No usable summary or evidence-backed structured sections were produced.")
        else:
            st.info("The backend did not return any evidence-backed structured sections.")
        return

    for key, content in sections.items():
        if not content:
            continue
        with st.expander(_section_label(str(key)), expanded=True):
            entries = content if isinstance(content, list) else [content]
            for entry in entries:
                if isinstance(entry, dict):
                    text = entry.get("text")
                    if text:
                        st.markdown(f"- {text}")
                    if entry.get("source_passage_ids"):
                        st.caption(
                            "Source passage IDs: "
                            + ", ".join(map(str, entry["source_passage_ids"]))
                        )
                    if entry.get("verification_result"):
                        st.caption(
                            f"Verification: {entry['verification_result']}"
                        )
                else:
                    st.write(entry)


def _render_claims(title: str, claims: Any) -> None:
    st.subheader(title)
    if not isinstance(claims, list) or not claims:
        st.info("No claim-level verification records were returned for this view.")
        return

    for index, record in enumerate(claims, start=1):
        if not isinstance(record, dict):
            with st.expander(f"Claim {index}"):
                st.write(record)
            continue
        claim = str(record.get("claim") or f"Claim {index}")
        status = str(record.get("verification_result") or "status unavailable")
        icon = STATUS_ICONS.get(status, "ℹ️")
        with st.expander(f"{icon} {status}: {claim[:100]}"):
            st.markdown(f"**Claim:** {claim}")
            if record.get("claim_category") is not None:
                st.write(f"**Category:** {record['claim_category']}")
            if record.get("verification_method") is not None:
                st.write(f"**Method:** {record['verification_method']}")
            if record.get("confidence") is not None:
                st.write(f"**Confidence / score:** {record['confidence']}")
            if record.get("confidence_interpretation"):
                st.caption(record["confidence_interpretation"])
            if record.get("source_document_id") is not None:
                st.write(f"**Source document ID:** {record['source_document_id']}")
            passage_ids = record.get("supporting_source_passage_ids")
            if passage_ids:
                st.write(
                    "**Supporting passage IDs:** "
                    + ", ".join(map(str, passage_ids))
                )
            references = record.get("supporting_source_references")
            if references:
                st.markdown("**Supporting source references**")
                for reference in references:
                    _render_source_reference(reference)
            reasons = record.get("abstention_reasons")
            if reasons:
                st.warning("Review notes: " + " ".join(map(str, reasons)))
            roles = record.get("evidence_role")
            if roles:
                st.caption(f"Evidence role(s): {', '.join(map(str, roles))}")


def _render_source_traceability(metadata: dict[str, Any]) -> None:
    references = metadata.get("source_references")
    if not isinstance(references, list) or not references:
        return

    st.subheader("Source traceability")
    st.caption(
        "The API returns source identifiers and locations where available; "
        "it does not include source passage text or PDF page images in these references."
    )
    with st.expander(f"Selected source references ({len(references)})"):
        for reference in references:
            _render_source_reference(reference)


def _render_metadata(metadata: dict[str, Any]) -> None:
    metric_fields = (
        ("passage_count", "Passages"),
        ("selected_passage_count", "Selected passages"),
        ("generation_levels", "Generation levels"),
        ("content_coverage_fraction", "Content coverage"),
    )
    metrics = [
        (label, metadata[key])
        for key, label in metric_fields
        if metadata.get(key) is not None
    ]
    if metrics:
        columns = st.columns(len(metrics))
        for column, (label, value) in zip(columns, metrics):
            display_value = (
                f"{value:.1%}"
                if label == "Content coverage" and isinstance(value, (int, float))
                else str(value)
            )
            column.metric(label, display_value)

    extraction = metadata.get("extraction")
    quality = extraction.get("quality") if isinstance(extraction, dict) else None
    warnings = quality.get("warnings") if isinstance(quality, dict) else None
    if warnings:
        for warning in warnings:
            st.warning(str(warning))

    grounding = metadata.get("grounding")
    if isinstance(grounding, dict):
        if grounding.get("unsupported_claims"):
            with st.expander(
                f"Unsupported generated claims ({grounding.get('unsupported_claim_count', len(grounding['unsupported_claims']))})"
            ):
                st.json(grounding["unsupported_claims"])
        if grounding.get("limitations"):
            st.caption("Backend grounding limitations: " + " ".join(grounding["limitations"]))

    withheld = metadata.get("role_attribution_filter")
    withheld_claims = withheld.get("withheld_claims") if isinstance(withheld, dict) else None
    if withheld_claims:
        with st.expander(f"Claims withheld by the backend ({len(withheld_claims)})"):
            for claim in withheld_claims:
                if isinstance(claim, dict):
                    st.markdown(f"**{claim.get('verification_result', 'Review required')}**")
                    st.write(claim.get("claim", ""))
                    if claim.get("abstention_reasons"):
                        st.warning(" ".join(map(str, claim["abstention_reasons"])))
                    for reference in claim.get("supporting_source_references", []):
                        _render_source_reference(reference)
                else:
                    st.write(claim)


def _render_result(result: dict[str, Any]) -> None:
    summary_status = result.get("metadata", {}).get("summary_status")
    if summary_status == "unverified_draft":
        st.warning(
            "The generated draft did not pass all role checks. Treat it as unverified."
        )
    elif summary_status == "no_usable_summary":
        st.error("Summary generation could not produce a usable draft.")
    else:
        st.success("Summary generated.")
    st.subheader("Summary")
    st.markdown(result["summary"] or "_The backend returned an empty summary._")

    metadata = result["metadata"]
    _render_structured_summary(metadata)

    returned_claims = metadata.get("returned_summary_claim_verification")
    if returned_claims is None:
        returned_claims = metadata.get("claim_verification")
    _render_claims("Claim-level verification", returned_claims)

    candidate_claims = metadata.get("claim_verification")
    if (
        isinstance(candidate_claims, list)
        and candidate_claims != returned_claims
    ):
        with st.expander("Final generation candidate checks (before return filtering)"):
            _render_claims("Candidate claim verification", candidate_claims)

    _render_source_traceability(metadata)
    _render_metadata(metadata)

    with st.expander("Complete API response JSON"):
        st.json(result)


st.set_page_config(
    page_title="Indian Legal Judgment Understanding Platform",
    page_icon="⚖️",
    layout="wide",
)

st.title("Indian Legal Judgment Understanding Platform")
st.caption("Developer interface for the existing judgment summarization API.")

with st.sidebar:
    st.subheader("Backend")
    st.code(BACKEND_URL, language=None)
    st.caption("Summarization is synchronous; this page waits for the full API response.")

uploaded_file = st.file_uploader(
    "Upload a court judgment PDF",
    type=["pdf"],
    accept_multiple_files=False,
    help="The backend accepts a multipart/form-data field named 'file'.",
)

pdf_bytes = uploaded_file.getvalue() if uploaded_file is not None else b""
valid_pdf = bool(
    uploaded_file is not None
    and uploaded_file.name.lower().endswith(".pdf")
    and pdf_bytes.startswith(b"%PDF-")
)
if uploaded_file is not None:
    st.write(f"**File:** {uploaded_file.name}")
    st.write(f"**Size:** {_format_bytes(len(pdf_bytes))}")
    if not valid_pdf:
        st.error("The selected file must have a .pdf extension and a valid PDF signature.")

submitted = st.button(
    "Generate Summary",
    type="primary",
    disabled=not valid_pdf,
    use_container_width=True,
)

upload_key = (
    hashlib.sha256(pdf_bytes).hexdigest()
    if valid_pdf
    else None
)
if st.session_state.get("upload_key") != upload_key:
    st.session_state.upload_key = upload_key
    st.session_state.summary_result = None
    st.session_state.summary_error = None
    st.session_state.elapsed_seconds = None

if submitted and valid_pdf and uploaded_file is not None:
    started_at = time.perf_counter()
    st.session_state.summary_result = None
    st.session_state.summary_error = None
    try:
        with st.spinner("Processing the judgment. This may take several minutes..."):
            st.session_state.summary_result = _request_summary(
                uploaded_file.name,
                pdf_bytes,
            )
    except BackendResponseError as error:
        st.session_state.summary_error = str(error)
    finally:
        st.session_state.elapsed_seconds = time.perf_counter() - started_at

if st.session_state.get("elapsed_seconds") is not None:
    st.caption(f"Backend request time: {st.session_state.elapsed_seconds:.1f} seconds")

if st.session_state.get("summary_error"):
    st.error(st.session_state.summary_error)
elif st.session_state.get("summary_result"):
    _render_result(st.session_state.summary_result)
