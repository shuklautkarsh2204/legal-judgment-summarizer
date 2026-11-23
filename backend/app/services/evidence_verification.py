"""Role-aware evidence organization and exploratory claim verification."""

import re


UNCLASSIFIED = "unclassified_uncertain"
VERIFICATION_METHOD = "lexical_overlap_heuristic"
_CONTENT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "for", "from", "had", "has", "have", "he", "her", "his", "in",
    "into", "is", "it", "its", "of", "on", "or", "our", "she", "that",
    "the", "their", "them", "there", "these", "they", "this", "those",
    "to", "was", "were", "which", "who", "will", "with", "would",
}

_PARTY_CUES = {
    "petitioner_argument": re.compile(
        r"\b(?:petitioner|appellant|applicant|plaintiff)\b.{0,100}"
        r"\b(?:argued|contended|submitted|claimed|alleged)\b",
        re.IGNORECASE,
    ),
    "respondent_argument": re.compile(
        r"\b(?:respondent|state|union|defendant)\b.{0,100}"
        r"\b(?:argued|contended|submitted|claimed|alleged)\b",
        re.IGNORECASE,
    ),
}
_COURT_CUE = re.compile(
    r"\b(?:(?:the\s+)?court|we)\b.{0,80}"
    r"\b(?:held|found|observed|reasoned|concluded|ordered|directed|decided|"
    r"allowed|dismissed|granted|rejected|affirmed|reversed|quashed)\b"
    r"|\bwe\s+(?:hold|find|observe|reason|conclude|order|direct|allow|dismiss)\b",
    re.IGNORECASE,
)
_DECISION_CUE = re.compile(
    r"\b(?:allowed|dismissed|granted|rejected|set aside|affirmed|"
    r"quashed|directed|ordered)\b",
    re.IGNORECASE,
)
_PRECEDENT_CUE = re.compile(
    r"\b(?:precedent|previous judgment|prior decision|relied upon|"
    r"followed in|distinguished in)\b",
    re.IGNORECASE,
)
_ISSUE_CUE = re.compile(
    r"\b(?:issue before the court|question for consideration|"
    r"issues for determination)\b",
    re.IGNORECASE,
)
_OVERVIEW_CUE = re.compile(
    r"\b(?:this case concerns|the matter arises from|the case relates to|"
    r"the dispute concerns)\b",
    re.IGNORECASE,
)
_FACT_CUE = re.compile(
    r"\b(?:procedural history|was arrested|was filed|filed a petition|"
    r"the facts are|the facts of the case)\b",
    re.IGNORECASE,
)
_OPPOSING_TERMS = (
    ("allowed", "dismissed"),
    ("granted", "rejected"),
    ("lawful", "unlawful"),
    ("valid", "invalid"),
    ("constitutional", "unconstitutional"),
    ("affirmed", "reversed"),
)
_SECTION_BY_CATEGORY = {
    "case_overview": "case_overview",
    "facts_and_procedure": "facts_and_procedural_history",
    "issues": "issues_considered",
    "petitioner_argument": "petitioner's_arguments",
    "respondent_argument": "respondent's_arguments",
    "court_reasoning": "court_reasoning",
    "court_decision": "final_decision_and_directions",
    "precedents": "important_precedents",
}


def _content_tokens(text):
    return {
        token for token in re.findall(r"[A-Za-z][A-Za-z'-]{2,}|\d+(?:\([\w]+\))?", text.lower())
        if token not in _CONTENT_STOPWORDS
    }


def classify_role(text):
    """Classify only explicit, unambiguous textual cues; otherwise abstain."""
    matches = [
        role for role, pattern in _PARTY_CUES.items()
        if pattern.search(text)
    ]
    has_court_cue = bool(_COURT_CUE.search(text))
    if has_court_cue:
        matches.append(
            "court_decision" if _DECISION_CUE.search(text) else "court_reasoning"
        )
    if _PRECEDENT_CUE.search(text):
        matches.append("precedents")
    if _ISSUE_CUE.search(text):
        matches.append("issues")
    if _OVERVIEW_CUE.search(text):
        matches.append("case_overview")
    if _FACT_CUE.search(text):
        matches.append("facts_and_procedure")

    unique_matches = set(matches)
    if len(unique_matches) != 1:
        return {
            "role": UNCLASSIFIED,
            "status": "ambiguous" if unique_matches else "unclassified",
            "method": "explicit_text_cue_heuristic",
        }
    return {
        "role": unique_matches.pop(),
        "status": "heuristic_assignment",
        "method": "explicit_text_cue_heuristic",
    }


def map_transferred_function(label):
    """Expose only approximate functional analogies, never speaker attribution."""
    normalized = (label or "").casefold()
    if "vorherige rechtsprechung" in normalized:
        role = "precedents"
    elif "subsumtion" in normalized or "verhältnismäßigkeitsprüfung" in normalized:
        role = "court_reasoning"
    elif "entscheidung des egmr" in normalized:
        role = "court_decision"
    else:
        role = UNCLASSIFIED
    return {
        "label": label,
        "approximate_function": role,
        "mapping_status": (
            "approximate_cross_domain_mapping"
            if role != UNCLASSIFIED else "unsupported_or_ambiguous"
        ),
        "provenance": "Exploratory RMU:ECHR transfer; not Indian ground truth.",
        "establishes_speaker": False,
    }


class RoleAwareEvidenceOrganizer:
    """Attach transparent role cues while keeping classifier transfer separate."""

    def organize(self, passages, transferred_predictions_by_id=None):
        transferred_predictions_by_id = transferred_predictions_by_id or {}
        organized = []
        for passage in passages:
            item = dict(passage)
            attribution = classify_role(item["text"])
            item["evidence_role"] = attribution["role"]
            item["role_assignment_status"] = attribution["status"]
            item["role_assignment_method"] = attribution["method"]
            prediction = transferred_predictions_by_id.get(item["passage_id"])
            item["transferred_function_mapping"] = (
                map_transferred_function(prediction) if prediction else None
            )
            organized.append(item)
        return organized

    @staticmethod
    def format_for_generation(passages):
        blocks = []
        for passage in passages:
            role = passage.get("evidence_role", UNCLASSIFIED)
            status = passage.get("role_assignment_status", "unclassified")
            blocks.append(
                f"[Passage {passage['passage_id']}; evidence role={role}; "
                f"role status={status}]\n{passage['text']}"
            )
        return "\n".join(blocks)


def _claim_category(claim):
    role = classify_role(claim)
    if role["role"] != UNCLASSIFIED:
        return role["role"]
    return UNCLASSIFIED


def _source_reference(passage):
    position = passage.get("original_position", {})
    return {
        "source_document_id": passage.get("document_id"),
        "source_passage_id": passage.get("passage_id"),
        "start_char": position.get("start_char"),
        "end_char": position.get("end_char"),
        "page_numbers": passage.get("page_numbers") or None,
        "paragraph_reference": None,
        "bounding_boxes": None,
        "unavailable_metadata": [
            field for field, value in (
                ("page_numbers", passage.get("page_numbers")),
                ("paragraph_reference", None),
                ("bounding_boxes", None),
            ) if value is None
        ],
    }


def _contradicts(claim, evidence):
    claim_lower = claim.lower()
    evidence_lower = evidence.lower()
    for first, second in _OPPOSING_TERMS:
        if (re.search(rf"\b{first}\b", claim_lower)
                and re.search(rf"\b{second}\b", evidence_lower)):
            return True
        if (re.search(rf"\b{second}\b", claim_lower)
                and re.search(rf"\b{first}\b", evidence_lower)):
            return True
    return False


def verify_claim(claim, passages, *, overlap_threshold=0.45, max_support_passages=3):
    """Return a claim-level, uncalibrated evidence signal with conservative abstention."""
    claim_tokens = _content_tokens(claim)
    category = _claim_category(claim)
    abstention_reasons = []
    candidates = []

    for passage in passages:
        source_tokens = _content_tokens(passage.get("text", ""))
        overlap_tokens = claim_tokens & source_tokens
        score = len(overlap_tokens) / len(claim_tokens) if claim_tokens else 0.0
        candidates.append({
            "passage": passage,
            "score": score,
            "overlap_tokens": overlap_tokens,
        })
    candidates.sort(key=lambda candidate: (-candidate["score"], candidate["passage"]["passage_id"]))

    selected = []
    covered_tokens = set()
    for candidate in candidates:
        if len(selected) >= max_support_passages or not candidate["overlap_tokens"]:
            break
        added_tokens = candidate["overlap_tokens"] - covered_tokens
        if added_tokens:
            selected.append(candidate)
            covered_tokens.update(added_tokens)
        if claim_tokens and len(covered_tokens) / len(claim_tokens) >= overlap_threshold:
            break
    overlap_score = len(covered_tokens) / len(claim_tokens) if claim_tokens else 0.0
    has_identifiable_support = overlap_score >= overlap_threshold
    selected_passages = (
        [candidate["passage"] for candidate in selected]
        if has_identifiable_support else []
    )
    evidence_roles = sorted({
        passage.get("evidence_role", UNCLASSIFIED) for passage in selected_passages
    })

    if not claim_tokens:
        abstention_reasons.append("Claim has no content tokens to compare.")
    if overlap_score < overlap_threshold:
        abstention_reasons.append("No passage set reached the exploratory lexical-overlap threshold.")

    result = "requires_human_review"
    if selected and any(_contradicts(claim, passage.get("text", "")) for passage in selected_passages):
        result = "possible_contradiction"
        abstention_reasons.append("Opposing outcome terms occur in the claim and candidate evidence.")
    elif overlap_score < overlap_threshold:
        result = "insufficient_evidence"
    elif any(
        role in {"petitioner_argument", "respondent_argument"}
        for role in evidence_roles
    ) and (
        category not in {"petitioner_argument", "respondent_argument"}
        or any(role != category for role in evidence_roles)
    ):
        result = "potential_role_attribution_problem"
        abstention_reasons.append(
            "Party-attributed evidence cannot support an unqualified or differently "
            "attributed claim."
        )
    elif not selected or UNCLASSIFIED in evidence_roles:
        result = "requires_human_review"
        abstention_reasons.append("Evidence role is not confidently identified.")
    else:
        result = "supported_by_identified_evidence"

    references = [_source_reference(passage) for passage in selected_passages]
    document_ids = sorted({
        passage.get("document_id") for passage in selected_passages
        if passage.get("document_id") is not None
    })
    return {
        "claim": claim,
        "claim_category": category,
        "supporting_source_passage_ids": [
            passage.get("passage_id") for passage in selected_passages
        ],
        "supporting_source_references": references,
        "source_document_id": document_ids[0] if len(document_ids) == 1 else None,
        "evidence_role": evidence_roles or [UNCLASSIFIED],
        "verification_result": result,
        "verification_method": VERIFICATION_METHOD,
        "confidence": round(overlap_score, 4),
        "confidence_interpretation": "Uncalibrated lexical-overlap signal; not entailment confidence.",
        "abstention_reasons": abstention_reasons,
    }


def verify_summary_claims(summary, passages):
    claims = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", summary)
        if sentence.strip()
    ]
    return [verify_claim(claim, passages) for claim in claims]


def retain_role_verified_claims(summary, claim_records):
    """Keep only claim sentences whose identified evidence passes the role check."""
    verified = {
        record["claim"]: record
        for record in claim_records
        if record["verification_result"] == "supported_by_identified_evidence"
    }
    retained = []
    withheld = []
    for claim in (
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", summary)
        if sentence.strip()
    ):
        record = verified.get(claim)
        if record is None:
            withheld.append(next(
                (item for item in claim_records if item["claim"] == claim),
                {
                    "claim": claim,
                    "verification_result": "requires_human_review",
                    "abstention_reasons": [
                        "No role-compatible evidence verification was available."
                    ],
                    "supporting_source_references": [],
                },
            ))
        else:
            retained.append(claim)
    return " ".join(retained), withheld


def build_structured_summary(claim_records):
    """Build only sections with generated claims tied to at least one source passage."""
    structured = {}
    for record in claim_records:
        if (
            not record["supporting_source_passage_ids"]
            or record["verification_result"] != "supported_by_identified_evidence"
        ):
            continue
        section = _SECTION_BY_CATEGORY.get(record["claim_category"])
        if section is None:
            continue
        structured.setdefault(section, []).append({
            "text": record["claim"],
            "source_passage_ids": record["supporting_source_passage_ids"],
            "verification_result": record["verification_result"],
        })
    return structured
