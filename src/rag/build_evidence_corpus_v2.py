"""Build a versioned, task-aligned evidence corpus candidate; dry by default.

Corpus V1 is read-only. Full document text is preserved. Content-quality
decisions apply to deterministic paragraph/page units for later Index V2
design; they never rewrite the extracted document or establish relevance.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from src.rag import build_evidence_corpus as v1


NEW_SOURCES = (
    dict(document_id="RESEARCH-LICHY-2023", title="When a RF Beats a CNN and GRU, Together - A Comparison of Deep Learning and Classical Machine Learning Approaches for Encrypted Malware Traffic Classification", source_type="peer_reviewed_research", publisher="", authors="", publication_date="", canonical_url="", doi="", local_path="data/raw/rag/evidence_sources_v2/task_aligned/LICHY-2023.pdf", traffic_topics="encrypted_traffic;malware_traffic;USTC-TFC2016", security_topics="traffic_classification;malware_detection;malware_family_classification", corpus_partition="trusted", source_provenance_status="VERIFIED", authorized_for_retrieval="YES", curation_notes="Researcher manually inspected the local PDF and approved it as a substantive extraction candidate; independent peer-review verification is not claimed by this stage."),
    dict(document_id="RESEARCH-SHEKHAWAT-2019", title="Feature Analysis of Encrypted Malicious Traffic", source_type="peer_reviewed_research", publisher="", authors="", publication_date="", canonical_url="", doi="", local_path="data/raw/rag/evidence_sources_v2/task_aligned/SHEKHAWAT-2019.pdf", traffic_topics="encrypted_traffic;benign_traffic;malicious_traffic", security_topics="traffic_analysis;traffic_classification;malicious_traffic_detection", corpus_partition="trusted", source_provenance_status="VERIFIED", authorized_for_retrieval="YES", curation_notes="Researcher manually inspected the local PDF and approved it as a substantive extraction candidate; independent peer-review verification is not claimed by this stage."),
)
OUTPUTS = (
    "data/processed/rag/evidence_documents_v2.csv",
    "metadata/rag/evidence_source_manifest_v2.csv",
    "metadata/rag/evidence_extraction_review_v2.csv",
    "metadata/rag/evidence_content_quality_review_v2.csv",
    "metadata/rag/evidence_corpus_v2_summary.csv",
    "metadata/rag/evidence_corpus_v2_validation_checklist.csv",
)
TERM_PATTERNS = {
    "benign": r"\bbenign\b", "malware": r"\bmalware\b", "malicious": r"\bmalicious\b",
    "attack": r"\battacks?\b", "encrypted_traffic": r"\bencrypted\s+traffic\b",
    "traffic_analysis": r"\btraffic\s+analysis\b", "tor": r"\btor\b", "vpn": r"\bvpn\b",
    "quic": r"\bquic\b", "tls": r"\btls\b",
}
TERM_COLUMNS = tuple(f"term_{name}_count" for name in TERM_PATTERNS)
DOCUMENT_HEADER = (*v1.DOCUMENT_HEADER, "corpus_revision", "is_new_document", *TERM_COLUMNS, "task_aligned")
QUALITY_HEADER = ("document_id", "review_unit_id", "unit_order", "source_location", "quality_decision", "quality_reason", "unit_word_count", "text_preview")
SUMMARY_HEADER = ("summary_level", "source_type", "is_new_document", "candidate_documents", "trusted_documents", "rejected_documents", "new_documents", "new_peer_reviewed_research_documents", "total_extracted_words", "task_aligned_documents", "documents_with_benign_term", "documents_with_malware_term", "documents_with_malicious_term", "quality_keep_units", "quality_excluded_units", "quality_requires_review_units")
CHECK_HEADER = v1.foundation.CHECKLIST_HEADER
V1_INPUTS = (v1.MANIFEST, v1.REVIEW, *v1.OUTPUTS)
QUALITY_DECISIONS = {"KEEP", "EXCLUDE_LOW_INFORMATION", "REQUIRES_REVIEW"}
require = v1.require
PROFILES = ("legacy", "evidence_quality")
IMPROVED_QUALITY_HEADER = (*QUALITY_HEADER, "quality_profile")
DECISION_CLAIM_HEADER = ("claim_id", "document_id", "source_dataset", "source_location", "evidence_quote",
                         "canonical_feature", "direction", "supported_class", "contrasted_class",
                         "review_status", "authorized_for_decision_evidence")
DECISION_CLASSES = {"USTC-TFC2016": ("benign", "malware"),
                    "CIC-Darknet2020": ("Non-Tor", "NonVPN", "Tor", "VPN")}


def validate_development_id(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value or "")
            and "test" not in {part.lower() for part in re.split(r"[._-]+", value)},
            "Improved profiles require a safe development-id path component")
    return value


def output_paths(development_id=None):
    if development_id is None:
        return OUTPUTS
    validate_development_id(development_id)
    return (f"data/processed/rag/{development_id}/evidence_documents_v2.csv",
            f"metadata/rag/{development_id}/stage02_corpus/evidence_source_manifest_v2.csv",
            f"metadata/rag/{development_id}/stage02_corpus/evidence_extraction_review_v2.csv",
            f"metadata/rag/{development_id}/stage02_corpus/evidence_content_quality_review_v2.csv",
            f"metadata/rag/{development_id}/stage02_corpus/evidence_corpus_v2_summary.csv",
            f"metadata/rag/{development_id}/stage02_corpus/evidence_corpus_v2_validation_checklist.csv")


CANONICAL_DECISION_FEATURES = {
    "USTC-TFC2016": frozenset({
        "bidirectional duration ms", "bidirectional packets", "bidirectional bytes",
        "source to destination duration ms", "source to destination packets",
        "source to destination bytes", "destination to source duration ms",
        "destination to source packets", "destination to source bytes",
        "bidirectional min packet size", "bidirectional mean packet size",
        "bidirectional standard deviation packet size", "bidirectional max packet size",
        "source to destination min packet size", "source to destination mean packet size",
        "source to destination standard deviation packet size",
        "source to destination max packet size", "destination to source min packet size",
        "destination to source mean packet size",
        "destination to source standard deviation packet size",
        "destination to source max packet size",
        "bidirectional min packet inter-arrival time ms",
        "bidirectional mean packet inter-arrival time ms",
        "bidirectional standard deviation packet inter-arrival time ms",
        "bidirectional max packet inter-arrival time ms",
        "source to destination min packet inter-arrival time ms",
        "source to destination mean packet inter-arrival time ms",
        "source to destination standard deviation packet inter-arrival time ms",
        "source to destination max packet inter-arrival time ms",
        "destination to source min packet inter-arrival time ms",
        "destination to source mean packet inter-arrival time ms",
        "destination to source standard deviation packet inter-arrival time ms",
        "destination to source max packet inter-arrival time ms",
        "bidirectional SYN packets", "bidirectional ACK packets", "bidirectional PSH packets",
        "bidirectional RST packets", "bidirectional FIN packets",
        "source to destination SYN packets", "source to destination ACK packets",
        "source to destination PSH packets", "source to destination RST packets",
        "source to destination FIN packets", "destination to source SYN packets",
        "destination to source ACK packets", "destination to source PSH packets",
        "destination to source RST packets", "destination to source FIN packets",
    }),
    "CIC-Darknet2020": frozenset({
        "flow duration", "total forward packet", "total backward packets",
        "total length of forward packet", "total length of backward packet",
        "forward packet length max", "forward packet length min", "forward packet length mean",
        "forward packet length standard deviation", "backward packet length max",
        "backward packet length min", "backward packet length mean",
        "backward packet length standard deviation", "flow bytes/s", "flow packets/s",
        "flow inter-arrival time mean", "flow inter-arrival time standard deviation",
        "flow inter-arrival time max", "flow inter-arrival time min",
        "forward inter-arrival time total", "forward inter-arrival time mean",
        "forward inter-arrival time standard deviation", "forward inter-arrival time max",
        "forward inter-arrival time min", "backward inter-arrival time total",
        "backward inter-arrival time mean", "backward inter-arrival time standard deviation",
        "backward inter-arrival time max", "backward inter-arrival time min",
        "forward PSH flags", "forward header length", "backward header length",
        "forward packets/s", "backward packets/s", "packet length min", "packet length max",
        "packet length mean", "packet length standard deviation", "packet length variance",
        "FIN flag count", "SYN flag count", "RST flag count", "PSH flag count",
        "ACK flag count", "down/up ratio", "average packet size",
        "forward segment size average", "backward segment size average",
        "backward packet/bulk average", "backward bulk rate average",
        "subflow forward packets", "subflow forward bytes", "subflow backward bytes",
        "forward init win bytes", "backward init win bytes", "forward act data packets",
        "forward seg size min", "idle mean", "idle standard deviation", "idle max", "idle min",
    }),
}

assert len(CANONICAL_DECISION_FEATURES["USTC-TFC2016"]) == 48
assert len(CANONICAL_DECISION_FEATURES["CIC-Darknet2020"]) == 61


def decision_claim_path(development_id):
    validate_development_id(development_id)
    return f"metadata/rag/{development_id}/stage02_corpus/decision_evidence_claims.csv"


def validate_decision_claim_rows(documents, quality, claims):
    docs = {row["document_id"]: row for row in documents}
    quality_by_document = {}
    for row in quality:
        quality_by_document.setdefault(row["document_id"], []).append(row)
    keep_units = {}
    for identity, document in docs.items():
        units = source_units(document["normalized_text"])
        reviews = sorted(quality_by_document.get(identity, []), key=lambda row: int(row["unit_order"]))
        require(len(units) == len(reviews), "Decision-claim quality units cannot be reconstructed")
        require(reviews and all(row.get("quality_profile") == "evidence_quality" for row in reviews),
                "Decision claims require evidence_quality review units")
        located_units = [(review["source_location"], unit[1]) for review, unit in zip(reviews, units)]
        expected_reviews = quality_rows(identity, located_units, profile="evidence_quality")
        for review, expected in zip(reviews, expected_reviews):
            require(all(str(review[key]) == str(expected[key]) for key in IMPROVED_QUALITY_HEADER),
                    "Decision claim references stale/altered quality review")
        keep_units[identity] = [
            (review["source_location"], text)
            for review, (_, text) in zip(reviews, units)
            if review["quality_decision"] == "KEEP"
        ]
    seen = set()
    normalized = []
    for claim in claims:
        require(set(claim) == set(DECISION_CLAIM_HEADER), "Invalid decision-claim schema")
        identity, dataset = claim["document_id"], claim["source_dataset"]
        require(claim["claim_id"].strip() and claim["claim_id"] not in seen, "Blank/duplicate decision claim_id")
        seen.add(claim["claim_id"])
        document = docs.get(identity)
        require(document is not None and v1.trusted(document) and document["authorized_for_retrieval"] == "YES",
                "Decision claim references an untrusted/unauthorized document")
        require(dataset in DECISION_CLASSES, "Invalid decision-claim dataset")
        require(
            claim["canonical_feature"] in CANONICAL_DECISION_FEATURES[dataset],
            "Invalid decision-claim canonical feature",
        )
        require(claim["direction"] in {"higher", "lower", "typical"}, "Invalid decision-claim direction")
        classes = DECISION_CLASSES[dataset]
        supported, contrasted = claim["supported_class"], claim["contrasted_class"]
        require(supported in classes and (not contrasted or contrasted in classes) and contrasted != supported,
                "Invalid decision-claim class relationship")
        require(claim["review_status"] == "APPROVED" and claim["authorized_for_decision_evidence"] == "YES",
                "Decision claim is not approved and authorized")
        require(claim["source_location"].strip() and claim["canonical_feature"].strip() and claim["evidence_quote"].strip(),
                "Decision-claim location, feature, and verbatim quote are required")
        require(
            claim["evidence_quote"] in document["normalized_text"]
            and any(
                claim["source_location"] == source_location
                and claim["evidence_quote"] in unit_text
                for source_location, unit_text in keep_units[identity]
            ),
                "Decision-claim quote is absent from the document or KEEP quality units")
        normalized.append({key: claim[key] for key in DECISION_CLAIM_HEADER})
    return sorted(normalized, key=lambda row: row["claim_id"])


def read_decision_claims(root, path):
    header, claims = v1.read_table(root, path)
    require(tuple(header) == DECISION_CLAIM_HEADER, "Decision-claim CSV header differs from required schema")
    return claims


def load_decision_claims(root, development_id):
    path = decision_claim_path(development_id)
    docs_path, _, _, quality_path, _, _ = output_paths(development_id)
    _, docs = v1.read_table(root, docs_path, DOCUMENT_HEADER)
    _, quality = v1.read_table(root, quality_path, IMPROVED_QUALITY_HEADER)
    if development_id == "v5e_validation":
        from src.rag.build_train_evidence_v5 import verify_documents
        verify_documents(root, development_id, docs)
    claims = read_decision_claims(root, path)
    return validate_decision_claim_rows(docs, quality, claims)


def file_metadata(root, names):
    return [dict(path=name, size_bytes=v1.local_path(root, name).stat().st_size, mtime_ns=v1.local_path(root, name).stat().st_mtime_ns) for name in names]


def term_counts(text):
    lower = text.lower()
    return {f"term_{name}_count": len(re.findall(pattern, lower)) for name, pattern in TERM_PATTERNS.items()}


def source_units(text, locations=None):
    blocks = []
    if locations is None:
        locations = [("document", text)]
    for location, page in locations:
        parts = [part.strip() for part in re.split(r"\n\s*\n", page) if part.strip()]
        if not parts and page.strip():
            parts = [page.strip()]
        blocks.extend((location, part) for part in parts)
    return blocks


def reference_heading(text):
    return bool(re.fullmatch(r"(?:\d+(?:\.\d+)*\.?\s+)?(?:references|normative references|informative references|bibliography|works cited)[:.]?", text.strip(), re.I))


def citation_only(text):
    """Require entry structure and publication metadata, not an RFC mention."""
    entries = re.split(r"\n(?=\s*\[[^\]\n]+\])", text.strip())
    for entry in entries:
        compact = " ".join(entry.split())
        # Explanation/instructions mixed into an entry require human review.
        if re.findall(r"\b(?:must|should|shall|implements?|requires?|provides?|uses?|contains?|discusses?|explains?|ensures?|allows?|validates?|sends?|receives?)\b", compact, re.I):
            return False
        if not re.fullmatch(r"\[[^\]]+\]\s+.+", compact):
            return False
        # A trailing explanatory paragraph is not part of a citation-only unit.
        if not re.findall(r"(?:\b(?:19|20)\d{2}|https?://[^\s<>]+|10\.\d{4,9}/[^\s<>]+)[>.,;)]*$", compact, re.I):
            return False
        markers = sum(bool(re.findall(pattern, compact, re.I)) for pattern in (
            r"\b(?:19|20)\d{2}\b", r"\bdoi\s*:?|10\.\d{4,9}/",
            r"https?://", r"\bRFC\s*\d+\b"))
        if markers < 2:
            return False
    return bool(entries)


def quality_decision(text, repeated=False, in_references=False, profile="legacy"):
    """Conservative structural review; a keyword alone never deletes content."""
    normalized = " ".join(text.split())
    require(profile in PROFILES, "Unknown quality profile")
    if profile == "evidence_quality":
        # Whole-unit vocabulary only: mixed technical prose is never removed.
        labels = r"(?:search|apply|about(?: us)?|datasets|publications|members|contact(?: us)?|home|menu|log in|sign in|toggle navigation|skip to (?:navigation|main content)|main navigation|next|previous|read more)"
        if re.fullmatch(rf"{labels}(?:[\s|;/,\u2022]+{labels})*[.!:]?", normalized, re.I):
            return "EXCLUDE_LOW_INFORMATION", "NAVIGATION_UI_ONLY"
    lower, words = normalized.lower(), normalized.split()
    if not words:
        return "EXCLUDE_LOW_INFORMATION", "EMPTY_UNIT"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    navigation = r"(?:skip to (?:main content|global navigation)|(?:global|main|site|toggle) navigation|(?:unb )?phone directory|give to unb|campus maps|careers at unb|web feedback)"
    if re.fullmatch(rf"{navigation}(?:\s*[|;\n]?\s*{navigation})*", normalized, re.I):
        return "EXCLUDE_LOW_INFORMATION", "NAVIGATION_BOILERPLATE"
    # Whole-unit matching prevents a page marker from deleting adjacent prose.
    author = r"[A-Z][A-Za-z'’-]+(?:,?\s+[A-Z]\.)?"
    footer = rf"{author}(?:\s+(?:&|and)\s+{author}|,?\s+et al\.)?\s+(?:Standards Track|Informational|Experimental|Best Current Practice)\s+(?:\[Page \d+\]|Page \d+)"
    header = r"RFC\s+\d+\s+[\w /()-]+\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}"
    # Bound metadata spans before matching: the author/title character classes
    # also accept ordinary prose after whitespace normalization. A whole-unit
    # match alone therefore cannot establish that a long paragraph is chrome.
    chrome_only = len(words) <= 24 and len(lines) <= 3 and not re.findall(
        r"\b(?:must|should|shall|remembered|parameters|PROTOCOL_VIOLATION|0-RTT|requires?|receives?|sends?)\b", normalized, re.I)
    if chrome_only and (re.fullmatch(footer, normalized, re.I) or re.fullmatch(header, normalized, re.I)):
        return "EXCLUDE_LOW_INFORMATION", "RFC_PAGE_HEADER_OR_FOOTER"
    heading = reference_heading(lines[0])
    body = "\n".join(lines[1:]) if heading else text
    if heading and not body:
        return "KEEP", "REFERENCE_SECTION_HEADING"
    if heading and citation_only(body):
        return "EXCLUDE_LOW_INFORMATION", "REFERENCE_ONLY_REGION"
    if heading:
        return "REQUIRES_REVIEW", "REFERENCE_HEADING_WITH_POTENTIALLY_SUBSTANTIVE_TEXT"
    if citation_only(body):
        return "EXCLUDE_LOW_INFORMATION", "BIBLIOGRAPHIC_ENTRY_ONLY"
    if in_references and re.match(r"\s*\[[^\]]+\]", text):
        return "REQUIRES_REVIEW", "POSSIBLE_REFERENCE_WITH_MIXED_CONTENT"
    if all(line.lower() == "table of contents" or re.fullmatch(r".+\.{3,}\s*\d+", line) for line in lines):
        return "EXCLUDE_LOW_INFORMATION", "TABLE_OF_CONTENTS_ONLY"
    contact_heading = bool(re.fullmatch(r"authors?['’]? addresses|author (?:contact|affiliation)(?: information)?|contact directory", lines[0], re.I))
    contact_line = r"(?:e-?mail\s*:\s*)?[\w.+-]+@[\w.-]+\.[a-z]{2,}|(?:phone|fax|tel)\s*:\s*[+\d ()-]+|\d+\s+[\w .]+\s+(?:street|st\.?|road|rd\.?|avenue|ave\.?)"
    if all(re.fullmatch(contact_line, line, re.I) for line in lines):
        return "EXCLUDE_LOW_INFORMATION", "AUTHOR_ADDRESS_OR_DIRECTORY_ONLY"
    if contact_heading:
        # Names/affiliations may accompany contact lines; prose is ambiguous.
        has_contact = any(re.fullmatch(contact_line, line, re.I) for line in lines[1:])
        labels_only = all(re.fullmatch(contact_line, line, re.I) or re.fullmatch(r"[A-Z][\w.'’-]*(?: [A-Z][\w.'’-]*){0,5}", line) for line in lines[1:])
        if has_contact and labels_only:
            return "EXCLUDE_LOW_INFORMATION", "AUTHOR_ADDRESS_OR_DIRECTORY_ONLY"
        return "REQUIRES_REVIEW", "POSSIBLE_AUTHOR_CONTACT_BLOCK"
    if re.fullmatch(r"(?:copyright\s+(?:\(c\)\s*)?\d{4}\s+[\w .,-]+[.]?\s*)?all rights reserved[.]?", lower):
        return "EXCLUDE_LOW_INFORMATION", "LICENSE_OR_FOOTER_ONLY"
    return "KEEP", "SUBSTANTIVE_OR_NOT_DETERMINISTICALLY_LOW_INFORMATION"


def quality_rows(document_id, units, profile="legacy"):
    normalized = [" ".join(text.split()).casefold() for _, text in units]
    repeated = Counter(value for value in normalized if value)
    rows = []
    in_references = False
    for order, ((location, text), key) in enumerate(zip(units, normalized)):
        first = next((line.strip() for line in text.splitlines() if line.strip()), "")
        if reference_heading(first):
            in_references = True
        elif re.fullmatch(r"(?:\d+(?:\.\d+)*\.?|Appendix\s+[A-Z])\s+[^\n]+", text.strip()):
            in_references = False
        decision, reason = quality_decision(text, repeated[key] >= 3, in_references, profile)
        rows.append(dict(document_id=document_id, review_unit_id=f"{document_id}-QUALITY-{order:05d}", unit_order=order,
                         source_location=location, quality_decision=decision, quality_reason=reason,
                         unit_word_count=len(text.split()), text_preview=" ".join(text.split())[:300]))
        if profile != "legacy":
            rows[-1]["quality_profile"] = profile
    require(rows and any(r["quality_decision"] == "KEEP" for r in rows), "Quality review would remove the full document: " + document_id)
    return rows


def extract_local_pdf(root, source):
    path = v1.local_path(root, source["local_path"])
    with path.open("rb") as stream:
        magic = stream.read(5)
    require(path.suffix.lower() == ".pdf" and magic == b"%PDF-", "Invalid PDF magic: " + source["document_id"])
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ValueError("Local pypdf text-layer parser required; no installation or OCR performed") from exc
    reader = PdfReader(str(path))
    require(not reader.is_encrypted and len(reader.pages) >= 1, "Unreadable/encrypted/empty PDF")
    pages = [v1.normalize_text(page.extract_text() or "") for page in reader.pages]
    require(all(pages), "PDF contains a page without extractable text; manual review required, no OCR")
    text = v1.normalize_text("\n\n".join(pages))
    require(len(text) >= 2000 and len(text.split()) >= 500, "PDF extraction is not substantive")
    return text, [(f"page:{i+1}", page) for i, page in enumerate(pages)], f"pypdf deterministic text-layer extraction; pages={len(pages)}; no OCR"


def validated_v1(root):
    _, manifest_rows = v1.read_table(root, v1.MANIFEST, v1.MANIFEST_HEADER)
    _, document_rows = v1.read_table(root, v1.OUTPUTS[0], v1.DOCUMENT_HEADER)
    reviews = v1.read_reviews(root, manifest_rows)
    manifest = {r["document_id"]: r for r in manifest_rows}
    documents = {r["document_id"]: r for r in document_rows}
    require(len(manifest) == len(manifest_rows) and len(documents) == len(document_rows), "Duplicate V1 identities")
    trusted = []
    for identity, document in documents.items():
        source = manifest.get(identity)
        require(source is not None, "V1 document missing manifest linkage")
        if identity == "TOR-SPEC-001" or reviews.get(identity, {}).get("review_status") == "REJECTED":
            require(document["authorized_for_retrieval"] == "NO", "TOR-SPEC-001 rejection was overridden")
            continue
        if v1.trusted(document):
            require(v1.trusted(source) and v1.usable_document(document), "V1 trusted decision cannot be established")
            trusted.append(document)
    require(len(trusted) == 10, f"Expected 10 trusted V1 documents; found {len(trusted)}")
    require("TOR-SPEC-001" in manifest and "TOR-SPEC-001" not in {d["document_id"] for d in trusted}, "Rejected TOR source policy differs")
    return sorted(trusted, key=lambda r: r["document_id"]), manifest_rows, reviews


def build(root, profile="legacy", additional_manifest=None, additional_documents=None, development_id=None):
    require(profile in PROFILES, "Unknown quality profile")
    require((profile == "legacy") == (development_id is None), "Improved profiles require --development-id; legacy mode must omit it")
    require(bool(additional_manifest) == bool(additional_documents), "Both curated manifest and document CSVs required")
    output_paths(development_id)
    require(not additional_manifest or profile == "evidence_quality", "Source expansion requires explicit evidence_quality profile")
    before = file_metadata(root, V1_INPUTS)
    trusted, manifest_v1, reviews_v1 = validated_v1(root)
    manifest = [dict(row, authorized_for_retrieval="NO") if row["document_id"] == "TOR-SPEC-001" else dict(row) for row in manifest_v1 if row["document_id"] in {d["document_id"] for d in trusted} | {"TOR-SPEC-001"}]
    manifest.extend(dict(source) for source in NEW_SOURCES)
    require(len({r["document_id"] for r in manifest}) == len(manifest) == 13, "V2 manifest population differs")
    review = [dict(document_id=k, review_status=v["review_status"], review_notes=v["review_notes"]) for k, v in sorted(reviews_v1.items()) if k in {r["document_id"] for r in manifest}]
    review.extend(dict(document_id=s["document_id"], review_status="APPROVED", review_notes="Researcher manually inspected the local PDF and approved the substantive text-layer extraction candidate before implementation.") for s in NEW_SOURCES)
    documents, quality = [], []
    for old in trusted:
        document = dict(old)
        counts = term_counts(document["normalized_text"])
        document.update(corpus_revision="V2", is_new_document="NO", **counts, task_aligned="YES" if counts["term_malware_count"] or counts["term_malicious_count"] or counts["term_benign_count"] else "NO")
        documents.append(document)
        quality.extend(quality_rows(document["document_id"], source_units(document["normalized_text"]), profile))
    for source in NEW_SOURCES:
        text, units, notes = extract_local_pdf(root, source)
        counts = term_counts(text)
        require(counts["term_malware_count"] or counts["term_malicious_count"] or counts["term_benign_count"], "New source lacks validated task-alignment terms")
        document = dict(source, extraction_status="PASS", extraction_review_status="APPROVED", character_count=len(text), word_count=len(text.split()), normalized_text=text,
                        extraction_notes=notes+"; manual source-level extraction disposition=APPROVED", corpus_revision="V2", is_new_document="YES", **counts, task_aligned="YES")
        documents.append(document)
        quality.extend(quality_rows(source["document_id"], source_units(text, units), profile))
    require(len(documents) == 12 and len({d["document_id"] for d in documents}) == 12, "Expected 12 V2 candidate documents")
    require({s["document_id"] for s in NEW_SOURCES} <= {d["document_id"] for d in documents} and "TOR-SPEC-001" not in {d["document_id"] for d in documents}, "New/rejected source population differs")
    require(before == file_metadata(root, V1_INPUTS), "V1 inputs changed during V2 construction")
    if additional_manifest:
        before.extend(file_metadata(root, (additional_manifest, additional_documents)))
        _, added_sources = v1.read_table(root, additional_manifest, v1.MANIFEST_HEADER)
        _, added_docs = v1.read_table(root, additional_documents, v1.DOCUMENT_HEADER)
        if any(d["source_type"] == "FROZEN_TRAIN_EMPIRICAL_EVIDENCE" for d in added_docs):
            from src.rag.build_train_evidence_v5 import verify_documents
            verify_documents(root, development_id, added_docs)
        sources = {r["document_id"]:r for r in added_sources}
        require(len(sources) == len(added_sources) == len(added_docs) == len({r["document_id"] for r in added_docs}), "Duplicate curated identities")
        require(set(sources) == {r["document_id"] for r in added_docs} and not set(sources) & {r["document_id"] for r in manifest}, "Curated source linkage/collision")
        for doc in added_docs:
            source = sources[doc["document_id"]]
            require(v1.trusted(source) and v1.usable_document(doc) and doc["extraction_review_status"] != "REJECTED", "Unauthorized curated evidence")
            require(all(doc[k] == source[k] for k in v1.MANIFEST_HEADER), "Curated manifest/document differs")
            counts = term_counts(doc["normalized_text"])
            documents.append(dict(doc,corpus_revision="V2",is_new_document="YES",**counts,
                task_aligned="YES" if any(counts.values()) else "NO"))
            manifest.append(source)
            review.append(dict(document_id=doc["document_id"],review_status=doc["extraction_review_status"],
                review_notes="Disposition preserved from manually curated corpus; no new authorization inferred"))
            quality.extend(quality_rows(doc["document_id"],source_units(doc["normalized_text"]),profile))
        require(before == file_metadata(root,[r["path"] for r in before]), "Curated inputs changed")
    return documents, sorted(manifest, key=lambda r: r["document_id"]), sorted(review, key=lambda r: r["document_id"]), sorted(quality, key=lambda r: (r["document_id"], r["unit_order"])), before


def summary_rows(documents, quality, rejected=1):
    groups = [("overall", "", "", documents)]
    groups += [("source_type", source, "", [d for d in documents if d["source_type"] == source]) for source in sorted({d["source_type"] for d in documents})]
    groups += [("revision", "", value, [d for d in documents if d["is_new_document"] == value]) for value in ("NO", "YES")]
    rows = []
    for level, source, new, members in groups:
        ids = {d["document_id"] for d in members}
        units = [r for r in quality if r["document_id"] in ids]
        counts = Counter(r["quality_decision"] for r in units)
        values = (level, source, new, len(members), sum(d["corpus_partition"] == "trusted" and d["authorized_for_retrieval"] == "YES" for d in members), rejected if level == "overall" else 0,
                  sum(d["is_new_document"] == "YES" for d in members), sum(d["is_new_document"] == "YES" and d["source_type"] == "peer_reviewed_research" for d in members),
                  sum(int(d["word_count"]) for d in members), sum(d["task_aligned"] == "YES" for d in members),
                  sum(int(d["term_benign_count"]) > 0 for d in members), sum(int(d["term_malware_count"]) > 0 for d in members), sum(int(d["term_malicious_count"]) > 0 for d in members),
                  counts["KEEP"], counts["EXCLUDE_LOW_INFORMATION"], counts["REQUIRES_REVIEW"])
        rows.append(dict(zip(SUMMARY_HEADER, values)))
    return rows


def overwrite_guard(root, manifest):
    exists = [v1.local_path(root, path, must_exist=False).exists() for path in OUTPUTS]
    if any(exists):
        require(all(exists), "Incomplete existing V2 output set requires manual review")
        _, old = v1.read_table(root, OUTPUTS[1], v1.MANIFEST_HEADER)
        identity = lambda rows: sorted((r["document_id"], r["local_path"], r["source_provenance_status"], r["authorized_for_retrieval"]) for r in rows)
        require(identity(old) == identity(manifest), "V2 source identity differs; refusing silent overwrite")


def checks(documents, manifest, review, quality, before, profile="legacy", development_id=None, decision_claims=None):
    facts = {
        "v1_preservation": "V1 metadata/corpus artifacts read-only and unchanged: "+json.dumps(before, sort_keys=True),
        "v1_trusted_population": "All 10 effectively authorized V1 documents retained",
        "tor_spec_rejected": "TOR-SPEC-001 retained as rejected manifest/review provenance only; absent from V2 document corpus",
        "new_source_population": "Exactly RESEARCH-LICHY-2023 and RESEARCH-SHEKHAWAT-2019",
        "local_pdf_validation": "PDF magic, page count, non-empty every-page text layer and substantive counts PASS; no OCR",
        "manual_extraction_review": "Both new candidate source-level dispositions APPROVED; extraction status remains separately preserved",
        "bibliographic_policy": "Unknown publisher/authors/date/DOI/URL remain blank; no independent peer-review claim",
        "content_quality_review": f"Deterministic conservative units={len(quality)}; decisions="+json.dumps(Counter(r["quality_decision"] for r in quality), sort_keys=True),
        "full_text_preserved": "Quality decisions do not delete or rewrite evidence_documents_v2 normalized_text",
        "task_alignment": "Descriptive term counts only; no relevance labels or trust scores",
        "stage_boundary": "No index, embedding, retrieval, provenance verification, screening, reranking, LLM, network, traffic data or figures",
        "canonical_v2_outputs": ";".join(output_paths(development_id)),
    }
    require(all(r["quality_decision"] in QUALITY_DECISIONS for r in quality), "Invalid quality vocabulary")
    require((len(documents) == 12 if profile == "legacy" else len(documents) >= 12)
            and len(manifest) == len(documents)+1 and len(review) >= 3, "Output population differs")
    if profile != "legacy":
        facts.update(quality_profile=profile, new_source_population="Two existing approved sources plus explicitly supplied authorized curated corpus rows")
        if decision_claims is not None:
            facts.update(decision_claim_count=len(decision_claims),
                         decision_claim_output=decision_claim_path(development_id),
                         decision_claim_policy="Only manually supplied APPROVED and authorized claims; no text-derived claim inference")
    return [v1.check("corpus_v2", name, "PASS", detail) for name, detail in facts.items()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--profile", choices=PROFILES, default="legacy")
    parser.add_argument("--development-id")
    parser.add_argument("--additional-manifest", help="Optional manually curated manifest CSV using the existing schema")
    parser.add_argument("--additional-documents", help="Matching existing-schema extracted document CSV; no automatic authorization")
    parser.add_argument("--decision-claims", help="Optional manually curated decision-evidence claims CSV")
    args = parser.parse_args(argv)
    try:
        root = args.project_root.resolve(strict=True)
        paths = output_paths(args.development_id) if args.profile != "legacy" else OUTPUTS
        documents, manifest, review, quality, before = build(root,args.profile,args.additional_manifest,args.additional_documents,args.development_id)
        require(not args.decision_claims or (args.profile == "evidence_quality" and args.development_id),
                "--decision-claims requires evidence_quality and a nonempty --development-id")
        claims = None
        claim_output = None
        if args.decision_claims:
            before.extend(file_metadata(root, (args.decision_claims,)))
            claims = validate_decision_claim_rows(documents, quality, read_decision_claims(root, args.decision_claims))
            claim_output = decision_claim_path(args.development_id)
        summaries = summary_rows(documents, quality)
        validation = checks(documents, manifest, review, quality, before,args.profile,args.development_id,claims)
        require((args.profile == "legacy") == (args.development_id is None), "Improved profiles require --development-id; legacy mode must omit it")
        if args.profile == "legacy":
            overwrite_guard(root, manifest)
        else:
            targets = (*paths, *((claim_output,) if claim_output else ()))
            require(not any(v1.local_path(root,p,must_exist=False).exists() for p in targets), "Corpus/decision-claim output already exists; refusing overwrite")
        if not args.execute:
            counts = Counter(row["quality_decision"] for row in quality)
            for decision in ("KEEP", "EXCLUDE_LOW_INFORMATION", "REQUIRES_REVIEW"):
                label = "EXCLUDED" if decision == "EXCLUDE_LOW_INFORMATION" else decision
                print(f"QUALITY_{label}_UNITS={counts[decision]}")
            print("EVIDENCE_CORPUS_V2=READY_FOR_CONSTRUCTION\nV1_TRUSTED_DOCUMENTS=10\nNEW_APPROVED_SOURCES=2\nCANDIDATE_DOCUMENTS=12\nLOCAL_PDF_VALIDATION=PASS\nNETWORK_ACCESS=NO\nTEST_ACCESS=NO")
            return 0
        quality_header = QUALITY_HEADER if args.profile == "legacy" else IMPROVED_QUALITY_HEADER
        output_tables = list(zip(paths, (DOCUMENT_HEADER, v1.MANIFEST_HEADER, v1.REVIEW_HEADER, quality_header, SUMMARY_HEADER, CHECK_HEADER), (documents, manifest, review, quality, summaries, validation)))
        if claims is not None:
            output_tables.append((claim_output, DECISION_CLAIM_HEADER, claims))
        for path, header, rows in output_tables:
            target = v1.local_path(root, path, must_exist=False)
            target.parent.mkdir(parents=True, exist_ok=True)
            v1.write_table(target, header, rows, mode="w" if args.profile == "legacy" else "x")
        require(before == file_metadata(root, [r["path"] for r in before]), "Input artifacts changed while writing V2")
        print("EVIDENCE_CORPUS_V2=PASS\nTRUSTED_DOCUMENTS=12\nNEW_DOCUMENTS=2\nLOCAL_PDF_VALIDATION=PASS\nCONTENT_QUALITY_REVIEW=PASS\nNETWORK_ACCESS=NO\nTEST_ACCESS=NO\nNEXT_STAGE=EVIDENCE_INDEX_V2_DESIGN")
        return 0
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        print("EVIDENCE_CORPUS_V2=BLOCKED\n"+str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
