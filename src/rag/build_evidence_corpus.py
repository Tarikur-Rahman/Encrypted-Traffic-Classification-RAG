"""Construct a local, curated document corpus. No chunking or downstream execution.

PDF support is optional and uses pypdf's text layer, never OCR. DOCX extraction
uses the standard library ZIP/XML readers and includes body, headers, footers,
footnotes and endnotes. Unapproved/adversarial entries remain metadata-only.
Blank counts mean not extracted; they never masquerade as observed zero counts.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
import zipfile
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

from src.rag import phase2_foundation as foundation


MANIFEST = "metadata/rag/evidence_source_manifest.csv"
REVIEW = "metadata/rag/evidence_extraction_review.csv"
REVIEW_HEADER = ("document_id", "review_status", "review_notes")
REVIEW_STATUSES = {"APPROVED", "REQUIRES_REVIEW", "REJECTED"}
MANIFEST_HEADER = tuple("document_id title source_type publisher authors publication_date canonical_url doi local_path traffic_topics security_topics corpus_partition source_provenance_status authorized_for_retrieval curation_notes".split())
DOCUMENT_HEADER = MANIFEST_HEADER[:-1] + ("extraction_status", "extraction_review_status", "character_count", "word_count", "normalized_text", "curation_notes", "extraction_notes")
SUMMARY_HEADER = tuple("summary_level source_type corpus_partition source_provenance_status authorized_for_retrieval document_count total_word_count total_character_count extracted_document_count".split())
OUTPUTS = ("data/processed/rag/evidence_documents.csv", "metadata/rag/evidence_corpus_summary.csv", "metadata/rag/evidence_corpus_validation_checklist.csv")
FOUNDATION_FILES = ("evidence_corpus_contract.csv", "evidence_corpus_inventory.csv", "phase2_readiness_checklist.csv", "phase2_experiment_logging_contract.csv")
INTERFACE_FILES = ("ustc_validation_interface.csv", "cic_validation_interface.csv", "interface_summary.csv", "interface_validation_checklist.csv")
EXTENSIONS = {".pdf", ".txt", ".md", ".html", ".htm", ".docx"}
PROVENANCE = {"VERIFIED", "REQUIRES_REVIEW", "REJECTED"}
MINIMUM = ("document_id", "title", "source_type", "local_path", "corpus_partition", "source_provenance_status", "authorized_for_retrieval")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local_path(root, relative, *, must_exist=True):
    """Only contained regular files; reject symlinks, junctions and hardlinks."""
    rel = Path(relative)
    require(not rel.is_absolute() and ".." not in rel.parts, "Unsafe relative path")
    path = root / rel
    require(path.resolve().is_relative_to(root), "Path escapes project")
    for part in (path, *path.parents):
        if part == root:
            break
        require(not foundation.linked_path(part), "Linked paths prohibited")
    if path.exists():
        require(path.is_file() and path.stat().st_nlink == 1, "Not an unaliased regular file")
    elif must_exist:
        raise ValueError("Missing local file: " + relative)
    return path


def configure_csv_field_limit():
    """Allow full-document fields using the largest platform-supported limit.

    Some platforms expose a Python sys.maxsize larger than the C integer
    accepted by csv. Reduce only on overflow; never truncate input text.
    """
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return limit
        except OverflowError:
            limit //= 10
            if limit == 0:
                raise


def read_table(root, relative, header=None):
    configure_csv_field_limit()
    actual, rows = foundation.read_csv(local_path(root, relative))
    if header is not None:
        require(actual == list(header), "Invalid schema: " + relative)
    return actual, rows


def check(section, name, status, details):
    return dict(section=section, check=name, status=status, details=details)


def prerequisites(root):
    headers = (foundation.EVIDENCE_HEADER, foundation.INVENTORY_HEADER, foundation.CHECKLIST_HEADER, foundation.LOG_HEADER)
    for name, header in zip(FOUNDATION_FILES, headers):
        # Inventory may correctly be empty; contracts and readiness may not.
        _, rows = read_table(root, "metadata/rag/" + name, header)
        require(rows or name == "evidence_corpus_inventory.csv", "Empty foundation contract: " + name)
    for name in INTERFACE_FILES:
        # Existence checks protect large frozen observation CSVs from rereading.
        local_path(root, "results/rag/interface/" + name)
    _, rows = read_table(root, "results/rag/interface/interface_validation_checklist.csv", foundation.CHECKLIST_HEADER)
    require(rows and all(r["status"] == "PASS" for r in rows), "Interface materialization not PASS")
    return [check("protection", "foundation_contracts_exist", "PASS", ", ".join(FOUNDATION_FILES)),
            check("protection", "interface_artifacts_exist", "PASS", "Four canonical artifacts present; validation checklist PASS")]


def trusted(row):
    return row["corpus_partition"] == "trusted" and row["source_provenance_status"] == "VERIFIED" and row["authorized_for_retrieval"] == "YES"


def read_reviews(root, rows):
    path = local_path(root, REVIEW, must_exist=False)
    if not path.exists():
        return {}
    _, records = read_table(root, REVIEW, REVIEW_HEADER)
    known = {row["document_id"] for row in rows}
    reviews = {}
    for record in records:
        identity = record["document_id"]
        require(identity in known, "Review references unknown or blank document ID: " + identity)
        require(identity not in reviews, "Duplicate extraction review: " + identity)
        require(record["review_status"] in REVIEW_STATUSES, "Invalid extraction review status")
        reviews[identity] = record
    return reviews


def usable_document(row):
    return trusted(row) and bool(row["normalized_text"]) and (
        row["extraction_status"] == "PASS" or
        (row["extraction_status"] == "REQUIRES_REVIEW" and row["extraction_review_status"] == "APPROVED"))


def review_findings(root, rows, reviews):
    """Report prior extraction outcomes without parsing source documents in dry mode."""
    previous = {}
    path = local_path(root, OUTPUTS[0], must_exist=False)
    if path.exists():
        _, saved = read_table(root, OUTPUTS[0])
        previous = {r["document_id"]: r for r in saved}
    findings = []
    for row in rows:
        if not trusted(row):
            continue
        identity = row["document_id"]
        review = reviews.get(identity, {}).get("review_status", "MISSING")
        old = previous.get(identity)
        matches = old and all(old.get(k) == row[k] for k in MANIFEST_HEADER if k != "authorized_for_retrieval")
        if matches and old.get("extraction_status") == "REQUIRES_REVIEW":
            detail = "Prior extraction requires manual completeness review; review_status=" + review + ". Current extraction will be revalidated during execute."
        elif matches and old.get("extraction_status") == "FAILED":
            detail = "Prior extraction FAILED; manual approval cannot authorize failed extraction."
        else:
            detail = "review_status=" + review + "; extraction completeness is checked during execute; any REQUIRES_REVIEW outcome needs explicit APPROVED review and non-empty text."
        findings.append(check(identity, "extraction_review_status", "WARN", detail))
    return findings


def source_path(root, row):
    relative = Path(row["local_path"])
    allowed = Path("data/rag/evidence_sources") / row["corpus_partition"]
    require(relative.is_relative_to(allowed) and relative != allowed, "Source partition/path mismatch")
    require(relative.suffix.lower() in EXTENSIONS, "Unsupported source extension")
    return local_path(root, row["local_path"])


def inspect(root):
    checks = prerequisites(root)
    path = local_path(root, MANIFEST, must_exist=False)
    if not path.exists():
        checks.append(check("manifest", "manifest_exists", "BLOCKED", "Manual corpus curation required"))
        return [], checks
    _, rows = read_table(root, MANIFEST, MANIFEST_HEADER)
    checks.append(check("manifest", "schema", "PASS", "Exact curated manifest columns"))
    ids, paths = set(), set()
    for row in rows:
        name = row["document_id"]
        require(all(row[f].strip() for f in MINIMUM), "Missing minimum metadata: " + name)
        require(re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)+", name) is not None, "Use a readable curated document ID: " + name)
        require(re.fullmatch(r"[a-fA-F0-9]{32,128}", name.replace("-", "")) is None, "Digest-like document IDs prohibited")
        require(name not in ids, "Duplicate document ID: " + name)
        ids.add(name)
        require(row["source_type"] in foundation.SOURCE_CATEGORIES, "Unsupported source type")
        require(row["corpus_partition"] in {"trusted", "adversarial"}, "Invalid corpus partition")
        require(row["source_provenance_status"] in PROVENANCE, "Invalid provenance status")
        require(row["authorized_for_retrieval"] in {"YES", "NO"}, "Invalid authorization")
        require(row["authorized_for_retrieval"] != "YES" or trusted(row), "Only VERIFIED trusted documents may be authorized")
        try:
            source = source_path(root, row)
            identity = str(source.resolve()).casefold()
            require(identity not in paths, "Duplicate local source path")
            paths.add(identity)
            checks.append(check(name, "local_source", "PASS", "Supported local source; explicit partition matches manifest"))
        except ValueError as exc:
            checks.append(check(name, "local_source", "BLOCKED", str(exc)))
        missing = [f for f in ("publisher", "authors", "publication_date", "canonical_url", "doi") if not row[f].strip()]
        if missing:
            checks.append(check(name, "bibliographic_completeness", "WARN", "Missing values preserved blank: " + ", ".join(missing)))
        if row["source_type"] == "peer_reviewed_research" and (not row["authors"] or not row["publication_date"] or not (row["doi"] or row["canonical_url"])):
            checks.append(check(name, "research_metadata", "WARN", "Authors, publication date and DOI or canonical URL preferred; curator review required"))
        if row["source_type"] == "standards_rfc" and not (row["publisher"] and row["canonical_url"]):
            checks.append(check(name, "standards_metadata", "WARN", "Publisher and canonical URL normally expected; no values inferred"))
        if trusted(row) and Path(row["local_path"]).suffix.lower() == ".pdf" and importlib.util.find_spec("pypdf") is None:
            checks.append(check(name, "pdf_parser", "BLOCKED", "Optional local pypdf parser unavailable; no installation or OCR performed"))
    for name in ("document_ids_unique", "no_blank_document_ids", "allowed_source_types", "explicit_partition", "trusted_adversarial_separation", "explicit_provenance_status", "explicit_authorization", "authorized_trusted_verified"):
        checks.append(check("manifest", name, "PASS", "Validated every curated record; directory/name confers no authority"))
    if not any(trusted(row) for row in rows):
        checks.append(check("manifest", "usable_trusted_sources", "BLOCKED", "No VERIFIED authorized trusted sources; curate manifest manually"))
    return rows, checks


def normalize_text(text):
    """Keep paragraph breaks and Unicode content; never summarize or truncate."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n\n")
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


class LocalHTML(HTMLParser):
    BLOCKS = {"p", "div", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "blockquote", "pre", "header", "footer"}
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0
        self.omitted = 0
        self.embedded = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"img", "iframe", "object", "embed", "canvas", "svg", "video", "audio"}:
            self.embedded += 1
        if tag in {"script", "style"}:
            self.hidden += 1
        elif not self.hidden and tag in self.BLOCKS | {"br"}:
            self.parts.append("\n\n" if tag in self.BLOCKS else "\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        elif not self.hidden and tag in self.BLOCKS:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if self.hidden:
            self.omitted += len(data)
        else:
            self.parts.append(data)


def extract_pdf(path):
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        return "REQUIRES_REVIEW", "", "Encrypted PDF; no decryption attempted"
    pages = [page.extract_text() or "" for page in reader.pages]
    empty = sum(not p.strip() for p in pages)
    # Any empty page might contain image-only evidence: do not silently approve.
    status = "PASS" if pages and not empty else "REQUIRES_REVIEW"
    return status, "\n\n".join(pages), f"pypdf text-layer extraction; pages={len(pages)}; empty_pages={empty}; no OCR"


def extract_docx(path):
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    texts, notes, review = [], [], False
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require("word/document.xml" in names, "DOCX body missing")
        parts = ["word/document.xml"] + sorted(n for n in names if re.fullmatch(r"word/(?:header\d+|footer\d+|footnotes|endnotes)\.xml", n))
        for part in parts:
            raw = archive.read(part)
            require(b"<!DOCTYPE" not in raw.upper() and b"<!ENTITY" not in raw.upper(), "XML entities prohibited")
            tree = ElementTree.fromstring(raw)
            for paragraph in tree.iter(ns + "p"):
                text = "".join((node.text or "") if node.tag == ns + "t" else ("\n" if node.tag in {ns + "br", ns + "cr"} else "\t" if node.tag == ns + "tab" else "") for node in paragraph.iter())
                texts.append(text)
            if any(node.tag in {ns + "drawing", ns + "pict", ns + "object", ns + "altChunk", ns + "del", ns + "ins"} for node in tree.iter()):
                review = True
        notes.append("ZIP/XML text extraction; parts=" + ",".join(parts))
        if review:
            notes.append("Images, embedded content or tracked changes require manual completeness review; no OCR")
    return "REQUIRES_REVIEW" if review else "PASS", "\n\n".join(texts), "; ".join(notes)


def extract(path):
    try:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            status, text, notes = extract_pdf(path)
        elif suffix == ".docx":
            status, text, notes = extract_docx(path)
        elif suffix in {".html", ".htm"}:
            parser = LocalHTML()
            parser.feed(path.read_text(encoding="utf-8-sig"))
            parser.close()
            status, text = "PASS", "".join(parser.parts)
            notes = f"HTML text extraction; omitted script/style characters={parser.omitted}; UTF-8 strict"
            if parser.hidden:
                status = "REQUIRES_REVIEW"
                notes += "; unclosed script/style region"
            if parser.embedded:
                status = "REQUIRES_REVIEW"
                notes += f"; embedded/non-text elements={parser.embedded}; manual completeness review required; external resources never fetched"
        else:
            require(suffix in {".txt", ".md"}, "Unsupported source extension")
            status, text, notes = "PASS", path.read_text(encoding="utf-8-sig"), "UTF-8 strict; Markdown markup retained; no rewriting"
        text = normalize_text(text)
        if not text:
            status = "REQUIRES_REVIEW"
            notes += "; no extractable text"
        return status, text, notes
    except ImportError:
        return "REQUIRES_REVIEW", "", "Optional pypdf parser unavailable; no installation or OCR"
    except Exception as exc:
        # Extraction errors are persisted, never represented as successful text.
        return "FAILED", "", type(exc).__name__ + ": " + str(exc)


def documents(root, rows, reviews=None):
    if reviews is None:
        reviews = read_reviews(root, rows)
    output = []
    for row in rows:
        document = dict(row)
        review = reviews.get(row["document_id"], {})
        document["extraction_review_status"] = review.get("review_status", "MISSING")
        if trusted(row):
            status, text, notes = extract(source_path(root, row))
            document.update(extraction_status=status, normalized_text=text, extraction_notes=notes,
                            character_count=len(text) if status != "FAILED" else "",
                            word_count=len(text.split()) if status != "FAILED" else "")
            if not usable_document(document):
                document["authorized_for_retrieval"] = "NO"
                document["extraction_notes"] += "; manifest authorization=YES; effective corpus authorization withheld pending extraction review"
            elif status == "REQUIRES_REVIEW":
                document["extraction_notes"] += "; manual completeness review APPROVED; original extraction status preserved"
        else:
            document.update(extraction_status="REQUIRES_REVIEW", normalized_text="", character_count="", word_count="",
                            extraction_notes="Not ingested: non-authorized or adversarial record; metadata preserved only")
        if review:
            document["extraction_notes"] += "; curated review notes: " + review["review_notes"]
        output.append(document)
    return output


def summarize(rows):
    dimensions = ("source_type", "corpus_partition", "source_provenance_status", "authorized_for_retrieval")
    groups = defaultdict(list)
    for row in rows:
        groups[("overall", "", "", "", "")].append(row)
        groups[("source_partition_provenance_authorization", *(row[d] for d in dimensions))].append(row)
    output = []
    for key, members in sorted(groups.items()):
        measured = [r for r in members if r["character_count"] != ""]
        result = dict(zip(("summary_level", *dimensions), key))
        result.update(document_count=len(members), extracted_document_count=len(measured),
                      total_word_count=sum(r["word_count"] for r in measured) if measured else "",
                      total_character_count=sum(r["character_count"] for r in measured) if measured else "")
        output.append(result)
    return output


def source_set_identity(rows):
    return json.dumps(sorted(rows, key=lambda r: r["document_id"]), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def output_preflight(root, rows):
    existing = [local_path(root, name, must_exist=False).exists() for name in OUTPUTS]
    if any(existing):
        require(all(existing), "Incomplete existing output set; explicit manual review required before overwrite")
        _, checks = read_table(root, OUTPUTS[2], foundation.CHECKLIST_HEADER)
        identities = [r["details"] for r in checks if r["check"] == "curated_source_set_identity"]
        require(identities == [source_set_identity(rows)], "Different curated manifest/source set; refusing silent overwrite")


def write_table(path, header, rows, mode="w"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def initialize(root):
    path = local_path(root, MANIFEST, must_exist=False)
    require(not path.exists(), "Manifest already exists; never overwritten")
    write_table(path, MANIFEST_HEADER, [], mode="x")


def initialize_review(root):
    path = local_path(root, REVIEW, must_exist=False)
    require(not path.exists(), "Review file already exists; never overwritten")
    write_table(path, REVIEW_HEADER, [], mode="x")


def execute(root, rows, checks):
    output_preflight(root, rows)
    reviews = read_reviews(root, rows)
    corpus = documents(root, rows, reviews)
    for row in corpus:
        checks.append(check(row["document_id"], "text_extraction_status", "PASS" if row["extraction_status"] == "PASS" else "WARN", row["extraction_status"] + ": " + row["extraction_notes"]))
        checks.append(check(row["document_id"], "extraction_review_status", "PASS" if usable_document(row) else "WARN", row["extraction_review_status"] + "; extraction=" + row["extraction_status"] + "; effective_authorization=" + row["authorized_for_retrieval"]))
    usable = sum(usable_document(r) for r in corpus)
    checks.append(check("corpus", "nonempty_usable_documents", "PASS" if usable else "BLOCKED", f"Authorized VERIFIED trusted documents with PASS extraction or APPROVED non-empty REQUIRES_REVIEW extraction: {usable}"))
    checks.append(check("identity", "curated_source_set_identity", "PASS", source_set_identity(rows)))
    for name, detail in {
        "no_automatically_inferred_trust": "Trusted partition, manifest authorization and VERIFIED status required; non-empty REQUIRES_REVIEW text additionally requires curated APPROVED completeness review; FAILED extraction never authorized",
        "no_fabricated_metadata": "Curated values preserved; missing bibliographic values blank",
        "no_phase_i_model_loading": "No model dependency or loading path",
        "no_phase_i_inference": "Existing interface artifacts only",
        "no_test_access": "No traffic dataset access",
        "no_ocr": "Text-layer extraction only", "no_network": "Local files only",
        "no_llm": "No rewriting or summarization", "no_embeddings": "Document text only",
        "no_retrieval": "No search execution", "no_indexing": "No index creation",
        "no_chunking": "Normalized full-document text", "no_hashes": "Manually curated readable IDs",
        "canonical_output_paths_only": ";".join(OUTPUTS),
        "figure_reproducibility": "Saved topic/bibliographic fields, document counts, numeric word/character counts; blank means unmeasured; aggregate totals cover extracted_document_count only",
    }.items():
        checks.append(check("policy", name, "PASS", detail))
    for name, header, data in zip(OUTPUTS, (DOCUMENT_HEADER, SUMMARY_HEADER, foundation.CHECKLIST_HEADER), (corpus, summarize(corpus), checks)):
        write_table(local_path(root, name, must_exist=False), header, data)
    return usable, sum(r["corpus_partition"] == "adversarial" for r in corpus)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--init-manifest", action="store_true")
    modes.add_argument("--init-review", action="store_true")
    args = parser.parse_args(argv)
    try:
        root = args.project_root.resolve(strict=True)
        if args.init_review:
            initialize_review(root)
            print("EVIDENCE_EXTRACTION_REVIEW=INITIALIZED")
            return 0
        if args.init_manifest:
            initialize(root)
            print("EVIDENCE_SOURCE_MANIFEST=INITIALIZED")
            return 0
        rows, checks = inspect(root)
        reviews = read_reviews(root, rows)
        checks.extend(review_findings(root, rows, reviews))
        for item in checks:
            print(f"{item['status']} {item['section']}/{item['check']}: {item['details']}")
        if any(item["status"] == "BLOCKED" for item in checks):
            print("EVIDENCE_CORPUS=BLOCKED_NEEDS_CURATED_SOURCES")
            return 2
        output_preflight(root, rows)
        if not args.execute:
            print("EVIDENCE_CORPUS=READY_FOR_CONSTRUCTION")
            return 0
        usable, adversarial = execute(root, rows, checks)
        if not usable:
            print("EVIDENCE_CORPUS=BLOCKED_NEEDS_CURATED_SOURCES")
            return 2
        print(f"EVIDENCE_CORPUS=PASS\nAUTHORIZED_TRUSTED_DOCUMENTS={usable}\nADVERSARIAL_DOCUMENTS={adversarial}\nTEST_ACCESS=NO\nNEXT_STAGE=EVIDENCE_INDEXING")
        return 0
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        print("EVIDENCE_CORPUS=BLOCKED_NEEDS_CURATED_SOURCES\n" + str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
