"""Synthetic Corpus V2 tests; no real PDF extraction or execution here."""
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.rag import build_evidence_corpus_v2 as v2
from src.rag import build_evidence_corpus as v1


def snapshot(root, names=None):
    paths = (root/name for name in names) if names else root.rglob("*")
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in paths if p.is_file()}


@pytest.fixture
def project(tmp_path, monkeypatch):
    manifest, documents, reviews = [], [], []
    for i in range(11):
        identity = "TOR-SPEC-001" if i == 10 else f"DOC-{i:03d}"
        source = dict.fromkeys(v1.MANIFEST_HEADER, "")
        source.update(document_id=identity, title="Synthetic trusted evidence", source_type="standards_rfc", publisher="Synthetic publisher",
                      local_path=f"data/rag/evidence_sources/trusted/{identity}.txt", traffic_topics="TLS;encrypted_traffic", security_topics="traffic_classification",
                      corpus_partition="trusted", source_provenance_status="VERIFIED", authorized_for_retrieval="YES")
        manifest.append(source)
        text = "Substantive encrypted traffic security discussion about TLS and attack behavior.\n\nTechnical protocol evidence for traffic analysis."
        document = dict(source, extraction_status="PASS", extraction_review_status="MISSING", character_count=len(text), word_count=len(text.split()), normalized_text=text, extraction_notes="Synthetic local extraction")
        if identity == "TOR-SPEC-001":
            document.update(authorized_for_retrieval="NO", extraction_status="REQUIRES_REVIEW", extraction_review_status="REJECTED")
            reviews.append(dict(document_id=identity, review_status="REJECTED", review_notes="Insufficient extraction"))
        documents.append(document)
    v1.write_table(tmp_path/v1.MANIFEST, v1.MANIFEST_HEADER, manifest)
    v1.write_table(tmp_path/v1.REVIEW, v1.REVIEW_HEADER, reviews)
    v1.write_table(tmp_path/v1.OUTPUTS[0], v1.DOCUMENT_HEADER, documents)
    v1.write_table(tmp_path/v1.OUTPUTS[1], v1.SUMMARY_HEADER, v1.summarize(documents))
    v1.write_table(tmp_path/v1.OUTPUTS[2], v1.foundation.CHECKLIST_HEADER, [v1.check("corpus", "nonempty_usable_documents", "PASS", "10")])
    for source in v2.NEW_SOURCES:
        path = tmp_path/source["local_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-synthetic-local-source")
    def extract(root, source):
        term = "malware" if source["document_id"] == "RESEARCH-LICHY-2023" else "malicious benign"
        paragraph = (f"Substantive encrypted traffic analysis and {term} classification evidence. " * 100).strip()
        return paragraph, [("page:1", paragraph)], "Synthetic deterministic text-layer extraction; pages=1; no OCR"
    monkeypatch.setattr(v2, "extract_local_pdf", extract)
    return tmp_path


def test_v1_trusted_population_and_tor_rejection(project):
    trusted, manifest, reviews = v2.validated_v1(project)
    assert len(trusted) == 10
    assert "TOR-SPEC-001" not in {r["document_id"] for r in trusted}
    assert next(r for r in manifest if r["document_id"] == "TOR-SPEC-001")["authorized_for_retrieval"] == "YES"
    assert reviews["TOR-SPEC-001"]["review_status"] == "REJECTED"


def test_two_stable_new_sources_without_invented_bibliography():
    assert [s["document_id"] for s in v2.NEW_SOURCES] == ["RESEARCH-LICHY-2023", "RESEARCH-SHEKHAWAT-2019"]
    assert all(s["source_type"] == "peer_reviewed_research" and s["corpus_partition"] == "trusted" for s in v2.NEW_SOURCES)
    assert all(s["publisher"] == s["authors"] == s["publication_date"] == s["canonical_url"] == s["doi"] == "" for s in v2.NEW_SOURCES)
    assert not {"USTC-WANG-2017", "ANDERSON-MCGREW-2017"} & {s["document_id"] for s in v2.NEW_SOURCES}


def test_build_population_metadata_and_full_text(project):
    documents, manifest, reviews, quality, _ = v2.build(project)
    assert len(documents) == 12 and len(manifest) == 13
    assert {s["document_id"] for s in v2.NEW_SOURCES} <= {d["document_id"] for d in documents}
    assert "TOR-SPEC-001" not in {d["document_id"] for d in documents}
    rejected = next(r for r in manifest if r["document_id"] == "TOR-SPEC-001")
    assert rejected["authorized_for_retrieval"] == "NO"
    assert next(r for r in reviews if r["document_id"] == "TOR-SPEC-001")["review_status"] == "REJECTED"
    for identity in ("RESEARCH-LICHY-2023", "RESEARCH-SHEKHAWAT-2019"):
        document = next(d for d in documents if d["document_id"] == identity)
        assert document["extraction_status"] == document["extraction_review_status"] == "APPROVED" or document["extraction_status"] == "PASS"
        assert document["extraction_status"] == "PASS" and document["extraction_review_status"] == "APPROVED"
        assert document["normalized_text"] and document["is_new_document"] == document["task_aligned"] == "YES"
        assert any(r["document_id"] == identity and r["quality_decision"] == "KEEP" for r in quality)
    assert all(set(d) == set(v2.DOCUMENT_HEADER) for d in documents)


def test_missing_candidate_file_and_bad_magic_block(project, monkeypatch):
    monkeypatch.undo()
    source = v2.NEW_SOURCES[0]
    (project/source["local_path"]).unlink()
    with pytest.raises((ValueError, OSError)):
        v2.extract_local_pdf(project, source)
    (project/source["local_path"]).write_bytes(b"not a PDF")
    with pytest.raises(ValueError, match="magic"):
        v2.extract_local_pdf(project, source)


def test_pdf_validation_empty_and_deterministic(project, monkeypatch):
    monkeypatch.undo()
    source = v2.NEW_SOURCES[0]
    class Reader:
        is_encrypted = False
        pages = [SimpleNamespace(extract_text=lambda: "")]
        def __init__(self, path): pass
    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=Reader))
    with pytest.raises(ValueError, match="without extractable"):
        v2.extract_local_pdf(project, source)
    paragraph = "Encrypted malware traffic analysis evidence. " * 150
    Reader.pages = [SimpleNamespace(extract_text=lambda: paragraph), SimpleNamespace(extract_text=lambda: paragraph)]
    first = v2.extract_local_pdf(project, source)
    second = v2.extract_local_pdf(project, source)
    assert first == second and first[1][0][0] == "page:1" and len(first[1]) == 2


@pytest.mark.parametrize("text,decision,reason", [
    ("Skip to Main Content Global Navigation", "EXCLUDE_LOW_INFORMATION", "NAVIGATION_BOILERPLATE"),
    ("References\n[1] Author 2020 https://example.invalid\n[2] Author 2021 doi:10.1234/example", "EXCLUDE_LOW_INFORMATION", "REFERENCE_ONLY_REGION"),
    ("References are important to this substantive encrypted traffic malware attack analysis and explain the experimental protocol in detail.", "KEEP", "SUBSTANTIVE_OR_NOT_DETERMINISTICALLY_LOW_INFORMATION"),
    ("References\nThis encrypted traffic analysis discusses malware, malicious attack behavior, TLS and VPN evidence in detail. [1] 2020", "REQUIRES_REVIEW", "REFERENCE_HEADING_WITH_POTENTIALLY_SUBSTANTIVE_TEXT"),
])
def test_conservative_quality_decisions(text, decision, reason):
    assert v2.quality_decision(text) == (decision, reason)


def test_repeated_headers_detected_and_no_full_document_deletion():
    units = [(f"page:{i}", "RFC 9000 QUIC Transport Protocol May 2021") for i in range(3)] + [("page:1", "Substantive encrypted traffic malware analysis with technical evidence and protocol behavior.")]
    rows = v2.quality_rows("PAPER-001", units)
    assert sum(r["quality_reason"] == "RFC_PAGE_HEADER_OR_FOOTER" for r in rows) == 3
    assert any(r["quality_decision"] == "KEEP" for r in rows)
    with pytest.raises(ValueError, match="full document"):
        v2.quality_rows("EMPTY-001", [("page:1", "Iyengar & Thomson Standards Track Page 7")])


@pytest.mark.parametrize("text", [
    "Darknet traffic includes Tor and VPN traffic. The dataset contains benign applications and supports malware monitoring and detection.",
    "IP address", "relay address", "source address", "destination address",
    "IPv4 address: four octets", "IPv6 address: sixteen octets", "address validation",
    "Introduction", "Dataset details", "TLS handshake", "Channel lifetime",
    "Creating circuits", "Forward Direction", "Backward Direction",
    "Field / Size / Description", "CircID", "Command", "Body", "ATYPEDescription",
    "RELAY", "CREATE2", "CREATED2", "CERTS", "AUTH_CHALLENGE", "AUTHENTICATE",
    "TLS_AES_128_GCM_SHA256", "QUIC CONNECTION_CLOSE 0x1c",
    "Channel negotiation uses TLS to authenticate the peer address.",
    "Circuit creation uses CREATE2 and CREATED2 cells.",
    "Relay-cell routing forwards cells along the circuit to the next relay address.",
    "TLS [RFC8446] provides encryption; QUIC address validation requires a token [RFC9000].",
    "References to other RFCs explain the protocol's security properties.",
    "Skip to Main Content\nDarknet traffic includes benign VPN and Tor applications.",
    "Iyengar & Thomson Standards Track Page 7\nQUIC validates the peer address.",
])
def test_substantive_short_and_mixed_units_retained(text):
    assert v2.quality_decision(text)[0] == "KEEP"
    assert v2.quality_decision(text, repeated=True)[0] == "KEEP"


@pytest.mark.parametrize("text", [
    "Skip to Main Content", "Skip to Global Navigation", "Toggle navigation",
    "UNB Phone Directory", "Give to UNB", "Campus Maps", "Careers at UNB", "Web feedback",
    "Authors' Addresses\nJane Doe\nExample University\n123 Main Street\nEmail: jane@example.invalid",
    "Iyengar & Thomson Standards Track Page 7",
])
def test_strong_standalone_boilerplate_excluded(text):
    assert v2.quality_decision(text)[0] == "EXCLUDE_LOW_INFORMATION"


@pytest.mark.parametrize("text", [
    "Search", "Apply", "About", "Datasets", "Publications", "Members",
    "Contact Us", "Toggle navigation", "Skip to navigation",
])
def test_evidence_quality_profile_excludes_navigation_only_units(text):
    assert v2.quality_decision(text,profile="evidence_quality") == ("EXCLUDE_LOW_INFORMATION","NAVIGATION_UI_ONLY")


def test_evidence_quality_profile_preserves_mixed_source_text_exactly():
    text = "About\n\nEncrypted flow duration is associated with the reported malware classification in this experiment."
    rows = v2.quality_rows("CURATED-001", [("page:2",text)],profile="evidence_quality")
    assert rows[0]["quality_decision"] == "KEEP"
    assert text == "About\n\nEncrypted flow duration is associated with the reported malware classification in this experiment."
    assert rows[0]["source_location"] == "page:2"
    assert rows[0]["quality_profile"] == "evidence_quality"


def test_namespaced_corpus_outputs_and_safe_development_id():
    assert v2.output_paths() == v2.OUTPUTS
    paths = v2.output_paths("v5_validation")
    assert paths[0] == "data/processed/rag/v5_validation/evidence_documents_v2.csv"
    assert all("/v5_validation/" in path and "/test/" not in path.lower() for path in paths)
    with pytest.raises(ValueError):
        v2.output_paths("final_test")
    with pytest.raises(ValueError):
        v2.build(Path("."), profile="evidence_quality")


def _decision_claim_fixture(
    quote="The bidirectional duration was typically observed in the benign traffic analysis.",
    *,
    source_dataset="USTC-TFC2016",
    canonical_feature="bidirectional duration ms",
    supported_class="benign",
    contrasted_class="malware",
):
    identity = "CLAIM-DOC-001"
    text = "Substantive network traffic analysis describes packet behavior.\n\n" + quote
    source = dict.fromkeys(v1.MANIFEST_HEADER, "")
    source.update(document_id=identity, title="Reviewed traffic analysis", source_type="standards_rfc",
                  local_path="data/raw/rag/claim_fixture.txt", corpus_partition="trusted",
                  source_provenance_status="VERIFIED", authorized_for_retrieval="YES")
    document = dict(source, normalized_text=text, extraction_status="PASS", extraction_review_status="APPROVED")
    units = v2.source_units(text)
    quality = v2.quality_rows(identity, units, profile="evidence_quality")
    quote_locations = [location for location, unit_text in units if quote in unit_text]
    assert len(quote_locations) == 1
    claim = dict(claim_id="CLAIM-001", document_id=identity, source_dataset=source_dataset,
                 source_location=quote_locations[0], evidence_quote=quote, canonical_feature=canonical_feature,
                 direction="typical", supported_class=supported_class, contrasted_class=contrasted_class,
                 review_status="APPROVED", authorized_for_decision_evidence="YES")
    return document, quality, claim


def test_manually_curated_decision_claim_passes_without_inference():
    document, quality, claim = _decision_claim_fixture()
    assert v2.validate_decision_claim_rows([document], quality, [claim]) == [claim]
    assert v2.validate_decision_claim_rows([document], quality, []) == []


@pytest.mark.parametrize("mutation", ["duplicate", "absent_quote", "non_keep_quote", "wrong_class", "same_class", "review", "authorization"])
def test_invalid_decision_claims_fail_closed(mutation):
    document, quality, claim = _decision_claim_fixture()
    claims = [dict(claim)]
    if mutation == "duplicate": claims.append(dict(claim))
    elif mutation == "absent_quote": claims[0]["evidence_quote"] = "Not present in this document."
    elif mutation == "non_keep_quote":
        document, quality, claims[0] = _decision_claim_fixture("Skip to Main Content")
        claims[0]["evidence_quote"] = "Skip to Main Content"
    elif mutation == "wrong_class": claims[0]["supported_class"] = "Tor"
    elif mutation == "same_class": claims[0]["contrasted_class"] = "benign"
    elif mutation == "review": claims[0]["review_status"] = "REQUIRES_REVIEW"
    elif mutation == "authorization": claims[0]["authorized_for_decision_evidence"] = "NO"
    with pytest.raises(ValueError):
        v2.validate_decision_claim_rows([document], quality, claims)


def test_exact_ustc_canonical_feature_passes():
    document, quality, claim = _decision_claim_fixture()
    assert claim["canonical_feature"] == "bidirectional duration ms"
    assert v2.validate_decision_claim_rows([document], quality, [claim]) == [claim]


def test_exact_cic_canonical_feature_passes():
    document, quality, claim = _decision_claim_fixture(
        quote="The flow duration was typically observed in the encrypted traffic analysis.",
        source_dataset="CIC-Darknet2020",
        canonical_feature="flow duration",
        supported_class="Non-Tor",
        contrasted_class="Tor",
    )
    assert v2.validate_decision_claim_rows([document], quality, [claim]) == [claim]


@pytest.mark.parametrize(
    "feature",
    ["not a canonical traffic feature", "Flow Duration", "Flow IAT Mean"],
)
def test_noncanonical_decision_features_fail_closed(feature):
    document, quality, claim = _decision_claim_fixture(canonical_feature=feature)
    with pytest.raises(ValueError):
        v2.validate_decision_claim_rows([document], quality, [claim])


def test_ustc_feature_under_cic_dataset_fails():
    document, quality, claim = _decision_claim_fixture(
        source_dataset="CIC-Darknet2020",
        canonical_feature="bidirectional duration ms",
        supported_class="Non-Tor",
        contrasted_class="Tor",
    )
    with pytest.raises(ValueError):
        v2.validate_decision_claim_rows([document], quality, [claim])


def test_cic_feature_under_ustc_dataset_fails():
    document, quality, claim = _decision_claim_fixture(canonical_feature="flow duration")
    with pytest.raises(ValueError):
        v2.validate_decision_claim_rows([document], quality, [claim])


def test_wrong_decision_claim_source_location_fails():
    document, quality, claim = _decision_claim_fixture()
    claim["source_location"] = "not-the-quoted-keep-unit"
    with pytest.raises(ValueError):
        v2.validate_decision_claim_rows([document], quality, [claim])


def test_exact_keep_unit_source_location_passes():
    document, quality, claim = _decision_claim_fixture()
    assert any(
        claim["source_location"] == location
        and claim["evidence_quote"] in unit_text
        for location, unit_text in v2.source_units(document["normalized_text"])
    )
    assert v2.validate_decision_claim_rows([document], quality, [claim]) == [claim]


def test_decision_claim_namespace_and_legacy_outputs_are_separate():
    assert v2.OUTPUTS == v2.output_paths()
    assert v2.decision_claim_path("v5_validation") == "metadata/rag/v5_validation/stage02_corpus/decision_evidence_claims.csv"
    assert v2.decision_claim_path("v5_validation") not in v2.OUTPUTS


@pytest.mark.parametrize("heading", ["References", "Normative References", "Informative References", "Bibliography", "16.1. Normative References"])
def test_reference_sections_require_citation_metadata(heading):
    citation = '[RFC9000] Iyengar, J. and M. Thomson, "QUIC: A UDP-Based Multiplexed and Secure Transport", RFC 9000, DOI 10.17487/RFC9000, May 2021.'
    assert v2.quality_decision(heading) == ("KEEP", "REFERENCE_SECTION_HEADING")
    assert v2.quality_decision(heading + "\n" + citation)[0] == "EXCLUDE_LOW_INFORMATION"
    rows = v2.quality_rows("RFC-9000", [("document", heading), ("document", citation),
        ("document", "[RFC9000] QUIC requires address validation; DOI 10.17487/RFC9000, 2021.")])
    assert [row["quality_decision"] for row in rows] == ["KEEP", "EXCLUDE_LOW_INFORMATION", "REQUIRES_REVIEW"]
    assert all(set(row) == set(v2.QUALITY_HEADER) for row in rows)


@pytest.mark.parametrize("identity,text", [
    ("RESEARCH-LICHY-2023", "We compare random forests, CNN and GRU models for encrypted malware traffic classification using USTC-TFC2016."),
    ("RESEARCH-SHEKHAWAT-2019", "Feature analysis distinguishes benign and malicious encrypted traffic using flow statistics."),
])
def test_research_sections_do_not_remove_substantive_text(identity, text):
    rows = v2.quality_rows(identity, [("page:1", text), ("page:2", "References"),
        ("page:2", '[1] Author, "Encrypted Malware Traffic", 2020, https://example.invalid/paper'),
        ("page:3", "Appendix A Further Analysis"), ("page:3", text)])
    assert [r["quality_decision"] for r in rows] == ["KEEP", "KEEP", "EXCLUDE_LOW_INFORMATION", "KEEP", "KEEP"]


def test_repetition_alone_is_not_page_chrome():
    rows = v2.quality_rows("TOR-CELLS", [("document", "RELAY")] * 3)
    assert all(row["quality_decision"] == "KEEP" for row in rows)


def test_citation_followed_by_explanation_is_not_excluded():
    text = '[RFC9000] Author, "QUIC", RFC 9000, May 2021.\n\nEncrypted packets protect the payload against passive observers.'
    assert v2.quality_decision(text)[0] == "KEEP"


@pytest.mark.parametrize("marker", ["RFC 9000 QUIC Transport Protocol May 2021", "Iyengar & Thomson Standards Track Page 7"])
def test_long_rfc_prose_with_embedded_chrome_kept(marker):
    # The former footer character class matched hundreds of prose words.
    text = "Remembered transport parameters govern early data and protocol behavior. " * 47
    text += "0-RTT parameters require PROTOCOL_VIOLATION handling. " + marker
    assert len(text.split()) > 400
    assert v2.quality_decision(text, repeated=True)[0] == "KEEP"
    assert v2.quality_decision(marker)[0] == "EXCLUDE_LOW_INFORMATION"


def test_term_counts_are_deterministic_descriptive_only():
    text = "Benign malware malicious attack encrypted traffic traffic analysis Tor VPN QUIC TLS malware."
    first = v2.term_counts(text)
    assert first == v2.term_counts(text)
    assert first["term_malware_count"] == 2
    assert all(type(value) is int for value in first.values())


def test_summary_counts_consistent(project):
    documents, _, _, quality, _ = v2.build(project)
    rows = v2.summary_rows(documents, quality)
    overall = next(r for r in rows if r["summary_level"] == "overall")
    assert overall["candidate_documents"] == overall["trusted_documents"] == 12
    assert overall["rejected_documents"] == 1
    assert overall["new_documents"] == overall["new_peer_reviewed_research_documents"] == 2
    assert overall["total_extracted_words"] == sum(int(d["word_count"]) for d in documents)
    assert overall["quality_keep_units"] + overall["quality_excluded_units"] + overall["quality_requires_review_units"] == len(quality)
    assert all(set(row) == set(v2.SUMMARY_HEADER) for row in rows)


def test_dry_mode_writes_nothing(project, capsys):
    before = snapshot(project)
    _, _, _, quality, _ = v2.build(project)
    assert v2.main(["--project-root", str(project)]) == 0
    output = capsys.readouterr().out
    assert "EVIDENCE_CORPUS_V2=READY_FOR_CONSTRUCTION" in output
    for decision, label in (("KEEP", "KEEP"), ("EXCLUDE_LOW_INFORMATION", "EXCLUDED"), ("REQUIRES_REVIEW", "REQUIRES_REVIEW")):
        count = sum(row["quality_decision"] == decision for row in quality)
        assert f"QUALITY_{label}_UNITS={count}" in output
    assert snapshot(project) == before


def test_execute_only_v2_paths_and_preserves_v1(project):
    before_v1 = snapshot(project, v2.V1_INPUTS)
    before = snapshot(project)
    assert v2.main(["--project-root", str(project), "--execute"]) == 0
    after = snapshot(project)
    assert set(after)-set(before) == set(v2.OUTPUTS)
    assert snapshot(project, v2.V1_INPUTS) == before_v1
    for name, header in zip(v2.OUTPUTS, (v2.DOCUMENT_HEADER, v1.MANIFEST_HEADER, v1.REVIEW_HEADER, v2.QUALITY_HEADER, v2.SUMMARY_HEADER, v2.CHECK_HEADER)):
        actual, rows = v1.read_table(project, name)
        assert actual == list(header) and rows
    saved = {name: (project/name).read_bytes() for name in v2.OUTPUTS}
    assert v2.main(["--project-root", str(project), "--execute"]) == 0
    assert all((project/name).read_bytes() == data for name, data in saved.items())


def test_changed_v2_source_identity_refuses_overwrite(project):
    assert v2.main(["--project-root", str(project), "--execute"]) == 0
    before = {name: (project/name).read_bytes() for name in v2.OUTPUTS}
    changed = [dict(source) for source in v2.NEW_SOURCES]
    changed[0]["local_path"] = "data/raw/rag/evidence_sources_v2/task_aligned/changed.pdf"
    with pytest.raises(ValueError, match="identity"):
        v2.overwrite_guard(project, [*v1.read_table(project, v2.OUTPUTS[1])[1][:-2], *changed])
    assert all((project/name).read_bytes() == data for name, data in before.items())


def test_static_stage_boundaries():
    source = Path(v2.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports, calls = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            calls.add(node.func.attr)
    assert not imports & {"requests", "httpx", "urllib", "socket", "openai", "sentence_transformers", "transformers", "hashlib", "matplotlib", "torch", "faiss"}
    assert not calls & {"search", "fit", "encode", "predict", "retrieve", "savefig", "sha256", "md5", "rerank"}
    assert not any(name in source for name in ("build_evidence_index", "run_rag_retrieval", "verify_retrieval_provenance"))
    assert "test.csv" not in source
    assert not any(term in source for term in ("USTC-WANG-2017", "ANDERSON-MCGREW-2017"))
    assert not any(path.startswith("artifacts/rag/index") or "retrieval" in path for path in v2.OUTPUTS)
