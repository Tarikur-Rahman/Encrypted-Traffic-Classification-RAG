"""Synthetic local corpus tests. Run manually; no external documents required."""
import ast
import csv
import zipfile
from pathlib import Path

import pytest

from src.rag import build_evidence_corpus as corpus
from src.rag import phase2_foundation as foundation


def write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def project(tmp_path):
    for name, (header, rows) in foundation.contracts().items():
        write(tmp_path / "metadata/rag" / name, header, rows)
    write(tmp_path / "metadata/rag/evidence_corpus_inventory.csv", foundation.INVENTORY_HEADER, [])
    checks = [dict(section="synthetic", check="fixture", status="PASS", details="Synthetic fixture only")]
    write(tmp_path / "metadata/rag/phase2_readiness_checklist.csv", foundation.CHECKLIST_HEADER, checks)
    for name in corpus.INTERFACE_FILES:
        write(tmp_path / "results/rag/interface" / name, foundation.CHECKLIST_HEADER, checks)
    return tmp_path


def curated(project, **changes):
    row = dict.fromkeys(corpus.MANIFEST_HEADER, "")
    row.update(document_id="RFC-EXAMPLE-001", title="Synthetic protocol evidence", source_type="standards_rfc",
               publisher="Synthetic publisher", canonical_url="https://example.invalid/spec",
               local_path="data/rag/evidence_sources/trusted/spec.txt", traffic_topics="encrypted transport",
               security_topics="authentication", corpus_partition="trusted", source_provenance_status="VERIFIED",
               authorized_for_retrieval="YES")
    row.update(changes)
    path = project / row["local_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("First paragraph.\n\nSecond paragraph.\n", encoding="utf-8")
    write(project / corpus.MANIFEST, corpus.MANIFEST_HEADER, [row])
    return row


@pytest.mark.parametrize("name", corpus.FOUNDATION_FILES)
def test_foundation_contracts_required(project, name):
    (project / "metadata/rag" / name).unlink()
    with pytest.raises(ValueError):
        corpus.prerequisites(project)


@pytest.mark.parametrize("name", corpus.INTERFACE_FILES)
def test_interface_artifacts_required(project, name):
    (project / "results/rag/interface" / name).unlink()
    with pytest.raises(ValueError):
        corpus.prerequisites(project)


def test_init_only_header_never_overwrites(project, capsys):
    before = snapshot(project)
    assert corpus.main(["--project-root", str(project), "--init-manifest"]) == 0
    assert "EVIDENCE_SOURCE_MANIFEST=INITIALIZED" in capsys.readouterr().out
    after = snapshot(project)
    assert set(after) - set(before) == {corpus.MANIFEST}
    assert foundation.read_csv(project / corpus.MANIFEST) == (list(corpus.MANIFEST_HEADER), [])
    assert corpus.main(["--project-root", str(project), "--init-manifest"]) == 2
    assert snapshot(project) == after
    curated(project)
    content = (project / corpus.MANIFEST).read_bytes()
    assert corpus.main(["--project-root", str(project), "--init-manifest"]) == 2
    assert (project / corpus.MANIFEST).read_bytes() == content


def test_missing_or_empty_manifest_needs_curation(project, capsys):
    before = snapshot(project)
    assert corpus.main(["--project-root", str(project)]) == 2
    assert "BLOCKED_NEEDS_CURATED_SOURCES" in capsys.readouterr().out
    assert snapshot(project) == before
    write(project / corpus.MANIFEST, corpus.MANIFEST_HEADER, [])
    assert corpus.main(["--project-root", str(project)]) == 2


@pytest.mark.parametrize("changes", [
    {"document_id": ""}, {"document_id": "not readable!"}, {"document_id": "a"*64},
    {"source_type": "unreviewed_blog"}, {"corpus_partition": "unknown"},
    {"source_provenance_status": "TRUSTED"}, {"authorized_for_retrieval": "true"},
    {"source_provenance_status": "REQUIRES_REVIEW"}, {"source_provenance_status": "REJECTED"},
    {"title": ""},
])
def test_invalid_manifest_metadata_rejected(project, changes):
    curated(project, **changes)
    with pytest.raises(ValueError):
        corpus.inspect(project)


def test_duplicate_ids_and_manifest_schema(project):
    row = curated(project)
    write(project / corpus.MANIFEST, corpus.MANIFEST_HEADER, [row, row])
    with pytest.raises(ValueError, match="Duplicate"):
        corpus.inspect(project)
    write(project / corpus.MANIFEST, ["document_id"], [dict(document_id="RFC-001")])
    with pytest.raises(ValueError, match="schema"):
        corpus.inspect(project)


@pytest.mark.parametrize("category", foundation.SOURCE_CATEGORIES)
def test_allowed_source_types_and_missing_bibliography_preserved(project, category):
    row = curated(project, source_type=category, publisher="", canonical_url="")
    rows, checks = corpus.inspect(project)
    assert rows == [row]
    assert any(c["status"] == "WARN" for c in checks)
    assert not any(c["status"] == "BLOCKED" for c in checks)


def test_missing_file_and_unsupported_extension_block(project):
    row = curated(project)
    (project / row["local_path"]).unlink()
    assert any(c["status"] == "BLOCKED" for c in corpus.inspect(project)[1])
    curated(project, local_path="data/rag/evidence_sources/trusted/spec.exe")
    assert any("Unsupported" in c["details"] for c in corpus.inspect(project)[1])


def test_partition_and_authorization_are_explicit(project):
    curated(project, corpus_partition="adversarial")
    with pytest.raises(ValueError, match="VERIFIED trusted"):
        corpus.inspect(project)
    curated(project, local_path="data/rag/evidence_sources/adversarial/spec.txt")
    assert any("partition/path" in c["details"] for c in corpus.inspect(project)[1])


def test_dry_mode_does_not_extract_or_write(project, monkeypatch, capsys):
    curated(project)
    before = snapshot(project)
    def forbidden(*args):
        raise AssertionError("Dry mode parsed a document")
    monkeypatch.setattr(corpus, "extract", forbidden)
    assert corpus.main(["--project-root", str(project)]) == 0
    assert "READY_FOR_CONSTRUCTION" in capsys.readouterr().out
    assert snapshot(project) == before


def test_deterministic_normalization():
    raw = "  alpha\t beta \r\n\r\n\r\n gamma\u00a0delta \r\n"
    expected = "alpha beta\n\ngamma delta"
    assert corpus.normalize_text(raw) == expected
    assert corpus.normalize_text(expected) == expected


@pytest.mark.parametrize("suffix", [".txt", ".md", ".html", ".htm"])
def test_text_formats(tmp_path, suffix):
    path = tmp_path / ("source" + suffix)
    content = "Alpha\n\nBeta" if suffix in {".txt", ".md"} else "<p>Alpha</p><p>Beta</p><script>never evidence</script>"
    path.write_text(content, encoding="utf-8")
    status, text, notes = corpus.extract(path)
    assert status == "PASS" and text == "Alpha\n\nBeta"
    if suffix in {".html", ".htm"}:
        assert "omitted script/style" in notes


def test_docx_body_tables_and_paragraphs(tmp_path):
    path = tmp_path / "source.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Alpha</w:t></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>Beta</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>')
    status, text, _ = corpus.extract(path)
    assert status == "PASS" and text == "Alpha\n\nBeta"


def test_html_embedded_content_requires_review_without_fetch(tmp_path):
    path = tmp_path / "source.html"
    path.write_text('<p>Visible evidence</p><img src="https://example.invalid/image.png">', encoding="utf-8")
    status, text, notes = corpus.extract(path)
    assert status == "REQUIRES_REVIEW" and text == "Visible evidence"
    assert "external resources never fetched" in notes


def test_empty_pdf_text_layer_requires_review(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    page = SimpleNamespace(extract_text=lambda: "")
    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=lambda path: SimpleNamespace(is_encrypted=False, pages=[page])))
    status, text, notes = corpus.extract(tmp_path / "empty.pdf")
    assert status == "REQUIRES_REVIEW" and not text
    assert "empty_pages=1" in notes and "no OCR" in notes


def test_extraction_failure_never_authorizes(project, monkeypatch):
    row = curated(project)
    monkeypatch.setattr(corpus, "extract", lambda path: ("FAILED", "", "Unreadable source"))
    document = corpus.documents(project, [row])[0]
    assert document["authorized_for_retrieval"] == "NO"
    assert document["word_count"] == ""
    assert row["authorized_for_retrieval"] == "YES"


def test_adversarial_and_nonapproved_sources_never_parsed(project, monkeypatch):
    row = curated(project, authorized_for_retrieval="NO", corpus_partition="adversarial",
                  local_path="data/rag/evidence_sources/adversarial/example.txt")
    def forbidden(*args):
        raise AssertionError("Unapproved text read")
    monkeypatch.setattr(corpus, "extract", forbidden)
    record = corpus.documents(project, [row])[0]
    assert record["normalized_text"] == "" and record["word_count"] == ""
    assert record["corpus_partition"] == "adversarial"


def test_execute_three_outputs_preserves_inputs_and_reruns(project, capsys):
    curated(project)
    before = snapshot(project)
    assert corpus.main(["--project-root", str(project), "--execute"]) == 0
    stdout = capsys.readouterr().out
    assert "AUTHORIZED_TRUSTED_DOCUMENTS=1" in stdout
    assert "NEXT_STAGE=EVIDENCE_INDEXING" in stdout
    after = snapshot(project)
    assert set(after) - set(before) == set(corpus.OUTPUTS)
    assert all(after[name] == content for name, content in before.items())
    assert corpus.main(["--project-root", str(project), "--execute"]) == 0
    assert snapshot(project) == after
    for name, header in zip(corpus.OUTPUTS, (corpus.DOCUMENT_HEADER, corpus.SUMMARY_HEADER, foundation.CHECKLIST_HEADER)):
        actual, rows = foundation.read_csv(project / name)
        assert actual == list(header) and rows
    _, docs = foundation.read_csv(project / corpus.OUTPUTS[0])
    assert int(docs[0]["word_count"]) == 4 and int(docs[0]["character_count"]) > 0
    assert not {"chunk_id", "chunk_text", "embedding", "vector", "retrieval_score", "reranking_score", "llm_output"} & set(corpus.DOCUMENT_HEADER)


def test_changed_source_set_refuses_silent_overwrite(project):
    curated(project)
    assert corpus.main(["--project-root", str(project), "--execute"]) == 0
    outputs = {p: (project / p).read_bytes() for p in corpus.OUTPUTS}
    curated(project, document_id="RFC-CHANGED-002")
    assert corpus.main(["--project-root", str(project), "--execute"]) == 2
    assert all((project / p).read_bytes() == content for p, content in outputs.items())


def test_summary_distinguishes_unknown_counts(project):
    row = curated(project)
    records = corpus.documents(project, [row])
    unmeasured = dict(records[0], document_id="ADV-001", corpus_partition="adversarial", authorized_for_retrieval="NO", character_count="", word_count="")
    summary = corpus.summarize(records + [unmeasured])
    total = next(r for r in summary if r["summary_level"] == "overall")
    assert total["document_count"] == 2 and total["extracted_document_count"] == 1
    assert total["total_word_count"] == 4
    adversarial = next(r for r in summary if r["corpus_partition"] == "adversarial")
    assert adversarial["total_word_count"] == ""


def test_no_executable_downstream_or_network_dependencies():
    source = Path(corpus.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = set()
    attributes = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attributes.add(node.func.attr)
    assert not imports & {"requests", "urllib", "socket", "httpx", "openai", "lightgbm", "torch", "faiss", "sentence_transformers", "matplotlib", "hashlib", "pytesseract"}
    assert not attributes & {"fit", "predict", "embed", "retrieve", "savefig", "sha256", "md5"}
    assert "test.csv" not in source


def test_source_cannot_escape_partition(project):
    row = curated(project)
    row["local_path"] = "data/rag/evidence_sources/trusted/../../../../outside.txt"
    with pytest.raises(ValueError):
        corpus.source_path(project, row)


def review_file(project, identity, status):
    write(project / corpus.REVIEW, corpus.REVIEW_HEADER,
          [dict(document_id=identity, review_status=status, review_notes="Manual completeness assessment")])


def test_init_review_header_only_and_no_overwrite(project, capsys):
    before = snapshot(project)
    assert corpus.main(["--project-root", str(project), "--init-review"]) == 0
    assert "EVIDENCE_EXTRACTION_REVIEW=INITIALIZED" in capsys.readouterr().out
    assert set(snapshot(project)) - set(before) == {corpus.REVIEW}
    assert foundation.read_csv(project / corpus.REVIEW) == (list(corpus.REVIEW_HEADER), [])
    assert corpus.main(["--project-root", str(project), "--init-review"]) == 2
    review_file(project, "RFC-001", "APPROVED")
    content = (project / corpus.REVIEW).read_bytes()
    assert corpus.main(["--project-root", str(project), "--init-review"]) == 2
    assert (project / corpus.REVIEW).read_bytes() == content


@pytest.mark.parametrize("status", ["APPROVED", "REQUIRES_REVIEW", "REJECTED", "MISSING"])
@pytest.mark.parametrize("extraction,text", [("PASS", "Evidence"), ("REQUIRES_REVIEW", "Evidence"), ("REQUIRES_REVIEW", ""), ("FAILED", "Evidence")])
def test_manual_review_authorization_matrix(project, monkeypatch, status, extraction, text):
    row = curated(project)
    if status != "MISSING":
        review_file(project, row["document_id"], status)
    monkeypatch.setattr(corpus, "extract", lambda path: (extraction, text, "Synthetic extraction"))
    document = corpus.documents(project, [row])[0]
    expected = extraction == "PASS" or (extraction == "REQUIRES_REVIEW" and status == "APPROVED" and bool(text))
    assert document["authorized_for_retrieval"] == ("YES" if expected else "NO")
    assert document["extraction_status"] == extraction
    assert document["extraction_review_status"] == status
    assert corpus.usable_document(document) == expected


@pytest.mark.parametrize("changes", [
    {"authorized_for_retrieval": "NO"},
    {"authorized_for_retrieval": "NO", "source_provenance_status": "REQUIRES_REVIEW"},
    {"authorized_for_retrieval": "NO", "source_provenance_status": "REJECTED"},
    {"authorized_for_retrieval": "NO", "corpus_partition": "adversarial", "local_path": "data/rag/evidence_sources/adversarial/spec.txt"},
])
def test_review_never_overrides_provenance_or_manifest(project, monkeypatch, changes):
    row = curated(project, **changes)
    review_file(project, row["document_id"], "APPROVED")
    def forbidden(path):
        raise AssertionError("Unapproved document extracted")
    monkeypatch.setattr(corpus, "extract", forbidden)
    document = corpus.documents(project, [row])[0]
    assert document["authorized_for_retrieval"] == "NO"
    assert not corpus.usable_document(document)


@pytest.mark.parametrize("failure", ["schema", "duplicate", "unknown", "status"])
def test_invalid_curated_reviews_fail(project, failure):
    row = curated(project)
    record = dict(document_id=row["document_id"], review_status="APPROVED", review_notes="Reviewed")
    header = corpus.REVIEW_HEADER
    records = [record]
    if failure == "schema":
        header, records = ("document_id",), [dict(document_id=row["document_id"])]
    elif failure == "duplicate":
        records = [record, record]
    elif failure == "unknown":
        record["document_id"] = "UNKNOWN-001"
    else:
        record["review_status"] = "YES"
    write(project / corpus.REVIEW, header, records)
    with pytest.raises(ValueError):
        corpus.read_reviews(project, [row])


def test_review_workflow_preserves_status_inputs_and_reports_dry_findings(project, monkeypatch, capsys):
    row = curated(project)
    monkeypatch.setattr(corpus, "extract", lambda path: ("REQUIRES_REVIEW", "Preserved evidence", "Completeness review needed"))
    assert corpus.main(["--project-root", str(project), "--execute"]) == 2
    capsys.readouterr()
    def forbidden(path):
        raise AssertionError("Dry mode extraction")
    monkeypatch.setattr(corpus, "extract", forbidden)
    before = snapshot(project)
    assert corpus.main(["--project-root", str(project)]) == 0
    output = capsys.readouterr().out
    assert row["document_id"] in output and "requires manual completeness review" in output
    assert snapshot(project) == before
    review_file(project, row["document_id"], "APPROVED")
    preserved = {name: (project / name).read_bytes() for name in (corpus.MANIFEST, corpus.REVIEW, row["local_path"])}
    monkeypatch.setattr(corpus, "extract", lambda path: ("REQUIRES_REVIEW", "Preserved evidence", "Completeness review needed"))
    assert corpus.main(["--project-root", str(project), "--execute"]) == 0
    _, documents = foundation.read_csv(project / corpus.OUTPUTS[0])
    assert documents[0]["extraction_status"] == "REQUIRES_REVIEW"
    assert documents[0]["extraction_review_status"] == "APPROVED"
    assert documents[0]["authorized_for_retrieval"] == "YES"
    _, checks = foundation.read_csv(project / corpus.OUTPUTS[2])
    assert any(r["check"] == "extraction_review_status" and "APPROVED" in r["details"] for r in checks)
    assert all((project / name).read_bytes() == value for name, value in preserved.items())


def test_large_csv_field_is_read_without_truncation(project):
    text = "Full document evidence.\n" * 10000
    assert len(text) > 131072
    path = project / corpus.OUTPUTS[0]
    write(path, ("normalized_text",), [dict(normalized_text=text)])
    previous = csv.field_size_limit()
    try:
        csv.field_size_limit(131072)
        header, rows = corpus.read_table(project, corpus.OUTPUTS[0])
        assert header == ["normalized_text"]
        assert rows[0]["normalized_text"] == text
        assert len(rows[0]["normalized_text"]) == len(text)
    finally:
        csv.field_size_limit(previous)


def test_csv_field_limit_retries_platform_overflow(monkeypatch):
    attempted = []
    def platform_limit(value):
        attempted.append(value)
        if len(attempted) < 3:
            raise OverflowError("Synthetic platform integer overflow")
        return 131072
    monkeypatch.setattr(csv, "field_size_limit", platform_limit)
    accepted = corpus.configure_csv_field_limit()
    assert attempted == [corpus.sys.maxsize, corpus.sys.maxsize // 10, corpus.sys.maxsize // 100]
    assert accepted == attempted[-1]


def test_large_corpus_dry_mode_and_manual_review_unchanged(project, monkeypatch, capsys):
    row = curated(project)
    text = "Preserved RFC evidence.\n\n" * 10000
    monkeypatch.setattr(corpus, "extract", lambda path: ("REQUIRES_REVIEW", text, "Manual completeness review needed"))
    previous = csv.field_size_limit()
    try:
        assert corpus.main(["--project-root", str(project), "--execute"]) == 2
        capsys.readouterr()
        before = snapshot(project)
        csv.field_size_limit(131072)
        assert corpus.main(["--project-root", str(project)]) == 0
        assert "requires manual completeness review" in capsys.readouterr().out
        assert snapshot(project) == before
        _, documents = corpus.read_table(project, corpus.OUTPUTS[0])
        assert documents[0]["normalized_text"] == text
        assert documents[0]["authorized_for_retrieval"] == "NO"
        review_file(project, row["document_id"], "APPROVED")
        csv.field_size_limit(131072)
        assert corpus.main(["--project-root", str(project), "--execute"]) == 0
        _, documents = corpus.read_table(project, corpus.OUTPUTS[0])
        assert documents[0]["normalized_text"] == text
        assert documents[0]["extraction_status"] == "REQUIRES_REVIEW"
        assert documents[0]["extraction_review_status"] == "APPROVED"
        assert documents[0]["authorized_for_retrieval"] == "YES"
    finally:
        csv.field_size_limit(previous)
