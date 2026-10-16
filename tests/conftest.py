"""Synthetic Phase-II fixtures: no real traffic, models, or experiment artifacts."""
from types import SimpleNamespace

import pytest

from src.rag import phase2_common as common
from src.rag import build_evidence_corpus_v2 as corpus
from src.rag import build_evidence_index_v2 as index
from src.rag import verify_retrieval_provenance_v3 as provenance


@pytest.fixture
def phase2_policy():
    return SimpleNamespace(limit=512, count=lambda text: len(text.split())+2)


@pytest.fixture
def phase2_case(phase2_policy):
    query = dict(query_id="USTC-VAL-000001", source_dataset="USTC-TFC2016", source_split="validation", rag_partition="development_validation",
                 predicted_class_label="benign", confidence="0.6", class_probabilities_json='{"benign":0.6,"malware":0.4}', retrieval_query_text="Encrypted traffic with typical packet sizes.", experiment_id="phase2_dev_retrieval_v3")
    chunks = []
    for i in range(12):
        text = f"Protocol evidence number {i} describing encrypted network traffic behavior."
        chunk = dict.fromkeys(index.CHUNK_HEADER, "")
        chunk.update(chunk_id=f"DOC-{i:02d}-QUALITY-00000-CHUNK-0000", document_id=f"DOC-{i:02d}", source_type="standards_rfc", source_provenance_status="VERIFIED", corpus_partition="trusted", authorized_for_retrieval="YES",
                     review_unit_id=f"DOC-{i:02d}-QUALITY-00000", quality_status="KEEP", source_location="document", chunk_text=text, chunk_word_count=len(text.split()), chunk_token_count=phase2_policy.count(text), chunk_character_count=len(text), chunk_order=0, unit_word_start=0, unit_word_end=len(text.split()))
        chunks.append(chunk)
    verified = common.unique(provenance.verify_chunks(chunks), "chunk_id")
    results = [dict(experiment_id="phase2_dev_retrieval_v3", query_id=query["query_id"], source_dataset=query["source_dataset"], retrieval_method=method, rank=str(i+1), chunk_id=chunk["chunk_id"], document_id=chunk["document_id"], retrieval_score=str(1/(i+1))) for method in provenance.METHODS for i, chunk in enumerate(chunks[:10])]
    config = dict(method="hybrid_rrf", top_k=10, retrieval_weight=1.0, source_preference_weight=0.0, task_alignment_weight=0.0, source_preferences={})
    return SimpleNamespace(query=query, chunks=common.unique(chunks, "chunk_id"), verified=verified, results=results, config=config, policy=phase2_policy)


@pytest.fixture
def phase2_index_project(tmp_path):
    documents, manifest, quality = [], [], []
    for i in range(12):
        identity = f"DOC-{i:02d}"
        source = dict.fromkeys(common.corpus.MANIFEST_HEADER, "")
        source.update(document_id=identity, title="Synthetic protocol evidence", source_type="standards_rfc", corpus_partition="trusted", source_provenance_status="VERIFIED", authorized_for_retrieval="YES")
        text = "Substantive TLS protocol evidence with encrypted packet transport.\n\nSkip to Main Content\n\nReferences\nPossibly bibliographic material requiring manual inspection."
        document = dict.fromkeys(corpus.DOCUMENT_HEADER, "")
        document.update(source, normalized_text=text, extraction_status="PASS", extraction_review_status="MISSING", word_count=len(text.split()), character_count=len(text))
        documents.append(document)
        manifest.append(source)
        quality.extend(corpus.quality_rows(identity, corpus.source_units(text)))
    for name, header, records in zip(index.INPUTS, (corpus.DOCUMENT_HEADER, common.corpus.MANIFEST_HEADER, corpus.QUALITY_HEADER), (documents, manifest, quality)):
        common.corpus.write_table(tmp_path/name, header, records)
    return tmp_path
