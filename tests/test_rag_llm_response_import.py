import pytest

from src.rag import phase2_common as common
from src.rag import import_constrained_llm_responses as responses
from src.rag import materialize_constrained_llm_experiment as prompts


@pytest.fixture
def response_case(phase2_case):
    query = phase2_case.query
    chunk = next(iter(phase2_case.chunks))
    prompt = dict(query_id=query["query_id"], source_dataset=query["source_dataset"], classifier_prediction="benign", allowed_chunk_ids=common.json_text([chunk]))
    value = dict(query_id=query["query_id"], reasoning_decision="SUPPORT_CLASS", evidence_status="SUPPORTED", supported_class="benign", explanation="Supplied evidence supports the observation.", cited_chunk_ids=[chunk])
    return prompt, value


def test_valid_response_round_trip(response_case):
    prompt, value = response_case
    row = responses.validate_response(value, prompt)
    assert set(row) == set(responses.HEADER)
    assert common.parse_json(row["cited_chunk_ids"]) == value["cited_chunk_ids"]
    assert row == responses.validate_all([value], {prompt["query_id"]: prompt})[0]


@pytest.mark.parametrize("field,value", [("query_id", "UNKNOWN"), ("evidence_status", "LIKELY"), ("supported_class", "Tor"), ("cited_chunk_ids", ["UNKNOWN"]), ("cited_chunk_ids", []), ("explanation", ""), ("reasoning_decision", "ABSTAIN")])
def test_bad_response_rejected(response_case, field, value):
    prompt, response = response_case
    with pytest.raises(ValueError):
        responses.validate_response(dict(response, **{field: value}), prompt)


def test_missing_extra_duplicate_and_incomplete_rejected(response_case):
    prompt, value = response_case
    mapping = {prompt["query_id"]: prompt}
    for altered in ({k: v for k, v in value.items() if k != "explanation"}, dict(value, extra="bad")):
        with pytest.raises(ValueError):
            responses.validate_response(altered, prompt)
    with pytest.raises(ValueError, match="duplicate"):
        responses.validate_all([value, value], mapping)
    with pytest.raises(ValueError, match="Missing"):
        responses.validate_all([], mapping)


def test_jsonl_csv_and_malformed_json(tmp_path, response_case):
    prompt, value = response_case
    relative = "results/rag/llm/imports/responses.jsonl"
    path = tmp_path/relative
    path.parent.mkdir(parents=True)
    path.write_text(common.json_text(value)+"\n", encoding="utf-8")
    assert list(responses.external_rows(tmp_path, relative)) == [value]
    csv_path = "results/rag/llm/imports/responses.csv"
    csv_value = dict(value, cited_chunk_ids=common.json_text(value["cited_chunk_ids"]))
    common.corpus.write_table(tmp_path/csv_path, prompts.REQUIRED_RESPONSE_FIELDS, [csv_value])
    assert list(responses.external_rows(tmp_path, csv_path)) == [value]
    path.write_text('{"query_id":', encoding="utf-8")
    with pytest.raises(ValueError):
        list(responses.external_rows(tmp_path, relative))
    with pytest.raises(ValueError):
        common.parse_json('{"query_id":"A","query_id":"B"}')


def test_no_evidence_requires_insufficient(response_case):
    prompt, response = response_case
    prompt = dict(prompt, allowed_chunk_ids="[]")
    value = dict(response, cited_chunk_ids=[], evidence_status="INSUFFICIENT", reasoning_decision="ABSTAIN", supported_class=None)
    assert responses.validate_response(value, prompt)["supported_class"] == ""
    with pytest.raises(ValueError):
        responses.validate_response(response, prompt)
