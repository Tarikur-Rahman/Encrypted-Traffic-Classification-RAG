"""Shared local contracts; development by default, TEST query validation opt-in."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

from src.rag import build_evidence_corpus as corpus

require = corpus.require
CONFIG_HEADER = ("field", "value")
SUMMARY_HEADER = ("source_dataset", "record_count")
CLASSES = {"USTC-TFC2016": ("benign", "malware"), "CIC-Darknet2020": ("Non-Tor", "NonVPN", "Tor", "VPN")}
QUERY_COUNTS = {"USTC-TFC2016": 97001, "CIC-Darknet2020": 31723}
STATUSES = ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT", "CONFLICTING")
QUERY_PATH = "results/rag/phase2_dev_retrieval_v3/query_records.csv"


def safe_input(relative):
    value = str(relative).replace("\\", "/")
    require(not any(part.lower() == "test" for part in re.split(r"[/_.-]+", value)), "TEST input prohibited")
    require(value.startswith(("results/rag/", "metadata/rag/", "data/processed/rag/", "artifacts/rag/index_v2/", "metadata/traffic/")), "Input outside Phase-II development contracts")
    return relative


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def parse_json(text):
    return json.loads(text, object_pairs_hook=unique_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + value)))


def rows(root, relative, required=()):
    corpus.configure_csv_field_limit()
    with corpus.local_path(root, safe_input(relative)).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        require(header and len(header) == len(set(header)) and set(required) <= set(header), "Invalid CSV schema: " + relative)
        for row in reader:
            require(None not in row and all(v is not None for v in row.values()), "Malformed CSV: " + relative)
            yield row


def unique(records, key):
    output = {}
    for row in records:
        identity = row[key]
        require(isinstance(identity, str) and identity.strip() and identity not in output, "Blank/duplicate " + key)
        output[identity] = row
    return output


def finite(value):
    require(not isinstance(value, bool), "Boolean numeric value")
    number = float(value)
    require(math.isfinite(number), "Nonfinite numeric value")
    return number


def validate_query(row, *, source_split="validation"):
    require(source_split in {"validation", "test"}, "Unknown explicit query scope")
    partition = "development_validation" if source_split == "validation" else "final_test"
    require(row["source_split"] == source_split and row["rag_partition"] == partition, "Query split/partition mismatch")
    if source_split == "test":
        forbidden = {"true_class_id", "true_class_label", "classifier_correct", "ground_truth", "target_id", "target_binary"}
        require(not forbidden.intersection(row), "Ground truth prohibited in TEST RAG records")
        require("-TEST-" in row["query_id"] and "-VAL-" not in row["query_id"], "Non-TEST query identity")
    dataset = row["source_dataset"]
    require(dataset in CLASSES and row["predicted_class_label"] in CLASSES[dataset], "Invalid dataset/class")
    require(0 <= finite(row["confidence"]) <= 1, "Invalid classifier confidence")
    probabilities = parse_json(row["class_probabilities_json"])
    if isinstance(probabilities, dict):
        require(set(probabilities) == set(CLASSES[dataset]), "Unknown probability class labels")
    values = list(probabilities.values()) if isinstance(probabilities, dict) else probabilities
    require(isinstance(values, list) and len(values) == len(CLASSES[dataset]), "Invalid class probabilities")
    require(all(0 <= finite(value) <= 1 for value in values) and abs(sum(map(float, values))-1) <= 1e-5, "Invalid probability distribution")


def queries(root, path=QUERY_PATH, expected=True):
    required = ("query_id", "source_dataset", "source_split", "rag_partition", "predicted_class_label", "confidence", "class_probabilities_json", "retrieval_query_text")
    result = unique(rows(root, path, required), "query_id")
    for row in result.values():
        validate_query(row)
        require(row.get("experiment_id") == "phase2_dev_retrieval_v3", "Wrong query experiment")
    require(result, "Empty query population")
    if expected:
        require(dict(Counter(row["source_dataset"] for row in result.values())) == QUERY_COUNTS, "Development query population differs")
    return result


def snapshot(root, paths):
    return [{"path": name, "size_bytes": corpus.local_path(root, safe_input(name)).stat().st_size,
             "mtime_ns": corpus.local_path(root, name).stat().st_mtime_ns} for name in paths]


def fresh_outputs(root, paths):
    # Revisions are intentional immutable experiment products. Never silently
    # replace completed or partial runs; users must resolve them explicitly.
    targets = [corpus.local_path(root, name, must_exist=False) for name in paths]
    require(not any(path.exists() for path in targets), "Output already exists; refusing canonical overwrite")
    return targets


def write_bundle(root, tables):
    targets = fresh_outputs(root, [name for name, _, _ in tables])
    for target, (_, header, records) in zip(targets, tables):
        corpus.write_table(target, header, records, mode="x")


def config_rows(values):
    return [dict(field=key, value=str(value)) for key, value in sorted(values.items())]


def checks(stage, details):
    return [corpus.check(stage, key, "PASS", str(value)) for key, value in sorted(details.items())]


def completed(root, configuration, checklist):
    config = {key: row["value"] for key, row in unique(rows(root, configuration, CONFIG_HEADER), "field").items()}
    validations = list(rows(root, checklist, corpus.foundation.CHECKLIST_HEADER))
    require(validations and all(row["status"] == "PASS" for row in validations), "Upstream stage is incomplete")
    identity = parse_json(config["input_metadata_json"])
    if configuration == "results/rag/llm/v4/llm_experiment_configuration.csv" and config.get("prompt_schema_version") == "constrained_evidence_v4":
        canonical = "results/rag/llm/v4/llm_prompt_records.csv"
        require(config.get("active_prompt_path") == canonical, "V4 requires its canonical prompt path")
        corpus.local_path(root, canonical)  # Keep regular-file and alias checks.
        # Query/context/provenance entries are genuine inputs. V1/V2/V3
        # prompt CSVs are preservation-only history, not V4 prerequisites.
        # Leave historical configuration/checklist/policy entries validated;
        # do not interpret v2_artifacts/v3_artifacts as dependency lists.
        # Keep the saved metadata unchanged so audit/resume identities survive.
        retired_prompt_paths = {
            "results/rag/llm/llm_prompt_records.csv",
            "results/rag/llm/v2/llm_prompt_records.csv",
            "results/rag/llm/v3/llm_prompt_records.csv",
        }
        identity = [row for row in identity if row["path"] not in retired_prompt_paths]
    require(identity == snapshot(root, [row["path"] for row in identity]), "Upstream inputs changed")
    return config


def summary(records):
    counts = Counter(row["source_dataset"] for row in records)
    return [dict(source_dataset=dataset, record_count=count) for dataset, count in sorted(counts.items())]


def parser(description):
    value = argparse.ArgumentParser(description=description)
    value.add_argument("--project-root", type=Path, default=Path("."))
    value.add_argument("--execute", action="store_true")
    return value


def run_tables(args, stage, build):
    try:
        root = args.project_root.resolve(strict=True)
        tables = build(root, args)
        if args.execute:
            write_bundle(root, tables)
        print(stage + ("=PASS" if args.execute else "=READY_FOR_EXECUTION"))
        return 0
    except (OSError, ValueError, KeyError, TypeError, csv.Error, ImportError) as exc:
        print(stage + "=BLOCKED\n" + str(exc))
        return 2
