"""Seed-42 reservoir sampling of V4 development prompts, without labels or inference.

Algorithm R visits canonical query_id order, using one local Random(42) stream.
Each of six prediction strata retains exactly 200 IDs; only 1,200 small metadata
rows are retained. Manifest order is query_id ascending, ordinal is one-based.
Validation recomputes selection from the entire validated V4 population.
"""
import csv
import random
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from src.rag import llm_v3_runtime as runtime

common, prompts = runtime.common, runtime.prompts
MANIFEST = "results/rag/llm/v4/subsets/stratified_200_per_prediction_class_seed42.csv"
HEADER = ("query_id", "source_dataset", "classifier_prediction", "stratum", "selection_ordinal", "sampling_seed")
STRATA = (("CIC-Darknet2020", "Non-Tor"), ("CIC-Darknet2020", "NonVPN"), ("CIC-Darknet2020", "Tor"),
          ("CIC-Darknet2020", "VPN"), ("USTC-TFC2016", "benign"), ("USTC-TFC2016", "malware"))
PER_STRATUM, SEED, TOTAL = 200, 42, 1200
SCOPE = "stratified_development_validation_subset"
STRATEGY = "200_per_source_dataset_prediction_class"


def select_rows(connection):
    rng = random.Random(SEED)
    reservoirs = {pair: [] for pair in STRATA}
    counts = dict.fromkeys(STRATA, 0)
    for identity, payload in connection.execute("SELECT id, payload FROM prompts ORDER BY id COLLATE BINARY"):
        prompt = common.parse_json(payload)
        pair = (prompt["source_dataset"], prompt["classifier_prediction"])
        common.require(pair in reservoirs, "Unexpected subset stratum")
        counts[pair] += 1
        bucket = reservoirs[pair]
        if len(bucket) < PER_STRATUM:
            bucket.append(identity)
        else:
            position = rng.randrange(counts[pair])
            if position < PER_STRATUM:
                bucket[position] = identity
    common.require(all(counts[pair] >= PER_STRATUM for pair in STRATA), "Each stratum requires at least 200 candidates")
    selected = sorted((identity, *pair) for pair, bucket in reservoirs.items() for identity in bucket)
    common.require(len(selected) == TOTAL and len({row[0] for row in selected}) == TOTAL, "Subset must contain 1200 unique IDs")
    return [dict(query_id=identity, source_dataset=dataset, classifier_prediction=prediction,
                 stratum=dataset + "/" + prediction, selection_ordinal=str(i), sampling_seed=str(SEED))
            for i, (identity, dataset, prediction) in enumerate(selected, 1)]


def validate_manifest(root, connection):
    expected = select_rows(connection)
    actual = common.rows(root, MANIFEST, HEADER)
    for row in expected:
        common.require(next(actual, None) == row, "Manifest differs from deterministic seed-42 selection or schema")
    common.require(next(actual, None) is None, "Manifest must contain exactly 1200 rows")
    return expected


def restrict_population(root, connection):
    """Restrict only the temporary executor database; canonical inputs untouched."""
    selected = validate_manifest(root, connection)
    connection.execute("CREATE TABLE selected_queries (id TEXT PRIMARY KEY, ordinal INTEGER UNIQUE)")
    connection.executemany("INSERT INTO selected_queries VALUES (?, ?)", [(row["query_id"], int(row["selection_ordinal"])-1) for row in selected])
    connection.execute("DELETE FROM prompts WHERE id NOT IN (SELECT id FROM selected_queries)")
    # Two passes avoid collisions with existing UNIQUE canonical ordinals.
    connection.execute("UPDATE prompts SET ordinal=-(SELECT ordinal+1 FROM selected_queries WHERE selected_queries.id=prompts.id)")
    connection.execute("UPDATE prompts SET ordinal=-ordinal-1")
    common.require(connection.execute("SELECT COUNT(*) FROM prompts").fetchone()[0] == TOTAL, "Selected executor population differs")
    connection.commit()
    return TOTAL


def main(argv=None):
    parser = common.parser(__doc__)
    parser.add_argument("--validate", action="store_true", help="Validate existing manifest without writing")
    args = parser.parse_args(argv)
    try:
        common.require(not (args.validate and args.execute), "Choose validation or materialization")
        root = args.project_root.resolve(strict=True)
        paths = [common.QUERY_PATH, *prompts.V4_OUTPUTS]
        if args.validate:
            paths.append(MANIFEST)
        before = common.snapshot(root, paths)
        if args.execute:
            common.fresh_outputs(root, [MANIFEST])
        with tempfile.TemporaryDirectory(prefix=".rag_llm_subset_", dir=root) as directory:
            with closing(sqlite3.connect(str(Path(directory)/"population.sqlite"))) as connection:
                runtime.prepare_population(root, connection, "v4")
                rows = validate_manifest(root, connection) if args.validate else select_rows(connection)
            common.require(before == common.snapshot(root, paths), "Subset inputs changed")
            if args.execute:
                staged = Path(directory)/"subset.csv"
                common.corpus.write_table(staged, HEADER, rows, mode="x")
                prompts.publish_staged(root, [(MANIFEST, staged)])
        print("LLM_EVALUATION_SUBSET=" + ("PASS" if args.validate or args.execute else "READY_FOR_EXECUTION") + "\nEVALUATION_SCOPE=" + SCOPE + "\nSELECTED_QUERIES=1200\nSAMPLING_SEED=42")
        return 0
    except (OSError, ValueError, TypeError, KeyError, csv.Error, sqlite3.Error) as exc:
        print("LLM_EVALUATION_SUBSET=FAIL\n" + str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
