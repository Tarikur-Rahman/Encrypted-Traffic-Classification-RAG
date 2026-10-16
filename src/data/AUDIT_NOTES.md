# Stage 01 CSV audits

These commands inventory sources only. They do not parse PCAP traffic, invoke
NFStream, normalize labels, create splits, or train models. No real-data audit
was executed during this refactor.

Runs live in `results/audits/<DATASET>/<raw_audit|schema_audit>/<UTC_RUN_ID>/`.
IDs use `YYYYMMDDTHHMMSSZ`. Directory creation is exclusive: a same-second
collision raises `FileExistsError`; retry in a later second. Partial runs remain
for inspection and must not be reused. `artifact_index.csv` is written last and
lists the relative path and byte size of every other CSV after publication.
CSV publication uses a temporary file
in the destination directory and an atomic, exclusive hard link. A filesystem
without hard-link support fails explicitly, rather than falling back to an
overwrite-prone operation.

Paths use forward slashes. Empty scalar values
are represented by empty CSV fields. CSVs are UTF-8 with headers and no dataframe
index. Each result table has the same run ID and UTC timestamp; the artifact
index contains only `relative_path` and `size_bytes`. Source files must remain
quiescent during an audit; this implementation does not lock external writers.

USTC classification uses only the exact first directory `Benign` or `Malware`.
Family names come from the next path component, removing the extension only
when that component is the capture filename. Capture groups use the complete
relative source path, including the extension, to distinguish identical stems
in different subdirectories. These identifiers are stable while the release
paths remain unchanged; identical content at different paths remains separate.
Provenance fields must never
be used as ML predictors.

Family counts count distinct `(binary_label, semantic_family)` pairs. Duplicate
capture-group counts count excess occurrences beyond the first. Release expectations are
checked against derived counts, never used to manufacture inventory entries.
Every check emits PASS or FAIL; the raw-audit CLI prints failed check names and
returns exit code 1 after finalizing all artifacts if any check fails.

CIC candidate detection is a column-name heuristic, not a target contract.
Candidate values are parsed as strings, preserving case, punctuation, spaces,
and numeric spellings such as `001`. Only empty source fields count as missing;
literal `NA`, `NULL`, and `<NA>` remain values. Unique counts exclude missing
fields; value counts include an empty-field row for missing values, and fractions
use all source records as denominator. Empty datasets use zero missing fractions.
All values are counted without the old top-100 truncation. To reread exact text
from CSV in pandas, use `keep_default_na=False` and `dtype` for text fields.

All output column identities use the exact names assigned by pandas when
reading the source CSV. For example, repeated `Label` headers become `Label`
and `Label.1`; existing suffixes are preserved, never stripped or canonicalized.
Column indexes are zero-based and retain source order. Candidate summaries,
value counts, schema rows, missing-value rows, and collision rows use the same
identities, so distinct source columns remain independently addressable.

Original headers are also read for collision detection. In the collision table,
`normalized_name` describes the raw-header collision while `column_name` and
`column_index` identify the pandas column. Exact duplicates also appear in the
case-insensitive collision report. Case folding is confined to descriptive
header collision detection; it is never applied to label values. Records with
inconsistent field counts raise an error and do not receive a completion index.
