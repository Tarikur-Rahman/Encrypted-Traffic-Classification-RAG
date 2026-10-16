# Provenance-Aware RAG for Evidence-Supported Encrypted Traffic Classification

Fresh implementation started from zero.

## Current completed engineering stage
- Repository structure created.
- Research protocol frozen before dataset processing.
- USTC acquisition, safe archive extraction, raw inventory/audit, NFStream extraction and feature-candidate materialization implemented.
- CIC raw/schema/label audit implemented; it deliberately does not invent label mappings.
- Group-disjoint split utility implemented.
- Common evaluation and figure-ready result store implemented.
- Synthetic tests included.

## Recommended environment
Python 3.11 or 3.12. NFStream 6.6.0 is pinned. On Windows, NFStream requires Npcap.

## First commands

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
python src/common/capture_environment.py
```

### USTC acquisition preflight
```powershell
python src/data/acquire_ustc.py --project-root . 
```

Execute:
```powershell
python src/data/acquire_ustc.py --project-root . --execute
python src/data/audit_raw.py --project-root . --dataset USTC-TFC2016
python src/data/extract_ustc_archives.py --project-root . --execute
python src/data/extract_ustc_nfstream.py --project-root .
```

The final command above is preflight-only unless `--execute` is added.

### CIC
Download the official CIC-Darknet2020 dataset from:
https://www.unb.ca/cic/datasets/darknet2020.html

Place the official CSV file(s) under:
`data/raw/CIC-Darknet2020/`

Then run:
```powershell
python src/data/audit_raw.py --project-root . --dataset CIC-Darknet2020
python src/data/audit_cic_schema.py --project-root .
```

Do not train CIC models until the audit output has been reviewed and the label/feature contract is frozen.
