
<p align="center">
  <strong>Encrypted Traffic Classification • Provenance-Aware RAG • Evidence Verification • Desktop Research Application</strong>
</p>

---

## 📌 Overview

**ProvenanceGuard ETI** is a two-phase cybersecurity research project for encrypted network traffic analysis.

The system first classifies encrypted traffic using machine-learning and deep-learning models. It then passes the frozen classifier output into a **provenance-aware Retrieval-Augmented Generation (RAG)** pipeline that retrieves supporting cybersecurity evidence, verifies its origin, reranks it, builds a trusted context, and produces an evidence-supported explanation.

A dedicated **desktop application** was also developed to make the complete workflow accessible through a visual research interface.

> **Core principle:** Phase II interprets and evaluates evidence around the classifier prediction, but it does **not** change the classifier label.

---

## ✨ Main Features

- 🔐 Encrypted traffic classification
- 🧪 Two benchmark datasets
- 🤖 Five classification models
- 🧩 Leakage-safe preprocessing
- 🧱 Family-disjoint / group-aware splitting
- 🔎 TF-IDF, BGE, and Hybrid RRF retrieval
- ✅ Explicit provenance verification
- 📊 Evidence reranking
- 📚 Trusted-context construction
- 🧠 Constrained local LLM reasoning
- 🧾 Evidence-supported explanations
- 🧭 Auditable final records
- 🖥️ Full desktop research application
- 📂 CSV and supported PCAP analysis
- 🔒 Frozen research-artifact protection
- 📤 Audit and result export

---

## 🧭 Table of Contents

- [Project Architecture](#-project-architecture)
- [Phase I — Encrypted Traffic Classification](#-phase-i--encrypted-traffic-classification)
- [Phase II — Provenance-Aware RAG](#-phase-ii--provenance-aware-rag)
- [Desktop Application](#-desktop-application)
- [Desktop Application Workflow](#-desktop-application-workflow)
- [Application Pages](#-application-pages)
- [Supported Inputs](#-supported-inputs)
- [Technology Stack](#-technology-stack)
- [Datasets](#-datasets)
- [Research Integrity](#-research-integrity)
- [Repository Structure](#-repository-structure)
- [Running the Project](#-running-the-project)
- [Reproducibility](#-reproducibility)
- [Known Limitations](#-known-limitations)
- [Future Work](#-future-work)

---

# 🏗️ Project Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                    INPUT NETWORK TRAFFIC                     │
│                       CSV / PCAP                             │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│        PHASE I — ENCRYPTED TRAFFIC CLASSIFICATION           │
│                                                              │
│  Flow Reconstruction / NFStream                              │
│            ↓                                                 │
│  Leakage-Safe Preprocessing                                  │
│            ↓                                                 │
│  Frozen Classifier                                           │
│            ↓                                                 │
│  Prediction + Confidence + Probabilities                     │
│            ↓                                                 │
│  Traffic Representation + Provenance                         │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│           PHASE II — PROVENANCE-AWARE RAG                   │
│                                                              │
│  Structured Security Query                                   │
│            ↓                                                 │
│  Cybersecurity Evidence Corpus                               │
│            ↓                                                 │
│  Evidence Indexing                                           │
│            ↓                                                 │
│  TF-IDF + Dense BGE + Hybrid RRF Retrieval                   │
│            ↓                                                 │
│  Provenance Verification                                     │
│            ↓                                                 │
│  Evidence Reranking                                          │
│            ↓                                                 │
│  Trusted Context                                             │
│            ↓                                                 │
│  Constrained LLM Reasoning                                   │
│            ↓                                                 │
│  Evidence-Supported Decision Fusion                          │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                       FINAL RECORD                           │
│                                                              │
│  Final Class                                                 │
│  Evidence Status                                             │
│  Supporting Evidence                                        │
│  Provenance                                                  │
│  Explanation                                                 │
│  Audit Trail                                                 │
└──────────────────────────────────────────────────────────────┘
```

---

# 🔐 Phase I — Encrypted Traffic Classification

Phase I performs encrypted network traffic classification while preserving strict separation between training, validation, and final evaluation data.

## Datasets

### USTC-TFC2016

Used as the primary encrypted-traffic benchmark.

The project reconstructs flows from the dataset, extracts network-flow features, applies leakage-safe preprocessing, and uses a **family-disjoint split** so traffic families do not leak across partitions.

Target classes:

```text
benign
malware
```

### CIC-Darknet2020

Used as the secondary benchmark and external validation dataset.

The project uses a fixed four-class target:

```text
Non-Tor
NonVPN
Tor
VPN
```

A group-aware split is used to reduce flow-level leakage.

---

## Classification Models

Five model families are included:

| Model | Type |
|---|---|
| Random Forest | Ensemble ML |
| 1D-CNN | Deep Learning |
| XGBoost | Gradient Boosting |
| LightGBM | Gradient Boosting |
| CNN-LSTM | Deep Learning |

The operational research pipeline retains **LightGBM** as the frozen classifier.

---

## Phase-I Output Contract

For every eligible traffic flow, Phase I produces:

```text
Predicted Traffic Class
Prediction Confidence
Class Probability Distribution
Traffic Representation
Traffic Provenance
```

These outputs become the input contract for Phase II.

---

# 🧠 Phase II — Provenance-Aware RAG

Phase II is the main research contribution.

Instead of asking an LLM to independently classify traffic, Phase II receives the **already-frozen classifier output** and asks a different question:

> What trustworthy evidence supports, contradicts, conflicts with, or fails to establish support for this prediction?

---

## Phase-II Pipeline

### 1. Structured Security Query

Classifier observations are converted into a structured security query containing:

- predicted traffic class
- confidence information
- traffic characteristics
- semantic traffic representation
- source metadata
- provenance information

### 2. Cybersecurity Evidence Corpus

The evidence corpus combines multiple source categories, including:

- TRAIN-derived empirical evidence
- peer-reviewed research
- official protocol documentation
- standards / RFC material
- dataset documentation

### 3. Evidence Indexing

Corpus documents are converted into searchable chunks and indexed for both lexical and semantic retrieval.

### 4. Evidence Retrieval

Three retrieval paths are available:

```text
TF-IDF
Dense BGE
Hybrid Reciprocal Rank Fusion (RRF)
```

### 5. Provenance Verification

Retrieved evidence is checked against the registered evidence inventory.

Only authorized and traceable evidence is allowed into the trusted-context stage.

### 6. Evidence Reranking

Verified evidence is reranked so that the most useful evidence is prioritized for context construction.

### 7. Trusted Context Construction

A bounded set of verified evidence chunks is selected for the reasoning model.

### 8. Constrained LLM Reasoning

A local LLM receives only the structured query and trusted context.

The reasoning output is required to follow a structured response contract.

### 9. Evidence-Supported Decision Fusion

The reasoning result is attached to the classifier output.

The classifier label remains frozen.

### 10. Final Record

The system records:

```text
Final Class
Evidence Status
Provenance
Supporting Evidence
Explanation
Fusion Reason
Audit Metadata
```

---

## Evidence States

The reasoning layer can return four evidence states:

| Status | Meaning |
|---|---|
| `SUPPORTED` | Retrieved evidence supports the classifier prediction |
| `CONTRADICTED` | Retrieved evidence conflicts with the classifier prediction |
| `INSUFFICIENT` | Available evidence is not sufficient to establish support |
| `CONFLICTING` | Retrieved evidence contains materially conflicting signals |

These states describe **evidence compatibility**, not ground-truth correctness.

---

# 🖥️ Desktop Application

## ProvenanceGuard ETI — Research Desktop

A complete desktop application was developed around the frozen research pipeline.

The application is not a separate classifier and does not replace the experimental implementation. It is a **research interface over the validated artifacts and inference contracts**.

The desktop application allows a researcher to move from traffic input to final evidence-supported decision without manually opening intermediate CSV, JSON, retrieval, provenance, or LLM artifacts.

---

## 🎯 Desktop App Goals

The interface was designed to provide:

- one visual workflow for both research phases
- transparent model output
- inspectable traffic flows
- visible confidence and probability information
- evidence retrieval inspection
- provenance verification
- reranking visibility
- trusted-context inspection
- constrained LLM output
- final decision traceability
- audit/export support

A major design requirement is **research integrity**:

> The original research project remains read-only while the desktop application consumes frozen artifacts and approved inference paths.

---

# 🔄 Desktop Application Workflow

```text
┌───────────────┐
│   New Input   │
│  CSV / PCAP   │
└───────┬───────┘
        │
        ▼
┌────────────────────┐
│  Input Validation  │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ Flow Reconstruction│
│     / NFStream     │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ Frozen Preprocessing│
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ Phase-I Classifier │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│    Flow Explorer   │
│ + Flow Inspection  │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ Structured Security│
│       Query        │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ Evidence Retrieval │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ Provenance Check + │
│     Reranking      │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Trusted Context   │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ Constrained Local  │
│    LLM Reasoning   │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│   Final Decision   │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Audit & Export    │
└────────────────────┘
```

---

# 🪟 Application Pages

## 1. Overview

The research workspace dashboard summarizes:

- backend availability
- classifier contracts
- evidence-index availability
- active analysis
- workflow progress
- current dataset
- classified-flow count
- retained research contribution

---

## 2. New Analysis

The input page supports the available research contracts and lets the user:

- upload supported CSV input
- upload supported PCAP input
- choose the dataset contract
- validate the input
- open existing completed research artifacts

The interface separates:

```text
Input Source
Dataset & Options
Validation
Ready
```

---

## 3. Analysis Progress

This page presents stage-level execution status.

Visible pipeline stages include:

- input validation
- flow reconstruction / NFStream
- schema validation
- frozen preprocessing
- Phase-I classification
- traffic representation
- traffic provenance
- five-model comparison
- structured security query
- evidence resources
- Phase-II processing

The page is designed to make pipeline execution traceable rather than appearing as a single opaque operation.

---

## 4. Classification

The classification page summarizes the active Phase-I analysis.

It includes:

- number of flows in scope
- mean confidence
- predicted-class count
- integrity scope
- class distribution
- confidence distribution
- access to individual flow inspection
- five-model comparison

The retained classifier is clearly distinguished from comparison models.

---

## 5. Flow Explorer

The Flow Explorer provides row-level navigation through classified flows.

Researchers can inspect:

- Flow ID
- predicted class
- prediction confidence
- input scope
- classifier identity

Filtering and search allow the user to locate specific flows or classes.

---

## 6. Flow Detail

The Flow Detail view exposes the complete Phase-I result for an individual flow.

It shows:

- predicted traffic class
- confidence
- full probability distribution
- flow identity
- retained classifier
- model run
- source scope
- source row
- Phase-II eligibility

This page provides the transition from classification to evidence reasoning.

---

## 7. Evidence Retrieval

This page visualizes the structured security query and the retrieved cybersecurity evidence.

The researcher can inspect:

- structured traffic observations
- observed class
- expected evidence relationship
- Hybrid RRF results
- Dense BGE results
- TF-IDF results
- relevance scores
- evidence excerpts
- document metadata
- evidence source type

---

## 8. Provenance & Reranking

The provenance page makes evidence verification explicit.

It includes:

- retrieved candidate ranking
- reranked evidence
- provenance status
- authorization status
- verification reason
- source identifiers
- verification ledger

No numerical provenance score is fabricated when one is not available.

---

## 9. Context & Reasoning

This page contains two main research objects:

### Trusted Context

The selected, verified evidence chunks used for reasoning.

### Constrained LLM Output

The structured reasoning response, including:

- evidence state
- supported class
- explanation
- response validity
- cited evidence IDs
- model and execution metadata

---

## 10. Final Decision

The final page combines classifier output and evidence interpretation.

It shows:

- final class
- Phase-I prediction
- Phase-I confidence
- evidence status
- verified-context count
- provenance information
- evidence-supported explanation
- LLM-supported class
- fusion reason
- evaluation scope
- provenance record
- audit navigation

The final classifier class remains governed by the **retain-classifier** fusion policy.

---

## 11. Audit & Export

The research record can be reviewed and exported for later inspection.

This supports:

- reproducibility
- experiment traceability
- evidence inspection
- provenance review
- result preservation

---

# 📸 Desktop Application Screenshots

For a polished GitHub repository, store application screenshots under a dedicated directory such as:

```text
docs/
└── screenshots/
    ├── 01_overview.png
    ├── 02_new_analysis.png
    ├── 03_analysis_progress.png
    ├── 04_classification.png
    ├── 05_flow_explorer.png
    ├── 06_flow_detail.png
    ├── 07_evidence_retrieval.png
    ├── 08_provenance_reranking.png
    ├── 09_context_reasoning.png
    └── 10_final_decision.png
```

Then the README can display them using:

```markdown
![Overview](docs/screenshots/01_overview.png)
![Classification](docs/screenshots/04_classification.png)
![Evidence Retrieval](docs/screenshots/07_evidence_retrieval.png)
![Final Decision](docs/screenshots/10_final_decision.png)
```

> Keep the original screenshots unchanged; only place copies in the repository screenshot directory.

---

# 📥 Supported Inputs

| Dataset | CSV | PCAP / PCAPNG |
|---|:---:|:---:|
| USTC-TFC2016 | ✅ | ✅ |
| CIC-Darknet2020 | ✅ | ❌ |

### CIC PCAP

Raw CIC-Darknet2020 PCAP inference is intentionally **not supported by the current frozen application contract**.

The application fails closed instead of silently applying an incompatible extraction pipeline.

---

# 🧰 Technology Stack

## Research / Backend

- Python
- pandas
- NumPy
- scikit-learn
- LightGBM
- XGBoost
- PyTorch
- NFStream
- sentence-transformer / BGE embeddings
- TF-IDF
- Reciprocal Rank Fusion

## Local Reasoning

- Qwen3-4B-Instruct-2507-FP8
- local HTTP inference service
- constrained structured response validation

## Desktop Application

- React
- Electron
- Python backend
- local API communication
- frozen research artifacts
- reusable chart and research-inspection components

---

# 📚 Datasets

### USTC-TFC2016

Primary encrypted traffic benchmark with benign and malware traffic families.

### CIC-Darknet2020

Secondary benchmark with the fixed target classes:

```text
Non-Tor
NonVPN
Tor
VPN
```

Dataset files are not necessarily included in the public repository because of size, licensing, and reproducibility considerations.

---

# 🔒 Research Integrity

The project deliberately separates model development from final evaluation.

## Core Rules

```text
TRAIN
  └── preprocessing fitting
  └── model training
  └── empirical evidence derivation

VALIDATION
  └── model development
  └── checkpoint selection
  └── configuration decisions

TEST
  └── final classifier evaluation
  └── final frozen Phase-II evaluation
```

Additional protections:

- USTC family-disjoint splitting
- CIC group-aware splitting
- frozen preprocessing contracts
- frozen classifier prediction before Phase II
- ground truth excluded from reasoning
- ground truth joined only after final reasoning
- explicit evidence provenance
- read-only original research artifacts
- no hidden Phase-II label override

---

# 🗂️ Repository Structure

A clean repository layout can follow this structure:

```text
ProvenanceGuard-ETI/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── research/
│   ├── phase1/
│   ├── phase2/
│   ├── configs/
│   ├── models/
│   └── artifacts/
│
├── desktop_app/
│   ├── frontend/
│   ├── backend/
│   ├── runtime/
│   ├── electron/
│   └── tests/
│
├── New_experiments/
   ├── table1_error_detection/
   ├── table2_ablation/
   ├── table3_retrieval_quality/
   ├── table6_runtime/
   └── explanation_evaluation/


```

> Adapt the folder names to the actual repository rather than renaming validated research directories only for cosmetic reasons.

---

# ▶️ Running the Project

Because the research pipeline uses frozen artifacts and dataset-specific contracts, setup should follow the repository's actual environment files.

A typical development workflow is:

```text
1. Prepare the Python environment
2. Install the registered dependencies
3. Verify frozen model/artifact paths
4. Start the Python backend
5. Start the local LLM service when Phase II is required
6. Launch the React/Electron interface
7. Select a supported CSV or PCAP input
8. Run the analysis
9. Inspect Phase I
10. Continue into Phase II
11. Review the final record and audit trail
```

Exact commands should be documented from the repository's real launcher scripts rather than guessed in the README.

---

# ♻️ Reproducibility

The project preserves important experiment artifacts such as:

- frozen dataset splits
- preprocessing metadata
- saved predictions
- model-selection records
- model artifacts
- retrieval candidates
- evidence inventory
- provenance metadata
- reranking outputs
- trusted contexts
- constrained reasoning outputs
- human annotation workbooks
- runtime measurements
- integrity hashes

Supervisor-requested analyses are kept separate from the original frozen research implementation.

---

# 🧪 Research vs Desktop Application

| Research Pipeline | Desktop Application |
|---|---|
| Defines validated experiments | Presents the workflow visually |
| Trains/evaluates research models | Uses frozen models/artifacts |
| Creates scientific outputs | Exposes outputs for inspection |
| Controls leakage and evaluation | Preserves those contracts |
| Stores authoritative research artifacts | Reads approved artifacts |
| Defines Phase-II reasoning | Provides interactive Phase-II execution |
| Scientific implementation | Research-facing interface |

The desktop application must not be treated as a replacement for the frozen experimental pipeline.

---

# ⚠️ Known Limitations

- Benchmark datasets cannot represent all modern encrypted traffic.
- CIC raw-PCAP inference is not supported by the current application contract.
- The evidence corpus is bounded and domain-specific.
- Local LLM reasoning introduces additional latency.
- Evidence compatibility should not be interpreted as ground-truth correctness.
- Explanation quality remains dependent on retrieval quality and available evidence.
- The current project is a research prototype rather than a production network-monitoring platform.

---

# 🚀 Future Work

Potential extensions include:

- newer encrypted-traffic datasets
- external real-world traffic
- additional independent cybersecurity evidence sources
- richer semantic retrieval
- improved reranking
- retrieval calibration
- expanded provenance policies
- larger human evaluation
- analyst usability studies
- streaming/live traffic support
- optimized local LLM inference
- broader cross-dataset validation
- production-oriented monitoring integration

---

# 📖 Research Title

**Provenance-Aware Retrieval-Augmented Reasoning for Evidence-Supported Encrypted Traffic Classification**

---

# 📝 Citation

If this repository is used in academic work, cite the associated paper or thesis after publication.

```bibtex
@{provenanceguard_eti,
  title  = {Provenance-Aware Retrieval-Augmented Reasoning for Evidence-Supported Encrypted Traffic Classification},
  author = {Project Authors},
  year   = {2026}
}
```

Replace the placeholder author field with the final approved author list before public release.

---

# 📄 License

Add the appropriate project license before making the repository public.

---

<p align="center">
  <strong>ProvenanceGuard ETI</strong><br>
  Encrypted Traffic Classification with Traceable Evidence, Provenance Verification, and Constrained Reasoning
</p>
