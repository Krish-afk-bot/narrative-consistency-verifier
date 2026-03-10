<div align="center">

# 📚 Narrative Consistency Validator

**Verify character backstories against long-form literary narratives using constraint-based reasoning**

![Status](https://img.shields.io/badge/status-active-success)
![Track](https://img.shields.io/badge/track-A%20Systems%20Reasoning-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

[Quick Start](#-quick-start) • [Architecture](#-system-architecture) • [Documentation](TRACK_A_SUBMISSION_REPORT.md)

</div>

---

## 🎯 Mission

Determine whether a proposed character backstory is logically consistent with events, relationships, and character development in a 100,000+ word novel—without reading the entire book manually.

---

## 🔍 The Challenge

Literary narratives establish complex constraints across hundreds of pages:

- **Temporal constraints**: Events must occur in valid chronological order
- **Causal constraints**: Character actions must align with stated beliefs and motivations  
- **Factual constraints**: Backstory claims must not contradict established facts
- **Relationship constraints**: Character dynamics must remain internally consistent

**The Problem**: A human reader would need hours to verify a single backstory claim against an entire novel. Existing NLP approaches fail because:

- LLMs cannot fit 100k+ words in context
- Simple search misses implicit contradictions
- QA systems retrieve facts but don't reason about consistency
- Summarization loses critical details

---

## 💡 Our Solution

A **constraint-based reasoning pipeline** that:

1. Breaks backstories into atomic, verifiable claims
2. Retrieves relevant evidence from the full novel using semantic search
3. Analyzes claim-evidence relationships using rule-based logic
4. Makes deterministic binary decisions: **Consistent** or **Contradictory**

**Key Insight**: Instead of asking an LLM "is this consistent?", we decompose the problem into structured reasoning steps with traceable logic.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧩 **Claim Decomposition** | Breaks backstories into atomic claims with categories (beliefs, events, relationships) and importance levels (core, significant, surface) |
| 🔎 **Long-Context Retrieval** | Handles 100k+ word novels through hierarchical chunking with overlap and timeline-aware indexing |
| ⚖️ **Constraint Reasoning** | Detects contradictions, support, and causal inconsistencies using pattern matching and temporal logic |
| 🎯 **Deterministic Classification** | Rule-based decisions ensure reproducibility—same input always produces same output |
| 📊 **Pathway Integration** | Uses Pathway for document ingestion, indexing, and retrieval orchestration |
| 🔬 **Explainable Outputs** | Every prediction includes a human-readable rationale derived from decision logic |

---

## 🔄 How It Works

```
┌─────────────┐
│  Backstory  │  "As a child, Edmond grew up wealthy..."
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Claim Extraction   │  → Claim 1: "Edmond grew up wealthy" [CORE]
└──────┬──────────────┘  → Claim 2: "Edmond had formal education" [SIGNIFICANT]
       │
       ▼
┌─────────────────────┐
│ Evidence Retrieval  │  → Search novel for relevant passages
└──────┬──────────────┘  → Retrieve top-k chunks with timeline coverage
       │
       ▼
┌─────────────────────┐
│ Constraint Analysis │  → Claim 1 + Evidence → CONTRADICTS (0.9 confidence)
└──────┬──────────────┘  → "Novel states Edmond was poor sailor's son"
       │
       ▼
┌─────────────────────┐
│  Classification     │  → Fatal contradiction in CORE claim
└──────┬──────────────┘  → Decision: CONTRADICTORY (label=0)
       │
       ▼
┌─────────────────────┐
│  Output Generation  │  story_id,prediction,rationale
└─────────────────────┘  1,0,"Proposed backstory contradicts established events"
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Novel Text  │  │  Backstories │  │  Test Data   │          │
│  │  (100k+ words)│  │     (CSV)    │  │    (CSV)     │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
└─────────┼──────────────────┼──────────────────┼──────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PATHWAY DOCUMENT STORE                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐ │
│  │   Novels   │  │   Chunks   │  │   Claims   │  │ Metadata │ │
│  │  (indexed) │  │ (embedded) │  │ (embedded) │  │(timeline)│ │
│  └────────────┘  └────────────┘  └────────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PROCESSING PIPELINE                         │
│                                                                   │
│  ┌──────────────────┐         ┌──────────────────┐             │
│  │ Claim Extractor  │────────▶│ Evidence Retriever│             │
│  │  (LLM-assisted)  │         │ (Semantic + KW)   │             │
│  └──────────────────┘         └─────────┬─────────┘             │
│                                          │                        │
│                                          ▼                        │
│                              ┌──────────────────────┐            │
│                              │ Constraint Engine    │            │
│                              │ (Rule-based patterns)│            │
│                              └─────────┬────────────┘            │
│                                        │                          │
│                                        ▼                          │
│                              ┌──────────────────────┐            │
│                              │ Deterministic        │            │
│                              │ Classifier           │            │
│                              └─────────┬────────────┘            │
└────────────────────────────────────────┼──────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUT LAYER                             │
│                    results.csv (submission)                      │
│  story_id | prediction | rationale                               │
│  ──────────────────────────────────────────────────────          │
│     1     |     1      | No narrative constraints violated       │
│     2     |     0      | Proposed backstory contradicts events   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 👤 User Journey

| Step | Action | System Response |
|------|--------|-----------------|
| 1️⃣ | User provides novel text + character backstory | System loads novel into document store |
| 2️⃣ | System chunks novel into overlapping segments | 5-paragraph chunks with 2-paragraph overlap, timeline tags |
| 3️⃣ | System extracts claims from backstory | "Grew up wealthy" → [EARLY_LIFE, CORE] |
| 4️⃣ | System retrieves evidence for each claim | Top-10 relevant passages with timeline coverage |
| 5️⃣ | System analyzes claim-evidence relationships | Pattern matching detects contradiction |
| 6️⃣ | System makes binary decision | Fatal contradiction → label=0 |
| 7️⃣ | System generates rationale | "Proposed backstory contradicts established events" |
| 8️⃣ | User receives CSV output | story_id, prediction, rationale |

---

## 🛠️ Technology Stack

### Core Framework
- **Python 3.8+** — Primary language
- **Pathway** — Document ingestion, indexing, and retrieval orchestration

### NLP & Embeddings
- **Sentence Transformers** — Semantic embeddings (all-MiniLM-L6-v2)
- **scikit-learn** — Cosine similarity computation

### LLM Integration (Optional)
- **Google Gemini 2.0 Flash** — Claim extraction and explanation generation only
- **Temperature = 0** — Deterministic outputs enforced
- **Note**: LLMs do NOT make classification decisions

### Data Processing
- **pandas** — CSV handling
- **regex** — Pattern matching for constraint detection

### Development
- **python-dotenv** — Environment configuration
- **pathlib** — Cross-platform file handling

---


## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip (Python package manager)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Krish-afk-bot/narrative-consistency-verifier.git
cd narrative-consistency-verifier
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables** (optional, for LLM features)
```bash
# Create .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

4. **Prepare data**
```bash
# Place your novels in data/ directory
data/
  ├── The Count of Monte Cristo.txt
  ├── In search of the castaways.txt
  ├── test.csv
  └── train.csv
```

### Running the Pipeline

```bash
python main.py
```

**Expected Output:**
```
============================================================
Track-A Backstory Consistency Validator
============================================================

[1/5] Initializing document store...
[2/5] Loading training data for character extraction...
[3/5] Loading and chunking novels...
[4/5] Loading test data...
[5/5] Processing test entries...

Writing results.csv...
  Output: results.csv
  Rows: 50

VALIDATION:
  test.csv rows: 50
  results.csv rows: 50
  ✓ Row count matches
  Consistent: 28
  Contradict: 22

Done.
```

### Output Format

`results.csv`:
```csv
story_id,prediction,rationale
1,1,No narrative constraints violated
2,0,Proposed backstory contradicts established events
3,1,Backstory aligns with established narrative events
```

---

## 📁 Project Structure

```
narrative-consistency-validator/
├── backstory_validator/
│   ├── config.py                    # Configuration and enums
│   ├── extraction/
│   │   └── claim_extractor.py       # Backstory → claims
│   ├── retrieval/
│   │   └── evidence_retriever.py    # Claim → evidence passages
│   ├── reasoning/
│   │   ├── constraint_engine.py     # Claim-evidence analysis
│   │   └── classifier.py            # Final decision logic
│   ├── pipeline/
│   │   ├── ingestion.py             # Novel loading
│   │   ├── chunking.py              # Hierarchical chunking
│   │   └── pathway_tables.py        # Document store
│   ├── llm/
│   │   └── gemini_client.py         # LLM interface (optional)
│   └── models/
│       └── schemas.py               # Data models
├── data/
│   ├── test.csv                     # Test backstories
│   ├── train.csv                    # Training data
│   └── *.txt                        # Novel files
├── main.py                          # Entry point
├── requirements.txt                 # Dependencies
├── results.csv                      # Output (generated)
├── TRACK_A_SUBMISSION_REPORT.md     # Technical documentation
└── README.md                        # This file
```

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes**
4. **Run tests** (if available)
5. **Commit your changes** (`git commit -m 'Add amazing feature'`)
6. **Push to the branch** (`git push origin feature/amazing-feature`)
7. **Open a Pull Request**

### Contribution Guidelines
- Write clear commit messages
- Add comments for complex logic
- Update documentation for new features
- Ensure code follows existing style
- Test your changes before submitting

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Kharagpur Data Science Hackathon 2026** — Track A: Systems Reasoning with NLP and Generative AI
- **Pathway** — Document processing framework
- **Sentence Transformers** — Semantic embedding models
- **Google Gemini** — LLM capabilities for claim extraction

---

## 📞 Contact

For questions, issues, or collaboration opportunities:

- Open an issue on [GitHub](https://github.com/Krish-afk-bot/narrative-consistency-verifier/issues)
- Project Link: [https://github.com/Krish-afk-bot/narrative-consistency-verifier](https://github.com/Krish-afk-bot/narrative-consistency-verifier)

---

<div align="center">

**Built with 🧠 for long-context narrative reasoning**

</div>
