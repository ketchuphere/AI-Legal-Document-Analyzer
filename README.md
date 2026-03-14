# 📄 AI Document Analyzer

> **AI-powered multi-agent document analysis — built on Anthropic Claude + LangGraph**

Upload any legal or business document (PDF, DOCX, TXT) and get an instant, structured analysis:
executive summary, risk breakdown, key clause extraction, compliance checklist, and improvement
suggestions — all in one clean Streamlit interface.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the App](#-running-the-app)
- [Running Tests](#-running-tests)
- [Docker Deployment](#-docker-deployment)
- [Cloud Deployment](#-cloud-deployment)
- [Example Output](#-example-output)
- [Evaluation Metrics](#-evaluation-metrics)
- [Roadmap](#-roadmap)
- [Disclaimer](#-disclaimer)

---

## 🎯 Overview

**Problem:** Legal and business documents are dense, jargon-heavy, and time-consuming to review.
Most people sign agreements without fully understanding the risks, missing clauses, or weak
protections they contain.

**Solution:** AI Document Analyzer uses a 7-node LangGraph agent pipeline
to automatically extract, summarise, and evaluate any document in seconds.

**Inputs:**  PDF / DOCX / TXT documents (up to 50 MB)
**Outputs:** Structured analysis report (viewable in-app, downloadable as Markdown or DOCX)
**Success Metrics:**
- All 7 pipeline nodes complete without error
- Structured JSON output parsed correctly for 95%+ of documents
- Full report generated in < 60 seconds for a 10-page document

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📝 Executive Summary | 5–12 bullet-point plain-English summary of the document |
| ⚠️ Risk Analysis | Structured risk cards with severity, likelihood, and mitigation |
| 📑 Clause Extraction | 10 key clauses extracted as structured JSON (parties, payment, IP, etc.) |
| ✅ Compliance Checklist | 12-point checklist with PASS / WARN / FAIL status |
| 💡 Improvement Suggestions | Clause-level recommendations organised by topic |
| 📊 Full Report | Combined Markdown report downloadable as `.md` or `.docx` |
| 💬 Document Q&A Chat | Multi-turn chatbot grounded to the uploaded document |
| 📄 Multi-format Support | PDF (via PyMuPDF), DOCX (python-docx), plain TXT |
| 🔄 Long Document Handling | Map-reduce chunking for documents exceeding context limits |
| ⚡ Result Caching | SHA-256 content hash prevents re-processing the same file |
| 🎨 Dark-mode UI | Custom Streamlit theme with metric cards and risk colour coding |

---

## 🏗️ Architecture

```
Upload
  │
  ▼
DocumentLoader ──► Text extraction (PDF/DOCX/TXT)
  │
  ▼
┌─────────────────────────────────────────────────────┐
│                  LangGraph Pipeline                 │
│                                                     │
│  condense ──► summarize ──► analyze_risks           │
│                                  │                  │
│                          extract_clauses            │
│                                  │                  │
│                        compliance_check             │
│                                  │                  │
│                      suggest_improvements           │
│                                  │                  │
│                         compile_report              │
└─────────────────────────────────────────────────────┘
                            │
                            ▼
              AgentState (final_report, clauses,
               risks_structured, compliance_items…)
                            │
                  ┌─────────┴──────────┐
                  │                    │
             Streamlit UI          Download
           (6 tabbed panels)     (MD / DOCX)
                  │
            Q&A Chat Panel
          (multi-turn chatbot)
```

Each LangGraph node:
1. Reads from `AgentState`
2. Calls Claude via `LLMClient`
3. Returns a partial state dict that LangGraph merges automatically

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit ≥ 1.35 |
| LLM | Anthropic Claude (Sonnet/Haiku/Opus) |
| Workflow | LangGraph + LangChain Core |
| PDF Parsing | PyMuPDF4LLM |
| DOCX Parsing | python-docx |
| Configuration | Pydantic Settings + python-dotenv |
| Export | python-docx (DOCX generation) |
| Testing | pytest + pytest-cov |
| Containerisation | Docker + Docker Compose |

---

## 📁 Project Structure

```
ai_document_analyzer/
│
├── app.py                          # Streamlit entry point
│
├── src/
│   ├── config/
│   │   └── settings.py             # Pydantic-Settings config (all tuneable values)
│   │
│   ├── agents/
│   │   ├── workflow.py             # LangGraph 7-node pipeline + AgentState
│   │   └── qa_chain.py             # Document Q&A chatbot chain
│   │
│   └── utils/
│       ├── llm_client.py           # Anthropic API wrapper
│       ├── document_loader.py      # PDF / DOCX / TXT loader + metadata
│       ├── text_splitter.py        # Character-level chunking utility
│       └── export.py               # DOCX report export
│
├── tests/
│   └── test_core.py                # 20+ unit & integration tests
│
├── docs/
│   └── sample_output.md            # Example analysis output
│
├── .streamlit/
│   └── config.toml                 # Streamlit theme & server settings
│
├── .env.example                    # Environment variable template
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Pytest + coverage config
├── Dockerfile                      # Container image definition
├── docker-compose.yml              # Multi-service compose file
└── README.md                       # This file
```

---

## 🔧 Installation

### Prerequisites
- Python 3.11 or later
- An [Anthropic API key](https://console.anthropic.com)
- `pip` (or a virtual-environment manager like `venv` / `conda`)

### Steps

```bash
# 1. Clone (or unzip) the project
git clone https://github.com/your-org/ai-document-analyzer.git
cd ai-document-analyzer

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Open .env and add your ANTHROPIC_API_KEY
```

---

## ⚙️ Configuration

Edit `.env` (copied from `.env.example`):

```env
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Optional overrides
DEFAULT_MODEL=claude-sonnet-4-20250514
DEFAULT_CHUNK_CHARS=6000
DEFAULT_CHUNK_OVERLAP=400
MAX_TOKENS_PER_CALL=2048
LOG_LEVEL=INFO
```

All settings are also tunable at runtime from the Streamlit sidebar.

### Available Models

| UI Label | Model String | Best For |
|----------|-------------|----------|
| Claude Sonnet 4 *(default)* | `claude-sonnet-4-20250514` | Best balance of speed and quality |
| Claude Haiku 4.5 *(fast)* | `claude-haiku-4-5-20251001` | Quick drafts, lower cost |
| Claude Opus 4 *(most capable)* | `claude-opus-4-20250514` | Complex, multi-party contracts |

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**.

**Workflow:**
1. Upload a PDF, DOCX, or TXT document in the sidebar
2. Optionally adjust model and chunk settings
3. Click **🔍 Analyse Document**
4. Explore the 6 result tabs (Summary → Risks → Clauses → Compliance → Suggestions → Report)
5. Use the **💬 Q&A Chat** tab to ask follow-up questions
6. Download the report as Markdown or DOCX

---

## 🧪 Running Tests

```bash
# All tests with coverage report
pytest tests/ -v

# Quick run (no coverage)
pytest tests/ -v --no-cov

# Single test class
pytest tests/test_core.py::TestAgentNodes -v
```

Expected output:
```
tests/test_core.py::TestTextSplitter::test_short_document_single_chunk   PASSED
tests/test_core.py::TestTextSplitter::test_long_document_multiple_chunks  PASSED
tests/test_core.py::TestTextSplitter::test_overlap_present                PASSED
tests/test_core.py::TestTextSplitter::test_estimate_tokens                PASSED
tests/test_core.py::TestDocumentLoader::test_basic_metadata_word_count    PASSED
... (20 tests total)

---------- coverage: src ----------
TOTAL  coverage: 78%
```

---

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t ai-doc-analyzer .

# Run with your .env file
docker run -p 8501:8501 --env-file .env ai-doc-analyzer

# Or with Docker Compose
docker compose up --build
```

Access at **http://localhost:8501**.

---

## ☁️ Cloud Deployment

### Streamlit Community Cloud (Free)
1. Push the repo to GitHub (remove `.env` — use Streamlit Secrets instead)
2. Go to [share.streamlit.io](https://share.streamlit.io) → "New app"
3. Add `ANTHROPIC_API_KEY` under **Secrets**
4. Deploy — done in < 2 minutes

### AWS / GCP / Azure (Container)
```bash
# Build and push
docker build -t ai-doc-analyzer .
docker tag ai-doc-analyzer your-registry/ai-doc-analyzer:latest
docker push your-registry/ai-doc-analyzer:latest

# Deploy as a container app / Cloud Run / ECS Task
# Map port 8501, pass ANTHROPIC_API_KEY as env var
```

### Heroku
```bash
heroku create your-app-name
heroku config:set ANTHROPIC_API_KEY=sk-ant-...
git push heroku main
```

---

## 📊 Example Output

See [`docs/sample_output.md`](docs/sample_output.md) for a full example analysis
of a sample consulting services agreement, including:
- 10-bullet executive summary
- 6 structured risk entries with severity table
- All 10 key clauses extracted
- 12-item compliance checklist (10 PASS, 1 WARN, 1 FAIL)
- Organised improvement suggestions across 7 topics

---

## 📈 Evaluation Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| Pipeline completion rate | ≥ 99% | Unit tests + manual testing |
| JSON parse success (risks/clauses/compliance) | ≥ 95% | `test_core.py` bad-JSON fallback tests |
| Summary quality (bullet count) | 5–12 bullets | Post-analysis assertion in workflow |
| Processing time (10-page PDF) | < 60 seconds | Manual benchmarks on Sonnet 4 |
| Test coverage | ≥ 70% | `pytest --cov` |
| DOCX export validity | 100% | Magic-byte check in tests |

---

## 🔮 Roadmap

- [ ] **Scanned PDF / OCR support** via Tesseract
- [ ] **Multi-document comparison** (diff two contract versions)
- [ ] **Streaming responses** for real-time output as each node finishes
- [ ] **Document history** (persist past analyses with SQLite)
- [ ] **Custom compliance templates** (GDPR, HIPAA, SOC 2, etc.)
- [ ] **REST API** (FastAPI wrapper for programmatic access)
- [ ] **Batch processing** via Anthropic Message Batches API
- [ ] **Parallel agent execution** (summarise + risks simultaneously via LangGraph branches)

---

## ⚠️ Disclaimer

This tool provides AI-assisted analysis for informational purposes only.
**It does not constitute legal advice.** Always consult a qualified attorney
before acting on any information contained in an AI-generated report.

---