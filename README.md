# OEM Compliance Mapping System

Maps RFP/tender requirements to the best-fit vendor products. It ingests OEM product datasheets into a searchable knowledge base, extracts requirements from a tender PDF, and produces a ranked, evidence-backed compliance report (JSON + Markdown).

See `OEM_Compliance_System_Architecture.md` (or `Architecture_Summary.pdf` for the short version) for a full design writeup.

## How it works

1. **Ingest** vendor datasheets (PDF) → text/OCR extraction → vendor & model identification → chunking → embedded into a ChromaDB knowledge base.
2. **Extract** requirements from an RFP page range → regex pass + LLM pass → deduplicated → embedded into a per-RFP ChromaDB collection.
3. **Compliance engine** retrieves evidence per requirement, identifies candidate products, evaluates every (requirement, product) pair, ranks products, and writes the final report.

## Requirements

- Python 3.10+
- A running [Ollama](https://ollama.com) instance (or any OpenAI-compatible endpoint) serving:
  - an embedding model — `bge-m3` by default
  - one or more chat models for extraction / compliance verdicts — `gemma2:9b` by default
- `pdffonts` (poppler-utils) is used opportunistically for PDF text-layer detection; it's optional and the code falls back to PyMuPDF if it's missing.

## Setup

```bash
pip install -r requirements.txt
```

Edit `config/settings.py` and point `LLMConfig.base_url` / `EmbeddingConfig.base_url` at your own Ollama (or compatible) host — they default to a specific private IP used during development.

If your endpoint needs an API key:

```bash
export LLM_API_KEY=your-key-here
```

## Usage

**Streamlit app** (upload RFPs, manage the knowledge base, browse reports):

```bash
streamlit run app.py
```

**CLI** (`main.py`):

```bash
python main.py ingest --dir datasheets/            # ingest a folder of OEM datasheets
python main.py ingest --file datasheet.pdf          # ingest one file
python main.py search --query "10Gbps NGFW with SD-WAN"
python main.py stats
python main.py list-docs
python main.py delete --doc-id <doc_id>
```

**Python API**:

```python
from api import OEMKnowledgeBase

kb = OEMKnowledgeBase()
kb.ingest("datasheets/")
results = kb.search_for_requirement("NGFW with 20Gbps threat prevention throughput")
```

**Compliance engine directly** (e.g. for scripting/testing):

```bash
python test_compliance.py --pdf rfp.pdf --start 1 --end 10 --top 3
python test_compliance.py --json data/requirements/some_rfp_pp1-10.json
```

## Project structure

```
config/          Tunables: LLM, OCR, PDF, chunking, embedding, vector store
models/          Pydantic schemas for datasheets, chunks, requirements
ingestion/       PDF/OCR extraction, vendor + model identification, chunking
knowledge_base/  ChromaDB wrapper for the OEM spec store
rfp/             RFP page extraction, requirement extraction, embedding
compliance/      Retrieval, matching, ranking, report generation
services/        Thin clients for the Ollama embedding & chat endpoints
ui/              Helpers for the Streamlit "Manage Database" tab
app.py           Streamlit UI
main.py          CLI entry point
api.py           Python facade (OEMKnowledgeBase)
```

## Data on disk

Created automatically on first run, under `data/`:

```
data/raw/                 ingested source PDFs
data/processed/           intermediate parsed-document JSON
data/vector_store/        ChromaDB persistence
data/requirements/        extracted RFP requirements (JSON)
data/compliance_reports/  generated compliance reports (JSON + Markdown)
logs/                     pipeline logs and run summaries
```

## Notes

- Image-only/scanned PDFs are OCR'd via PaddleOCR if installed; otherwise pages with no extractable text are flagged with a warning rather than failing the run.
- Both pipelines must use the same embedding model so requirement vectors and spec vectors are comparable — don't change `EmbeddingConfig.model_name` for one without the other.
- Vendor name → model-prefix canonicalization is pre-configured for a handful of networking/security vendors (Fortinet, Palo Alto, Cisco, Juniper, etc.) in `config/settings.py`; unrecognized vendors still work, just without that extra canonicalization.
