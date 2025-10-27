# Mortgage Document Q&A System

AI-powered Retrieval-Augmented Generation (RAG) pipeline for mortgage document analysis. Upload multi-document PDFs and ask natural language questions with intelligent retrieval and source attribution.

## Features

- **Multi-Document Processing**: Handles up to 20 documents with 1000+ pages each
- **Hybrid Text Extraction**: Native PDF text + OCR for scanned documents (EasyOCR)
- **Intelligent Document Classification**: Automatic detection of 23+ mortgage document types
- **Three-Stage Retrieval Pipeline**:
  - Dense semantic search (all-mpnet-base-v2 embeddings)
  - Sparse keyword matching (BM25)
  - Cross-encoder reranking (ms-marco)
- **Query Expansion**: Converts vague queries into specific mortgage terminology
- **Smart Routing**: Automatically routes queries to relevant document types
- **Source Attribution**: Every answer includes page numbers and relevance scores

## Architecture
```
┌─────────────────┐
│   FastAPI REST  │
│       API       │
└────────┬────────┘
         │
    ┌────▼────┐
    │ Uploads │
    └────┬────┘
         │
    ┌────▼──────────────┐
    │  PDF Processing   │
    │  - Text Extract   │
    │  - OCR (EasyOCR)  │
    │  - Classification │
    └────┬──────────────┘
         │
    ┌────▼──────────┐
    │   Chunking    │
    │  (800/150)    │
    └────┬──────────┘
         │
    ┌────▼─────────────┐
    │  Vector Storage  │
    │  - FAISS Index   │
    │  - BM25 Index    │
    └────┬─────────────┘
         │
    ┌────▼──────────────┐
    │   Query Pipeline  │
    │  1. Expansion     │
    │  2. Dense Search  │
    │  3. BM25 Boost    │
    │  4. Reranking     │
    └────┬──────────────┘
         │
    ┌────▼──────────────┐
    │  LLM Generation   │
    │  (Llama 3.3 70B)  │
    └───────────────────┘
```

## Tech Stack

- **API Framework**: FastAPI
- **Vector Database**: FAISS
- **Metadata Storage**: PostgreSQL
- **Embeddings**: sentence-transformers (all-mpnet-base-v2, 768-dim)
- **Reranker**: cross-encoder (ms-marco-MiniLM-L-6-v2)
- **LLM**: Groq API (Llama 3.3 70B for generation, Llama 3.1 8B for classification)
- **OCR**: EasyOCR + OpenCV preprocessing
- **Chunking**: LlamaIndex SentenceSplitter
- **Containerization**: Docker + Docker Compose

## Quick Start

### Prerequisites

- Docker & Docker Compose
- 8GB RAM minimum
- Groq API key ([Get one here](https://console.groq.com/))

### Installation

1. **Clone repository**:
```bash
git clone <your-repo-url>
cd mortgage-doc-qa
```

2. **Set environment variables**:
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

3. **Start services**:
```bash
docker-compose up --build
```

4. **Access API**:
- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

## API Endpoints

### Upload Document
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@mortgage_application.pdf"
```

**Response**:
```json
{
  "status": "success",
  "doc_id": "uuid-here",
  "filename": "mortgage_application.pdf",
  "statistics": {
    "total_pages": 7,
    "documents_found": 3,
    "total_chunks": 12,
    "document_types": ["Loan Estimate", "Pay Stub", "Resume"],
    "processing_time": "14.2s"
  }
}
```

### Query Documents
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the loan amount?",
    "top_k": 5,
    "auto_route": true
  }'
```

### List Documents
```bash
curl -X GET "http://localhost:8000/api/v1/documents"
```

### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

## Development

### Run Locally (Without Docker)

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set environment variables**:
```bash
export GROQ_API_KEY="your_key_here"
export DATABASE_URL="sqlite:///data/mortgage_docs.db"
```

3. **Run API**:
```bash
uvicorn api.main:app --reload
```

### Run Tests
```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=api --cov=core --cov-report=html

# Specific test file
pytest tests/test_api.py -v
```

## Project Structure
```
mortgage-doc-qa/
├── api/
│   ├── __init__.py
│   └── main.py                 # FastAPI endpoints
├── core/
│   ├── __init__.py
│   └── document_store.py       # RAG pipeline implementation
├── database/
│   ├── __init__.py
│   └── models.py               # SQLAlchemy models
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_document_processing.py
├── data/
│   └── uploads/                # Uploaded PDFs stored here
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
└── README.md
```

## Performance Metrics

Based on test data (7-page multi-document PDF):

- **Document Processing**: ~15 seconds
- **Query Response Time**: 2-4 seconds
- **Routing Accuracy**: 88.4% average confidence on test queries

## Supported Document Types

The system automatically classifies 23+ mortgage-related document types:

**Financial Documents**: Loan Application, Loan Estimate, Closing Disclosure, Mortgage Note, Deed of Trust

**Income Verification**: Pay Stub, W-2, Tax Return, Employment Verification, Bank Statement, Asset Statement

**Property Documents**: Appraisal Report, Home Inspection, Title Insurance, Purchase Agreement

**Identity Documents**: Driver License, Social Security Card, Passport

**Other**: Credit Report, Gift Letter, Insurance Policy, Resume, Employment Contract




 License

MIT License

 Contact

Project Link: https://github.com/envy312/mortgage-doc-qa
