# Financial Research AI Agent

## Overview

Financial Research AI Agent is an AI-powered system designed to research and analyze publicly traded companies using financial data, recent news, official SEC filings, and AI-generated insights.

The system combines information from multiple sources and uses Retrieval-Augmented Generation (RAG) to provide more evidence-supported financial research through a user-friendly web interface.

## Objectives

* Collect company and financial information
* Retrieve relevant and recent financial news
* Access official SEC financial filings
* Process and retrieve relevant financial document information
* Generate structured, evidence-supported financial analysis using AI
* Present research results clearly through the frontend

## Technology Stack

| Component          | Technology                               |
| ------------------ | ---------------------------------------- |
| Frontend           | Next.js, React, TypeScript, Tailwind CSS |
| Backend            | Python, FastAPI                          |
| AI Analysis        | Google Gemini                            |
| Embedding Model    | Gemini Embeddings (production path) + BGE `BAAI/bge-small-en-v1.5` (local evaluation path) |
| Vector Database    | ChromaDB                                 |
| Financial Data     | yfinance (via Yahoo Finance)             |
| News               | yfinance / financial news sources        |
| Official Documents | SEC EDGAR                              |

## Embedding and Retrieval Paths

The project supports two embedding paths for SEC document retrieval:

* **Gemini Embeddings (existing production path)** — document chunks from SEC filings are embedded with the Gemini embedding model and stored in the `financial_documents` ChromaDB collection. This remains the primary production retrieval path.
* **Local BGE embeddings (local evaluation/integration path)** — `BAAI/bge-small-en-v1.5` produces 384-dimensional embeddings on-device via SentenceTransformers (CPU). These are stored in a separate `financial_documents_bge` ChromaDB collection used to evaluate a local retrieval and question-answering path. This work reduces dependence on Gemini embedding quota and supports dynamic company research without relying on a preloaded set of companies.

The BGE path is an **additional, isolated evaluation path**. It is not a complete production migration from Gemini embeddings.

## Current Architecture

```text
    User
     |
     v
Next.js Frontend
     |
     v
FastAPI Backend
     |
+----------------------------------------------+
|  Company   Recent   SEC Filings             |
| Financial    News       |                    |
|    Data            Document Processing       |
|     |                  |                     |
|     |            Chunking/Embedding         |
|     |                  |                     |
|     |  +----------------+----------------+   |
|     |  |                |                |   |
|     |  v                v                |   |
|     | Gemini      Local BGE (bge-         |   |
|     | Embeddings  small-en-v1.5, 384-d)   |   |
|     |  |                |                |   |
|     |  | ChromaDB (Gemini)  ChromaDB     |   |
|     |  | (financial_documents)  |        |   |
|     |  |            (financial_documents_bge) |  |
+-----|-----------------------------|--------+   |
     |                     |                     |
     v                     v                     |
RAG Evidence Retrieval  (BGE QA path)           |
     |                     |                     |
     +----------+----------+                     |
               |                                  |
               v                                  |
        Gemini AI Analysis                        |
               |                                  |
               v                                  |
    Structured Research Report                     |
               |                                  |
               v                                  |
          Frontend Display                           |
```

## Current Development

The system builds on the integrated RAG-enhanced research flow and continues to expand retrieval quality, local embedding evaluation, and the reliability of the research pipeline.

### RAG Reliability Improvements

Reliability around SEC document processing, ingestion, retrieval, and embedding-related operations has been improved. The system degrades gracefully when individual stages fail — for example, retrieval failures no longer block the overall analysis response, and embedding quota or API failures are handled with fallback behavior so that available evidence and news can still be surfaced.

### SEC Document Processing Improvements

SEC EDGAR filings continue to serve as official, primary-source financial evidence. Document processing and chunking for these filings has been improved, including more structured document and chunk metadata, more stable chunk identifiers, and ingestion/reuse handling that reduces duplicate or stale chunks. Filings are treated as authoritative evidence sources for risk and opportunity analysis.

### Local BGE Embedding Evaluation and Integration

`BAAI/bge-small-en-v1.5` is evaluated as a local, on-device embedding option using SentenceTransformers (CPU, 384-dimensional vectors). A dedicated BGE ChromaDB collection (`financial_documents_bge`) is used to keep this work separate from the production Gemini path. The BGE path is intended to reduce dependence on Gemini embedding quota and to prototype a local-first retrieval workflow. This is an **evaluation integration path, not a completed production migration from Gemini embeddings.**

### Dynamic BGE Financial Research / QA Flow

The BGE-based research workflow can dynamically research and ingest a company when the required information is not already available in the local index, then retry retrieval. This broadens the set of companies that can be researched beyond a preloaded list, while still grounding the analysis in retrieved SEC evidence.

### Financial Metric Period Context

The analysis instructions now surface clearer context about the period associated with each financial metric: trailing twelve month (TTM) versus latest-period and live market values. The model is instructed to flag period differences or uncertainty rather than reconciling figures across mismatched bases. This is an **improvement to period context, not a guaranteed resolution** of every possible financial-period interpretation issue.

## Current Project Structure

```text
financial-research-ai-agent/
│
├── Backend/
│   │
│   ├── main.py
│   │
│   ├── routes/
│   │   └── company.py
│   │
│   ├── services/
│   │   ├── financial_data.py
│   │   ├── news_service.py
│   │   ├── gemini_service.py
│   │   ├── embedding_service.py
│   │   ├── rag_service.py
│   │   ├── sec_service.py
│   │   ├── document_service.py
│   │   ├── document_processor.py
│   │   ├── ingestion_service.py
│   │   ├── bge_embedding_service.py
│   │   ├── bge_rag_service.py
│   │   ├── bge_query_service.py
│   │   ├── company_availability_service.py
│   │   └── dynamic_research_service.py
│   │
│   └── tests/
│       ├── test_company_routes.py
│       ├── test_gemini_service.py
│       ├── test_rag_integration.py
│       ├── test_rag_service.py
│       ├── test_bge_service.py
│       ├── test_bge_query_service.py
│       ├── test_company_availability_service.py
│       ├── test_financial_qa_service.py
│       └── test_document_processor.py
│
├── Data/
│   ├── financial_documents/
│   │   ├── raw/
│   │   └── processed/
│   └── chroma/                  # Production ChromaDB (collection: financial_documents)

├── frontend/
│   ├── app/
│   ├── public/
│   ├── package.json
│   └── tsconfig.json
│
├── docs/
│   ├── project-overview.md
│   └── rag_architecture.md
│
├── README.md
├── .gitignore
└── LICENSE
```

### Backend

The `Backend` directory contains the main research and AI processing logic.

* **`main.py`** — Starts the FastAPI application.
* **`routes/company.py`** — Handles company research requests and connects the different research services.
* **`financial_data.py`** — Retrieves company and financial information.
* **`news_service.py`** — Retrieves and processes recent company-related news.
* **`gemini_service.py`** — Sends research context to Gemini and generates the structured analysis.
* **`embedding_service.py`** — Generates vector embeddings for financial document chunks.
* **`rag_service.py`** — Handles vector storage and relevant document retrieval using ChromaDB.
* **`sec_service.py`** — Communicates with SEC EDGAR to identify and retrieve official company filings.
* **`document_service.py`** — Handles financial document retrieval and management.
* **`document_processor.py`** — Processes documents and divides them into useful text chunks.
* **`ingestion_service.py`** — Connects document processing, embedding generation, and ChromaDB ingestion.
* **`bge_embedding_service.py`** — Generates local BGE (bge-small-en-v1.5, 384-dim) embeddings via SentenceTransformers.
* **`bge_rag_service.py`** — BGE-backed retrieval and vector storage in the local evaluation ChromaDB collection.
* **`bge_query_service.py`** — Semantic retrieval over the BGE collection.
* **`company_availability_service.py`** — Checks whether indexed SEC evidence exists for a given company.
* **`dynamic_research_service.py`** — Dynamically researches and ingests a company for the BGE QA flow when needed.

### Data

The `Data` directory contains the project's financial documents and the persistent ChromaDB database (committed).

* **`financial_documents/raw/`** — Retrieved SEC filings before processing.
* **`financial_documents/processed/`** — Cleaned and structured document data ready for ingestion.
* **`chroma/`** — Production ChromaDB (collection `financial_documents`, Gemini embeddings).

BGE evaluation uses a **separate local ChromaDB collection** (`financial_documents_bge`) in an evaluation/integration environment on the local machine. This collection and the `chroma_bge/` and `chroma_eval/` directories are local, untracked artifacts — not committed project directories.

### Frontend

The `frontend` directory contains the web interface through which users search for companies and view the generated research analysis.

The frontend communicates with the FastAPI backend and displays the structured AI-generated results.

### Tests

The `Backend/tests` directory contains automated tests for:

* Company API and availability functionality
* Gemini analysis
* RAG and BGE services
* Financial QA with BGE retrieval
* SEC document processing
* Complete RAG and QA integration

Test suites for the BGE path are isolated from the production BGE collection so that no test data is treated as production financial data.

### Documentation

The `docs` directory contains project documentation and the RAG architecture description.

## Current Project Status

**Status: Active Development — RAG-Enabled Research System Integrated**

### Currently Working

* Company information retrieval
* Financial data retrieval
* Recent news retrieval
* Gemini AI analysis
* Structured financial health analysis
* SEC filing retrieval
* SEC document processing and chunking
* Structured document/chunk metadata
* ChromaDB vector storage (production)
* RAG-based evidence retrieval (Gemini embeddings)
* RAG-enhanced Gemini analysis
* Local BGE (bge-small-en-v1.5) embedding evaluation/integration
* Dynamic BGE company research and ingestion for the QA flow
* Finance-metric period context improvements in analysis output
* Automated tests with BGE/test isolation from production data

### Current Improvements

* Improved reliability around SEC document processing, ingestion, retrieval, and embedding-related failures, with graceful degradation and embedding quota/failure fallback handling.
* Structured metadata and chunk identity improvements for more stable, deduplicated SEC ingestion.
* Local BGE evaluation/integration path to reduce dependence on Gemini embedding quota.
* Clearer financial-period context for metrics in AI analysis output.

## Next Development Direction

The next development phase continues to focus on improving research intelligence through **multi-source evidence** and stronger financial reasoning.

Planned focus areas:

* Improve SEC evidence retrieval quality and ranking.
* Continue evaluating the local BGE embedding path against the existing Gemini embedding path.
* Strengthen financial QA and dynamic company research on the BGE workflow.
* Broaden evidence sources to support more reliable research.
* Advance the multi-source evidence toward an agentic research workflow.

```text
Multi-Source Evidence
        ↓
Better Retrieval (BGE vs. Gemini embeddings)
        ↓
More Reliable Financial Research
        ↓
Agentic Research Workflow
```

## Disclaimer

This project is intended for financial research and educational purposes only. The generated analysis is informational and does not constitute personalized investment advice.
