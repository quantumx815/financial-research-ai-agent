# RAG Architecture

## Overview

Stage 7 introduces a Retrieval-Augmented Generation (RAG) layer to the Financial Research AI Agent. The goal is to augment Gemini's analysis with relevant company documents such as SEC filings, investor relations materials, and other trusted financial sources.

## Why ChromaDB

ChromaDB was selected because:
- It is a lightweight, local-first vector database suitable for internship projects.
- It supports persistent storage out of the box.
- It supports metadata filtering, which is required for multi-company support.
- It does not require a separate server process.
- It integrates cleanly with Python and the existing backend architecture.

## Shared Vector Database

All companies share a single ChromaDB collection. Company identity is represented through metadata rather than separate databases or collections.

Metadata schema:
- `company_symbol` (str): uppercase stock symbol, e.g. `AAPL`
- `company_name` (str): full company name
- `document_type` (str): e.g. `annual_report`, `10_k`, `press_release`
- `document_year` (int): year of the document
- `source` (str): e.g. `SEC`, `IR`
- `source_url` (str): original document URL
- `document_id` (str): unique document identifier

## Embeddings

Embeddings are generated using Google's `gemini-embedding-001` model through the existing `google-genai` SDK.

The embedding layer is abstracted behind `embedding_service.py` so the model can be changed later without rewriting the RAG system.

## Directory Structure

```
Data/
├── financial_documents/
│   ├── raw/            # original downloaded documents (future)
│   └── processed/      # cleaned/chunked text (future)
└── chroma/             # ChromaDB persistent storage
```

## RAG Service

`rag_service.py` provides:
- `add_document_chunk()`: store a text chunk with company metadata
- `query()`: retrieve relevant chunks filtered by `company_symbol`
- `list_companies()`: discover which companies have indexed documents
- `delete_company_documents()`: remove all documents for a company

The RAG service is independent from the Gemini analysis pipeline. Stage 6 analysis continues to work without RAG.

## Trusted Sources

Future document ingestion will prioritize:
1. Government/regulatory filings (e.g. SEC)
2. Official company investor-relations websites
3. Other reliable financial sources

Every stored chunk preserves source metadata for traceability.

## Dynamic Document Flow (Future)

The intended future flow is:
1. User searches a company.
2. Check whether relevant documents exist in ChromaDB.
3. If documents exist, retrieve relevant chunks.
4. If documents do not exist, fetch from trusted online sources.
5. Process, chunk, embed, and store in ChromaDB.
6. Retrieve relevant chunks and inject into the research package.

Stage 7.1 establishes the storage and retrieval foundation only.

## Missing Documents

If a company has no indexed documents, the system should continue functioning using financial data and news only. RAG is an enhancement, not a dependency.

## Next Steps (Stage 7.2)

Stage 7.2 will build on this foundation by:
- Implementing document discovery and downloading.
- Implementing chunking and ingestion.
- Connecting retrieved documents to the research package.
- Injecting RAG evidence into the Gemini prompt.
