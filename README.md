# Financial Research AI Agent

## Overview

Financial Research AI Agent is an AI-powered system designed to assist users in researching and analyzing companies using financial data, company information, recent news, and AI-generated insights.

The goal of the project is to reduce the effort required to collect financial information from multiple sources and transform it into a structured research analysis.

## Problem Statement

Financial research often requires collecting information from multiple sources, reviewing financial data and news, calculating relevant metrics, and interpreting the information to understand a company's overall position.

This project aims to simplify this process by combining financial data collection, news retrieval, company research, and AI-powered analysis into a single research workflow.

## Objectives

* Research company information
* Collect relevant financial data
* Retrieve recent financial and company-related news
* Process and organize research data
* Analyze financial information using AI
* Generate structured research insights
* Provide a user-friendly interface for financial research

## Current Technology Stack

### Frontend

* Next.js
* React
* TypeScript
* Tailwind CSS

### Backend

* Python
* FastAPI

### AI

* Google Gemini
* Gemini 3.5 Flash
* AI-powered financial research analysis

### Data

* Financial data services
* Financial/news data sources
* Company information

## Current System Architecture

The current system follows a simple research workflow:

```text
User
  |
  v
Frontend (Next.js)
  |
  v
FastAPI Backend
  |
  +----> Company Information
  |
  +----> Financial Data
  |
  +----> News Data
  |
  v
Gemini AI
  |
  v
Research Analysis
```

## Current Project Progress

### Week 1 — Project Initialization

* Project requirements identified
* Initial architecture planned
* Repository created
* Frontend and backend structure established

### Week 2 — Backend and Gemini Integration

* FastAPI backend implemented
* Company API route added
* Financial data service implemented
* News service implemented
* Gemini service integrated
* Gemini 3.5 Flash model configured
* Successfully tested AI-generated analysis using AAPL (Apple)
* Initial Next.js frontend implemented
* Frontend connected to the research workflow

## Current Research Workflow

The current workflow is:

```text
Company / Stock Symbol
        |
        v
Financial Data
        |
        +----> Company Information
        |
        +----> Recent News
        |
        v
Research Context
        |
        v
Gemini 3.5 Flash
        |
        v
AI Research Analysis
```

The current implementation has successfully demonstrated the Gemini analysis workflow using **AAPL** as the test company.

## Project Structure

```text
financial-research-ai-agent/
│
├── Backend/
│   ├── main.py
│   ├── routes/
│   │   └── company.py
│   └── services/
│       ├── financial_data.py
│       ├── gemini_service.py
│       └── news_service.py
│
├── frontend/
│   ├── app/
│   ├── public/
│   ├── package.json
│   └── tsconfig.json
│
├── Data/
├── docs/
├── README.md
└── .gitignore
```

## Next Development Phase

The next phase will focus on improving the quality and usefulness of the research context provided to the Gemini model.

Planned work includes:

* Collecting more recent financial news
* Cleaning and normalizing news data
* Removing duplicate and irrelevant information
* Combining financial data and news into a unified research context
* Improving the Gemini research prompt
* Producing more structured AI research reports
* Improving the frontend presentation of research results
* Adding stronger financial analysis capabilities

## Future Work

* Financial document processing
* Public filing analysis
* Financial ratio calculations
* Company comparison
* Retrieval-Augmented Generation (RAG)
* Source-backed research reports
* Advanced AI research workflows
* Database integration
* Testing and deployment

## Project Status

**Current Stage: Active Development**

The initial project architecture, backend services, frontend foundation, and Gemini AI integration have been implemented. The current focus is on expanding and improving the financial research data pipeline to provide richer context for AI-generated analysis.
