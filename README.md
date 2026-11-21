
# Smart Product Finder

Smart Product Finder is a RAG-based product recommendation system developed as part of the "Hybrid Artificial Intelligence Systems Development Technologies" course at Kharkiv National University of Radio Electronics (NURE).

## Project Description

The system helps users find suitable products based on natural language queries in English.

Users can specify their preferences, including product purpose, characteristics, and price limits. The system searches a product catalog and generates recommendations based on retrieved information.

The project uses Retrieval-Augmented Generation (RAG) to combine semantic search with large language model capabilities.

## Technology Stack

- **Python** — programming language
- **ChromaDB** — vector database
- **Sentence Transformers (all-MiniLM-L6-v2)** — text embeddings
- **Qwen 2.5 3B** — language model
- **Ollama** — local LLM execution
- **BM25 + RRF** — hybrid search
- **CLI** — command-line interface

## Dataset

- **Source:** Amazon Berkeley Objects (ABO)
- **Dataset size:** 1,000 product records (planned subset)
- **Language:** English
- **Categories:** Various product categories
- **License:** CC BY 4.0

Dataset: https://registry.opendata.aws/amazon-berkeley-objects/

Product price availability will be verified during data preparation.

## Project Structure

```text
rag-project-topoliuk/
├── src/                 # Source code
├── config/              # Configuration files
├── reports/
│   └── lab-1/           # Laboratory reports
├── main.py              # Application entry point
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/TopoliukNure/rag-project-topoliuk.git
cd rag-project-topoliuk
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Download Ollama from:

https://ollama.com/

Download the language model:

```bash
ollama pull qwen2.5:3b
```

## Development Status

The project is currently in the design phase (Lab 1).

The data processing pipeline, vector indexing, retrieval algorithms, and generation functionality will be implemented in subsequent laboratory works.

## Laboratory Works

1. **Lab 1:** RAG System Design
2. **Lab 2:** Data Collection and Processing
3. **Lab 3:** Vector Representations
4. **Lab 4:** Retrieval Algorithms and Generation
5. **Lab 5:** System Testing

## Author

GitHub: [TopoliukNure](https://github.com/TopoliukNure)
