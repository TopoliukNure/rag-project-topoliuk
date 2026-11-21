
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

[Flipkart Products v1](https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products)
is the main corpus for Labs 2–5. It contains 1,000 selected English-language product
documents across 28 top-level categories. Source listings were crawled in 2015–2016.
The data is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

Catalog budgets are expressed in INR. Filter by the structured `discounted_price`
field and retain `retail_price` for display. See the [data card](reports/lab-2/data-card.md)
for selection, attribution, preprocessing, and the limits of the historical prices.

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

## Prepare the Lab 2 corpus

Run from the repository root with Python 3.11 or newer after installing the dependencies:

```bash
python src/prepare_flipkart.py
```

The script downloads the original Kaggle ZIP and checks the CSV's SHA-256. Supply
`--source PATH` to use the same archive or CSV from disk and `--output-root PATH`
to generate an independent copy. Outputs are written under `data/lab-2/` and
`reports/lab-2/`.

## Development Status

Lab 1 covers RAG system design, and Lab 2 adds the reproducible Flipkart data
pipeline. Vector indexing, hybrid retrieval, generation, and evaluation follow
in Labs 3–5.

## Laboratory Works

1. **Lab 1:** RAG System Design
2. **Lab 2:** Data Collection and Processing
3. **Lab 3:** Vector Representations
4. **Lab 4:** Retrieval Algorithms and Generation
5. **Lab 5:** System Testing

## Author

GitHub: [TopoliukNure](https://github.com/TopoliukNure)
