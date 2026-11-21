
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

## Lab 3: Vector Representations

Use Python 3.12. Install the verified CPU runtime and project dependencies
in the existing virtual environment before running the Lab 3 commands:

```powershell
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run from the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe -m src.index_lab3
.\.venv\Scripts\python.exe -m src.search_lab3 "A bendable USB lamp for my laptop"
.\.venv\Scripts\python.exe -m src.evaluate_lab3
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_lab3.py"
```

On Windows, if the repository path contains non-ASCII characters, use a
persistent ASCII directory for Chroma. To persist local paths across shells,
create the ignored file `config/lab3.local.json`:

```json
{
  "index_root": "C:/path/with/ascii/SmartProductFinder/chroma",
  "model_cache": "C:/path/with/ascii/SmartProductFinder/huggingface/hub"
}
```

The first run downloads the pinned MiniLM model into the configured cache.
`SMART_PRODUCT_INDEX_ROOT` or `--index-root` can override the index location.
`HF_HOME` or `HF_HUB_CACHE` can override the model cache.

Chroma uses cosine distance with HNSW `ef_search` set to 10 or 100. The CLI
displays cosine similarity as `1 - distance`. Use `--config ef100` to select
the other index. Model inputs are truncated at 256 wordpieces. Prices are
historical INR values. Evaluation defaults to `data/lab-3/evaluation-results.json`.
Pass `--output PATH` to preserve an existing result during a comparison run.

## Author

GitHub: [TopoliukNure](https://github.com/TopoliukNure)
