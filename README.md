# Real-World Benchmarking: WaveflowDB vs Pinecone

A comprehensive benchmarking framework for comparing vector database retrieval performance using real-world RAG (Retrieval-Augmented Generation) queries and documents.

## 📋 Overview

The benchmark includes 127 diverse queries across multiple domains (medical reports, insurance policies, legal documents, academic papers, and more) with ground-truth relevant document mappings.

## 🎯 Key Features

- **Multi-system Comparison**: Side-by-side evaluation of Pinecone and WaveflowDB
- **Comprehensive Metrics**: Precision, Recall, F1-score, MRR, and nDCG@k
- **Scalable Processing**: Multiprocessing support for parallel query evaluation
- **Hybrid Filtering**: Optional query transformation for semantic VQL filtering
- **Performance Tracking**: Detailed timing metrics (embedding, query, total)
- **Flexible Configuration**: Environment-based configuration via `.env` file

## 📁 Project Structure

```
real_world_benchmarking/
├── README.md                          # This file
├── Instructions.txt                   # Step-by-step execution guide
├── utils.py                           # Utility functions
├── 1_prepare_data_id.py              # Data preparation script
├── 2_waveflow_upload.py              # WaveflowDB data upload
├── 3_pinecone_upload.py              # Pinecone data upload & metrics
├── 4_run_pipeline.py                 # Main evaluation pipeline
├── source_data/
│   └── query_map.csv                 # Test queries with ground truth
├── staging_data/                     # Sample data (provided)
│   └── test.ipynb
└── (auto-generated directories)
    ├── processed_data/               # Output from step 1
    ├── results/                      # Final results & metrics
    └── logs/                         # Execution logs
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Required packages: `pandas`, `spacy`, `sentence-transformers`, `pinecone`, `waveflowdb-client`,`python-dotenv`, `PyPDF2`, `python-docx`, `openpyxl`

### Installation

1. **Clone or navigate to the project**

```bash
cd real_world_benchmarking
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
   Create a `.env` file in the root directory:

```env
# Data paths
DATA_DIR_SOURCE="./source_data"
DATA_DIR_TARGET="./staging_data"
DATA_DIR_FORMATTED="./processed_data"
LOGS=logs
RESULTS_DIR="./results"
QUERY_MAP_SOURCE_FILE=query_map.csv
DELIMITER=XOXO

# Pinecone configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
BATCH_SIZE=50

# WaveflowDB configuration
WAVEFLOWDB_API_KEY=your_waveflow_api_key
BASE_URL=your_waveflow_base_url
NAMESPACE=your_namespace
USER_ID=your_user_id
SESSION_ID=your_session_id

# Model configuration
MODEL_NAME=all-MiniLM-L6-v2
TYPE=" OR "

# Pipeline configuration
TOP_K=10
MAX_WORKERS_QUERY=2
```

## Obtaining API Keys

- **Pinecone:**

  - Visit the Pinecone console at `https://app.pinecone.io` and sign in or create an account.
  - Create a project (or select an existing one) and navigate to the "API Keys" section.
  - Create a new API key, copy the key value and add it to your `.env` as:

    ```env
    PINECONE_API_KEY=your_pinecone_api_key
    PINECONE_INDEX_NAME=your_index_name
    ```

  - Note: Pinecone may provide separate keys for different environments (dev/prod). Keep keys secret.

- **WaveflowDB:**

  - Visit the Pinecone console at `https://db.agentanalytics.ai`and sign in or create an account.
  - Create a database and then navigate to "API Endpoints"

- Create a new API key, copy the key value and add it to your `.env` as:

  ```env
  WAVEFLOWDB_API_KEY=your_waveflow_api_key
  BASE_URL=https://your-waveflow-host.example.com
  NAMESPACE=your_namespace(Database)
  USER_ID=your_user_id
  SESSION_ID=your_session_id
  ```

## 📊 Execution Steps

### Step 1: Prepare Data

Processes the source query map and document files, creating normalized filenames and matching queries to documents.

```bash
python 1_prepare_data_id.py
```

**Output**: `processed_data/` folder with organized documents and `prepare_data_output.csv`

### Step 2: Upload to WaveflowDB

Uploads processed documents to the WaveflowDB instance.

```bash
python 2_waveflow_upload.py
```

### Step 3: Upload to Pinecone

Generates embeddings and uploads documents to Pinecone with performance metrics.

```bash
python 3_pinecone_upload.py
```

**Output**: `results/pinecone_upload_logs.xlsx` with batch performance metrics

### Step 4: Run Evaluation Pipeline

Executes the main benchmarking pipeline with parallel query evaluation across multiple configurations.

```bash
python 4_run_pipeline.py
```

**Output**:

- `results/waveflow_results_top{k}_hybrid{filter}.csv` - Waveflow results
- `results/pinecone_results_top{k}_hybrid{filter}.csv` - Pinecone results
- `results/merged_results_top{k}_hybrid{filter}.csv` - Combined results per setting
- `results/all_results.xlsx` - Final results with raw and aggregated sheets

## 📈 Evaluation Metrics

### Per-Query Metrics

- **Precision**: Proportion of retrieved documents that are relevant
- **Recall**: Proportion of relevant documents that are retrieved
- **F1-Score**: Harmonic mean of precision and recall
- **MRR (Mean Reciprocal Rank)**: Position of first relevant document (1/rank)
- **nDCG@k (Normalized Discounted Cumulative Gain)**: Ranking quality accounting for position

### Performance Metrics

- **Embedding Time**: Time to generate query embeddings
- **Query Time**: Time to search the vector database
- **Total Time**: Combined embedding + query time

## 🔍 Query Dataset

The `query_map.csv` contains 127 test queries across diverse domains:

| Category   | Examples                                        |
| ---------- | ----------------------------------------------- |
| Medical    | Lab reports (CBC, Lipid Profile, Thyroid, etc.) |
| Insurance  | ICICI Pru policies and brochures                |
| Legal      | Indian Penal Code, contract clauses, case laws  |
| Corporate  | Annual reports, policy documents                |
| Academic   | Research papers, dissertations                  |
| Literature | Classic novels, short stories                   |
| Other      | Children's books, technical papers              |

Each query includes:

- **Query ID**: Unique identifier
- **Question**: Natural language query
- **Doc IDs**: Ground truth relevant documents (space-separated)

## ⚙️ Configuration Options

### Top-K Values

Evaluate retrieval quality at different result set sizes:

```python
TOP_K_LIST = [2, 5, 10]
```

### Hybrid Filtering

Test with and without semantic filtering:

```python
HYBRID_FILTER_LIST = [True, False]
```

- **True**: Waveflow uses VQL transformation; Pinecone skipped
- **False**: Both systems use semantic search only

### Parallel Processing

Control parallel query evaluation:

```env
MAX_WORKERS_QUERY=4  # Number of processes for parallel evaluation
```

## 📝 Utility Functions

### `utils.py` Key Functions

- **`extract_keywords(text)`**: Extracts ranked keywords using SpaCy (PROPN > NOUN)
- **`convert_to_sql_vql(query, type)`**: Transforms natural language to VQL with keyword pairs
- **`clean_filename_base(fname)`**: Normalizes filenames (ASCII, lowercase, safe chars)
- **`rename_files_in_folder(folder_path)`**: Batch renames files for consistency
- **`parse_passage_ids(raw)`**: Parses document IDs from various formats

## 📤 Results Format

### Raw Results CSV

Contains per-query evaluation metrics:

```
query_id, query_text, precision, recall, f1, mrr, ndcg,
embedding_time, query_time, total_time, retrieved_docs,
relevant_docs, status, system
```

### Aggregated Results (Excel)

Summary statistics grouped by system, top_k, and hybrid_filter:

```
system, top_k, hybrid_filter, avg_precision, avg_recall,
avg_f1, avg_mrr, avg_ndcg
```

## 🔧 Troubleshooting

### File Encoding Issues

If you encounter encoding errors with source documents, the code handles `cp1252` encoding for CSV files and `utf-8` with error ignore for text files.

### Empty Results

Verify that:

- Documents are properly uploaded to both systems
- Query embeddings are being generated correctly
- API credentials are valid in `.env` file

### Performance Optimization

- Reduce `BATCH_SIZE` for lower memory usage
- Increase `MAX_WORKERS_QUERY` for faster parallel processing (if CPU allows)
- Use smaller embedding models for faster processing

## 📋 Log Files

Execution logs are saved to the `logs/` directory:

- `pipeline.log` - Main pipeline execution logs
- `prepare_data_output.csv` - Data preparation summary

