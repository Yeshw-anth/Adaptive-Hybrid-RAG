# src/config/settings.py

import os
from pathlib import Path

# --- Core Path Settings ---
# Dynamically determine the project's base directory.
# This allows for flexible path construction throughout the application.
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# --- General Settings ---
APP_NAME = "Auto-Adaptive RAG"
APP_VERSION = "0.1.0"

# --- Logging Settings ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = BASE_DIR / "logs"
LOG_FILE_PATH = LOG_DIR / "app.log"

# --- RAG Pipeline Settings ---
# Determines whether to clear the vector store on application restart.
# Set to True during development to ensure a fresh index.
# Set to False in production to persist the index across restarts.
CLEAR_ON_RESTART = True

# The target size for text chunks in tokens.
CHUNK_SIZE = 512

# Number of tokens to overlap between chunks.
CHUNK_OVERLAP = 50

# The number of most similar chunks to retrieve from the vector store before reranking.
RETRIEVER_TOP_K = 10

# The number of chunks to keep after reranking.
RERANKER_TOP_N = 5

# The number of most similar chunks to retrieve for the Fast pipeline.
FAST_PIPELINE_TOP_K = 5

# --- Ingestion Settings ---
# Select the ingestion strategy:
# 'unstructured': (Recommended) Uses the 'unstructured' library for intelligent,
#                 structure-aware parsing of all document types.
# 'manual':       Uses a set of simpler, file-type-specific loaders.
INGESTION_STRATEGY = "unstructured"

# --- LLM & Embedding Model Settings ---
# Specifies the embedding model to use for document and query vectorization.
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
DEFAULT_LLM_MODEL = "phi3"
SMALL_LLM_MODEL = "phi3"
LARGE_LLM_MODEL = "phi3"

# --- Vector Store Settings ---
# Directory to persist the FAISS vector store index.
PERSIST_DIR = BASE_DIR / "storage"

# --- Data Settings ---
# Directory for user-uploaded files.
UPLOAD_DIR = BASE_DIR / "uploaded_files"

# --- Tree-sitter Settings ---
# Path to the compiled language library for code parsing.
LANGUAGE_LIBRARY_PATH = BASE_DIR / "treesitter_build" / "languages.dll"

# --- Evaluation & Logging Settings ---
# Path to the structured JSONL file for logging outputlogs traces.
OUTPUT_LOG_FILE = BASE_DIR / "src" / "core" / "outputlogs" / "logs" / "response_log.jsonl"
# Directory to save evaluation reports.
EVAL_REPORT_DIR = BASE_DIR / "src" / "evaluation" / "evaluation_reports"

# --- Experimental Features ---
# Enable this flag to activate the image captioning module.
# This requires a multi-modal model (e.g., llava) to be running via Ollama.
MULTIMODAL_LLM_MODEL = "moondream"
IMAGE_OUTPUT_DIR = BASE_DIR / "image_cache"