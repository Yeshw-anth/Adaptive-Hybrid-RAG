# Auto-Adaptive RAG: A Production-Ready, Multi-Modal RAG System

This project implements an advanced, auto-adaptive Retrieval-Augmented Generation (RAG) system designed for production use. It intelligently analyzes documents and queries to select the optimal processing strategy, delivering accurate, context-aware answers from a diverse range of file types.

This is not just a simple RAG pipeline; it is a sophisticated orchestration engine that combines multi-modal understanding, adaptive chunking, and dynamic retrieval strategies to tackle real-world document complexity.

## System Architecture

![System Architecture](assets/architecture.png)

## Core Features

- **Intelligent, Structure-Aware Ingestion**: Automatically parses complex documents (PDFs, DOCX, etc.) using the `unstructured` library to identify and preserve structural elements like headings, tables, text blocks, and images.
- **Adaptive Chunking Engine**: Goes far beyond simple fixed-size chunking. It applies specialized strategies for different content types:
  - **Text**: Uses a text splitter to split text into sentence and paragraph boundaries.
  - **Code**: Employs `Tree-sitter` for syntax-aware chunking, keeping logical code blocks (functions, classes) intact.
  - **Images**: Utilizes a multi-modal Vision-Language Model (VLM) to generate detailed captions, making visual information searchable.
  - **Tables**: Processes tables as structured data to preserve their tabular context.
- **Dynamic Strategy Router**: The "brain" of the system. It analyzes each user query to determine its intent, complexity, and required content types, then dynamically selects the most effective retrieval pipeline.
- **Multiple Retrieval Pipelines**: 
  - **Fast Pipeline**: For simple queries, using a quick vector search.
  - **Accurate Pipeline**: For complex questions, using hybrid search (vector + keyword) and a Cross-Encoder reranker for maximum relevance.
  - **Specialized Pipelines**: Dedicated routes for `code`, `keyword`, and `structured` (table) queries.
- **Configurable Ingestion**: Supports two ingestion backends, switchable via a single setting for benchmarking:
  - `unstructured`: The modern, powerful, and recommended default.
  - `manual`: A legacy system of file-specific loaders.
- **Full Observability**: Detailed logging for every stage, from ingestion to final response generation, enabling easy debugging and performance analysis.

## What We Achieved

This project successfully evolved from a basic RAG prototype into a robust, production-ready system. Key achievements include:

1.  **Eliminated Over-Engineering**: We systematically removed complex, unnecessary components (like the `SemanticSplitter` and `Adjudicator`) in favor of a more direct and powerful architecture.
2.  **Unified Ingestion Pipeline**: We replaced a clunky, manual system of individual file loaders with a single, elegant `ChunkingEngine` powered by `unstructured` and a factory of specialized chunking strategies.
3.  **Intelligent Content-Aware Processing**: The system no longer treats all content as plain text. It understands the difference between prose, code, tables, and images, and processes each accordingly.
4.  **Robustness and Configurability**: We removed hardcoded "magic numbers" and introduced clear configuration options in `settings.py`, making the system maintainable and easy to tune.
5.  **Pragmatic, Production-Focused Design**: Every architectural decision was made with a focus on real-world performance, scalability, and maintainability.

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd auto-adaptive-rag
    ```

2.  **Install Tesseract OCR:**
    The `unstructured` library requires Tesseract for processing images and scanned PDFs. This is a system-level dependency.

    - **Windows:** Download and run the installer from the [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) page. **Important:** Add Tesseract to your system `PATH` during installation.
    - **macOS:** `brew install tesseract`
    - **Debian/Ubuntu:** `sudo apt-get install tesseract-ocr`

3.  **Create a Virtual Environment and Install Dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

4.  **Build Code Parsing Grammars:**
    This one-time step compiles the `tree-sitter` grammars needed for intelligent code chunking.
    ```bash
    python treesitter_build/build_grammars.py
    ```

## How to Use the System

### 1. Run the FastAPI Server

This command starts the main application. For development, it's recommended to use `uvicorn` for features like hot-reloading.

**For development (with auto-reload):**
```bash
uvicorn main:app --reload
```

**For  simple execution:**
```bash
python main.py
```

The server will be running at `http://127.0.0.1:8000`.

### 2. Run the Streamlit UI

In a separate terminal, start the Streamlit web interface.

```bash
streamlit run st_app.py
```

### 3. Upload and Process Documents

Use the "Upload Documents" section in the sidebar of the Streamlit application to upload your files. After uploading, click the "Process Uploaded Files" button to begin ingestion. You will see the processing status in the UI.

*File upload and processing status*
![File Upload and Processing](assets/file_upload.png)

### 4. Ask a Question

Once your files are processed, you can ask questions using the chat input. The system will analyze your query, select the best strategy, and generate a response.

*Example of a query and its response*
![Query and Response](assets/query_and_response.png)

### 5. Review Sources and Confidence

For each response, the system provides the source chunks it used for generation and a confidence score. This allows you to verify the accuracy of the answer and understand its origin.

*Sources and confidence score for a response*
![Sources and Confidence](assets/sources_and_confidence.png)

## Configuration

Key settings can be modified in `src/config/settings.py`:

- `INGESTION_STRATEGY`: Choose the document processing backend.
  - `"unstructured"`: (Default) The modern, intelligent, structure-aware pipeline.
  - `"manual"`: The legacy pipeline with individual file loaders. Useful for benchmarking.
- `CHUNK_SIZE` / `CHUNK_OVERLAP`: Control the size and overlap of text chunks.
- `EMBED_MODEL_NAME`: Specify the embedding model to use.
- `CLEAR_ON_RESTART`: Set to `True` during development to clear the vector store on each server start.

## Future Scope

This project provides a powerful foundation that can be extended in many ways:

- **Knowledge Graph Integration**: Augment the vector store with a knowledge graph to enable more complex, multi-hop reasoning.
- **Agent-Based Workflows**: Develop autonomous agents that can use the RAG system as a tool to perform research, analysis, and report generation.
- **Advanced Evaluation Suite**: Expand the evaluation framework to continuously measure performance on metrics like answer relevance, faithfulness, and latency.
- **UI Enhancements**: Build a more advanced user interface (e.g., using the included `st_app.py` Streamlit app) to visualize retrieval steps and allow for interactive feedback.

## Project Structure

```
auto-adaptive-hybrid-rag/
│
├── src/
│   ├── api/                  # FastAPI endpoints
│   ├── chunking/             # Adaptive chunking logic
│   │   └── strategies/       # Strategies for different content types (text, code, image)
│   ├── config/               # Configuration settings
│   ├── core/                 # Core orchestration and system logic
│   │   ├── decision/         # Decision engine for routing
│   │   ├── llm/              # Language model wrappers
│   │   ├── pipelines/        # RAG pipelines (fast, accurate, etc.)
│   │   ├── retrieval/        # Retrieval and reranking logic
│   │   └── strategy/         # Query analysis and strategy selection
│   ├── data/                 # Data schemas and models
│   │   └── ingestion/        # Document loading and parsing
│   ├── evaluation/           # Evaluation scripts and datasets
│   └── experimental/         # Experimental features
│
├── tests/                    # Test suite
├── assets/                   # Images and other static assets
├── main.py                   # FastAPI application entry point
├── st_app.py                 # Streamlit UI application
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```