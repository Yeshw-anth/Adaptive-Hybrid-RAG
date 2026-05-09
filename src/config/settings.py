# src/config/settings.py

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    # Pydantic V2 model configuration
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # --- Core Path Settings ---
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    # --- General Settings ---
    APP_NAME: str = "Auto-Adaptive RAG"
    APP_VERSION: str = "0.1.0"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    RELOAD: bool = False

    # --- Loguru Settings ---
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"
    LOG_FILENAME: str = "app.log"
    LOG_ROTATION: str = "10 MB"
    LOG_RETENTION: str = "10 days"
    LOG_FORMAT_CONSOLE: str = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{extra[query_id]}</cyan> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    LOG_FORMAT_FILE: str = "{time} | {level} | {extra[query_id]} | {name}:{function}:{line} - {message}"
    LOG_STRUCTURED: bool = False

    @computed_field
    @property
    def LOG_FILE_PATH(self) -> Path:
        return self.BASE_DIR / self.LOG_DIR / self.LOG_FILENAME

    # --- RAG Pipeline Settings ---
    CLEAR_ON_RESTART: bool = False
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    RETRIEVER_TOP_K: int = 10
    RERANKER_TOP_N: int = 5
    FAST_PIPELINE_TOP_K: int = 5

    # --- LLM Settings ---
    GOOGLE_API_KEY: str
    GOOGLE_MODEL_NAME: str = "gemini-3.1-flash-lite-preview" # or any other model supported by Google Gemini
    DEFAULT_LLM_MODEL: str = "phi3"
    SMALL_LLM_MODEL: str = "phi3"
    LARGE_LLM_MODEL: str = "phi3" # used a small model based on device limitations can use a better model if possible
    MULTIMODAL_LLM_MODEL: str = "moondream"

    # --- Embedding Model Settings ---
    EMBED_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM: int = 384

    # --- Vector Store & Graph Settings ---
    STORAGE_DIR: str = "storage"
    GRAPH_DIR: str = "graph_store"
    VECTOR_STORE_DIR: str = "vector_store"
    CACHE_DIR: str = "cache"
    GRAPH_FILENAME: str = "knowledge_graph.graphml"
    GRAPH_STORE_TYPE: str = "networkx"  # "networkx" or "neo4j"

    # --- Neo4j Settings ---
    NEO4J_URI: str
    NEO4J_USER: str
    NEO4J_PASSWORD: str

    @computed_field
    @property
    def STORAGE_PATH(self) -> Path:
        return self.BASE_DIR / self.STORAGE_DIR

    @computed_field
    @property
    def GRAPH_PATH(self) -> Path:
        return self.STORAGE_PATH / self.GRAPH_DIR

    @computed_field
    @property
    def VECTOR_STORE_PATH(self) -> Path:
        return self.STORAGE_PATH / self.VECTOR_STORE_DIR
        
    @computed_field
    @property
    def CACHE_PATH(self) -> Path:
        return self.STORAGE_PATH / self.CACHE_DIR

    @computed_field
    @property
    def GRAPH_FILE_PATH(self) -> Path:
        return self.GRAPH_PATH / self.GRAPH_FILENAME

    # --- Data Settings ---
    UPLOAD_DIR: str = "uploaded_files"

    @computed_field
    @property
    def UPLOAD_PATH(self) -> Path:
        return self.BASE_DIR / self.UPLOAD_DIR

    # --- Tree-sitter Settings ---
    LANGUAGE_LIBRARY_FILENAME: str = "languages.dll"

    @computed_field
    @property
    def LANGUAGE_LIBRARY_PATH(self) -> Path:
        return self.BASE_DIR / "treesitter_build" / self.LANGUAGE_LIBRARY_FILENAME

    # --- Evaluation & Logging Settings ---
    OUTPUT_LOG_FILENAME: str = "response_log.jsonl"
    EVAL_REPORT_DIR: str = "src/evaluation/evaluation_reports"
    EVALUATION_DATASET_FILENAME: str = "evaluation_dataset.json"

    @computed_field
    @property
    def OUTPUT_LOG_FILE(self) -> Path:
        return self.BASE_DIR / self.LOG_DIR / self.OUTPUT_LOG_FILENAME

    @computed_field
    @property
    def EVAL_REPORT_PATH(self) -> Path:
        return self.BASE_DIR / self.EVAL_REPORT_DIR

    @computed_field
    @property
    def EVALUATION_DATASET_PATH(self) -> Path:
        return self.BASE_DIR / "src" / "evaluation" / self.EVALUATION_DATASET_FILENAME


    # --- Experimental Features ---
    IMAGE_OUTPUT_DIR: str = "image_cache"

    @computed_field
    @property
    def IMAGE_OUTPUT_PATH(self) -> Path:
        return self.BASE_DIR / self.IMAGE_OUTPUT_DIR

settings = Settings()