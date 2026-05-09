import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import uvicorn
import shutil
import spacy
import subprocess
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI


from src.api.routes import router as api_router
from src.config.settings import settings
from src.core.logging_config import logger
from src.core.system_builder import SystemBuilder
from treesitter_build.build_grammars import build_tree_sitter_grammars
from src.core.graph.graph_recovery import graph_recovery

# --- Pre-startup checks and setup ---

def clear_volatile_directories():
    """Clears directories that should be reset on startup if configured."""
    if settings.CLEAR_ON_RESTART:
        logger.info("CLEAR_ON_RESTART is True. Clearing volatile directories.")
        
        dirs_to_clear = [
            settings.VECTOR_STORE_PATH,
            settings.CACHE_PATH,
            settings.GRAPH_PATH, # Commented out to preserve the graph
        ]
        
        for d in dirs_to_clear:
            if d.exists():
                try:
                    shutil.rmtree(d)
                    logger.info(f"Successfully cleared directory: {d}")
                except Exception as e:
                    logger.error(f"Failed to clear directory {d}: {e}")
            
            # Recreate the directory after clearing
            d.mkdir(parents=True, exist_ok=True)
    else:
        logger.info("CLEAR_ON_RESTART is False. Skipping directory clearing.")
        # Still ensure directories exist
        settings.VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
        settings.GRAPH_PATH.mkdir(parents=True, exist_ok=True)
        settings.CACHE_PATH.mkdir(parents=True, exist_ok=True)
        settings.IMAGE_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# --- Pre-startup setup ---
# Add Tesseract to PATH
tesseract_path = r"C:\Program Files\Tesseract-OCR"
if tesseract_path not in os.environ["PATH"]:
    os.environ["PATH"] = tesseract_path + os.pathsep + os.environ["PATH"]
# --- End pre-startup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan. This is the modern replacement for
    startup and shutdown events.
    """
    # --- Startup Logic ---
    # The order of these startup tasks is critical.

    # 0. Check for and download spaCy model if necessary
    try:
        spacy.load("en_core_web_sm")
        logger.info("spaCy model 'en_core_web_sm' already installed.")
    except OSError:
        logger.warning("spaCy model 'en_core_web_sm' not found. Downloading...")
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            logger.info("Successfully downloaded spaCy model 'en_core_web_sm'.")
        except subprocess.CalledProcessError as e:
            logger.critical(f"--- FATAL: Failed to download spaCy model: {e} ---")
            logger.critical("Please install it manually by running: python -m spacy download en_core_web_sm")
            raise

    # 1. Clear volatile directories if configured
    clear_volatile_directories()

    # 2. Recover graph from a temporary file if it exists.
    logger.info("Checking for graph recovery from temporary file...")
    graph_recovery()
   

    # 3. Ensure storage directory exists
    storage_dir = settings.GRAPH_PATH.parent
    logger.info(f"Ensuring storage directory exists at: {storage_dir}")
    storage_dir.mkdir(parents=True, exist_ok=True)

    # 4. Build Tree-sitter Grammars
    logger.info("Building tree-sitter grammars...")
    try:
        build_tree_sitter_grammars()
        logger.info("Tree-sitter grammars built successfully.")
    except Exception as e:
        logger.critical(f"--- FATAL: Failed to build tree-sitter grammars: {e} ---", exc_info=True)
        raise

    # 5. Build and initialize system components
    builder = None
    logger.info("--- Starting Application Initialization ---")
    try:
        logger.info("Building system components...")
        builder = SystemBuilder(settings)
        rag_orchestrator, vector_index, ingestion_pipeline = builder.build_all()

        logger.info("Syncing retrievers...")
        if rag_orchestrator:
            rag_orchestrator.load_and_sync_retrievers()

        app.state.rag_orchestrator = rag_orchestrator
        app.state.vector_index = vector_index
        app.state.ingestion_pipeline = ingestion_pipeline
        logger.info("--- Application Initialization Complete ---")
    except Exception as e:
        logger.critical(f"--- FATAL: Application failed to initialize: {e} ---", exc_info=True)
        # Ensure the app can start in a degraded state
        app.state.rag_orchestrator = None
        app.state.vector_index = None
        app.state.ingestion_pipeline = None

    yield

    # --- Shutdown logic ---
    logger.info("--- Application Shutting Down ---")
    if builder:
        builder.shutdown()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="An auto-adaptive RAG system for intelligent document processing.",
    lifespan=lifespan
)

app.include_router(api_router, prefix="/api")

@app.get("/", tags=["Health"])
def health_check():
    """Basic health check endpoint."""
    return {"status": "ok", "version": settings.APP_VERSION}

if __name__ == "__main__":
    logger.info(f"Starting server at http://{settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )