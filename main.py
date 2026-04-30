import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import uvicorn
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI


from src.api.routes import router as api_router
from src.config.settings import settings
from src.core.logging_config import logger
from src.core.system_builder import SystemBuilder
from treesitter_build.build_grammars import build_tree_sitter_grammars

# --- Pre-startup checks and setup ---

def clear_volatile_directories():
    """Clears directories that should be reset on startup if configured."""
    if settings.CLEAR_ON_RESTART:
        logger.info("CLEAR_ON_RESTART is True. Clearing volatile directories.")
        
        dirs_to_clear = [
                    settings.PERSIST_PATH,
                    settings.CACHE_PATH,
                    settings.IMAGE_OUTPUT_PATH,
                    settings.GRAPH_PATH
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
        settings.PERSIST_PATH.mkdir(parents=True, exist_ok=True)
        settings.CACHE_PATH.mkdir(parents=True, exist_ok=True)
        settings.IMAGE_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# --- Pre-startup setup ---
clear_volatile_directories()

# 2. Add Tesseract to PATH
tesseract_path = r"C:\Program Files\Tesseract-OCR"
if tesseract_path not in os.environ["PATH"]:
    os.environ["PATH"] = tesseract_path + os.pathsep + os.environ["PATH"]


# 3. Check and build tree-sitter grammars if necessary
# This is now handled in the lifespan function.
# --- End pre-startup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan. This is the modern replacement for
    startup and shutdown events.
    """
    # --- Ensure storage directory exists ---
    # This is done here to ensure the environment is ready before any components
    # that might need it are initialized.
    storage_dir = settings.GRAPH_PATH.parent
    logger.info(f"Ensuring storage directory exists at: {storage_dir}")
    storage_dir.mkdir(parents=True, exist_ok=True)

    # --- Build Tree-sitter Grammars ---
    logger.info("Building tree-sitter grammars...")
    try:
        build_tree_sitter_grammars()
        logger.info("Tree-sitter grammars built successfully.")
    except Exception as e:
        logger.critical(f"--- FATAL: Failed to build tree-sitter grammars: {e} ---", exc_info=True)
        raise

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