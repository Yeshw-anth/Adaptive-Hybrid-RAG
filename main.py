import uvicorn
import logging
import os
from fastapi import FastAPI
from src.api.routes import router as api_router
from src.config import settings
from src.core.logging_config import setup_logging
from src.core.system_builder import SystemBuilder
from treesitter_build.build_grammars import build_tree_sitter_grammars

import shutil
from src.config.settings import IMAGE_OUTPUT_DIR

from contextlib import asynccontextmanager

# --- Pre-startup checks and setup ---

def clear_image_cache():
    """Clears the image cache directory."""
    if os.path.exists(IMAGE_OUTPUT_DIR):
        logging.info(f"Clearing image cache at: {IMAGE_OUTPUT_DIR}")
        try:
            shutil.rmtree(IMAGE_OUTPUT_DIR)
            os.makedirs(IMAGE_OUTPUT_DIR)
            logging.info("Image cache cleared successfully.")
        except Exception as e:
            logging.error(f"Failed to clear image cache: {e}")
    else:
        os.makedirs(IMAGE_OUTPUT_DIR)
        logging.info(f"Image cache directory created at: {IMAGE_OUTPUT_DIR}")

# 1. Clear image cache on restart
clear_image_cache()

# 2. Add Tesseract to PATH
tesseract_path = r"C:\Program Files\Tesseract-OCR"
if tesseract_path not in os.environ["PATH"]:
    os.environ["PATH"] = tesseract_path + os.pathsep + os.environ["PATH"]

# 3. Configure logging
setup_logging()

# 4. Check and build tree-sitter grammars if necessary
build_tree_sitter_grammars()
# --- End pre-startup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan. This is the modern replacement for
    startup and shutdown events.
    """
    logging.info("--- Starting Application Initialization ---")
    try:
        builder = SystemBuilder()
        rag_orchestrator, vector_index = builder.build_all()
        
        # Manually sync retrievers after the index is loaded
        if rag_orchestrator:
            rag_orchestrator.load_and_sync_retrievers()

        app.state.rag_orchestrator = rag_orchestrator
        app.state.vector_index = vector_index
        logging.info("--- Application Initialization Complete ---")
    except Exception as e:
        logging.critical(f"--- FATAL: Application failed to initialize ---", exc_info=True)
        app.state.rag_orchestrator = None
        app.state.vector_index = None
    
    yield
    
    # --- Shutdown logic would go here ---
    logging.info("--- Application Shutting Down ---")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="An auto-adaptive RAG system for intelligent document processing.",
    lifespan=lifespan
)

app.include_router(api_router, prefix="/api")

@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok", "version": settings.APP_VERSION}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)