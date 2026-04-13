import logging
from fastapi import APIRouter, HTTPException, File, UploadFile, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Set
import asyncio
import os
import shutil

# Import the shared objects from settings.py
from src.config import settings

# Import other necessary components
import hashlib
from src.core.ingestion import IngestionRouter
from src.chunking.chunking_engine import ChunkingEngine
from src.chunking.metadata_enricher import MetadataEnricher


# --- API Models ---
class QueryRequest(BaseModel):
    query: str
    session_id: str  # To manage conversation history, if needed
    model: str | None = None # Optional field to override the model from the frontend

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_score: float | None = None
    strategy: str | None = None

# --- Globals ---
router = APIRouter()



# --- API Endpoints ---
@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: Request, query_request: QueryRequest):
    """
    Receives a query, processes it through the RAG pipeline, and returns the response.
    """
    rag_orchestrator = request.app.state.rag_orchestrator
    if not rag_orchestrator:
        raise HTTPException(status_code=503, detail="RAG orchestrator is not initialized.")
    
    try:
        result = await rag_orchestrator.query(
            original_query=query_request.query,
            model_override=query_request.model
        )
        
        return QueryResponse(
            answer=result.get("answer", "No response generated."),
            sources=result.get("sources", []),
            confidence_score=result.get("confidence_score"),
            strategy=result.get("pipeline") # Use the 'pipeline' key which holds the string name
        )
    except Exception as e:
        logging.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing query.")

@router.post("/upload")
async def upload(request: Request, files: List[UploadFile] = File(...)):
    """Uploads and processes multiple files concurrently."""
    vector_index = request.app.state.vector_index
    chunking_engine = request.app.state.chunking_engine

    if not all([vector_index, chunking_engine]):
        raise HTTPException(status_code=503, detail="Core components are not initialized.")

    upload_dir = settings.UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)

    metadata_enricher = MetadataEnricher()

    async def process_file(file: UploadFile):
        file_path = os.path.join(upload_dir, file.filename)
        file_name = file.filename
        try:
            content = await file.read()
            with open(file_path, "wb") as buffer:
                buffer.write(content)

            # 1. Loader
            loader = IngestionRouter.get_loader(file_path)
            elements = loader.process()
            if not elements:
                return {"filename": file_name, "status": "skipped", "message": "No content extracted."}

            # 2. Chunking
            nodes = await chunking_engine.chunk_document(elements, file_path=file_path)

            # 3. Metadata Enrichment
            enriched_nodes = metadata_enricher.enrich_nodes(nodes, file_name=file_name, file_path=file_path, section_title="General")
            
            # 4. Storage
            vector_index.insert_nodes(enriched_nodes)
            vector_index.storage_context.persist(persist_dir=settings.PERSIST_DIR)
            
            logging.info(f"Successfully indexed {len(enriched_nodes)} nodes from '{file_name}'.")
            if enriched_nodes:
                logging.info(f"Sample enriched metadata for {file_name}: {enriched_nodes[0].metadata}")

            return {"filename": file_name, "status": "success", "node_count": len(enriched_nodes)}
        except Exception as e:
            logging.error(f"Error processing file {file_name} in batch: {e}", exc_info=True)
            error_message = f"An internal error occurred: {str(e)}"
            return {"filename": file_name, "status": "failed", "error": error_message}
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    tasks = [process_file(file) for file in files]
    results = await asyncio.gather(*tasks)

    logging.info("Batch processing complete.")
    return results

# --- Feedback Endpoint (Future Scope) ---
# class FeedbackRequest(BaseModel):
#     query_id: str
#     rating: int = Field(..., ge=-1, le=1)  # -1 for downvote, 1 for upvote
#     comment: str | None = None

# @router.post("/feedback")
# async def feedback_endpoint(request: Request, feedback_request: FeedbackRequest):
#     """
#     Receives user feedback and logs it against the corresponding query.
#     """
#     feedback_store = request.app.state.feedback_store
#     if not feedback_store:
#         raise HTTPException(status_code=503, detail="FeedbackStore is not initialized.")
    
#     try:
#         feedback_store.update_feedback(
#             query_id=feedback_request.query_id,
#             user_rating=feedback_request.rating,
#             user_comment=feedback_request.comment
#         )
#         return {"status": "success", "message": "Feedback received."}
#     except FileNotFoundError:
#         raise HTTPException(status_code=404, detail=f"Log file not found. Cannot save feedback.")
#     except ValueError as e:
#         # This could happen if the query_id is not found
#         raise HTTPException(status_code=404, detail=str(e))
#     except Exception as e:
#         logging.error(f"Error processing feedback: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="Error processing feedback.")