from src.core.logging_config import logger
from fastapi import APIRouter, HTTPException, File, UploadFile, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Set
import asyncio
import os
import shutil

# Import the shared objects from settings.py
from src.config.settings import settings

# Import other necessary components
import hashlib
from src.core.ingestion import IngestionPipeline


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
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing query.")

@router.post("/upload")
async def upload(request: Request, files: List[UploadFile] = File(...)):
    """Uploads and processes multiple files concurrently."""
    ingestion_pipeline = request.app.state.ingestion_pipeline

    if not ingestion_pipeline:
        raise HTTPException(status_code=503, detail="Core components are not initialized.")

    upload_dir = settings.UPLOAD_PATH

    async def process_file(file: UploadFile):
        file_path = os.path.join(upload_dir, file.filename)
        file_name = file.filename
        try:
            content = await file.read()
            with open(file_path, "wb") as buffer:
                buffer.write(content)

            # Use the IngestionPipeline to process the file
            nodes = await ingestion_pipeline.ingest_file(file_path)

            if not nodes:
                return {"filename": file_name, "status": "skipped", "message": "No content extracted or processed."}

            logger.info(f"Successfully ingested and processed {len(nodes)} nodes from '{file_name}'.")
            
            return {"filename": file_name, "status": "success", "node_count": len(nodes)}
        except Exception as e:
            logger.error(f"Error processing file {file_name} in batch: {e}", exc_info=True)
            error_message = f"An internal error occurred: {str(e)}"
            return {"filename": file_name, "status": "failed", "error": error_message}
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    tasks = [process_file(file) for file in files]
    results = await asyncio.gather(*tasks)

    logger.info("Batch processing complete.")
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
#         logger.error(f"Error processing feedback: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="Error processing feedback.")