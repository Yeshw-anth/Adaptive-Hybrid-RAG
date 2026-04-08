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
processed_files_hashes: Set[str] = set()

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
        result = await rag_orchestrator.orchestrate_query(
            query=query_request.query,
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
async def upload_file(request: Request, file: UploadFile = File(...)):
    """
    Uploads a file, processes it through the ingestion pipeline, and adds it to the index.
    """
    ingestion_pipeline = request.app.state.rag_orchestrator.ingestion_pipeline
    vector_index = request.app.state.vector_index

    if not ingestion_pipeline or not vector_index:
        raise HTTPException(status_code=503, detail="Core components are not initialized.")
    
    try:
        # Save the file temporarily to process it
        upload_dir = settings.UPLOAD_DIR
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)

        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)

        # Use a simple hash check to prevent re-processing
        file_hash = hashlib.sha256(content).hexdigest()
        if file_hash in processed_files_hashes:
            os.remove(file_path)
            return {"message": f"File '{file.filename}' with this content has already been processed."}

        # --- Ingestion Strategy Routing ---
        strategy = settings.INGESTION_STRATEGY
        nodes = []
        if strategy == "unstructured":
            nodes = await ingestion_pipeline.ingest_file_unstructured(file_path)
        elif strategy == "manual":
            nodes = ingestion_pipeline.ingest_file_manual(file_path)
        else:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=f"Invalid INGESTION_STRATEGY: '{strategy}'")

        if not nodes:
            os.remove(file_path)
            return {"message": f"File '{file.filename}' was processed, but no content was extracted. It might be empty or unsupported."}

        # Insert the nodes into the index and persist
        vector_index.insert_nodes(nodes)
        vector_index.storage_context.persist(persist_dir=settings.PERSIST_DIR)
        
        processed_files_hashes.add(file_hash)
        logging.info(f"Successfully indexed {len(nodes)} nodes from '{file.filename}'.")

        # Clean up the temporary file
        os.remove(file_path)

        return {"message": f"File '{file.filename}' uploaded and indexed successfully."}
    except Exception as e:
        logging.error(f"Unhandled exception in upload_file: {e}", exc_info=True)
        # Clean up in case of error
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process and index the file: {e}")

@router.post("/upload_batch")
async def upload_batch(request: Request, files: List[UploadFile] = File(...)):
    """Uploads and processes multiple files concurrently."""
    ingestion_pipeline = request.app.state.rag_orchestrator.ingestion_pipeline
    vector_index = request.app.state.vector_index

    if not ingestion_pipeline or not vector_index:
        raise HTTPException(status_code=503, detail="Core components are not initialized.")

    upload_dir = settings.UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)

    async def process_file(file: UploadFile):
        file_path = os.path.join(upload_dir, file.filename)
        try:
            content = await file.read()
            with open(file_path, "wb") as buffer:
                buffer.write(content)

            # Use a simple hash check to prevent re-processing
            file_hash = hashlib.sha256(content).hexdigest()
            if file_hash in processed_files_hashes:
                os.remove(file_path)
                return {"filename": file.filename, "status": "skipped", "message": "Content already processed."}

            # --- Ingestion Strategy Routing ---
            strategy = settings.INGESTION_STRATEGY
            nodes = []
            if strategy == "unstructured":
                nodes = await ingestion_pipeline.ingest_file_unstructured(file_path)
            elif strategy == "manual":
                loop = asyncio.get_running_loop()
                nodes = await loop.run_in_executor(
                    None, ingestion_pipeline.ingest_file_manual, file_path
                )
            else:
                os.remove(file_path)
                return {"filename": file.filename, "status": "failed", "error": f"Invalid INGESTION_STRATEGY: '{strategy}'"}

            if not nodes:
                os.remove(file_path)
                return {"filename": file.filename, "status": "skipped", "message": "No content extracted."}

            # This part is tricky for concurrency. For simplicity, we'll let LlamaIndex handle it.
            # In a high-throughput system, you might collect all nodes and insert them in one batch.
            vector_index.insert_nodes(nodes)
            processed_files_hashes.add(file_hash)
            
            os.remove(file_path)
            logging.info(f"Successfully indexed {len(nodes)} nodes from '{file.filename}'.")
            return {"filename": file.filename, "status": "success", "node_count": len(nodes)}
        except Exception as e:
            # Log the full error for debugging, but return a cleaner message to the user
            logging.error(f"Error processing file {file.filename} in batch: {e}", exc_info=True)
            error_message = f"An internal error occurred: {str(e)}"
            if os.path.exists(file_path):
                os.remove(file_path)
            return {"filename": file.filename, "status": "failed", "error": error_message}

    tasks = [process_file(file) for file in files]
    results = await asyncio.gather(*tasks)

    # Persist all changes at the end of the batch
    vector_index.storage_context.persist(persist_dir=settings.PERSIST_DIR)
    logging.info("Batch processing complete. Index persisted.")

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