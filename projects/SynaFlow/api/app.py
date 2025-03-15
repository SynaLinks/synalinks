import os
import asyncio
import uvicorn
import uuid
import time
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from cachetools import TTLCache
import logging
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import synalinks
from src.data_models import ScientificQuery, ScientificAnswer, Citation
from src.language_models import setup_api_keys, get_local_model, get_cloud_model
from src.programs import ScientificQAProgram
from src.config import CACHE_ENABLED, CACHE_TTL, MAX_CONCURRENT_REQUESTS
from src.utils import setup_logging

# Initialize logging
logger = setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_file="api.log"
)

# Set up API keys
setup_api_keys()

# Initialize cache
request_cache = TTLCache(maxsize=1000, ttl=CACHE_TTL)

# Initialize FastAPI app
app = FastAPI(
    title="Scientific Q&A API",
    description="API for answering scientific questions with citations",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Middleware to track request processing time.
    
    Args:
        request: The incoming request
        call_next: The next middleware or route handler
        
    Returns:
        The response with a processing time header added
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Load the Synalinks program
loop = asyncio.get_event_loop()

# Try to load trained program, fall back to new program if not found
try:
    program = loop.run_until_complete(
        synalinks.Program.load("models/trained_qa_program.json")
    )
    logger.info("Loaded trained program")
except Exception as e:
    logger.warning(f"Could not load trained program: {e}. Creating new program.")
    # Initialize with local model and fallback to cloud if needed
    try:
        model = get_local_model()
        logger.info("Using local language model")
    except Exception:
        logger.warning("Local model unavailable. Using cloud model.")
        model = get_cloud_model()
    
    program = ScientificQAProgram(language_model=model)

# Pydantic models for API
class QueryRequest(BaseModel):
    """Request model for a scientific question."""
    question: str = Field(..., description="The scientific question to be answered")
    domain: Optional[str] = Field(None, description="Scientific domain (e.g., physics, biology)")
    context: Optional[str] = Field(None, description="Additional context for the question")

class CitationResponse(BaseModel):
    """Response model for a scientific citation."""
    title: str
    authors: List[str]
    year: Optional[int] = None
    source: str
    url: Optional[str] = None

class AnswerResponse(BaseModel):
    """Response model for a scientific answer."""
    background: str
    reasoning: str
    answer: str
    confidence: float
    citations: List[CitationResponse]
    further_reading: Optional[List[str]] = None

class BatchQueryRequest(BaseModel):
    """Request model for batch scientific questions."""
    queries: List[QueryRequest]

# API endpoints
@app.post("/query", response_model=AnswerResponse)
async def process_query(request: QueryRequest) -> AnswerResponse:
    """
    Process a single scientific question.
    
    Args:
        request: The query request containing the question and optional context
        
    Returns:
        A comprehensive scientific answer
        
    Raises:
        HTTPException: If there's an error processing the query
    """
    request_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    # Log request
    logger.info(json.dumps({
        "event": "request_received",
        "request_id": request_id,
        "question": request.question,
        "timestamp": start_time.isoformat()
    }))
    
    # Check cache if enabled
    if CACHE_ENABLED:
        cache_key = f"{request.question}:{request.domain or ''}:{request.context or ''}"
        cached_response = request_cache.get(cache_key)
        if cached_response:
            logger.info(json.dumps({
                "event": "cache_hit",
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }))
            return cached_response
    
    try:
        # Create a Synalinks Query instance
        query_input = ScientificQuery(
            question=request.question,
            domain=request.domain,
            context=request.context
        )
        
        # Process with the program
        result = await program(query_input)
        
        # Extract fields from the result
        response = AnswerResponse(
            background=result.get("background"),
            reasoning=result.get("reasoning"),
            answer=result.get("answer"),
            confidence=result.get("confidence"),
            citations=[
                CitationResponse(**citation) 
                for citation in result.get("citations", [])
            ],
            further_reading=result.get("further_reading")
        )
        
        # Store in cache if enabled
        if CACHE_ENABLED:
            cache_key = f"{request.question}:{request.domain or ''}:{request.context or ''}"
            request_cache[cache_key] = response
        
        # Log success
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        logger.info(json.dumps({
            "event": "request_processed",
            "request_id": request_id,
            "processing_time": processing_time,
            "timestamp": end_time.isoformat()
        }))
        
        return response
    except Exception as e:
        # Log error
        logger.error(json.dumps({
            "event": "request_error",
            "request_id": request_id,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }))
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/batch", response_model=List[AnswerResponse])
async def process_batch(request: BatchQueryRequest) -> List[AnswerResponse]:
    """
    Process multiple queries in parallel.
    
    Args:
        request: Batch request containing multiple queries
        
    Returns:
        List of scientific answers corresponding to each query
        
    Raises:
        HTTPException: If the batch size exceeds the maximum allowed
    """
    if len(request.queries) > MAX_CONCURRENT_REQUESTS:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds maximum allowed ({MAX_CONCURRENT_REQUESTS})"
        )
    
    # Process each query in parallel
    tasks = []
    for query in request.queries:
        tasks.append(process_query(query))
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any errors
    responses = []
    for result in results:
        if isinstance(result, Exception):
            # Add error placeholder
            responses.append({
                "error": str(result),
                "background": "",
                "reasoning": "",
                "answer": "Error processing question",
                "confidence": 0.0,
                "citations": []
            })
        else:
            responses.append(result)
    
    return responses

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    
    Returns:
        Dictionary with health status information
    """
    return {
        "status": "healthy",
        "model_type": program.language_model.model,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False) 