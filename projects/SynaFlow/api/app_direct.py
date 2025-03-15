import os
import asyncio
import uvicorn
import uuid
import time
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from cachetools import TTLCache
import logging
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import synalinks and other modules
import synalinks
from synalinks import ChatMessage, ChatRole, ChatMessages
from src.data_models import ScientificQuery, ScientificAnswer, Citation
from src.programs import lookup_citations
from src.utils import setup_logging

# Initialize logging
logger = setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_file="api_direct.log"
)

# Initialize cache
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
request_cache = TTLCache(maxsize=1000, ttl=CACHE_TTL)

# Initialize FastAPI app
app = FastAPI(
    title="Scientific Q&A API (Direct synalinks)",
    description="API for answering scientific questions with citations using synalinks directly",
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

# Initialize the language model once
language_model = synalinks.LanguageModel(
    model="openai/gpt-4",
)
# Store temperature as an attribute for later use during inference
language_model._temperature = 0.2

async def generate_scientific_answer(question: str, domain: Optional[str] = None, context: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a scientific answer using synalinks and OpenAI.
    
    Args:
        question: The scientific question
        domain: Optional domain (e.g., physics, biology)
        context: Optional additional context
    
    Returns:
        Dictionary with structured answer components
    """
    # Create the prompt
    system_message = """You are a scientific question answering system. 
    Provide comprehensive, accurate answers to scientific questions.
    
    Format your response as a structured JSON with these fields:
    {
      "background": "Background information and context for the question",
      "reasoning": "Step-by-step reasoning process and explanation",
      "answer": "Clear and concise answer to the question",
      "confidence": 0.95, // A number between 0 and 1
      "citations": [
        {
          "title": "Title of source",
          "authors": ["Author 1", "Author 2"],
          "year": 2023,
          "source": "Journal or publication",
          "url": "https://example.com/source"
        }
      ],
      "further_reading": ["Suggested reading 1", "Suggested reading 2"]
    }
    """
    
    user_message = f"Question: {question}\n"
    if domain:
        user_message += f"Domain: {domain}\n"
    if context:
        user_message += f"Additional Context: {context}\n"
    
    # Create individual chat messages
    system_msg = ChatMessage(role=ChatRole.SYSTEM, content=system_message)
    user_msg = ChatMessage(role=ChatRole.USER, content=user_message)
    
    # Create proper ChatMessages object
    messages = ChatMessages(messages=[system_msg, user_msg])
    
    # Generate response
    response = await language_model(messages)
    
    # Extract content from the response and parse JSON
    try:
        content = response.get("content", "")
        # Find JSON in the content (might be surrounded by markdown code blocks)
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            json_str = content[json_start:json_end].strip()
        elif "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            json_str = content[json_start:json_end].strip()
        else:
            json_str = content
        
        # Parse the JSON
        result = json.loads(json_str)
        
        # Ensure all required fields are present
        result.setdefault("background", "No background information provided.")
        result.setdefault("reasoning", "No reasoning provided.")
        result.setdefault("answer", "No answer provided.")
        result.setdefault("confidence", 0.5)
        result.setdefault("citations", [])
        result.setdefault("further_reading", [])
        
        return result
    except Exception as e:
        logger.error(f"Error parsing response: {e}")
        # Return a default response in case of parsing error
        return {
            "background": "Error parsing the AI response.",
            "reasoning": "The system encountered an error while processing your question.",
            "answer": content,  # Return the raw content as the answer
            "confidence": 0.0,
            "citations": [],
            "further_reading": []
        }

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
        # Generate the answer
        result = await generate_scientific_answer(
            question=request.question,
            domain=request.domain,
            context=request.context
        )
        
        # Create the response
        response = AnswerResponse(
            background=result["background"],
            reasoning=result["reasoning"],
            answer=result["answer"],
            confidence=result["confidence"],
            citations=[
                CitationResponse(**citation) 
                for citation in result["citations"]
            ],
            further_reading=result["further_reading"]
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

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    
    Returns:
        Dictionary with health status information
    """
    return {
        "status": "healthy",
        "model_type": language_model.model,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    # Set the API key from environment if not already set
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    
    # Start the API server on port 8001 instead of 8000
    # Start the API server
    uvicorn.run("app_direct:app", host="0.0.0.0", port=8001, reload=False) 