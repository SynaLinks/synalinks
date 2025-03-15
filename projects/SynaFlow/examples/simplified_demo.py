import os
import asyncio
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from src.simplified_program import SimplifiedScientificQA
from src.data_models import ScientificQuery

# Load environment variables
load_dotenv()

# Ensure OpenAI API key is available
if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
    print("Error: OPENAI_API_KEY not found in environment variables")
    print("Please add it to your .env file or set it as an environment variable")
    exit(1)

async def process_question(
    qa_program: SimplifiedScientificQA,
    question: str,
    domain: Optional[str] = None,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a scientific question with the given domain and context.
    
    Args:
        qa_program: The scientific QA program
        question: The scientific question
        domain: Optional domain (e.g., physics, biology)
        context: Optional additional context
        
    Returns:
        Dictionary with the answer components
    """
    # Create a query object
    query = {
        "question": question,
        "domain": domain,
        "context": context
    }
    
    try:
        # Process the query
        result = await qa_program(query)
        return result
    except Exception as e:
        print(f"Error generating answer: {e}")
        return {
            "background": "An error occurred while processing your question.",
            "reasoning": f"Error: {str(e)}",
            "answer": "Unable to generate an answer at this time.",
            "confidence": 0.0,
            "citations": [],
            "further_reading": []
        }

async def interactive_demo():
    """Run an interactive demo for the scientific QA program."""
    print("\n=== Scientific QA Demo ===")
    print("Loading trained model...")
    
    # Load the trained model
    model_path = "models/trained_qa_program.json"
    
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}")
        print("Please train the model first by running 'python main.py --train'")
        return
    
    print(f"Found trained model at {model_path}")
    qa_program = SimplifiedScientificQA.from_file(model_path)
    
    print(f"Loaded model with {len(qa_program.examples)} examples")
    
    print("\nEnter scientific questions. Type 'quit' to exit.")
    
    while True:
        # Get user input
        print("\n" + "-" * 40)
        question = input("Enter a scientific question: ")
        
        # Check if user wants to quit
        if question.lower() in ["quit", "exit", "q"]:
            break
        
        # Get optional domain and context
        domain = input("Domain (optional, press Enter to skip): ")
        domain = domain if domain else None
        
        context = input("Additional context (optional, press Enter to skip): ")
        context = context if context else None
        
        print("\nProcessing question...")
        
        # Process the question
        result = await process_question(qa_program, question, domain, context)
        
        # Display the answer
        print("\n=== Answer ===")
        
        print("\nBackground:")
        print(result.get("background", "No background information provided."))
        
        print("\nReasoning:")
        print(result.get("reasoning", "No reasoning provided."))
        
        print("\nAnswer:")
        print(result.get("answer", "No answer provided."))
        
        print(f"\nConfidence: {result.get('confidence', 0.0):.2f}")
        
        # Display citations
        citations = result.get("citations", [])
        if citations:
            print("\nCitations:")
            for i, citation in enumerate(citations):
                authors = ", ".join(citation.get("authors", []))
                title = citation.get("title", "")
                year = citation.get("year", "")
                source = citation.get("source", "")
                url = citation.get("url", "")
                
                print(f"{i+1}. {title} ({year}) by {authors}. {source}")
                if url:
                    print(f"   URL: {url}")
        
        # Display further reading
        further_reading = result.get("further_reading", [])
        if further_reading:
            print("\nFurther Reading:")
            for i, reading in enumerate(further_reading):
                print(f"{i+1}. {reading}")

if __name__ == "__main__":
    asyncio.run(interactive_demo()) 