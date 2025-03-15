import os
import asyncio
import json
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to the Python path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import synalinks components
import synalinks
from synalinks import ChatMessage, ChatRole, ChatMessages

# Set up OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not set. Please set it in your .env file or environment variables.")
    sys.exit(1)
else:
    print(f"Found API key: {api_key[:5]}...{api_key[-4:]}")
    # Ensure it's set in the environment
    os.environ["OPENAI_API_KEY"] = api_key

async def generate_scientific_answer(question: str, domain: str = None) -> Dict[str, Any]:
    """Generate a scientific answer using direct OpenAI calls."""
    print(f"Generating answer for: {question}")
    
    # Create the language model
    model = synalinks.LanguageModel(
        model="openai/gpt-4",
    )
    
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
    
    # Create chat messages
    messages = ChatMessages(messages=[
        ChatMessage(role=ChatRole.SYSTEM, content=system_message),
        ChatMessage(role=ChatRole.USER, content=user_message)
    ])
    
    try:
        # Generate response
        response = await model(messages)
        
        # Extract and parse content
        content = response.get("content", "")
        
        # Find JSON in content
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
        
        return result
    except Exception as e:
        print(f"Error generating answer: {e}")
        return {
            "background": f"Error: {e}",
            "reasoning": "An error occurred during processing",
            "answer": "Unable to generate answer due to technical issues",
            "confidence": 0.0,
            "citations": [],
            "further_reading": []
        }

async def main():
    """Run the direct test."""
    print("\n===== Direct Scientific Q&A Test =====\n")
    
    # Test with H2O question
    question = "What is H2O and why is it important?"
    domain = "chemistry"
    
    print(f"Question: {question}")
    print(f"Domain: {domain}")
    print("\nGenerating answer...\n")
    
    result = await generate_scientific_answer(question, domain)
    
    # Pretty print the result
    print("\n----- RESULT -----\n")
    print(f"Background: {result.get('background')}\n")
    print(f"Reasoning: {result.get('reasoning')}\n")
    print(f"Answer: {result.get('answer')}\n")
    print(f"Confidence: {result.get('confidence')}\n")
    
    print("Citations:")
    for citation in result.get("citations", []):
        authors = ", ".join(citation.get("authors", []))
        title = citation.get("title", "")
        year = citation.get("year", "")
        source = citation.get("source", "")
        print(f"- {title} ({year}) by {authors}. {source}")
    
    print("\nFurther reading:")
    for reading in result.get("further_reading", []):
        print(f"- {reading}")
    
    print("\n===== Test Completed =====")

if __name__ == "__main__":
    asyncio.run(main()) 