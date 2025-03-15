import os
import asyncio
import sys
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import synalinks and other modules
import synalinks
from synalinks import ChatMessage, ChatRole, ChatMessages
from src.data_models import ScientificQuery, ScientificAnswer, Citation
from src.programs import lookup_citations

async def generate_answer(question, domain=None, context=None):
    """Generate a scientific answer using synalinks and OpenAI."""
    # Setup language model with OpenAI
    model = synalinks.LanguageModel(
        model="openai/gpt-4",
    )
    # Store temperature as an attribute for later use during inference
    model._temperature = 0.2
    
    # Create the prompt
    system_message = """You are a scientific question answering system. 
    Provide comprehensive, accurate answers to scientific questions with the following format:
    1. Background: Provide context and background information
    2. Reasoning: Explain the scientific reasoning and process
    3. Answer: Give a clear, concise answer
    4. Citations: List relevant scientific sources
    """
    
    user_message = f"Question: {question}\n"
    if domain:
        user_message += f"Domain: {domain}\n"
    if context:
        user_message += f"Additional Context: {context}\n"
    
    # Create individual chat messages
    system_msg = ChatMessage(role=ChatRole.SYSTEM, content=system_message)
    user_msg = ChatMessage(role=ChatRole.USER, content=user_message)
    
    # Create proper ChatMessages object with keyword arguments
    messages = ChatMessages(messages=[system_msg, user_msg])
    
    # Generate response
    response = await model(messages)
    return response

async def enhanced_demo():
    """An enhanced demo to test synalinks with OpenAI."""
    print("\n===== SynaFlow Scientific Q&A Enhanced Demo =====\n")
    
    # Test citation lookup first (doesn't require OpenAI API)
    try:
        print("Testing citation lookup...")
        result = await lookup_citations("quantum physics")
        print(f"Found {len(result['citations'])} citations:")
        for citation in result['citations']:
            print(f"- {citation['title']} by {', '.join(citation['authors'])}")
        print("\nCitation lookup test successful!\n")
    except Exception as e:
        print(f"Error in citation lookup: {e}")
    
    # Now test the full QA system with OpenAI
    try:
        print("\nTesting full scientific QA system with OpenAI...")
        print("Processing question: How do black holes emit Hawking radiation?")
        
        response = await generate_answer(
            question="How do black holes emit Hawking radiation?",
            domain="physics"
        )
        
        print("\n----- Generated Answer -----\n")
        print(response)
        
        print("\nOpenAI integration test successful!\n")
    except Exception as e:
        print(f"Error in OpenAI integration: {e}")
        import traceback
        traceback.print_exc()
    
    print("Enhanced demo completed.")

if __name__ == "__main__":
    # Set the API key from environment if not already set
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    
    # Run the demo
    asyncio.run(enhanced_demo()) 