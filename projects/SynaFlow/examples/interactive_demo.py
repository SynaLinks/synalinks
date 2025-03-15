import os
import sys
import json
import asyncio
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import required modules
import synalinks
from synalinks import ChatMessage, ChatRole, ChatMessages
from src.language_models import setup_api_keys, get_cloud_model
from src.programs import ScientificQAProgram
from src.data_models import ScientificQuery, ScientificAnswer, Citation
from src.utils import format_citation

def load_trained_examples(model_path: str) -> List[Dict[str, Any]]:
    """
    Load examples from the trained model file.
    
    Args:
        model_path: Path to the trained model JSON file
        
    Returns:
        List of examples from the model file, or default examples if not found
    """
    try:
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        if 'config' in model_data and 'examples' in model_data['config']:
            return model_data['config']['examples']
        else:
            print("No examples found in model file. Loading examples from dataset...")
            return load_examples_from_dataset()
    except Exception as e:
        print(f"Error loading examples from model file: {e}")
        return load_examples_from_dataset()

def load_examples_from_dataset() -> List[Dict[str, Any]]:
    """
    Load examples from the dataset file.
    
    Returns:
        List of examples formatted for the ScientificQAProgram
    """
    try:
        dataset_path = "data/scientific_qa_dataset.json"
        if not os.path.exists(dataset_path):
            print(f"Dataset file not found at {dataset_path}")
            return []
            
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
            
        # Extract examples from the train set
        examples = []
        if 'train' in dataset and isinstance(dataset['train'], list):
            for item in dataset['train']:
                if 'question' in item and 'answer' in item:
                    examples.append({
                        "inputs": item['question'],
                        "outputs": item['answer']
                    })
            print(f"Loaded {len(examples)} examples from dataset")
            return examples
        else:
            print("No train examples found in dataset")
            return []
    except Exception as e:
        print(f"Error loading examples from dataset: {e}")
        return []

async def answer_scientific_question(question: str, domain: Optional[str] = None, context: Optional[str] = None, model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a scientific answer using examples from the trained model.
    
    Args:
        question: The scientific question to answer
        domain: Optional domain (e.g., physics, biology)
        context: Optional additional context
        model_path: Path to the trained model file
        
    Returns:
        Dictionary with the answer components
    """
    print(f"\nProcessing question: {question}")
    
    # Create a new program instance
    program = ScientificQAProgram(
        language_model=get_cloud_model(),
        use_citation_lookup=True
    )
    
    # Load examples from the trained model
    if model_path and os.path.exists(model_path):
        examples = load_trained_examples(model_path)
        if examples:
            print(f"Using {len(examples)} examples from trained model.")
            program.examples = examples
    
    # Create the query
    query = ScientificQuery(
        question=question,
        domain=domain,
        context=context
    )
    
    try:
        # Generate the answer
        result = await program(query)
        return result
    except Exception as e:
        print(f"Error generating answer with program: {e}")
        print("Falling back to direct API call...")
        
        # Fallback to direct API call if program call fails
        return await direct_api_call(question, domain, context)

async def direct_api_call(question: str, domain: Optional[str] = None, context: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate answer using direct API call without program abstraction.
    
    Args:
        question: The scientific question to answer
        domain: Optional domain (e.g., physics, biology)
        context: Optional additional context
        
    Returns:
        Dictionary with the answer components
    """
    # Create the language model
    model = get_cloud_model()
    
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
        
        # Ensure default fields
        result.setdefault("background", "No background information provided.")
        result.setdefault("reasoning", "No reasoning provided.")
        result.setdefault("answer", "No answer provided.")
        result.setdefault("confidence", 0.5)
        result.setdefault("citations", [])
        result.setdefault("further_reading", [])
        
        return result
    except Exception as e:
        print(f"Error in direct API call: {e}")
        return {
            "background": "Error processing the question.",
            "reasoning": "An error occurred during processing.",
            "answer": f"Unable to answer due to: {str(e)}",
            "confidence": 0.0,
            "citations": [],
            "further_reading": []
        }

async def run_interactive_demo():
    """Run the interactive demo."""
    print("\n===== Scientific Q&A Interactive Demo =====\n")
    print("This demo uses a trained model to answer scientific questions.")
    print("Type 'exit' to quit the demo.\n")
    
    # Setup
    setup_api_keys()
    model_path = "models/trained_qa_program.json"
    
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        print("Continuing with default examples.")
    else:
        print(f"Found trained model: {model_path}")
    
    # Main loop
    while True:
        # Get user input
        question = input("\nEnter a scientific question: ")
        if question.lower() == 'exit':
            break
        
        domain = input("Domain (optional, press Enter to skip): ")
        domain = domain if domain else None
        
        context = input("Additional context (optional, press Enter to skip): ")
        context = context if context else None
        
        # Generate answer
        result = await answer_scientific_question(question, domain, context, model_path)
        
        # Print results
        print("\n----- BACKGROUND -----")
        print(result.get("background", "No background information available."))
        
        print("\n----- REASONING -----")
        print(result.get("reasoning", "No reasoning available."))
        
        print("\n----- ANSWER -----")
        print(result.get("answer", "No answer available."))
        
        print(f"\n----- CONFIDENCE: {result.get('confidence', 0.0):.2f} -----")
        
        print("\n----- CITATIONS -----")
        citations = result.get("citations", [])
        if citations:
            for citation in citations:
                authors = ", ".join(citation.get("authors", []))
                title = citation.get("title", "")
                year = citation.get("year", "")
                source = citation.get("source", "")
                print(f"- {title} ({year}) by {authors}. {source}")
        else:
            print("No citations available.")
        
        print("\n----- FURTHER READING -----")
        further_reading = result.get("further_reading", [])
        if further_reading:
            for reading in further_reading:
                print(f"- {reading}")
        else:
            print("No further reading suggestions available.")

if __name__ == "__main__":
    # Ensure API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Please set it in your .env file or environment variables.")
        sys.exit(1)
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Run the demo
    asyncio.run(run_interactive_demo()) 