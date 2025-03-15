import os
import sys
import json
import asyncio
from typing import Dict, Any, Optional

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import required modules with proper paths
import synalinks
from src.language_models import setup_api_keys, get_cloud_model
from src.programs import ScientificQAProgram
from src.data_models import ScientificQuery, ScientificAnswer
from src.utils import format_citation

def print_separator(text):
    """Print a section separator with text."""
    print(f"\n{'=' * 10} {text} {'=' * 10}\n")

def print_model_structure(model_path):
    """Print the structure of the saved model file."""
    print_separator("MODEL FILE STRUCTURE")
    try:
        with open(model_path, 'r') as f:
            model_data = json.load(f)
            
        # Print top-level keys
        print(f"Top-level keys: {list(model_data.keys())}")
        
        # Print examples if they exist
        if 'config' in model_data and 'examples' in model_data['config']:
            print(f"Number of examples: {len(model_data['config']['examples'])}")
            print(f"Example structure: {json.dumps(model_data['config']['examples'][0], indent=2)[:300]}...")
        
        # Print other interesting parts
        if 'compile_config' in model_data:
            print(f"Compile config: {json.dumps(model_data['compile_config'], indent=2)}")
    except Exception as e:
        print(f"Error analyzing model file: {e}")

async def test_with_load_examples(model_path):
    """Try to use the examples from the trained model with a new program."""
    print_separator("TESTING WITH LOADED EXAMPLES")
    try:
        # Load model data from file
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        # Extract examples
        if 'config' in model_data and 'examples' in model_data['config']:
            examples = model_data['config']['examples']
            print(f"Loaded {len(examples)} examples from model file")
            
            # Create a fresh program with the loaded examples
            program = ScientificQAProgram()
            program.examples = examples
            
            # Test with a sample question
            query = ScientificQuery(
                question="What is H2O?",
                domain="chemistry"
            )
            
            print("Testing with query: What is H2O?")
            result = await program(query)
            
            print("\nResult:")
            print(f"Background: {result.get('background', 'N/A')[:100]}...")
            print(f"Answer: {result.get('answer', 'N/A')[:100]}...")
            
            return True
    except Exception as e:
        print(f"Error using loaded examples: {e}")
        return False

async def test_new_program():
    """Test a completely new program instance."""
    print_separator("TESTING NEW PROGRAM")
    try:
        # Create a new program with cloud model for better results
        program = ScientificQAProgram(
            language_model=get_cloud_model(),
            use_citation_lookup=True
        )
        
        # Test with a sample question
        query = ScientificQuery(
            question="What is H2O?",
            domain="chemistry"
        )
        
        print("Testing with query: What is H2O?")
        result = await program(query)
        
        print("\nResult:")
        print(f"Background: {result.get('background', 'N/A')[:100]}...")
        print(f"Answer: {result.get('answer', 'N/A')[:100]}...")
        
        return True
    except Exception as e:
        print(f"Error with new program: {e}")
        return False

async def main():
    # Setup
    print_separator("SETUP")
    setup_api_keys()
    model_path = "models/trained_qa_program.json"
    
    # Print model information
    if os.path.exists(model_path):
        print(f"Model file exists at {model_path}")
        print_model_structure(model_path)
    else:
        print(f"Model file does not exist at {model_path}")
        return
    
    # Try using examples from trained model
    success = await test_with_load_examples(model_path)
    
    # If that fails, test with a new program
    if not success:
        print("Falling back to testing with a new program")
        await test_new_program()

if __name__ == "__main__":
    # Run the test
    asyncio.run(main()) 