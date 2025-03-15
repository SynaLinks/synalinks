import asyncio
import argparse
import logging
from typing import Dict, Any, Optional, Tuple
import sys
import os

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Now import with correct paths
import synalinks
from src.language_models import setup_api_keys
from src.programs import ScientificQAProgram
from src.training import train_program, evaluate_program
from src.data_models import ScientificQuery
from src.utils import setup_logging, format_citation

logger = setup_logging(log_level="INFO")

async def run_interactive_demo(model_path: Optional[str] = None) -> None:
    """
    Run an interactive demo of the scientific QA system.
    
    Args:
        model_path: Optional path to a trained model file
    """
    # Set up API keys
    setup_api_keys()
    
    # Load the program
    try:
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading trained program from {model_path}")
            try:
                # Try to load the program asynchronously
                program = await synalinks.Program.load(model_path)
            except TypeError as e:
                if "can't be used in 'await' expression" in str(e):
                    # Try synchronous loading if await isn't supported
                    logger.info("Trying synchronous loading")
                    program = synalinks.Program.load(model_path)
                else:
                    raise
        else:
            if model_path:
                logger.warning(f"Model file {model_path} not found. Creating new program.")
            else:
                logger.info("Creating new program")
            program = ScientificQAProgram()
    except Exception as e:
        logger.error(f"Error loading trained model: {e}")
        logger.info("Creating new program as fallback")
        program = ScientificQAProgram()
    
    print("\n===== Scientific Q&A System =====\n")
    print("Type 'exit' to quit the demo\n")
    
    while True:
        question = input("\nEnter a scientific question: ")
        if question.lower() == 'exit':
            break
        
        domain = input("Domain (optional, press Enter to skip): ")
        domain = domain if domain else None
        
        context = input("Additional context (optional, press Enter to skip): ")
        context = context if context else None
        
        print("\nProcessing your question...\n")
        
        try:
            # Create query
            query = ScientificQuery(
                question=question,
                domain=domain,
                context=context
            )
            
            # Get answer
            result = await program(query)
            
            # Print the answer
            print(f"\n----- BACKGROUND -----\n")
            print(result.get("background"))
            
            print(f"\n----- REASONING -----\n")
            print(result.get("reasoning"))
            
            print(f"\n----- ANSWER -----\n")
            print(result.get("answer"))
            
            print(f"\n----- CONFIDENCE: {result.get('confidence'):.2f} -----\n")
            
            print(f"\n----- CITATIONS -----\n")
            for citation in result.get("citations", []):
                formatted_citation = format_citation(citation)
                print(f"- {formatted_citation}")
            
            further_reading = result.get("further_reading", [])
            if further_reading:
                print(f"\n----- FURTHER READING -----\n")
                for reading in further_reading:
                    print(f"- {reading}")
                    
        except Exception as e:
            print(f"Error: {e}")

async def main() -> None:
    """
    Main entry point for the scientific Q&A system.
    """
    parser = argparse.ArgumentParser(description="Scientific Q&A System")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--demo', action='store_true', help='Run interactive demo')
    parser.add_argument('--model-path', type=str, default="models/trained_qa_program.json", 
                        help='Path to the model file')
    parser.add_argument('--data-path', type=str, default="data/scientific_qa_dataset.json",
                        help='Path to the dataset file')
    
    args = parser.parse_args()
    
    if args.train:
        logger.info("Starting training")
        program, history = await train_program(
            program_path=args.model_path if args.evaluate else None,
            data_path=args.data_path,
            output_path=args.model_path
        )
        logger.info(f"Training completed and model saved to {args.model_path}")
    
    if args.evaluate:
        logger.info("Starting evaluation")
        metrics = await evaluate_program(
            program_path=args.model_path,
            data_path=args.data_path
        )
        logger.info(f"Evaluation metrics: {metrics}")
    
    if args.demo:
        await run_interactive_demo(args.model_path)
    
    # If no arguments provided, run the demo
    if not (args.train or args.evaluate or args.demo):
        await run_interactive_demo(args.model_path)

if __name__ == "__main__":
    asyncio.run(main()) 