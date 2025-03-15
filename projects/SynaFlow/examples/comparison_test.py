import os
import asyncio
import json
import time
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from src.simplified_program import SimplifiedScientificQA
from src.language_models import get_cloud_model
from synalinks import ChatMessage, ChatRole, ChatMessages

# Load environment variables
load_dotenv()

# Ensure OpenAI API key is available
if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
    print("Error: OPENAI_API_KEY not found in environment variables")
    print("Please add it to your .env file or set it as an environment variable")
    exit(1)

# Test questions
TEST_QUESTIONS = [
    {"question": "What is quantum entanglement?", "domain": "physics"},
    {"question": "How do neurons communicate?", "domain": "neuroscience"},
    {"question": "What causes climate change?", "domain": "environmental science"},
    {"question": "How do vaccines work?", "domain": "immunology"},
    {"question": "What is dark matter?", "domain": "astrophysics"}
]

async def direct_fallback_approach(question: str, domain: Optional[str] = None) -> Dict[str, Any]:
    """
    Direct fallback approach without trained examples.
    
    Args:
        question: Scientific question
        domain: Optional domain
        
    Returns:
        Dictionary with answer components
    """
    language_model = get_cloud_model()
    
    # Create system message without examples
    system_message = """You are a scientific question answering system. 
    Provide comprehensive, accurate answers to scientific questions.
    
    Format your response as a structured JSON with these fields:
    {
      "background": "Background information and context for the question",
      "reasoning": "Step-by-step reasoning process and explanation",
      "answer": "Clear and concise answer to the question",
      "confidence": 0.95,
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
    
    # Create user message
    user_message = f"Question: {question}\n"
    if domain:
        user_message += f"Domain: {domain}\n"
    
    # Create messages
    messages = ChatMessages(messages=[
        ChatMessage(role=ChatRole.SYSTEM, content=system_message),
        ChatMessage(role=ChatRole.USER, content=user_message)
    ])
    
    # Generate response
    response = await language_model(messages)
    
    # Parse response
    try:
        content = response.get('content', '')
        
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
        
        # Parse JSON
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
        print(f"Error parsing response: {e}")
        return {
            "background": "Error parsing the response.",
            "reasoning": "An error occurred during processing.",
            "answer": content,
            "confidence": 0.0,
            "citations": [],
            "further_reading": []
        }

async def trained_model_approach(question: str, domain: Optional[str] = None) -> Dict[str, Any]:
    """
    Approach using the trained model with examples.
    
    Args:
        question: Scientific question
        domain: Optional domain
        
    Returns:
        Dictionary with answer components
    """
    # Load the trained model
    model_path = "models/trained_qa_program.json"
    
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}")
        return {"error": "Model not found"}
    
    qa_program = SimplifiedScientificQA.from_file(model_path)
    
    # Process the question
    query = {
        "question": question,
        "domain": domain
    }
    
    try:
        result = await qa_program(query)
        return result
    except Exception as e:
        print(f"Error processing with trained model: {e}")
        return {
            "background": "Error processing with trained model.",
            "reasoning": f"Error: {str(e)}",
            "answer": "Unable to generate an answer with the trained model.",
            "confidence": 0.0,
            "citations": [],
            "further_reading": []
        }

def compare_responses(fallback_response: Dict[str, Any], trained_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare responses from both approaches.
    
    Args:
        fallback_response: Response from fallback approach
        trained_response: Response from trained model approach
        
    Returns:
        Dictionary with comparison metrics
    """
    # Compare answer length (more detailed answers might be better)
    fallback_answer_length = len(fallback_response.get("answer", ""))
    trained_answer_length = len(trained_response.get("answer", ""))
    
    # Compare number of citations
    fallback_citations = len(fallback_response.get("citations", []))
    trained_citations = len(trained_response.get("citations", []))
    
    # Compare confidence
    fallback_confidence = fallback_response.get("confidence", 0.0)
    trained_confidence = trained_response.get("confidence", 0.0)
    
    # Compare reasoning length
    fallback_reasoning_length = len(fallback_response.get("reasoning", ""))
    trained_reasoning_length = len(trained_response.get("reasoning", ""))
    
    return {
        "answer_length_comparison": {
            "fallback": fallback_answer_length,
            "trained": trained_answer_length,
            "difference": trained_answer_length - fallback_answer_length,
            "percent_difference": (trained_answer_length - fallback_answer_length) / max(1, fallback_answer_length) * 100
        },
        "citation_comparison": {
            "fallback": fallback_citations,
            "trained": trained_citations,
            "difference": trained_citations - fallback_citations
        },
        "confidence_comparison": {
            "fallback": fallback_confidence,
            "trained": trained_confidence,
            "difference": trained_confidence - fallback_confidence
        },
        "reasoning_length_comparison": {
            "fallback": fallback_reasoning_length,
            "trained": trained_reasoning_length,
            "difference": trained_reasoning_length - fallback_reasoning_length,
            "percent_difference": (trained_reasoning_length - fallback_reasoning_length) / max(1, fallback_reasoning_length) * 100
        }
    }

async def run_tests():
    """Run comparison tests between the two approaches."""
    print("\n=== Comparison Test: Fallback vs. Trained Model ===\n")
    
    # Check if model exists
    model_path = "models/trained_qa_program.json"
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}")
        return
    
    # Load the model to check examples
    with open(model_path, 'r') as f:
        model_data = json.load(f)
    
    examples = []
    if 'config' in model_data and 'examples' in model_data['config']:
        examples = model_data['config']['examples']
    
    print(f"Testing with trained model containing {len(examples)} examples")
    print(f"Running tests on {len(TEST_QUESTIONS)} questions...\n")
    
    overall_metrics = {
        "fallback_time": 0,
        "trained_time": 0,
        "answer_length_diff_total": 0,
        "reasoning_length_diff_total": 0,
        "citations_diff_total": 0,
        "confidence_diff_total": 0
    }
    
    for i, test in enumerate(TEST_QUESTIONS):
        question = test["question"]
        domain = test.get("domain")
        
        print(f"Test {i+1}: {question}")
        print(f"Domain: {domain}")
        
        # Run fallback approach
        print("Running fallback approach...")
        fallback_start = time.time()
        fallback_response = await direct_fallback_approach(question, domain)
        fallback_time = time.time() - fallback_start
        
        # Run trained model approach
        print("Running trained model approach...")
        trained_start = time.time()
        trained_response = await trained_model_approach(question, domain)
        trained_time = time.time() - trained_start
        
        # Compare the results
        comparison = compare_responses(fallback_response, trained_response)
        
        # Update overall metrics
        overall_metrics["fallback_time"] += fallback_time
        overall_metrics["trained_time"] += trained_time
        overall_metrics["answer_length_diff_total"] += comparison["answer_length_comparison"]["difference"]
        overall_metrics["reasoning_length_diff_total"] += comparison["reasoning_length_comparison"]["difference"]
        overall_metrics["citations_diff_total"] += comparison["citation_comparison"]["difference"]
        overall_metrics["confidence_diff_total"] += comparison["confidence_comparison"]["difference"]
        
        # Print results for this question
        print("\nResults:")
        print(f"  Fallback time: {fallback_time:.2f}s")
        print(f"  Trained model time: {trained_time:.2f}s")
        print(f"  Time difference: {trained_time - fallback_time:.2f}s")
        
        print("\nFallback answer snippet:")
        print(f"  {fallback_response.get('answer', '')[:100]}...")
        
        print("\nTrained model answer snippet:")
        print(f"  {trained_response.get('answer', '')[:100]}...")
        
        print("\nMetrics:")
        print(f"  Answer length: Fallback={comparison['answer_length_comparison']['fallback']}, Trained={comparison['answer_length_comparison']['trained']}")
        print(f"  Answer length difference: {comparison['answer_length_comparison']['difference']} chars ({comparison['answer_length_comparison']['percent_difference']:.1f}%)")
        
        print(f"  Citations: Fallback={comparison['citation_comparison']['fallback']}, Trained={comparison['citation_comparison']['trained']}")
        print(f"  Citations difference: {comparison['citation_comparison']['difference']}")
        
        print(f"  Confidence: Fallback={comparison['confidence_comparison']['fallback']:.2f}, Trained={comparison['confidence_comparison']['trained']:.2f}")
        print(f"  Confidence difference: {comparison['confidence_comparison']['difference']:.2f}")
        
        print(f"  Reasoning length: Fallback={comparison['reasoning_length_comparison']['fallback']}, Trained={comparison['reasoning_length_comparison']['trained']}")
        print(f"  Reasoning length difference: {comparison['reasoning_length_comparison']['difference']} chars ({comparison['reasoning_length_comparison']['percent_difference']:.1f}%)")
        
        print("\n" + "-" * 80 + "\n")
    
    # Calculate averages
    num_tests = len(TEST_QUESTIONS)
    avg_fallback_time = overall_metrics["fallback_time"] / num_tests
    avg_trained_time = overall_metrics["trained_time"] / num_tests
    avg_answer_length_diff = overall_metrics["answer_length_diff_total"] / num_tests
    avg_reasoning_length_diff = overall_metrics["reasoning_length_diff_total"] / num_tests
    avg_citations_diff = overall_metrics["citations_diff_total"] / num_tests
    avg_confidence_diff = overall_metrics["confidence_diff_total"] / num_tests
    
    # Print overall results
    print("\n=== Overall Results ===")
    print(f"Average fallback approach time: {avg_fallback_time:.2f}s")
    print(f"Average trained model time: {avg_trained_time:.2f}s")
    print(f"Average time difference: {avg_trained_time - avg_fallback_time:.2f}s")
    print(f"Average answer length difference: {avg_answer_length_diff:.1f} chars")
    print(f"Average reasoning length difference: {avg_reasoning_length_diff:.1f} chars")
    print(f"Average citations difference: {avg_citations_diff:.1f}")
    print(f"Average confidence difference: {avg_confidence_diff:.2f}")
    
    if avg_trained_time < avg_fallback_time:
        speed_advantage = "trained model"
        speed_percent = ((avg_fallback_time - avg_trained_time) / avg_fallback_time) * 100
    else:
        speed_advantage = "fallback approach"
        speed_percent = ((avg_trained_time - avg_fallback_time) / avg_trained_time) * 100
    
    print(f"\nSpeed advantage: {speed_advantage} is {speed_percent:.1f}% faster")
    
    if avg_answer_length_diff > 0:
        print(f"Content advantage: trained model provides {abs(avg_answer_length_diff):.1f} more characters on average")
    else:
        print(f"Content advantage: fallback approach provides {abs(avg_answer_length_diff):.1f} more characters on average")
    
    if avg_citations_diff > 0:
        print(f"Citation advantage: trained model provides {avg_citations_diff:.1f} more citations on average")
    else:
        print(f"Citation advantage: fallback approach provides {abs(avg_citations_diff):.1f} more citations on average")
    
    print("\nConclusion:")
    advantages_trained = []
    advantages_fallback = []
    
    if avg_trained_time < avg_fallback_time:
        advantages_trained.append("faster response time")
    else:
        advantages_fallback.append("faster response time")
    
    if avg_answer_length_diff > 0:
        advantages_trained.append("longer answers")
    else:
        advantages_fallback.append("longer answers")
    
    if avg_reasoning_length_diff > 0:
        advantages_trained.append("more detailed reasoning")
    else:
        advantages_fallback.append("more detailed reasoning")
    
    if avg_citations_diff > 0:
        advantages_trained.append("more citations")
    else:
        advantages_fallback.append("more citations")
    
    if avg_confidence_diff > 0:
        advantages_trained.append("higher confidence")
    else:
        advantages_fallback.append("higher confidence")
    
    if advantages_trained:
        print(f"Trained model advantages: {', '.join(advantages_trained)}")
    
    if advantages_fallback:
        print(f"Fallback approach advantages: {', '.join(advantages_fallback)}")
    
    if len(advantages_trained) > len(advantages_fallback):
        print("\nOverall recommendation: Use the trained model approach")
    elif len(advantages_fallback) > len(advantages_trained):
        print("\nOverall recommendation: Use the fallback approach")
    else:
        print("\nOverall recommendation: Both approaches have equal advantages, consider specific use case requirements")

if __name__ == "__main__":
    asyncio.run(run_tests()) 