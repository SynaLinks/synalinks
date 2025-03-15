import asyncio
import sys
import os

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from src.data_models import ScientificQuery, ScientificAnswer
from src.programs import lookup_citations

async def simple_demo():
    """A simple demo to test basic functionality."""
    print("\n===== SynaFlow Scientific Q&A Simple Demo =====\n")
    
    # Test citation lookup
    try:
        print("Testing citation lookup...")
        result = await lookup_citations("quantum physics")
        print(f"Found {len(result['citations'])} citations:")
        for citation in result['citations']:
            print(f"- {citation['title']} by {', '.join(citation['authors'])}")
        print("\nCitation lookup test successful!\n")
    except Exception as e:
        print(f"Error in citation lookup: {e}")
    
    print("Simple demo completed.")

if __name__ == "__main__":
    asyncio.run(simple_demo()) 