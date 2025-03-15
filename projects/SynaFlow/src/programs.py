import synalinks
import asyncio
from typing import Optional, Dict, Any, List, Callable
from src.data_models import ScientificQuery, ScientificAnswer, Citation
from src.language_models import get_model_with_fallback, get_cloud_model

async def lookup_citations(topic: str, max_results: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """
    Look up scientific citations for a given topic.
    
    Args:
        topic: The scientific topic to find citations for
        max_results: Maximum number of citations to return
        
    Returns:
        Dictionary containing a list of citation information
    """
    # In a real implementation, this would query a scientific database/API
    # This is a simplified example
    citations = [
        {
            "title": f"Understanding {topic}: A Comprehensive Review",
            "authors": ["A. Researcher", "B. Scientist"],
            "year": 2023,
            "source": "Journal of Science",
            "url": f"https://example.com/science/{topic.lower().replace(' ', '-')}"
        },
        {
            "title": f"Advanced {topic} Techniques",
            "authors": ["C. Expert", "D. Scholar"],
            "year": 2022,
            "source": "Scientific Reports",
            "url": f"https://example.com/reports/{topic.lower().replace(' ', '-')}"
        }
    ]
    return {"citations": citations[:max_results]}

class ScientificQAProgram(synalinks.Program):
    """
    A scientific question answering program using the hybrid approach.
    
    This program processes scientific questions and generates comprehensive
    answers with citations using the Synalinks framework.
    """
    
    def __init__(
        self,
        language_model: Optional[synalinks.LanguageModel] = None,
        use_citation_lookup: bool = True,
        name: str = "scientific_qa",
        description: str = "Scientific question answering system",
        trainable: bool = True,
    ):
        """
        Initialize the Scientific Q&A Program.
        
        Args:
            language_model: The language model to use for generation
            use_citation_lookup: Whether to use citation lookup functionality
            name: Name of the program
            description: Description of the program
            trainable: Whether the program can be trained
        """
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        self.language_model = language_model or get_model_with_fallback()
        self.use_citation_lookup = use_citation_lookup
        
        # Define reference examples
        self.examples = [
            {
                "inputs": {
                    "question": "How do black holes emit Hawking radiation?",
                    "domain": "physics"
                },
                "outputs": {
                    "background": "Black holes are regions of spacetime where gravity is so strong that nothing can escape from them, not even light.",
                    "reasoning": "According to quantum field theory in curved spacetime, vacuum fluctuations near the event horizon can result in particle-antiparticle pairs. When this happens near the event horizon, one particle may fall in while the other escapes, appearing as radiation.",
                    "answer": "Black holes emit Hawking radiation due to quantum effects near the event horizon where particle-antiparticle pairs can be separated, with one falling into the black hole and the other escaping as radiation.",
                    "confidence": 0.95,
                    "citations": [
                        {
                            "title": "A Brief History of Time",
                            "authors": ["Stephen Hawking"],
                            "year": 1988,
                            "source": "Bantam Books",
                            "url": "https://example.com/brief-history-time"
                        }
                    ]
                }
            }
        ]
        
        # Define hints for better generation
        self.hints = [
            "Provide comprehensive scientific explanations",
            "Include relevant mathematical formulas when applicable",
            "Cite reliable scientific sources",
            "Distinguish between established facts and theoretical models",
            "Consider multiple perspectives if there are scientific disagreements"
        ]
    
    async def build(self, inputs: Dict[str, Any]) -> None:
        """
        Build the program structure using the Functional API.
        
        Args:
            inputs: Input data for the program
        """
        # Extract the question topic for citation lookup
        if self.use_citation_lookup:
            # First step: Look up citations
            action_output = await synalinks.Action(
                fn=lookup_citations,
                language_model=self.language_model,
            )(inputs)
        
        # Generate the complete scientific answer
        outputs = await synalinks.Generator(
            data_model=ScientificAnswer,
            language_model=self.language_model,
            examples=self.examples,
            hints=self.hints,
            use_inputs_schema=True,
            use_outputs_schema=True,
        )(inputs if not self.use_citation_lookup else action_output)
        
        # Create the program using the functional API
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            name=self.name,
            description=self.description,
            trainable=self.trainable,
        ) 