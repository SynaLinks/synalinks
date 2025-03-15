import os
import asyncio
import json
import re
from typing import Dict, Any, List, Optional

import synalinks
from synalinks import ChatMessage, ChatRole, ChatMessages
from src.data_models import ScientificQuery, ScientificAnswer, Citation
from src.language_models import get_cloud_model

class SimplifiedScientificQA:
    """
    A simplified scientific Q&A program that works reliably.
    
    This class uses the examples from training but bypasses the complex
    program execution flow that causes errors.
    """
    
    def __init__(
        self,
        examples: Optional[List[Dict[str, Any]]] = None,
        name: str = "scientific_qa",
        description: str = "Scientific question answering system"
    ):
        """
        Initialize the simplified scientific Q&A program.
        
        Args:
            examples: List of examples for few-shot learning
            name: Name of the program
            description: Description of the program
        """
        self.examples = examples or []
        self.name = name
        self.description = description
        self.language_model = get_cloud_model()
    
    @classmethod
    def from_file(cls, file_path: str):
        """
        Create a program instance from a saved model file.
        
        Args:
            file_path: Path to the saved model file
            
        Returns:
            Instance of SimplifiedScientificQA
        """
        # Load the model file
        with open(file_path, 'r') as f:
            model_data = json.load(f)
        
        # Extract examples
        examples = []
        if 'config' in model_data and 'examples' in model_data['config']:
            examples = model_data['config']['examples']
        
        # Create a new instance
        return cls(
            examples=examples,
            name=model_data.get('config', {}).get('name', 'scientific_qa'),
            description=model_data.get('config', {}).get('description', 'Scientific question answering system')
        )
    
    async def __call__(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a scientific query.
        
        Args:
            query: Scientific query (can be a ScientificQuery object or a dict)
            
        Returns:
            Dictionary with answer components
        """
        # Convert query to dict if it's an object
        if hasattr(query, '__dict__'):
            query_dict = {k: v for k, v in query.__dict__.items() 
                         if not k.startswith('_')}
        else:
            query_dict = query
        
        # Extract query components
        question = query_dict.get('question', '')
        domain = query_dict.get('domain')
        context = query_dict.get('context')
        
        return await self.generate_answer(question, domain, context)
    
    async def generate_answer(self, question: str, domain: Optional[str] = None, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a scientific answer.
        
        Args:
            question: The scientific question
            domain: Optional domain (e.g., physics, biology)
            context: Optional additional context
            
        Returns:
            Dictionary with answer components
        """
        # Create the prompt with examples
        system_message = self._create_system_message()
        user_message = self._create_user_message(question, domain, context)
        
        # Create chat messages
        messages = ChatMessages(messages=[
            ChatMessage(role=ChatRole.SYSTEM, content=system_message),
            ChatMessage(role=ChatRole.USER, content=user_message)
        ])
        
        # Generate response
        response = await self.language_model(messages)
        
        # Parse the response
        return self._parse_response(response)
    
    def _create_system_message(self) -> str:
        """Create the system message with examples."""
        base_message = """You are a scientific question answering system. 
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
        
        CRITICAL INSTRUCTIONS:
        1. Your response MUST be valid JSON - check for missing commas, incorrect quoting, etc.
        2. Do not include code block formatting like ```json or ``` around your response
        3. Ensure all JSON fields have valid values - no trailing commas, properly quoted strings
        4. The "confidence" value must be a number between 0 and 1, not a string
        5. Authors and further_reading must be valid arrays, even if empty
        6. Only include the JSON object with no other text before or after
        """
        
        # Add examples if available
        if self.examples:
            base_message += "\n\nHere are some examples of how to answer scientific questions:\n\n"
            for i, example in enumerate(self.examples):
                inputs = example.get('inputs', {})
                outputs = example.get('outputs', {})
                
                # Format example
                base_message += f"Example {i+1}:\n"
                base_message += f"Question: {inputs.get('question', '')}\n"
                if inputs.get('domain'):
                    base_message += f"Domain: {inputs.get('domain')}\n"
                
                base_message += f"\nBackground: {outputs.get('background', '')}\n"
                base_message += f"Reasoning: {outputs.get('reasoning', '')}\n"
                base_message += f"Answer: {outputs.get('answer', '')}\n"
                base_message += f"Confidence: {outputs.get('confidence', 0.95)}\n"
                
                # Add citations
                citations = outputs.get('citations', [])
                if citations:
                    base_message += "Citations:\n"
                    for citation in citations:
                        authors = ", ".join(citation.get('authors', []))
                        title = citation.get('title', '')
                        year = citation.get('year', '')
                        source = citation.get('source', '')
                        base_message += f"- {title} ({year}) by {authors}. {source}\n"
                
                base_message += "\n"
        
        return base_message
    
    def _create_user_message(self, question: str, domain: Optional[str] = None, context: Optional[str] = None) -> str:
        """Create the user message with the question."""
        message = f"Question: {question}\n"
        if domain:
            message += f"Domain: {domain}\n"
        if context:
            message += f"Additional Context: {context}\n"
        return message
    
    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the response from the language model with improved reliability.
        
        This method implements several fallback strategies to handle malformed JSON responses:
        1. Try to extract JSON content from markdown code blocks
        2. Fix common JSON syntax errors
        3. Use regex to extract keys and values if JSON parsing fails
        4. Provide sensible defaults for missing fields
        
        Args:
            response: The response from the language model
            
        Returns:
            Dictionary with structured answer components
        """
        try:
            # Extract content from the response
            content = response.get('content', '')
            
            # 1. Try to extract from markdown code blocks
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            else:
                # Try to find an object starting with { and ending with }
                match = re.search(r'(\{.*\})', content, re.DOTALL)
                if match:
                    json_str = match.group(1)
                else:
                    json_str = content
            
            # 2. Fix common JSON syntax errors
            # Remove JavaScript comments
            json_str = re.sub(r'//.*', '', json_str)
            # Fix trailing commas in arrays
            json_str = re.sub(r',\s*]', ']', json_str)
            # Fix trailing commas in objects
            json_str = re.sub(r',\s*}', '}', json_str)
            
            # Try to parse the fixed JSON
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError as e:
                # 3. Use regex to extract keys and values
                print(f"JSON decode error: {e}. Attempting regex extraction...")
                result = self._extract_fields_with_regex(content)
            
            # 4. Ensure default fields
            result.setdefault("background", "No background information provided.")
            result.setdefault("reasoning", "No reasoning provided.")
            result.setdefault("answer", "No answer provided.")
            result.setdefault("confidence", 0.5)
            result.setdefault("citations", [])
            result.setdefault("further_reading", [])
            
            # Ensure confidence is a float between 0 and 1
            if isinstance(result.get("confidence"), str):
                try:
                    result["confidence"] = float(result["confidence"])
                except ValueError:
                    result["confidence"] = 0.5
            
            # Cap confidence value
            if result.get("confidence") > 1.0:
                result["confidence"] = 1.0
            elif result.get("confidence") < 0.0:
                result["confidence"] = 0.0
            
            # Ensure citations is a list
            if not isinstance(result.get("citations"), list):
                result["citations"] = []
            
            # Ensure further_reading is a list
            if not isinstance(result.get("further_reading"), list):
                result["further_reading"] = []
            
            return result
            
        except Exception as e:
            print(f"Error parsing response: {e}")
            # Fallback to providing a response with the raw content
            return {
                "background": "Error parsing the response.",
                "reasoning": f"An error occurred during processing: {str(e)}",
                "answer": content,  # Return raw content as fallback
                "confidence": 0.0,
                "citations": [],
                "further_reading": []
            }
    
    def _extract_fields_with_regex(self, content: str) -> Dict[str, Any]:
        """
        Extract fields from content using regex when JSON parsing fails.
        
        Args:
            content: The raw content string
            
        Returns:
            Dictionary with extracted fields
        """
        result = {}
        
        # Extract key fields
        background_match = re.search(r'"background"\s*:\s*"([^"]*)"', content)
        if background_match:
            result["background"] = background_match.group(1)
        
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', content)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1)
        
        answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', content)
        if answer_match:
            result["answer"] = answer_match.group(1)
        
        confidence_match = re.search(r'"confidence"\s*:\s*([\d\.]+)', content)
        if confidence_match:
            try:
                result["confidence"] = float(confidence_match.group(1))
            except ValueError:
                result["confidence"] = 0.5
        
        # For citations and further_reading, we'll use empty lists as they're complex to extract with regex
        result["citations"] = []
        result["further_reading"] = []
        
        return result 