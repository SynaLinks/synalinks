import os
import json
import logging
from typing import Dict, Any, List, Optional, Union, Tuple

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        
    Returns:
        Configured logger instance
    """
    # Set up logging level from string
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Configure handlers
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger("scientific_qa")

def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load and parse a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON content as dictionary
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data: Dict[str, Any], file_path: str, pretty: bool = True) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the file
        pretty: Whether to format the JSON with indentation
        
    Raises:
        IOError: If the file cannot be written
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Write the file
    with open(file_path, 'w', encoding='utf-8') as f:
        if pretty:
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            json.dump(data, f, ensure_ascii=False)

def format_citation(citation: Dict[str, Any]) -> str:
    """
    Format a citation dictionary into a readable string.
    
    Args:
        citation: Citation dictionary with fields like title, authors, year, etc.
        
    Returns:
        Formatted citation string
    """
    authors = citation.get("authors", [])
    title = citation.get("title", "")
    year = citation.get("year", "")
    source = citation.get("source", "")
    url = citation.get("url", "")
    
    # Format authors
    if len(authors) == 1:
        author_str = authors[0]
    elif len(authors) == 2:
        author_str = f"{authors[0]} and {authors[1]}"
    elif len(authors) > 2:
        author_str = f"{', '.join(authors[:-1])}, and {authors[-1]}"
    else:
        author_str = ""
    
    # Build the citation string
    citation_parts = []
    if author_str:
        citation_parts.append(author_str)
    if year:
        citation_parts.append(f"({year})")
    if title:
        citation_parts.append(f"{title}.")
    if source:
        citation_parts.append(f"{source}.")
    if url:
        citation_parts.append(f"Available at: {url}")
    
    return " ".join(citation_parts) 