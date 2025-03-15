import synalinks
import os
from typing import Optional, Any, Dict
from src.config import OPENAI_API_KEY, ANTHROPIC_API_KEY
from src.config import DEFAULT_LOCAL_MODEL, DEFAULT_CLOUD_MODEL, FALLBACK_MODEL

def setup_api_keys() -> None:
    """
    Set up API keys for LLM providers.
    
    This function configures the API keys for OpenAI and Anthropic
    if they are available in the environment.
    """
    # Just check if API keys are available, but don't try to set them directly
    # since synalinks might handle this differently
    if OPENAI_API_KEY:
        print(f"OpenAI API key is available")
    if ANTHROPIC_API_KEY:
        print(f"Anthropic API key is available")
    
    # The actual keys are already set in the environment variables
    # which synalinks should read automatically

def get_local_model(model_name: str = DEFAULT_LOCAL_MODEL, temperature: float = 0.2) -> synalinks.LanguageModel:
    """
    Get a local LLM via Ollama.
    
    Args:
        model_name: Name of the local model to use
        temperature: Temperature parameter for generation
        
    Returns:
        A configured local language model
    """
    # Create the model with only the model parameter
    model = synalinks.LanguageModel(
        model=model_name,
    )
    # Store temperature as an attribute for later use during inference
    model._temperature = temperature
    return model

def get_cloud_model(model_name: str = DEFAULT_CLOUD_MODEL, temperature: float = 0.2) -> synalinks.LanguageModel:
    """
    Get a cloud-based LLM.
    
    Args:
        model_name: Name of the cloud model to use
        temperature: Temperature parameter for generation
        
    Returns:
        A configured cloud language model
    """
    # Create the model with only the model parameter
    model = synalinks.LanguageModel(
        model=model_name,
    )
    # Store temperature as an attribute for later use during inference
    model._temperature = temperature
    return model

def get_model_with_fallback() -> synalinks.LanguageModel:
    """
    Try to get a local model, fall back to cloud if not available.
    
    This function attempts to initialize a local language model first,
    and if that fails, it falls back to a cloud model, and finally
    to a designated fallback model if both primary options fail.
    
    Returns:
        A configured language model
    
    Raises:
        Exception: If all model initialization attempts fail
    """
    try:
        return get_local_model()
    except Exception as e:
        print(f"Local model unavailable: {e}. Falling back to cloud model.")
        try:
            return get_cloud_model()
        except Exception as e2:
            print(f"Primary cloud model unavailable: {e2}. Using fallback model.")
            model = synalinks.LanguageModel(model=FALLBACK_MODEL)
            model._temperature = 0.2
            return model 