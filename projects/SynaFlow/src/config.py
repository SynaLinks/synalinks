import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")

# LLM Configuration
DEFAULT_LOCAL_MODEL: str = "ollama_chat/llama3"
DEFAULT_CLOUD_MODEL: str = "openai/gpt-4"
FALLBACK_MODEL: str = "anthropic/claude-3-sonnet-20240229"

# Application Settings
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour by default
MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))

# Training Settings
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "16"))
EPOCHS: int = int(os.getenv("EPOCHS", "5")) 