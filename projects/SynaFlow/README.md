# SynaFlow

A scientific question answering system built with SynaLinks, leveraging language models to provide accurate scientific answers with citations.

## Features

- Scientific question answering with domain-specific knowledge
- Citation and reference tracking for answers
- API for integration with other applications
- Customizable examples for improved responses
- Docker support for easy deployment

## Getting Started

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (for containerized deployment)
- OpenAI API key

### Installation

1. Clone the repository
   ```
   git clone https://github.com/yourusername/synaflow.git
   cd synaflow
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables
   ```
   cp .env.example .env
   # Edit .env with your API keys
   ```

### Running Locally

1. Train the model (optional, use pre-trained model if available)
   ```
   python main.py --train
   ```

2. Run the interactive demo
   ```
   python examples/simplified_demo.py
   ```

3. Start the API server
   ```
   python api/app_direct.py
   ```

### Docker Deployment

1. Build and start the containers
   ```
   docker-compose up -d
   ```

2. Check the API is running
   ```
   curl http://localhost:8000/health
   ```

## API Usage

### Generate a scientific answer

```
POST /query

{
  "question": "What is quantum entanglement?",
  "domain": "physics",
  "context": "I'm a university student"
}
```

Response:
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2023-11-18T12:34:56.789Z",
  "result": {
    "background": "Quantum entanglement is a fundamental concept in quantum physics...",
    "reasoning": "When we consider the mathematical framework of quantum mechanics...",
    "answer": "Quantum entanglement occurs when pairs or groups of particles interact...",
    "confidence": 0.95,
    "citations": [
      {
        "title": "Quantum entanglement and information",
        "authors": ["Einstein, A.", "Podolsky, B.", "Rosen, N."],
        "year": 1935,
        "source": "Physical Review",
        "url": "https://example.com/paper"
      }
    ],
    "further_reading": [
      "Quantum Computation and Quantum Information by Nielsen and Chuang"
    ]
  }
}
```

## Project Structure

```
synaflow/
├── api/                 # API implementation
│   └── app_direct.py    # FastAPI application
├── examples/            # Example scripts and demos
│   ├── simplified_demo.py
│   ├── comparison_test.py
│   └── interactive_demo.py
├── models/              # Trained models
│   └── trained_qa_program.json
├── src/                 # Source code
│   ├── config.py        # Configuration settings
│   ├── data_models.py   # Data models and schemas
│   ├── language_models.py # LLM integration
│   ├── programs.py      # Program implementations
│   ├── simplified_program.py # Simplified implementation
│   ├── training.py      # Training utilities
│   └── utils.py         # Utility functions
├── tests/               # Test suite
├── .env.example         # Example environment variables
├── docker-compose.yml   # Docker Compose configuration
├── Dockerfile           # Docker build configuration
├── main.py              # Main entry point
└── requirements.txt     # Python dependencies
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- SynaLinks for the underlying framework
- OpenAI for language model APIs 