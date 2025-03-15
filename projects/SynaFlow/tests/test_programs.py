import pytest
import asyncio
from typing import Dict, Any, TYPE_CHECKING
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.programs import ScientificQAProgram
from src.data_models import ScientificQuery

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture

@pytest.fixture
def mock_language_model(mocker):
    """
    Create a mock language model for testing.
    
    Args:
        mocker: pytest-mock fixture
        
    Returns:
        Mock language model
    """
    mock_lm = mocker.MagicMock()
    mock_lm.model = "test-model"
    mock_lm.generate.return_value = "test generation"
    return mock_lm

@pytest.fixture
def mock_citation_data() -> Dict[str, Any]:
    """
    Create mock citation data for testing.
    
    Returns:
        Dictionary with mock citation data
    """
    return {
        "citations": [
            {
                "title": "Test Citation",
                "authors": ["Test Author"],
                "year": 2023,
                "source": "Test Journal",
                "url": "https://example.com/test"
            }
        ]
    }

@pytest.mark.asyncio
async def test_scientific_qa_program_initialization() -> None:
    """Test that the ScientificQAProgram initializes correctly."""
    program = ScientificQAProgram(name="test_program")
    
    # Check basic properties
    assert program.name == "test_program"
    assert program.description == "Scientific question answering system"
    assert program.trainable is True
    
    # Check examples and hints exist
    assert isinstance(program.examples, list)
    assert len(program.examples) > 0
    assert isinstance(program.hints, list)
    assert len(program.hints) > 0

@pytest.mark.asyncio
async def test_lookup_citations(mocker: "MockerFixture") -> None:
    """
    Test the citation lookup functionality.
    
    Args:
        mocker: pytest-mock fixture
    """
    from src.programs import lookup_citations
    
    # Call the function
    result = await lookup_citations("quantum mechanics")
    
    # Check the result
    assert "citations" in result
    assert isinstance(result["citations"], list)
    assert len(result["citations"]) > 0
    
    # Check the structure of a citation
    citation = result["citations"][0]
    assert "title" in citation
    assert "authors" in citation
    assert "year" in citation
    assert "source" in citation
    assert "url" in citation
    
    # Verify content
    assert "quantum mechanics" in citation["title"].lower()

@pytest.mark.asyncio
async def test_program_build(mocker: "MockerFixture", mock_language_model, mock_citation_data) -> None:
    """
    Test the build method of ScientificQAProgram.
    
    Args:
        mocker: pytest-mock fixture
        mock_language_model: Mock language model fixture
        mock_citation_data: Mock citation data fixture
    """
    # Mock the Action and Generator
    mock_action = mocker.patch("synalinks.Action")
    mock_action_instance = mock_action.return_value
    mock_action_instance.return_value = mock_citation_data
    
    mock_generator = mocker.patch("synalinks.Generator")
    mock_generator_instance = mock_generator.return_value
    mock_generator_instance.return_value = {
        "background": "Test background",
        "reasoning": "Test reasoning",
        "answer": "Test answer",
        "confidence": 0.9,
        "citations": [
            {
                "title": "Test Citation",
                "authors": ["Test Author"],
                "year": 2023,
                "source": "Test Journal",
                "url": "https://example.com/test"
            }
        ]
    }
    
    # Create program and test the build method
    program = ScientificQAProgram(language_model=mock_language_model)
    
    # Create input
    inputs = {
        "question": "How does quantum entanglement work?",
        "domain": "physics"
    }
    
    # Call build
    await program.build(inputs)
    
    # Verify Action was called
    mock_action.assert_called_once()
    mock_action_instance.assert_called_once_with(inputs)
    
    # Verify Generator was called
    mock_generator.assert_called_once()
    mock_generator_instance.assert_called_once() 