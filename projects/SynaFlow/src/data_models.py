import synalinks
from typing import List, Optional

class ScientificQuery(synalinks.DataModel):
    """Scientific question input model."""
    question: str = synalinks.Field(
        description="The scientific question to be answered"
    )
    domain: Optional[str] = synalinks.Field(
        description="Scientific domain (e.g., physics, biology, chemistry)",
        default=None
    )
    context: Optional[str] = synalinks.Field(
        description="Additional context for the question",
        default=None
    )

class Citation(synalinks.DataModel):
    """Citation for scientific sources."""
    title: str = synalinks.Field(
        description="Title of the source"
    )
    authors: List[str] = synalinks.Field(
        description="List of authors"
    )
    year: Optional[int] = synalinks.Field(
        description="Publication year",
        default=None
    )
    source: str = synalinks.Field(
        description="Source (journal, book, website, etc.)"
    )
    url: Optional[str] = synalinks.Field(
        description="URL if available",
        default=None
    )

class ScientificAnswer(synalinks.DataModel):
    """Comprehensive scientific answer model."""
    background: str = synalinks.Field(
        description="Background information and context for the question"
    )
    reasoning: str = synalinks.Field(
        description="Step-by-step reasoning process and explanation"
    )
    answer: str = synalinks.Field(
        description="Clear and concise answer to the question"
    )
    confidence: float = synalinks.Field(
        description="Confidence score between 0 and 1",
        default=0.0
    )
    citations: List[Citation] = synalinks.Field(
        description="Scientific citations supporting the answer"
    )
    further_reading: Optional[List[str]] = synalinks.Field(
        description="Suggested further reading on the topic",
        default=None
    ) 