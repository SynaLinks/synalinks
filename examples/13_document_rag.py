"""
# Document RAG (Retrieval-Augmented Generation)

This example demonstrates how to build a classic RAG system using Synalinks.
RAG combines document retrieval with language model generation to answer
questions based on your own documents.

## How RAG Works

1. **Index**: Store documents in a knowledge base with embeddings
2. **Retrieve**: When a question is asked, find relevant documents
3. **Generate**: Use the retrieved context to generate an accurate answer

## Creating a Document Store

```python
class Document(synalinks.DataModel):
    id: str = synalinks.Field(description="Document ID")
    title: str = synalinks.Field(description="Document title")
    content: str = synalinks.Field(description="Document content")

knowledge_base = synalinks.KnowledgeBase(
    uri="duckdb://./documents.db",
    data_models=[Document],
    embedding_model=embedding_model,  # For semantic search
)
```

## Building the RAG Pipeline

```python
inputs = synalinks.Input(data_model=Query)

# Retrieve relevant documents
retrieved = await synalinks.RetrieveKnowledge(
    knowledge_base=knowledge_base,
    language_model=language_model,
    search_type="hybrid",
    k=3,
)(inputs)

# Generate answer from retrieved context
answer = await synalinks.Generator(
    data_model=Answer,
    language_model=language_model,
    instructions="Answer based on the retrieved documents.",
)(retrieved)
```

### Key Takeaways

- **Hybrid Search**: Combines keyword (BM25) and semantic (vector) search
    for better retrieval accuracy.
- **Chunking**: For large documents, split into smaller chunks for better
    retrieval granularity.
- **Context Window**: Retrieved documents are passed as context to the LM
    for grounded generation.
- **Trainable**: The retrieval and generation modules can be optimized
    using Synalinks training.

## Program Visualization

![document_rag](../assets/examples/document_rag.png)
"""

import asyncio
import os

from dotenv import load_dotenv

import synalinks


# =============================================================================
# Define the data models
# =============================================================================


class Document(synalinks.DataModel):
    """A document stored in the knowledge base."""

    id: str = synalinks.Field(
        description="Unique document identifier",
    )
    title: str = synalinks.Field(
        description="Document title",
    )
    content: str = synalinks.Field(
        description="The main text content of the document",
    )
    source: str = synalinks.Field(
        description="Source or category of the document",
    )


class Query(synalinks.DataModel):
    """A user question."""

    query: str = synalinks.Field(
        description="The user's question",
    )


class Answer(synalinks.DataModel):
    """An answer generated from retrieved documents."""

    answer: str = synalinks.Field(
        description="The answer to the question based on retrieved documents",
    )
    sources: str = synalinks.Field(
        description="The document titles used to generate the answer",
    )


async def main():
    load_dotenv()

    # Enable observability for tracing
    synalinks.enable_observability(
        tracking_uri="http://localhost:5000",
        experiment_name="document_rag_pipeline",
    )

    # Initialize models
    language_model = synalinks.LanguageModel(
        model="openai/gpt-4.1",
    )

    embedding_model = synalinks.EmbeddingModel(
        model="openai/text-embedding-3-small",
    )

    # Clean up any existing database
    db_path = "./examples/documents.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    # ==========================================================================
    # Step 1: Create the Document Knowledge Base
    # ==========================================================================
    print("Step 1: Creating Document Knowledge Base")
    print("=" * 60)

    knowledge_base = synalinks.KnowledgeBase(
        uri=f"duckdb://{db_path}",
        data_models=[Document],
        embedding_model=embedding_model,
        metric="cosine",
    )

    print(f"Knowledge base created: {db_path}")

    # ==========================================================================
    # Step 2: Add Documents to the Knowledge Base
    # ==========================================================================
    print("\nStep 2: Adding Documents to Knowledge Base")
    print("=" * 60)

    # Sample documents (in a real scenario, these would come from files, APIs, etc.)
    documents = [
        Document(
            id="doc-001",
            title="Introduction to Machine Learning",
            content="""
            Machine learning is a subset of artificial intelligence that enables
            systems to learn and improve from experience without being explicitly
            programmed. It focuses on developing algorithms that can access data
            and use it to learn for themselves. The process begins with
            observations or data, such as examples, direct experience, or
            instruction, to look for patterns in data and make better decisions
            in the future. The primary aim is to allow computers to learn
            automatically without human intervention.
            """,
            source="AI Fundamentals",
        ),
        Document(
            id="doc-002",
            title="Types of Machine Learning",
            content="""
            There are three main types of machine learning: supervised learning,
            unsupervised learning, and reinforcement learning. In supervised
            learning, the algorithm learns from labeled training data and makes
            predictions. Unsupervised learning works with unlabeled data to find
            hidden patterns. Reinforcement learning involves an agent learning
            to make decisions by performing actions and receiving rewards or
            penalties. Each type has specific use cases and applications in
            real-world scenarios.
            """,
            source="AI Fundamentals",
        ),
        Document(
            id="doc-003",
            title="Neural Networks Explained",
            content="""
            Neural networks are computing systems inspired by biological neural
            networks in the human brain. They consist of interconnected nodes
            (neurons) organized in layers: an input layer, one or more hidden
            layers, and an output layer. Each connection has a weight that
            adjusts as learning proceeds. Deep learning uses neural networks
            with many hidden layers, enabling the model to learn complex
            patterns. Popular architectures include CNNs for images and RNNs
            for sequential data.
            """,
            source="Deep Learning",
        ),
        Document(
            id="doc-004",
            title="Natural Language Processing Overview",
            content="""
            Natural Language Processing (NLP) is a field of AI that focuses on
            the interaction between computers and human language. Key tasks
            include text classification, named entity recognition, sentiment
            analysis, machine translation, and question answering. Modern NLP
            relies heavily on transformer models like BERT and GPT. These models
            use attention mechanisms to understand context and relationships
            between words in a sentence.
            """,
            source="NLP Guide",
        ),
        Document(
            id="doc-005",
            title="Large Language Models",
            content="""
            Large Language Models (LLMs) are neural networks trained on massive
            amounts of text data. They can generate human-like text, answer
            questions, summarize documents, and perform various language tasks.
            Examples include GPT-4, Claude, and Llama. LLMs use the transformer
            architecture and are trained using self-supervised learning on
            internet-scale datasets. They demonstrate emergent capabilities as
            they scale in size and training data.
            """,
            source="NLP Guide",
        ),
        Document(
            id="doc-006",
            title="RAG: Retrieval-Augmented Generation",
            content="""
            Retrieval-Augmented Generation (RAG) is a technique that combines
            information retrieval with text generation. Instead of relying
            solely on the knowledge embedded in model weights, RAG retrieves
            relevant documents from a knowledge base and uses them as context
            for generation. This approach reduces hallucinations, allows for
            up-to-date information, and enables domain-specific applications.
            RAG systems typically use vector databases for efficient similarity
            search.
            """,
            source="Advanced AI",
        ),
        Document(
            id="doc-007",
            title="Vector Databases and Embeddings",
            content="""
            Vector databases store and index high-dimensional vectors (embeddings)
            for efficient similarity search. Embeddings are numerical
            representations of text, images, or other data that capture semantic
            meaning. Similar items have vectors that are close together in the
            embedding space. Popular vector databases include Pinecone, Weaviate,
            and Chroma. They use algorithms like HNSW or IVF for approximate
            nearest neighbor search, enabling fast retrieval at scale.
            """,
            source="Advanced AI",
        ),
    ]

    # Store documents
    print("\nStoring documents...")
    for doc in documents:
        await knowledge_base.update(doc.to_json_data_model())
        print(f"  - Stored: {doc.title}")

    print(f"\nTotal documents stored: {len(documents)}")

    # ==========================================================================
    # Step 3: Build the RAG Pipeline
    # ==========================================================================
    print("\nStep 3: Building RAG Pipeline")
    print("=" * 60)

    inputs = synalinks.Input(data_model=Query)

    # Retrieve relevant documents using hybrid search
    retrieved = await synalinks.RetrieveKnowledge(
        knowledge_base=knowledge_base,
        language_model=language_model,
        search_type="hybrid",
        k=3,  # Retrieve top 3 documents
        return_inputs=True,
        return_query=True,
    )(inputs)

    # Generate answer based on retrieved context
    answer = await synalinks.Generator(
        data_model=Answer,
        language_model=language_model,
        instructions="""
        Answer the user's question based ONLY on the retrieved documents.
        If the information is not in the documents, say "I don't have information about that."
        Include the titles of the documents you used in the 'sources' field.
        """,
    )(retrieved)

    # Create the RAG program
    rag_program = synalinks.Program(
        inputs=inputs,
        outputs=answer,
        name="document_rag",
        description="Answer questions using retrieved documents",
    )

    # Plot the program
    synalinks.utils.plot_program(
        rag_program,
        to_folder="examples",
        show_module_names=True,
        show_schemas=True,
        show_trainable=True,
    )

    print("RAG pipeline created successfully!")

    # ==========================================================================
    # Step 4: Test the RAG System
    # ==========================================================================
    print("\nStep 4: Testing RAG System")
    print("=" * 60)

    test_questions = [
        "What is machine learning?",
        "What are the three types of machine learning?",
        "How do neural networks work?",
        "What is RAG and why is it useful?",
        "What are vector databases used for?",
        "What is the capital of France?",  # Not in documents
    ]

    for question in test_questions:
        print(f"\n{'─' * 60}")
        print(f"Q: {question}")
        print(f"{'─' * 60}")

        result = await rag_program(Query(query=question))

        print(f"\nA: {result.get('answer')}")
        print(f"\nSources: {result.get('sources')}")

    # ==========================================================================
    # Step 5: Direct Search Examples
    # ==========================================================================
    print("\n\nStep 5: Direct Search Examples")
    print("=" * 60)

    # Full-text search
    print("\nFull-text search for 'transformer':")
    results = await knowledge_base.fulltext_search("transformer", k=3)
    for r in results:
        print(f"  - Document ID: {r.get('id')}, Score: {r.get('score', 'N/A'):.4f}")

    # Hybrid search
    print("\nHybrid search for 'how computers learn from data':")
    results = await knowledge_base.hybrid_search("how computers learn from data", k=3)
    for r in results:
        print(f"  - Document ID: {r.get('id')}, Score: {r.get('score', 'N/A'):.4f}")

    # ==========================================================================
    # Step 6: List All Documents
    # ==========================================================================
    print("\n\nStep 6: All Documents in Knowledge Base")
    print("=" * 60)

    doc_model = knowledge_base.get_symbolic_data_models()[0]
    all_docs = await knowledge_base.getall(doc_model, limit=20)

    for doc in all_docs:
        data = doc.get_json()
        print(f"\n[{data['id']}] {data['title']}")
        print(f"   Source: {data['source']}")
        # Print first 100 chars of content
        content_preview = data["content"].strip()[:100].replace("\n", " ")
        print(f"   Content: {content_preview}...")

    print("\n\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
