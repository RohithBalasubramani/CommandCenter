"""
Tests for Layer 2 RAG Pipeline.

Tests vector storage, embedding, and retrieval components.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))


class TestRAGDocument:
    """Test RAGDocument data class."""

    def test_create_document(self):
        """Test creating a RAG document."""
        from layer2.rag_pipeline import RAGDocument

        doc = RAGDocument(
            id="test_001",
            content="Test content for document",
            metadata={"type": "test", "equipment_id": "EQ-001"},
        )

        assert doc.id == "test_001"
        assert doc.content == "Test content for document"
        assert doc.metadata["type"] == "test"
        assert doc.embedding is None

    def test_document_with_embedding(self):
        """Test document with pre-computed embedding."""
        from layer2.rag_pipeline import RAGDocument

        embedding = [0.1] * 768  # Fake 768-dim embedding
        doc = RAGDocument(
            id="test_002",
            content="Test with embedding",
            metadata={},
            embedding=embedding,
        )

        assert doc.embedding is not None
        assert len(doc.embedding) == 768


class TestRAGSearchResult:
    """Test RAGSearchResult data class."""

    def test_create_search_result(self):
        """Test creating a search result."""
        from layer2.rag_pipeline import RAGSearchResult

        result = RAGSearchResult(
            id="result_001",
            content="Matching content",
            metadata={"equipment_type": "pump"},
            score=0.95,
        )

        assert result.id == "result_001"
        assert result.score == 0.95
        assert result.metadata["equipment_type"] == "pump"


class TestRAGResponse:
    """Test RAGResponse data class."""

    def test_create_rag_response(self):
        """Test creating a RAG response."""
        from layer2.rag_pipeline import RAGResponse

        response = RAGResponse(
            query="What is pump status?",
            retrieved_docs=[],
            context="Pump 1 is running at 80% load",
            llm_response="Pump 1 is currently running normally at 80% load.",
            sources=[{"id": "pump_001", "score": 0.9}],
        )

        assert response.query == "What is pump status?"
        assert "80%" in response.llm_response
        assert len(response.sources) == 1


class TestEmbeddingService:
    """Test the embedding service."""

    def test_embedding_service_lazy_load(self, mock_embedding_model):
        """Test that embedding model is lazy loaded."""
        from layer2.rag_pipeline import EmbeddingService

        service = EmbeddingService()
        # Model should not be loaded yet
        assert service._model is None

    def test_embed_single_text(self, mock_embedding_model):
        """Test embedding a single text."""
        from layer2.rag_pipeline import EmbeddingService

        service = EmbeddingService()
        embedding = service.embed("Test text for embedding")

        assert embedding is not None
        # Should be a list of floats
        assert isinstance(embedding, list)

    def test_embed_batch(self, mock_embedding_model):
        """Test embedding multiple texts."""
        from layer2.rag_pipeline import EmbeddingService

        service = EmbeddingService()
        texts = ["Text 1", "Text 2", "Text 3"]

        # Mock returns numpy array
        import numpy as np
        mock_embedding_model.return_value.encode.return_value = np.random.rand(3, 768)

        embeddings = service.embed_batch(texts)
        assert embeddings is not None


class TestVectorStoreService:
    """Test the vector store service."""

    def test_vector_store_lazy_load(self, mock_chromadb):
        """Test that ChromaDB client is lazy loaded."""
        from layer2.rag_pipeline import VectorStoreService

        service = VectorStoreService()
        assert service._client is None

    def test_get_or_create_collection(self, mock_chromadb):
        """Test getting or creating a collection."""
        from layer2.rag_pipeline import VectorStoreService

        service = VectorStoreService()
        collection = service.get_or_create_collection("test_collection")

        assert collection is not None
        mock_chromadb.PersistentClient.return_value.get_or_create_collection.assert_called_once()

    def test_add_documents(self, mock_chromadb, mock_embedding_model):
        """Test adding documents to collection."""
        from layer2.rag_pipeline import VectorStoreService, RAGDocument

        service = VectorStoreService()
        docs = [
            RAGDocument(id="doc1", content="Content 1", metadata={"type": "test"}),
            RAGDocument(id="doc2", content="Content 2", metadata={"type": "test"}),
        ]

        service.add_documents("test_collection", docs)

        # Verify collection.add was called
        collection = mock_chromadb.PersistentClient.return_value.get_or_create_collection.return_value
        collection.add.assert_called_once()

    def test_search_documents(self, mock_chromadb, mock_embedding_model):
        """Test searching documents."""
        from layer2.rag_pipeline import VectorStoreService

        service = VectorStoreService()
        results = service.search("test_collection", "pump status", n_results=5)

        assert results is not None
        assert len(results) == 2  # Mock returns 2 results
        assert results[0].id == "eq_1"
        assert results[0].score == 0.9  # 1 - 0.1 distance

    def test_search_with_filter(self, mock_chromadb, mock_embedding_model):
        """Test searching with metadata filter."""
        from layer2.rag_pipeline import VectorStoreService

        service = VectorStoreService()
        results = service.search(
            "test_collection",
            "pump status",
            n_results=5,
            filter_metadata={"equipment_type": "pump"},
        )

        # Verify filter was passed
        collection = mock_chromadb.PersistentClient.return_value.get_or_create_collection.return_value
        call_args = collection.query.call_args
        assert call_args[1]["where"] == {"equipment_type": "pump"}

    def test_get_collection_count(self, mock_chromadb):
        """Test getting collection document count."""
        from layer2.rag_pipeline import VectorStoreService

        service = VectorStoreService()
        count = service.get_collection_count("test_collection")

        assert count == 100  # Mock returns 100

    def test_delete_collection(self, mock_chromadb):
        """Test deleting a collection."""
        from layer2.rag_pipeline import VectorStoreService

        service = VectorStoreService()
        service.delete_collection("test_collection")

        mock_chromadb.PersistentClient.return_value.delete_collection.assert_called_with("test_collection")

    def test_search_multiple_collections(self, mock_chromadb, mock_embedding_model):
        """Test searching across multiple collections."""
        from layer2.rag_pipeline import VectorStoreService

        service = VectorStoreService()
        results = service.search_multiple_collections(
            "pump status",
            collections=["equipment", "alerts"],
            n_results_per=3,
        )

        assert results is not None
        # Results should be sorted by score
        if len(results) > 1:
            assert results[0].score >= results[1].score


class TestOllamaLLMService:
    """Test the Ollama LLM service."""

    def test_llm_generate(self, mock_ollama):
        """Test LLM text generation."""
        from layer2.rag_pipeline import OllamaLLMService

        llm = OllamaLLMService()
        response = llm.generate("What is pump status?")

        assert response == "Test LLM response"
        mock_ollama.post.assert_called_once()

    def test_llm_generate_with_system_prompt(self, mock_ollama):
        """Test LLM generation with system prompt."""
        from layer2.rag_pipeline import OllamaLLMService

        llm = OllamaLLMService()
        response = llm.generate(
            "What is pump status?",
            system_prompt="You are an industrial assistant.",
        )

        call_args = mock_ollama.post.call_args
        assert "system" in call_args[1]["json"]

    def test_llm_chat(self, mock_ollama):
        """Test LLM chat interface."""
        mock_ollama.post.return_value.json.return_value = {
            "message": {"content": "Chat response"}
        }

        from layer2.rag_pipeline import OllamaLLMService

        llm = OllamaLLMService()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "What is pump status?"},
        ]
        response = llm.chat(messages)

        assert response == "Chat response"

    def test_llm_generate_json(self, mock_ollama):
        """Test LLM JSON generation."""
        mock_ollama.post.return_value.json.return_value = {
            "response": '{"type": "query", "domain": "industrial"}'
        }

        from layer2.rag_pipeline import OllamaLLMService

        llm = OllamaLLMService()
        result = llm.generate_json("Classify this query")

        assert result is not None
        assert result["type"] == "query"

    def test_llm_is_available(self, mock_ollama):
        """Test LLM availability check."""
        from layer2.rag_pipeline import OllamaLLMService

        llm = OllamaLLMService()
        available = llm.is_available()

        assert available is True

    def test_llm_unavailable(self, mock_ollama):
        """Test handling when LLM is unavailable."""
        mock_ollama.get.side_effect = Exception("Connection refused")

        from layer2.rag_pipeline import OllamaLLMService

        llm = OllamaLLMService()
        available = llm.is_available()

        assert available is False

    def test_llm_generate_connection_error(self):
        """Test handling of connection errors during generation."""
        import requests
        from unittest.mock import patch

        with patch("layer2.rag_pipeline.requests") as mock_requests:
            # Set up the ConnectionError properly
            mock_requests.exceptions.ConnectionError = requests.exceptions.ConnectionError
            mock_requests.post.side_effect = requests.exceptions.ConnectionError()

            from layer2.rag_pipeline import OllamaLLMService

            llm = OllamaLLMService()
            response = llm.generate("Test query")

            assert "[LLM unavailable" in response


class TestIndustrialRAGPipeline:
    """Test the complete Industrial RAG Pipeline."""

    def test_pipeline_initialization(self, mock_chromadb, mock_ollama):
        """Test pipeline initialization."""
        from layer2.rag_pipeline import IndustrialRAGPipeline

        pipeline = IndustrialRAGPipeline()

        assert pipeline.vector_store is not None
        assert pipeline.llm is not None
        assert pipeline.llm_fast is not None
        assert pipeline.llm_quality is not None

    def test_get_stats(self, mock_chromadb, mock_ollama):
        """Test getting pipeline statistics."""
        from layer2.rag_pipeline import IndustrialRAGPipeline

        pipeline = IndustrialRAGPipeline()
        stats = pipeline.get_stats()

        assert "equipment_count" in stats
        assert "alerts_count" in stats
        assert "llm_available" in stats
        assert "llm_model" in stats

    def test_query_pipeline(self, mock_chromadb, mock_embedding_model, mock_ollama):
        """Test querying the RAG pipeline."""
        from layer2.rag_pipeline import IndustrialRAGPipeline

        pipeline = IndustrialRAGPipeline()
        response = pipeline.query("What is the pump status?")

        assert response is not None
        assert response.query == "What is the pump status?"
        assert response.llm_response is not None

    def test_query_with_options(self, mock_chromadb, mock_embedding_model, mock_ollama):
        """Test querying with various options."""
        from layer2.rag_pipeline import IndustrialRAGPipeline

        pipeline = IndustrialRAGPipeline()
        response = pipeline.query(
            "Show alerts",
            n_results=10,
            include_alerts=True,
            include_maintenance=False,
            include_documents=False,
        )

        assert response is not None

    def test_clear_index(self, mock_chromadb, mock_ollama):
        """Test clearing the index."""
        from layer2.rag_pipeline import IndustrialRAGPipeline

        pipeline = IndustrialRAGPipeline()
        pipeline.clear_index()

        # Verify delete was called for all collections
        assert mock_chromadb.PersistentClient.return_value.delete_collection.call_count >= 1


class TestRAGPipelineSingleton:
    """Test the RAG pipeline singleton."""

    def test_singleton_returns_same_instance(self, mock_chromadb, mock_ollama):
        """Test that get_rag_pipeline returns the same instance."""
        from layer2.rag_pipeline import get_rag_pipeline, _rag_pipeline
        import layer2.rag_pipeline as rag_module

        # Reset singleton
        rag_module._rag_pipeline = None

        pipeline1 = get_rag_pipeline()
        pipeline2 = get_rag_pipeline()

        assert pipeline1 is pipeline2
