"""
Layer 2 RAG Pipeline — Retrieval Augmented Generation for Industrial Data

This module provides:
1. Document embedding using sentence-transformers (BGE-M3 or similar)
2. Vector storage using ChromaDB
3. Retrieval and context preparation for LLM
4. LLM integration via Ollama (Qwen3/Phi-4)

Tech Stack:
- Embeddings: sentence-transformers (all-MiniLM-L6-v2 default, BGE-M3 for production)
- Vector DB: ChromaDB (local, persistent)
- LLM: Ollama API (Qwen3, Phi-4, or similar)
"""

import os
import json
import logging
from typing import Optional
from dataclasses import dataclass

# Optional imports - gracefully degrade if not installed
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# BM25 for hybrid search
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

# Embedding model — BGE-base-en-v1.5: best balance of quality vs speed for industrial RAG.
# MTEB retrieval score ~63 (vs 56 for MiniLM). 768-dim, 110MB, sentence-transformers compatible.
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

# ChromaDB persistence directory (absolute path to avoid CWD issues)
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PERSIST_DIR = os.getenv("RAG_CHROMA_DIR", os.path.join(_BACKEND_DIR, "chroma_db"))

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi4")  # legacy single-model (used by existing code)

# Dual-model config for Pipeline v2
# FAST: small model for intent parsing + widget selection (~5GB VRAM)
# QUALITY: large model for voice response generation (~40-55GB VRAM)
OLLAMA_MODEL_FAST = os.getenv("OLLAMA_MODEL_FAST", "llama3.1:8b")
OLLAMA_MODEL_QUALITY = os.getenv("OLLAMA_MODEL_QUALITY", "llama3.3")

# Collection names
EQUIPMENT_COLLECTION = "industrial_equipment"
ALERTS_COLLECTION = "industrial_alerts"
MAINTENANCE_COLLECTION = "maintenance_records"
OPERATIONAL_DOCS_COLLECTION = "operational_documents"
SHIFT_LOGS_COLLECTION = "shift_logs"
WORK_ORDERS_COLLECTION = "work_orders"


# ============================================================
# Data Classes
# ============================================================

@dataclass
class RAGDocument:
    """Document for RAG indexing."""
    id: str
    content: str
    metadata: dict
    embedding: Optional[list] = None


@dataclass
class RAGSearchResult:
    """Search result from RAG."""
    id: str
    content: str
    metadata: dict
    score: float


@dataclass
class RAGResponse:
    """Complete RAG response with context and LLM answer."""
    query: str
    retrieved_docs: list
    context: str
    llm_response: str
    sources: list


# ============================================================
# Embedding Service
# ============================================================

class EmbeddingService:
    """Service for generating text embeddings."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        return self._model

    def embed(self, text: str) -> list:
        """Generate embedding for a single text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list) -> list:
        """Generate embeddings for multiple texts."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


# ============================================================
# Vector Store Service
# ============================================================

class VectorStoreService:
    """Service for vector storage and retrieval using ChromaDB."""

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        self.persist_dir = persist_dir
        self._client = None
        self._embedding_service = None
        # BM25 indices for hybrid search (collection_name -> (bm25, doc_ids, doc_contents))
        self._bm25_indices: dict[str, tuple] = {}

    @property
    def client(self):
        """Lazy load ChromaDB client."""
        if self._client is None:
            if not CHROMADB_AVAILABLE:
                raise ImportError(
                    "chromadb not installed. "
                    "Run: pip install chromadb"
                )
            logger.info(f"Initializing ChromaDB at: {self.persist_dir}")
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            logger.info("ChromaDB initialized successfully")
        return self._client

    @property
    def embedding_service(self):
        """Get or create embedding service."""
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    def get_or_create_collection(self, name: str):
        """Get or create a collection."""
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, collection_name: str, documents: list[RAGDocument]):
        """Add documents to a collection."""
        collection = self.get_or_create_collection(collection_name)

        ids = []
        embeddings = []
        metadatas = []
        contents = []

        for doc in documents:
            ids.append(doc.id)
            contents.append(doc.content)
            metadatas.append(doc.metadata)

            # Generate embedding if not provided
            if doc.embedding:
                embeddings.append(doc.embedding)
            else:
                embeddings.append(self.embedding_service.embed(doc.content))

        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=contents,
        )

        # Build BM25 index for hybrid search
        if BM25_AVAILABLE:
            self._build_bm25_index(collection_name, ids, contents)

        logger.info(f"Added {len(documents)} documents to collection: {collection_name}")

    def search(
        self,
        collection_name: str,
        query: str,
        n_results: int = 5,
        filter_metadata: dict = None,
    ) -> list[RAGSearchResult]:
        """Search for similar documents."""
        collection = self.get_or_create_collection(collection_name)

        # Generate query embedding
        query_embedding = self.embedding_service.embed(query)

        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"],
        )

        # Convert to RAGSearchResult
        search_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                search_results.append(RAGSearchResult(
                    id=doc_id,
                    content=results["documents"][0][i] if results["documents"] else "",
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    score=1 - results["distances"][0][i] if results["distances"] else 0,
                ))

        return search_results

    def delete_collection(self, collection_name: str):
        """Delete a collection."""
        try:
            self.client.delete_collection(collection_name)
            # Clear BM25 index
            if collection_name in self._bm25_indices:
                del self._bm25_indices[collection_name]
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.warning(f"Failed to delete collection {collection_name}: {e}")

    # ============================================================
    # BM25 Hybrid Search
    # ============================================================

    def _build_bm25_index(self, collection_name: str, doc_ids: list[str], doc_contents: list[str]):
        """Build BM25 index for a collection."""
        if not BM25_AVAILABLE:
            return

        # Tokenize documents (simple whitespace + lowercase)
        tokenized = [self._tokenize(doc) for doc in doc_contents]

        # Build BM25 index
        bm25 = BM25Okapi(tokenized)
        self._bm25_indices[collection_name] = (bm25, doc_ids, doc_contents)
        logger.info(f"Built BM25 index for {collection_name} with {len(doc_ids)} documents")

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for BM25."""
        import re
        # Lowercase, split on non-alphanumeric, filter short tokens
        tokens = re.split(r'[^a-zA-Z0-9]+', text.lower())
        return [t for t in tokens if len(t) > 1]

    def _ensure_bm25_index(self, collection_name: str):
        """Ensure BM25 index exists for collection, build if needed."""
        if not BM25_AVAILABLE:
            return False

        if collection_name in self._bm25_indices:
            return True

        # Build from existing collection
        try:
            collection = self.get_or_create_collection(collection_name)
            count = collection.count()
            if count == 0:
                return False

            # Fetch all documents
            results = collection.get(include=["documents"])
            if results and results["ids"] and results["documents"]:
                self._build_bm25_index(collection_name, results["ids"], results["documents"])
                return True
        except Exception as e:
            logger.warning(f"Failed to build BM25 index for {collection_name}: {e}")

        return False

    def search_hybrid(
        self,
        collection_name: str,
        query: str,
        n_results: int = 5,
        alpha: float = 0.7,
        filter_metadata: dict = None,
    ) -> list[RAGSearchResult]:
        """
        Hybrid search combining vector similarity and BM25.

        Args:
            collection_name: Collection to search
            query: Search query
            n_results: Number of results to return
            alpha: Weight for vector search (1-alpha for BM25). Default 0.7 favors vectors.
            filter_metadata: Optional metadata filter

        Returns:
            List of RAGSearchResult ordered by hybrid score
        """
        # Get more candidates from each method for better fusion
        n_candidates = n_results * 3

        # Vector search
        vector_results = self.search(collection_name, query, n_candidates, filter_metadata)

        # BM25 search
        bm25_results = self._search_bm25(collection_name, query, n_candidates)

        # If BM25 not available, return pure vector results
        if not bm25_results:
            return vector_results[:n_results]

        # Reciprocal Rank Fusion (RRF)
        k = 60  # RRF constant
        scores = {}  # doc_id -> score

        # Add vector search contributions
        for rank, result in enumerate(vector_results):
            rrf_score = 1.0 / (k + rank + 1)
            scores[result.id] = scores.get(result.id, 0) + alpha * rrf_score

        # Add BM25 contributions
        for rank, (doc_id, bm25_score) in enumerate(bm25_results):
            rrf_score = 1.0 / (k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0) + (1 - alpha) * rrf_score

        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Build results from vector_results (which have full metadata)
        result_map = {r.id: r for r in vector_results}
        final_results = []

        for doc_id in sorted_ids[:n_results]:
            if doc_id in result_map:
                result = result_map[doc_id]
                # Update score to hybrid score
                result.score = scores[doc_id]
                final_results.append(result)
            else:
                # Result only from BM25, need to fetch from collection
                try:
                    collection = self.get_or_create_collection(collection_name)
                    fetched = collection.get(ids=[doc_id], include=["documents", "metadatas"])
                    if fetched and fetched["ids"]:
                        final_results.append(RAGSearchResult(
                            id=doc_id,
                            content=fetched["documents"][0] if fetched["documents"] else "",
                            metadata=fetched["metadatas"][0] if fetched["metadatas"] else {},
                            score=scores[doc_id],
                        ))
                except Exception:
                    pass

        return final_results

    def _search_bm25(
        self, collection_name: str, query: str, n_results: int
    ) -> list[tuple[str, float]]:
        """
        Search using BM25 only.

        Returns:
            List of (doc_id, bm25_score) tuples
        """
        if not self._ensure_bm25_index(collection_name):
            return []

        bm25, doc_ids, _ = self._bm25_indices[collection_name]
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        # Get BM25 scores for all documents
        scores = bm25.get_scores(query_tokens)

        # Get top N by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]

        return [(doc_ids[i], scores[i]) for i in top_indices if scores[i] > 0]

    def query_entity(
        self,
        entity: str,
        metric: str = "",
        collection_name: str = EQUIPMENT_COLLECTION,
        n_results: int = 3,
    ) -> list[RAGSearchResult]:
        """Targeted query for a specific entity + metric combination.

        Used by the schema-driven data collector to fetch data for individual widgets.
        """
        query_text = f"{entity} {metric}".strip()
        return self.search(collection_name, query_text, n_results=n_results)

    def search_multiple_collections(
        self,
        query: str,
        collections: list[str],
        n_results_per: int = 3,
    ) -> list[RAGSearchResult]:
        """Search across multiple collections, merging results sorted by score."""
        all_results = []
        for coll in collections:
            try:
                results = self.search(coll, query, n_results=n_results_per)
                all_results.extend(results)
            except Exception:
                continue
        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results

    def get_collection_count(self, collection_name: str) -> int:
        """Get document count in a collection."""
        try:
            collection = self.get_or_create_collection(collection_name)
            return collection.count()
        except Exception:
            return 0


# ============================================================
# LLM Service (Ollama)
# ============================================================

class OllamaLLMService:
    """Service for LLM inference via Ollama."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        cache: "LLMCache" = None,
    ):
        self.base_url = base_url
        self.model = model
        self.cache = cache

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        cache_key: str = None,
    ) -> str:
        """Generate response from LLM.

        Args:
            cache_key: If provided, used as the semantic cache key instead of
                       the full prompt (useful when the prompt contains large
                       templates that drown out the variable query portion).
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests not installed. Run: pip install requests")

        # Check cache first
        if self.cache is not None:
            cached = self.cache.get(cache_key or prompt, system_prompt or "")
            if cached is not None:
                return cached

        url = f"{self.base_url}/api/generate"

        # Prepend /no_think for qwen3 models to disable thinking mode
        actual_prompt = prompt
        if "qwen3" in self.model.lower():
            actual_prompt = "/no_think\n" + prompt

        payload = {
            "model": self.model,
            "prompt": actual_prompt,
            "stream": False,
            "keep_alive": -1,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            raw = result.get("response", "")
            # Strip any remaining <think>...</think> tags (qwen3, deepseek-r1)
            import re as _re
            clean = _re.sub(r'<think>.*?</think>', '', raw, flags=_re.DOTALL).strip()
            result_text = clean if clean else raw.strip()

            # Store in cache
            if self.cache is not None:
                self.cache.put(cache_key or prompt, result_text, system_prompt or "")

            return result_text
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to Ollama at {self.base_url}")
            return "[LLM unavailable - Ollama not running]"
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"[LLM error: {str(e)}]"

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Chat with LLM using message format."""
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests not installed. Run: pip install requests")

        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to Ollama at {self.base_url}")
            return "[LLM unavailable - Ollama not running]"
        except Exception as e:
            logger.error(f"LLM chat failed: {e}")
            return f"[LLM error: {str(e)}]"

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        cache_key: str = None,
    ) -> dict | None:
        """Generate structured JSON response from LLM.

        Uses Ollama's JSON format mode and validates the output.
        Returns parsed dict on success, None on failure.

        Args:
            cache_key: If provided, used as the semantic cache key instead of
                       the full prompt.
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests not installed. Run: pip install requests")

        # Check cache first
        if self.cache is not None:
            cached = self.cache.get(cache_key or prompt, system_prompt or "")
            if cached is not None:
                return cached

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": -1,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        for attempt in range(2):
            try:
                response = requests.post(url, json=payload, timeout=90)
                response.raise_for_status()
                raw = response.json().get("response", "")
                parsed = json.loads(raw)

                # Store in cache
                if self.cache is not None:
                    self.cache.put(cache_key or prompt, parsed, system_prompt or "")

                return parsed
            except json.JSONDecodeError:
                logger.warning(f"JSON parse failed (attempt {attempt + 1}): {raw[:200]}")
                continue
            except requests.exceptions.ConnectionError:
                logger.error(f"Failed to connect to Ollama at {self.base_url}")
                return None
            except Exception as e:
                logger.error(f"LLM JSON generation failed: {e}")
                return None
        return None

    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False


# ============================================================
# Semantic LLM Cache
# ============================================================

class LLMCache:
    """
    Semantic cache for LLM responses using embedding similarity.

    Uses EmbeddingService to compute cache keys via cosine similarity.
    In-memory with TTL — no Redis required.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 300,
        max_entries: int = 200,
    ):
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._embedding_service = None

        # Cache: list of (embedding_ndarray, response, timestamp)
        self._cache: list = []
        self._lock = __import__("threading").Lock()

        # Stats
        self.hits = 0
        self.misses = 0

    @property
    def embedding_service(self):
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    def _cosine_similarity(self, a, b) -> float:
        import numpy as np
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / norm) if norm > 0 else 0.0

    def _partition_key(self, system_prompt: str) -> str:
        """Create a short partition key from the system prompt so entries
        with different system prompts never match each other."""
        import hashlib
        return hashlib.md5(system_prompt.encode()).hexdigest()[:8] if system_prompt else ""

    def get(self, prompt: str, system_prompt: str = "") -> object:
        """Look up a cached response for a semantically similar prompt.

        Only the user prompt is embedded for similarity (the system prompt is
        used as a partition key so different call-sites don't collide).
        """
        import time as _time
        import numpy as np

        now = _time.time()
        partition = self._partition_key(system_prompt)

        try:
            query_emb = np.array(self.embedding_service.embed(prompt))
        except Exception:
            return None

        with self._lock:
            # Evict expired
            self._cache = [e for e in self._cache if now - e[2] < self.ttl_seconds]

            best_sim = 0.0
            best_response = None
            for emb, response, ts, part in self._cache:
                if part != partition:
                    continue
                sim = self._cosine_similarity(query_emb, emb)
                if sim > best_sim:
                    best_sim = sim
                    best_response = response

            if best_sim >= self.similarity_threshold:
                self.hits += 1
                logger.debug(f"LLMCache HIT: similarity={best_sim:.3f} partition={partition}")
                return best_response

        self.misses += 1
        return None

    def put(self, prompt: str, response: object, system_prompt: str = ""):
        """Store a response in the cache."""
        import time as _time
        import numpy as np

        partition = self._partition_key(system_prompt)
        try:
            embedding = np.array(self.embedding_service.embed(prompt))
        except Exception:
            return

        now = _time.time()
        with self._lock:
            if len(self._cache) >= self.max_entries:
                self._cache.sort(key=lambda e: e[2])
                self._cache = self._cache[-(self.max_entries - 1):]
            self._cache.append((embedding, response, now, partition))

    def get_stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total > 0 else 0,
            "entries": len(self._cache),
        }


# ============================================================
# Industrial RAG Pipeline
# ============================================================

class IndustrialRAGPipeline:
    """
    Complete RAG pipeline for industrial equipment data.

    Combines:
    - Vector search for relevant equipment/alerts
    - LLM for natural language response generation
    """

    def __init__(self):
        self.vector_store = VectorStoreService()
        # Shared semantic cache for both LLM instances
        self._llm_cache = LLMCache()
        self.llm = OllamaLLMService(cache=self._llm_cache)  # legacy single-model (backward compat)
        # Pipeline v2: dual-model LLM instances
        self.llm_fast = OllamaLLMService(model=OLLAMA_MODEL_FAST, cache=self._llm_cache)
        self.llm_quality = OllamaLLMService(model=OLLAMA_MODEL_QUALITY, cache=self._llm_cache)
        self._indexed = False

    def index_equipment_from_db(self):
        """Index all equipment from Django database."""
        from industrial.models import (
            Transformer, DieselGenerator, ElectricalPanel, UPS,
            Chiller, AHU, CoolingTower, Pump, Compressor, Motor,
            EnergyMeter, Alert, MaintenanceRecord,
        )

        logger.info("Starting equipment indexing...")

        # Index each equipment type
        equipment_models = [
            (Transformer, "transformer"),
            (DieselGenerator, "diesel_generator"),
            (ElectricalPanel, "electrical_panel"),
            (UPS, "ups"),
            (Chiller, "chiller"),
            (AHU, "ahu"),
            (CoolingTower, "cooling_tower"),
            (Pump, "pump"),
            (Compressor, "compressor"),
            (Motor, "motor"),
            (EnergyMeter, "energy_meter"),
        ]

        all_docs = []
        for model, eq_type in equipment_models:
            for eq in model.objects.all():
                doc = self._equipment_to_document(eq, eq_type)
                all_docs.append(doc)

        if all_docs:
            self.vector_store.add_documents(EQUIPMENT_COLLECTION, all_docs)
            logger.info(f"Indexed {len(all_docs)} equipment documents")

        # Index alerts
        alert_docs = []
        for alert in Alert.objects.all()[:2000]:  # All alerts (resolved + unresolved) for historical context
            doc = self._alert_to_document(alert)
            alert_docs.append(doc)

        if alert_docs:
            self.vector_store.add_documents(ALERTS_COLLECTION, alert_docs)
            logger.info(f"Indexed {len(alert_docs)} alert documents")

        # Index maintenance records
        maint_docs = []
        for record in MaintenanceRecord.objects.all()[:4000]:  # Up to 4000 records
            doc = self._maintenance_to_document(record)
            maint_docs.append(doc)

        if maint_docs:
            self.vector_store.add_documents(MAINTENANCE_COLLECTION, maint_docs)
            logger.info(f"Indexed {len(maint_docs)} maintenance documents")

        # Index operational documents (SOPs, inspection reports, etc.)
        op_docs = self._index_operational_documents()
        logger.info(f"Indexed {op_docs} operational documents")

        # Index shift logs
        shift_docs = self._index_shift_logs()
        logger.info(f"Indexed {shift_docs} shift log documents")

        # Index work orders
        wo_docs = self._index_work_orders()
        logger.info(f"Indexed {wo_docs} work order documents")

        self._indexed = True
        logger.info("Equipment indexing complete!")

    def _equipment_to_document(self, equipment, eq_type: str) -> RAGDocument:
        """Convert equipment model to RAG document."""
        # Build a rich text description
        content_parts = [
            f"{equipment.name} ({equipment.equipment_id})",
            f"Type: {eq_type.replace('_', ' ').title()}",
            f"Location: {equipment.location}",
            f"Building: {equipment.building}" if equipment.building else "",
            f"Status: {equipment.status}",
            f"Health Score: {equipment.health_score}%",
            f"Criticality: {equipment.criticality}",
            equipment.description,
        ]

        # Add type-specific fields
        if hasattr(equipment, 'capacity_kva'):
            content_parts.append(f"Capacity: {equipment.capacity_kva} kVA")
        if hasattr(equipment, 'capacity_kw'):
            content_parts.append(f"Capacity: {equipment.capacity_kw} kW")
        if hasattr(equipment, 'capacity_tr'):
            content_parts.append(f"Capacity: {equipment.capacity_tr} TR")
        if hasattr(equipment, 'load_percent') and equipment.load_percent:
            content_parts.append(f"Current Load: {equipment.load_percent:.1f}%")
        if hasattr(equipment, 'power_consumption_kw') and equipment.power_consumption_kw:
            content_parts.append(f"Power Consumption: {equipment.power_consumption_kw:.1f} kW")

        content = " | ".join([p for p in content_parts if p])

        return RAGDocument(
            id=f"{eq_type}_{equipment.equipment_id}",
            content=content,
            metadata={
                "equipment_id": equipment.equipment_id,
                "equipment_type": eq_type,
                "name": equipment.name,
                "location": equipment.location,
                "building": equipment.building or "",
                "status": equipment.status,
                "criticality": equipment.criticality,
                "health_score": equipment.health_score,
            },
        )

    def _alert_to_document(self, alert) -> RAGDocument:
        """Convert alert model to RAG document."""
        content = (
            f"Alert: {alert.message} | "
            f"Equipment: {alert.equipment_name} ({alert.equipment_id}) | "
            f"Severity: {alert.severity} | "
            f"Type: {alert.alert_type} | "
            f"Parameter: {alert.parameter} | "
            f"Value: {alert.value} {alert.unit} | "
            f"Threshold: {alert.threshold} {alert.unit}"
        )

        return RAGDocument(
            id=f"alert_{alert.id}",
            content=content,
            metadata={
                "equipment_id": alert.equipment_id,
                "equipment_name": alert.equipment_name,
                "severity": alert.severity,
                "alert_type": alert.alert_type,
                "acknowledged": alert.acknowledged,
                "resolved": alert.resolved,
            },
        )

    def _maintenance_to_document(self, record) -> RAGDocument:
        """Convert maintenance record to RAG document."""
        content = (
            f"Maintenance: {record.description} | "
            f"Equipment: {record.equipment_name} ({record.equipment_id}) | "
            f"Type: {record.maintenance_type} | "
            f"Work Done: {record.work_done} | "
            f"Parts Replaced: {record.parts_replaced} | "
            f"Technician: {record.technician}"
        )

        return RAGDocument(
            id=f"maint_{record.id}",
            content=content,
            metadata={
                "equipment_id": record.equipment_id,
                "equipment_name": record.equipment_name,
                "maintenance_type": record.maintenance_type,
            },
        )

    def _index_operational_documents(self) -> int:
        """Index operational documents (SOPs, inspection reports, etc.) from raw SQL table."""
        from django.db import connection
        try:
            with connection.cursor() as c:
                c.execute("SELECT doc_id, doc_type, title, equipment_type, equipment_id, content, author, department FROM operational_documents")
                rows = c.fetchall()
        except Exception:
            logger.warning("operational_documents table not found — skipping")
            return 0

        if not rows:
            return 0

        docs = []
        for row in rows:
            doc_id, doc_type, title, eq_type, eq_id, content, author, department = row
            docs.append(RAGDocument(
                id=f"opdoc_{doc_id}",
                content=f"{title} | Type: {doc_type} | {content}",
                metadata={
                    "doc_id": doc_id or "",
                    "doc_type": doc_type or "",
                    "title": title or "",
                    "equipment_type": eq_type or "",
                    "equipment_id": eq_id or "",
                    "author": author or "",
                    "department": department or "",
                },
            ))

        self.vector_store.add_documents(OPERATIONAL_DOCS_COLLECTION, docs)
        return len(docs)

    def _index_shift_logs(self) -> int:
        """Index shift handover logs from raw SQL table."""
        from django.db import connection
        try:
            with connection.cursor() as c:
                c.execute("SELECT log_id, shift_date, shift_name, supervisor, events, pending_actions, notes FROM shift_logs")
                rows = c.fetchall()
        except Exception:
            logger.warning("shift_logs table not found — skipping")
            return 0

        if not rows:
            return 0

        docs = []
        for row in rows:
            log_id, shift_date, shift_name, supervisor, events, pending_actions, notes = row
            content = (
                f"Shift Log {shift_date} {shift_name} | "
                f"Supervisor: {supervisor} | "
                f"Events: {events} | "
                f"Pending: {pending_actions} | "
                f"Notes: {notes}"
            )
            docs.append(RAGDocument(
                id=f"shift_{log_id}",
                content=content,
                metadata={
                    "log_id": log_id or "",
                    "shift_date": shift_date or "",
                    "shift_name": shift_name or "",
                    "supervisor": supervisor or "",
                },
            ))

        self.vector_store.add_documents(SHIFT_LOGS_COLLECTION, docs)
        return len(docs)

    def _index_work_orders(self) -> int:
        """Index work orders from raw SQL table."""
        from django.db import connection
        try:
            with connection.cursor() as c:
                c.execute("""
                    SELECT wo_id, equipment_id, equipment_name, equipment_type,
                           work_type, priority, status, description, assigned_to, vendor, notes
                    FROM work_orders
                """)
                rows = c.fetchall()
        except Exception:
            logger.warning("work_orders table not found — skipping")
            return 0

        if not rows:
            return 0

        docs = []
        for row in rows:
            wo_id, eq_id, eq_name, eq_type, work_type, priority, status, desc, assigned, vendor, notes = row
            content = (
                f"Work Order {wo_id} | Equipment: {eq_name} ({eq_id}) | "
                f"Type: {work_type} | Priority: {priority} | Status: {status} | "
                f"Description: {desc} | Assigned: {assigned}"
            )
            if vendor:
                content += f" | Vendor: {vendor}"
            if notes:
                content += f" | Notes: {notes}"

            docs.append(RAGDocument(
                id=f"wo_{wo_id}",
                content=content,
                metadata={
                    "wo_id": wo_id or "",
                    "equipment_id": eq_id or "",
                    "equipment_name": eq_name or "",
                    "work_type": work_type or "",
                    "priority": priority or "",
                    "status": status or "",
                },
            ))

        self.vector_store.add_documents(WORK_ORDERS_COLLECTION, docs)
        return len(docs)

    def query_energy_sql(self, equipment_id: str = None, days: int = 30) -> list[dict]:
        """Direct SQL query for energy time-series data (not vector search).

        Accepts meter_id (e.g. "EM-INC-01") or name fragment (e.g. "Main Incomer").
        When nothing is given, returns data for the main incomer.
        """
        from django.db import connection
        try:
            with connection.cursor() as c:
                resolved_meter = None

                if equipment_id:
                    # Try exact meter_id match first
                    c.execute("""
                        SELECT meter_id FROM energy_readings
                        WHERE meter_id = %s LIMIT 1
                    """, [equipment_id])
                    row = c.fetchone()
                    if row:
                        resolved_meter = row[0]
                    else:
                        # Try fuzzy match on meter_name or meter_id
                        c.execute("""
                            SELECT DISTINCT meter_id FROM energy_readings
                            WHERE meter_name LIKE %s OR meter_id LIKE %s
                            LIMIT 1
                        """, [f"%{equipment_id}%", f"%{equipment_id}%"])
                        row = c.fetchone()
                        if row:
                            resolved_meter = row[0]

                if not resolved_meter:
                    # Pick the main incomer meter (highest cumulative energy)
                    c.execute("""
                        SELECT meter_id FROM energy_readings
                        ORDER BY energy_kwh_cumulative DESC
                        LIMIT 1
                    """)
                    row = c.fetchone()
                    if not row:
                        return []
                    resolved_meter = row[0]

                c.execute("""
                    SELECT meter_id, meter_name, timestamp, power_kw, power_factor, voltage_avg
                    FROM energy_readings
                    WHERE meter_id = %s
                    ORDER BY timestamp ASC
                    LIMIT 200
                """, [resolved_meter])
                rows = c.fetchall()
                return [
                    {"meter_id": r[0], "meter_name": r[1], "timestamp": r[2],
                     "power_kw": r[3], "power_factor": r[4], "voltage_avg": r[5]}
                    for r in rows
                ]
        except Exception as e:
            logger.warning(f"Energy SQL query failed: {e}")
            return []

    def query(
        self,
        question: str,
        n_results: int = 5,
        include_alerts: bool = True,
        include_maintenance: bool = True,
        include_documents: bool = True,
        include_work_orders: bool = True,
        include_shift_logs: bool = False,
        use_hybrid: bool = True,
    ) -> RAGResponse:
        """
        Query the RAG pipeline with a natural language question.

        Searches across all indexed collections:
        - Equipment (always)
        - Alerts (default on)
        - Maintenance records (default on)
        - Operational documents / SOPs (default on)
        - Work orders (default on)
        - Shift logs (on demand)

        Args:
            use_hybrid: If True, use hybrid BM25+vector search for better
                        accuracy on technical queries (default: True)

        Returns retrieved context and LLM-generated response.
        """
        logger.info(f"RAG Query: {question} (hybrid={use_hybrid})")

        # Choose search method
        search_fn = self.vector_store.search_hybrid if use_hybrid else self.vector_store.search

        # Search equipment (always use hybrid for equipment - critical for IDs like TX-001)
        equipment_results = search_fn(
            EQUIPMENT_COLLECTION,
            question,
            n_results=n_results,
        )

        # Search alerts
        alert_results = []
        if include_alerts:
            alert_results = search_fn(
                ALERTS_COLLECTION,
                question,
                n_results=3,
            )

        # Search maintenance
        maint_results = []
        if include_maintenance:
            maint_results = search_fn(
                MAINTENANCE_COLLECTION,
                question,
                n_results=3,
            )

        # Search operational documents (SOPs, inspection reports, etc.)
        doc_results = []
        if include_documents:
            try:
                doc_results = search_fn(
                    OPERATIONAL_DOCS_COLLECTION,
                    question,
                    n_results=3,
                )
            except Exception:
                pass  # Collection may not exist yet

        # Search work orders
        wo_results = []
        if include_work_orders:
            try:
                wo_results = search_fn(
                    WORK_ORDERS_COLLECTION,
                    question,
                    n_results=3,
                )
            except Exception:
                pass

        # Search shift logs
        shift_results = []
        if include_shift_logs:
            try:
                shift_results = search_fn(
                    SHIFT_LOGS_COLLECTION,
                    question,
                    n_results=3,
                )
            except Exception:
                pass

        # Combine all results
        all_results = (
            equipment_results + alert_results + maint_results +
            doc_results + wo_results + shift_results
        )

        # Build context for LLM
        context_parts = []
        sources = []

        for result in all_results:
            context_parts.append(f"- {result.content}")
            sources.append({
                "id": result.id,
                "type": result.metadata.get("equipment_type", "unknown"),
                "name": result.metadata.get("name", result.metadata.get("equipment_name", "")),
                "score": result.score,
            })

        context = "\n".join(context_parts)

        # Generate response with LLM
        system_prompt = """You are Command Center, an industrial operations voice assistant.
You help operators with equipment monitoring, production lines, quality control, supply chain, and safety.
IMPORTANT: Keep responses to 2-3 sentences maximum. This will be spoken aloud by a voice system.
Be specific — cite equipment IDs, metric values, and current statuses when available.
Use natural spoken language, not written prose. If you lack data to answer, say so briefly."""

        prompt = f"""Based on the following industrial data:
{context}

Answer the operator's question: {question}

Provide a concise, data-rich spoken response."""

        llm_response = self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=256,
        )

        return RAGResponse(
            query=question,
            retrieved_docs=all_results,
            context=context,
            llm_response=llm_response,
            sources=sources,
        )

    def get_stats(self) -> dict:
        """Get statistics about indexed data."""
        return {
            "equipment_count": self.vector_store.get_collection_count(EQUIPMENT_COLLECTION),
            "alerts_count": self.vector_store.get_collection_count(ALERTS_COLLECTION),
            "maintenance_count": self.vector_store.get_collection_count(MAINTENANCE_COLLECTION),
            "operational_docs_count": self.vector_store.get_collection_count(OPERATIONAL_DOCS_COLLECTION),
            "shift_logs_count": self.vector_store.get_collection_count(SHIFT_LOGS_COLLECTION),
            "work_orders_count": self.vector_store.get_collection_count(WORK_ORDERS_COLLECTION),
            "llm_available": self.llm.is_available(),
            "llm_model": self.llm.model,
            "llm_model_fast": self.llm_fast.model,
            "llm_model_quality": self.llm_quality.model,
        }

    def clear_index(self):
        """Clear all indexed data."""
        for collection in [
            EQUIPMENT_COLLECTION, ALERTS_COLLECTION, MAINTENANCE_COLLECTION,
            OPERATIONAL_DOCS_COLLECTION, SHIFT_LOGS_COLLECTION, WORK_ORDERS_COLLECTION,
        ]:
            self.vector_store.delete_collection(collection)
        self._indexed = False
        logger.info("Cleared all indexed data")


# ============================================================
# Singleton Instance
# ============================================================

_rag_pipeline: Optional[IndustrialRAGPipeline] = None


def get_rag_pipeline() -> IndustrialRAGPipeline:
    """Get or create the RAG pipeline singleton."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = IndustrialRAGPipeline()
    return _rag_pipeline
