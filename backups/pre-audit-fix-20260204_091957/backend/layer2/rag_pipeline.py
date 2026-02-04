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
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.warning(f"Failed to delete collection {collection_name}: {e}")

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
    ):
        self.base_url = base_url
        self.model = model

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate response from LLM."""
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests not installed. Run: pip install requests")

        url = f"{self.base_url}/api/generate"

        # Prepend /no_think for qwen3 models to disable thinking mode
        actual_prompt = prompt
        if "qwen3" in self.model.lower():
            actual_prompt = "/no_think\n" + prompt

        payload = {
            "model": self.model,
            "prompt": actual_prompt,
            "stream": False,
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
            return clean if clean else raw.strip()
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
    ) -> dict | None:
        """Generate structured JSON response from LLM.

        Uses Ollama's JSON format mode and validates the output.
        Returns parsed dict on success, None on failure.
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests not installed. Run: pip install requests")

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
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
                return json.loads(raw)
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
        self.llm = OllamaLLMService()  # legacy single-model (backward compat)
        # Pipeline v2: dual-model LLM instances
        self.llm_fast = OllamaLLMService(model=OLLAMA_MODEL_FAST)
        self.llm_quality = OllamaLLMService(model=OLLAMA_MODEL_QUALITY)
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
        """Direct SQL query for energy time-series data (not vector search)."""
        from django.db import connection
        try:
            with connection.cursor() as c:
                if equipment_id:
                    c.execute("""
                        SELECT meter_id, meter_name, timestamp, power_kw, power_factor, voltage_avg
                        FROM energy_readings
                        WHERE meter_id = %s
                        ORDER BY timestamp DESC
                        LIMIT 100
                    """, [equipment_id])
                else:
                    c.execute("""
                        SELECT meter_id, meter_name, timestamp, power_kw, power_factor, voltage_avg
                        FROM energy_readings
                        ORDER BY timestamp DESC
                        LIMIT 100
                    """)
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

        Returns retrieved context and LLM-generated response.
        """
        logger.info(f"RAG Query: {question}")

        # Search equipment
        equipment_results = self.vector_store.search(
            EQUIPMENT_COLLECTION,
            question,
            n_results=n_results,
        )

        # Search alerts
        alert_results = []
        if include_alerts:
            alert_results = self.vector_store.search(
                ALERTS_COLLECTION,
                question,
                n_results=3,
            )

        # Search maintenance
        maint_results = []
        if include_maintenance:
            maint_results = self.vector_store.search(
                MAINTENANCE_COLLECTION,
                question,
                n_results=3,
            )

        # Search operational documents (SOPs, inspection reports, etc.)
        doc_results = []
        if include_documents:
            try:
                doc_results = self.vector_store.search(
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
                wo_results = self.vector_store.search(
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
                shift_results = self.vector_store.search(
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
