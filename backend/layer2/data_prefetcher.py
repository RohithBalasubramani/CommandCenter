"""
Data Pre-Fetcher — fast parallel ChromaDB scan before widget selection.

Gives Stage 3 (widget selector) a summary of what data actually exists
for the entities mentioned in the user's query, so the LLM can make
informed decisions about which widgets to build.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# Collections to probe and what metadata to extract
_PROBE_COLLECTIONS = {
    "industrial_equipment": {
        "fields": ["equipment_type", "status", "health_score", "location", "criticality", "name"],
        "label": "Equipment",
    },
    "industrial_alerts": {
        "fields": ["severity", "equipment_name"],
        "label": "Alerts",
    },
    "maintenance_records": {
        "fields": ["maintenance_type", "shift_date", "equipment_name"],
        "label": "Maintenance",
    },
}


class DataPrefetcher:
    """Parallel ChromaDB scanner that produces a compact entity data summary."""

    def __init__(self):
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            from layer2.rag_pipeline import get_rag_pipeline
            self._pipeline = get_rag_pipeline()
        return self._pipeline

    def prefetch(self, intent) -> str:
        """
        Search ChromaDB collections for mentioned entities and return a
        compact text summary suitable for prompt injection (~300-500 tokens).
        """
        entities = getattr(intent, "entities", {}) or {}
        devices = entities.get("devices", [])

        if not devices:
            # No specific entities — do a general probe
            return self._general_probe(intent)

        results: dict[str, dict] = {}
        vs = self.pipeline.vector_store

        # Parallel search: one task per (device × collection)
        tasks = []
        with ThreadPoolExecutor(max_workers=6) as pool:
            for device in devices[:4]:  # cap at 4 entities
                for coll_name, coll_info in _PROBE_COLLECTIONS.items():
                    fut = pool.submit(self._search_one, vs, coll_name, device)
                    tasks.append((fut, device, coll_name, coll_info))

            for fut, device, coll_name, coll_info in tasks:
                try:
                    search_results = fut.result(timeout=2)
                    if device not in results:
                        results[device] = {}
                    results[device][coll_info["label"]] = self._summarize(
                        search_results, coll_info["fields"], coll_info["label"]
                    )
                except Exception as e:
                    logger.debug(f"Pre-fetch failed for {device}/{coll_name}: {e}")

        return self._format(results)

    def _general_probe(self, intent) -> str:
        """When no entities mentioned, summarize what's in the primary domain."""
        query = getattr(intent, "raw_text", "") or ""
        if not query:
            return "No specific entities mentioned."

        vs = self.pipeline.vector_store
        try:
            eq_results = vs.search("industrial_equipment", query, n_results=3)
            if eq_results:
                names = [r.metadata.get("name", "?") for r in eq_results]
                return f"Related equipment found: {', '.join(names)}. Query the data collection stage for details."
        except Exception:
            pass
        return "No specific entities mentioned."

    def _search_one(self, vs, collection: str, entity: str):
        """Single ChromaDB search — runs in thread pool."""
        return vs.search(collection, entity, n_results=3)

    def _summarize(self, search_results, fields: list, label: str) -> str:
        """Extract metadata fields from search results into a compact string."""
        if not search_results:
            return "none"

        if label == "Equipment":
            meta = search_results[0].metadata
            parts = []
            if meta.get("equipment_type"):
                parts.append(meta["equipment_type"])
            if meta.get("health_score"):
                parts.append(f"health={meta['health_score']}%")
            if meta.get("status"):
                parts.append(f"status={meta['status']}")
            if meta.get("criticality"):
                parts.append(f"criticality={meta['criticality']}")
            if meta.get("location"):
                parts.append(f"location={meta['location']}")
            return ", ".join(parts) if parts else "found"

        elif label == "Alerts":
            count = len(search_results)
            severities = [r.metadata.get("severity", "info") for r in search_results]
            highest = "critical" if "critical" in severities else (
                "warning" if "warning" in severities else "info"
            )
            msg = search_results[0].content[:60] if search_results else ""
            return f"{count} found (highest: {highest}) — \"{msg}\""

        elif label == "Maintenance":
            count = len(search_results)
            types = list({r.metadata.get("maintenance_type", "") for r in search_results if r.metadata.get("maintenance_type")})
            date = search_results[0].metadata.get("shift_date", "")
            parts = [f"{count} records"]
            if types:
                parts.append(f"types: {', '.join(types[:2])}")
            if date:
                parts.append(f"latest: {date}")
            return ", ".join(parts)

        return f"{len(search_results)} results"

    def _format(self, results: dict[str, dict]) -> str:
        """Format all entity results into prompt-ready text."""
        if not results:
            return "No specific entities mentioned."

        lines = []
        for entity, collections in results.items():
            lines.append(f'Entity "{entity}":')
            for label, summary in collections.items():
                lines.append(f"  {label}: {summary}")
        return "\n".join(lines)
