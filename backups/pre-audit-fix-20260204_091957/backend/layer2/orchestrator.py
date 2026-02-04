"""
Layer 2 Orchestrator — The Brain of Command Center

This module handles:
- 2A: Intent parsing from transcripts
- 2B: Parallel RAG query coordination
- 2C: Response generation + Layout decisions

The orchestrator receives transcripts from Layer 1, processes them,
and returns:
1. voice_response: Text for Layer 1 to speak (TTS)
2. layout_json: Widget commands for Layer 3 (Blob)
3. context_update: Updated context for future queries
"""

import os
import re
import time
import uuid
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import RAG pipeline
from layer2.rag_pipeline import get_rag_pipeline, RAGResponse
from layer2.fixture_selector import FixtureSelector

# Pipeline v2 imports
from layer2.intent_parser import IntentParser, ParsedIntent
from layer2.widget_selector import WidgetSelector, WidgetPlan
from layer2.data_collector import SchemaDataCollector
from layer2.widget_catalog import CATALOG_BY_SCENARIO

logger = logging.getLogger(__name__)

# Pipeline v2 flag — set PIPELINE_V2=1 env var to enable
PIPELINE_V2 = os.environ.get("PIPELINE_V2", "1") == "1"

# Domain keywords for intent detection
DOMAIN_KEYWORDS = {
    "industrial": [
        # General equipment
        "pump", "pumps", "motor", "motors", "temperature", "temp", "pressure",
        "voltage", "current", "power", "energy", "device", "devices", "machine",
        "machines", "sensor", "sensors", "meter", "meters", "grid", "production",
        "line", "manufacturing", "oee", "efficiency", "running", "status",
        "operational", "maintenance", "equipment", "plc", "scada",
        # Electrical equipment
        "transformer", "transformers", "generator", "generators", "diesel",
        "panel", "panels", "switchgear", "ups", "electrical",
        # HVAC equipment
        "chiller", "chillers", "ahu", "ahus", "cooling", "tower", "towers",
        "hvac", "air handling", "compressor", "compressors",
        # Energy / power quality
        "energy meter", "consumption", "load", "kva", "kw", "kwh",
        "harmonic", "harmonics", "thd", "power factor", "sag", "swell",
        "dip", "dips", "peak", "demand", "tariff", "cost",
        # Water / utilities
        "water", "nitrogen", "compressed air", "scrubber", "di plant",
        # Specific asset names from the plant
        "ht-", "amf", "cell lt", "main panel", "feeder",
    ],
    "supply": [
        "inventory", "stock", "supplier", "vendor", "purchase",
        "procurement", "rfq", "po", "shipment", "delivery", "warehouse", "logistics",
        "material", "materials", "parts", "components", "lead time", "shortage",
        "supply chain", "sourcing", "import", "export", "container", "freight",
    ],
    "people": [
        "employee", "employees", "worker", "workers", "staff", "team", "shift",
        "shifts", "schedule", "attendance", "leave", "vacation", "hr", "hiring",
        "training", "safety", "overtime", "workforce", "headcount", "contractor",
        "technician", "operator", "supervisor", "engineer",
    ],
    "tasks": [
        "task", "tasks", "project", "projects", "milestone", "deadline", "due",
        "assignment", "todo", "work order", "ticket", "issue", "priority",
        "progress", "complete", "pending", "overdue", "backlog",
    ],
    "alerts": [
        "alert", "alerts", "alarm", "alarms", "warning", "warnings", "error",
        "errors", "critical", "urgent", "notification", "issue", "problem",
        "fault", "failure", "anomaly", "threshold", "breach", "trip", "tripped",
    ],
}

# Height hints per scenario — controls row-span on the frontend grid.
# short=1 row (~70px), medium=2 rows (~150px), tall=3 rows (~230px), x-tall=4 rows (~310px)
SCENARIO_HEIGHT_HINTS = {
    "kpi":                "short",
    "alerts":             "medium",
    "comparison":         "medium",
    "trend":              "tall",
    "trend-multi-line":   "tall",
    "trends-cumulative":  "tall",
    "distribution":       "tall",
    "composition":        "tall",
    "category-bar":       "tall",
    "timeline":           "tall",
    "peopleview":         "tall",
    "peoplehexgrid":      "tall",
    "peoplenetwork":      "tall",
    "supplychainglobe":   "tall",
    "flow-sankey":        "x-tall",
    "matrix-heatmap":     "x-tall",
    "eventlogstream":     "x-tall",
    "edgedevicepanel":    "x-tall",
    "chatstream":         "x-tall",
}

# Filler templates for different scenarios
FILLER_TEMPLATES = {
    "greeting": [
        "Hello!",
        "Hey there!",
        "Hi, how can I help?",
    ],
    "checking": [
        "Let me check that for you.",
        "Checking the latest data now.",
        "One moment while I look that up.",
        "Pulling up the equipment data.",
        "Checking the monitoring systems.",
    ],
    "processing": [
        "Processing your request.",
        "Analyzing the data.",
        "Running the query now.",
        "Running that through the operations pipeline.",
        "Processing the production data.",
    ],
    "fetching": [
        "Fetching the latest metrics.",
        "Getting the current status.",
        "Retrieving the information.",
        "Looking up the latest readings.",
        "Retrieving the maintenance records.",
    ],
}

# Casual conversation patterns (not domain queries, but still in-scope interaction)
CONVERSATION_PATTERNS = [
    r"\b(how are you|how're you|how do you do|how have you been)\b",
    r"\b(what's up|whats up|sup)\b",
    r"\b(thank you|thanks|thank|appreciate)\b",
    r"\b(what can you do|what do you do|what are you|who are you|what's your name|whats your name)\b",
    r"\b(help me|can you help)\b",
    r"\b(you're welcome|no problem|never mind|nevermind|forget it)\b",
    r"\b(bye|goodbye|good night|see you|take care)\b",
    r"\b(nice|awesome|great|cool|ok|okay|got it|understood|sure)\b",
    r"\b(tell me a joke|are you a robot|are you real|are you ai)\b",
]

OUT_OF_SCOPE_MESSAGE = (
    "That's outside what I can help with. "
    "I'm your industrial operations assistant — I can help with "
    "equipment monitoring, alerts, maintenance, supply chain, "
    "workforce management, and task tracking. "
    "What would you like to know?"
)

# Response templates for different intents
RESPONSE_TEMPLATES = {
    "status_query": "Based on the latest data, {summary}",
    "metric_query": "The current {metric_name} is {value} {unit}. {trend_info}",
    "alert_query": "{alert_count} active alerts. {alert_summary}",
    "list_query": "Here's what I found: {items}",
    "action_confirm": "I've {action_description}. {result}",
    "no_data": "I couldn't find data for that query. Would you like me to check something else?",
    "error": "I encountered an issue while processing your request. {error_detail}",
}


@dataclass
class Intent:
    """Parsed intent from user transcript."""
    type: str  # query, action, clarification, greeting, etc.
    domains: list = field(default_factory=list)
    entities: dict = field(default_factory=dict)
    confidence: float = 0.0
    raw_text: str = ""


@dataclass
class RAGResult:
    """Result from a RAG pipeline query."""
    domain: str
    success: bool
    data: dict = field(default_factory=dict)
    error: Optional[str] = None
    execution_time_ms: int = 0


@dataclass
class OrchestratorResponse:
    """Complete response from the orchestrator."""
    voice_response: str  # Text for Layer 1 TTS
    layout_json: dict | None  # Commands for Layer 3 Blob (None = keep current layout)
    context_update: dict  # Updated context for future queries
    intent: Intent = None
    rag_results: list = field(default_factory=list)
    processing_time_ms: int = 0
    filler_text: str = ""  # Filler to speak while processing


class Layer2Orchestrator:
    """
    Main orchestrator for Layer 2.

    Coordinates intent parsing, RAG queries, and response generation.
    """

    def __init__(self, context: dict = None):
        self.context = context or {}
        self.executor = ThreadPoolExecutor(max_workers=5)
        # Pipeline v2 components (lazy-initialized)
        self._intent_parser = None
        self._widget_selector = None
        self._data_collector = None

    def process_transcript(self, transcript: str, session_context: dict = None,
                           user_id: str = "default_user") -> OrchestratorResponse:
        """
        Main entry point: process a transcript and generate response.

        Routes to v2 pipeline (LLM-driven) or v1 (regex) based on PIPELINE_V2 flag.
        """
        if PIPELINE_V2:
            return self._process_transcript_v2(transcript, session_context, user_id)
        return self._process_transcript_v1(transcript, session_context)

    # ══════════════════════════════════════════════════════════════
    # Pipeline v2 — LLM-driven
    # ══════════════════════════════════════════════════════════════

    def _process_transcript_v2(self, transcript: str, session_context: dict = None,
                                user_id: str = "default_user") -> OrchestratorResponse:
        """
        Pipeline v2: LLM intent parsing → data pre-fetch → LLM widget selection
        → schema-driven data collection → LLM fixture selection → height budget
        → 70B voice response → save to user memory.
        """
        start_time = time.time()

        if session_context:
            self.context.update(session_context)

        # Stage 1: LLM Intent Parsing
        if self._intent_parser is None:
            self._intent_parser = IntentParser()
        parsed = self._intent_parser.parse(transcript)

        logger.info(
            f"[v2] Intent: type={parsed.type} domains={parsed.domains} "
            f"primary={parsed.primary_characteristic} method={parsed.parse_method} "
            f"confidence={parsed.confidence:.2f}"
        )

        # Build v1-compatible Intent for backward compat
        intent = Intent(
            type=parsed.type if parsed.type in ("query", "greeting", "conversation", "out_of_scope", "action") else
                  "action" if parsed.type.startswith("action_") else parsed.type,
            domains=parsed.domains,
            entities=parsed.entities,
            confidence=parsed.confidence,
            raw_text=transcript,
        )

        # Short-circuit: out-of-scope
        if parsed.type == "out_of_scope":
            processing_time = int((time.time() - start_time) * 1000)
            return OrchestratorResponse(
                voice_response=OUT_OF_SCOPE_MESSAGE,
                layout_json=None,
                context_update=self._update_context(intent, []),
                intent=intent,
                processing_time_ms=processing_time,
            )

        # Short-circuit: conversation / greeting
        if parsed.type == "conversation":
            voice_response = self._generate_conversation_response(intent)
            processing_time = int((time.time() - start_time) * 1000)
            return OrchestratorResponse(
                voice_response=voice_response,
                layout_json=None,
                context_update=self._update_context(intent, []),
                intent=intent,
                processing_time_ms=processing_time,
            )

        if parsed.type == "greeting":
            voice_response = self._generate_greeting()
            processing_time = int((time.time() - start_time) * 1000)
            return OrchestratorResponse(
                voice_response=voice_response,
                layout_json=None,
                context_update=self._update_context(intent, []),
                intent=intent,
                processing_time_ms=processing_time,
            )

        # Short-circuit: action types → execute action, no dashboard
        if parsed.type.startswith("action_"):
            return self._handle_action_v2(parsed, intent, start_time)

        # ── Query path: build dashboard ──

        filler = self._generate_filler(intent)

        # Stage 2.5: Data Pre-Fetch — tell the LLM what data exists for mentioned entities
        from layer2.data_prefetcher import DataPrefetcher
        try:
            data_summary = DataPrefetcher().prefetch(parsed)
            logger.info(f"[v2] Pre-fetch: {len(data_summary)} chars of entity context")
        except Exception as e:
            logger.warning(f"Pre-fetch failed: {e}")
            data_summary = ""

        # Get user memory context
        from layer2.user_memory import UserMemoryManager
        memory_mgr = UserMemoryManager()
        try:
            user_context = memory_mgr.format_for_prompt(user_id)
        except Exception as e:
            logger.warning(f"User memory read failed: {e}")
            user_context = ""

        # Stage 3: LLM Widget Selection (now with data + user context)
        if self._widget_selector is None:
            self._widget_selector = WidgetSelector()
        widget_plan = self._widget_selector.select(parsed, data_summary, user_context)

        logger.info(
            f"[v2] Widget plan: {len(widget_plan.widgets)} widgets, "
            f"height={widget_plan.total_height_units}/{18}, "
            f"method={widget_plan.select_method}"
        )

        # Stage 3: Schema-Driven Data Collection (parallel)
        if self._data_collector is None:
            self._data_collector = SchemaDataCollector()
        widget_data = self._data_collector.collect_all(widget_plan.widgets, transcript)

        # Inject _query_context into data_override so fixture selection can match
        # on semantic content (not just equipment names from RAG)
        for i, w in enumerate(widget_data):
            plan_item = widget_plan.widgets[i] if i < len(widget_plan.widgets) else None
            if plan_item:
                dor = w.get("data_override") or {}
                demo = dor.get("demoData", {})
                # Ensure demo is a dict (some collectors return lists)
                if not isinstance(demo, dict):
                    demo = {"_raw": demo}
                # Build context string from widget plan's query + why + user transcript
                ctx_parts = []
                dr = plan_item.data_request
                if isinstance(dr, dict):
                    for key in ("query", "metric"):
                        val = dr.get(key)
                        if val:
                            ctx_parts.append(val if isinstance(val, str) else " ".join(str(v) for v in val))
                if plan_item.why:
                    ctx_parts.append(plan_item.why)
                ctx_parts.append(transcript)
                ctx_string = " ".join(str(p) for p in ctx_parts).lower()
                demo["_query_context"] = ctx_string
                dor["demoData"] = demo
                # Also set label from context if current label is generic
                label = demo.get("label", "")
                if not label or label == "N/A" or label == "Unknown":
                    metric = dr.get("metric", "") if isinstance(dr, dict) else ""
                    demo["label"] = metric or (plan_item.why[:50] if plan_item.why else "")
                w["data_override"] = dor

        # Stage 4: Fixture Selection (LLM-based with rule-based fallback)
        from layer2.llm_fixture_selector import LLMFixtureSelector
        story = f"{widget_plan.heading} — answering '{transcript[:80]}'"
        llm_fixture_sel = LLMFixtureSelector()
        widget_data = llm_fixture_sel.select_all(widget_data, story, transcript)

        # Inject heightHint from catalog
        for w in widget_data:
            scenario = w["scenario"]
            w["heightHint"] = SCENARIO_HEIGHT_HINTS.get(scenario, "medium")

        # Inject per-widget description ("why" text) from widget plan
        plan_items_by_idx = {i: item for i, item in enumerate(widget_plan.widgets)}
        for i, w in enumerate(widget_data):
            if i in plan_items_by_idx and plan_items_by_idx[i].why:
                w["description"] = plan_items_by_idx[i].why

        # Size-heightHint coherence (same as v1)
        for w in widget_data:
            hint = w.get("heightHint", "medium")
            size = w.get("size", "normal")
            if size == "hero":
                continue
            if hint == "x-tall" and size in ("normal", "compact"):
                w["size"] = "expanded"
            if hint == "tall" and size == "normal" and w["scenario"] in (
                "composition", "category-bar", "flow-sankey", "matrix-heatmap",
                "comparison", "timeline", "eventlogstream",
            ):
                w["size"] = "expanded"

        # Max 1 hero — demote additional heroes to expanded
        hero_seen = False
        for w in widget_data:
            if w.get("size") == "hero":
                if hero_seen:
                    w["size"] = "expanded"
                    logger.info(f"Demoted extra hero '{w['scenario']}' to expanded")
                else:
                    hero_seen = True

        # Row-packing: upsize widgets to fill 12-column grid rows
        widget_data = self._pack_row_gaps(widget_data)

        layout_json = {
            "heading": widget_plan.heading,
            "widgets": widget_data,
            "transitions": {},
        }

        # Stage 5: Layout-Aware Voice Response (70B quality model)
        voice_response = self._generate_voice_response_v2(parsed, layout_json, transcript)

        processing_time = int((time.time() - start_time) * 1000)
        logger.info(
            f"[v2] Complete: {processing_time}ms — {len(widget_data)} widgets — "
            f"'{voice_response[:60]}...'"
        )

        # Save to user memory for future context-aware selections
        try:
            scenarios_used = [w["scenario"] for w in widget_data]
            memory_mgr.record(user_id, transcript, parsed, scenarios_used)
            logger.info(f"[v2] Saved to user memory: user={user_id}, scenarios={scenarios_used}")
        except Exception as e:
            logger.warning(f"User memory save failed: {e}")

        return OrchestratorResponse(
            voice_response=voice_response,
            layout_json=layout_json,
            context_update=self._update_context(intent, []),
            intent=intent,
            processing_time_ms=processing_time,
            filler_text=filler,
        )

    def _pack_row_gaps(self, widgets: list) -> list:
        """Adjust widget sizes to eliminate column gaps in 12-col grid.

        Groups widgets into rows, then upsizes smaller widgets to fill
        any leftover columns. Never downsizes.
        """
        SIZE_COLS = {"hero": 12, "expanded": 6, "normal": 4, "compact": 3}
        UPSIZE = {"compact": "normal", "normal": "expanded"}

        rows: list[list] = []
        current_row: list = []
        current_cols = 0

        for w in widgets:
            cols = SIZE_COLS.get(w.get("size", "normal"), 4)
            if w.get("size") == "hero":
                if current_row:
                    rows.append(current_row)
                rows.append([w])
                current_row = []
                current_cols = 0
                continue
            if current_cols + cols > 12:
                rows.append(current_row)
                current_row = [w]
                current_cols = cols
            else:
                current_row.append(w)
                current_cols += cols

        if current_row:
            rows.append(current_row)

        # Fill gaps in each row by upsizing smallest widgets
        for row in rows:
            if len(row) == 1 and row[0].get("size") == "hero":
                continue
            total = sum(SIZE_COLS.get(w.get("size", "normal"), 4) for w in row)
            gap = 12 - total
            while gap >= 1:
                # Find smallest widget that can be upsized
                candidates = [w for w in row if w.get("size") in UPSIZE]
                if not candidates:
                    break
                candidates.sort(key=lambda x: SIZE_COLS.get(x.get("size", "normal"), 4))
                target = candidates[0]
                old_size = target["size"]
                new_size = UPSIZE[old_size]
                gained = SIZE_COLS[new_size] - SIZE_COLS[old_size]
                if gained > gap:
                    break
                target["size"] = new_size
                gap -= gained
                logger.debug(f"Row-pack: upsized {target['scenario']} {old_size}→{new_size}")

        return [w for row in rows for w in row]

    def _handle_action_v2(self, parsed: ParsedIntent, intent: Intent, start_time: float) -> OrchestratorResponse:
        """Handle action intents via the actions Django app."""
        try:
            from actions.handlers import ActionHandler
            handler = ActionHandler()
            result = handler.execute(parsed)
            voice_response = result.voice_response
        except Exception as e:
            logger.error(f"[v2] Action handler error: {e}")
            voice_response = "Sorry, I couldn't complete that action. Please try again."

        processing_time = int((time.time() - start_time) * 1000)
        return OrchestratorResponse(
            voice_response=voice_response,
            layout_json=None,  # actions don't change the dashboard
            context_update=self._update_context(intent, []),
            intent=intent,
            processing_time_ms=processing_time,
        )

    def _generate_voice_response_v2(self, parsed: ParsedIntent, layout: dict, transcript: str) -> str:
        """Generate voice response using the quality LLM, aware of what dashboard was built."""
        try:
            pipeline = get_rag_pipeline()
            llm = pipeline.llm_quality

            widgets = layout.get("widgets", [])
            widget_summary = ", ".join(
                f"{w['scenario']} ({w.get('why', '')})" for w in widgets[:5]
            )

            # Get RAG context for grounding
            rag_response = pipeline.query(transcript, n_results=3)
            rag_context = rag_response.context[:2000] if rag_response.context else ""

            prompt = f"""You are Command Center, an industrial operations voice assistant.

Dashboard built: "{layout.get('heading', 'Dashboard')}" with {len(widgets)} widgets showing: {widget_summary}

RULES:
1. 2-3 sentences maximum — this will be spoken aloud via TTS.
2. Lead with the direct answer to the question.
3. Cite equipment IDs, metric values, and units from the data when available.
4. Briefly mention what the dashboard shows for context.
5. Never speculate — if data is missing, say so.
6. Use natural spoken language, not written prose.

Context data:
{rag_context}

User question: {transcript}

Response:"""

            system_prompt = (
                "You are Command Center, an industrial operations voice assistant. "
                "Keep responses to 2-3 sentences maximum. Be specific and data-rich."
            )

            response = llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=384,
            )

            if response and not response.startswith("[LLM"):
                return response.strip()

        except Exception as e:
            logger.warning(f"[v2] Voice response generation failed: {e}")

        # Fallback: simple response
        heading = layout.get("heading", "your query")
        n_widgets = len(layout.get("widgets", []))
        return f"Here's what I found for {heading}. I've prepared a dashboard with {n_widgets} widgets showing the relevant data."

    # ══════════════════════════════════════════════════════════════
    # Pipeline v1 — Regex-based (original code, kept as fallback)
    # ══════════════════════════════════════════════════════════════

    def _process_transcript_v1(self, transcript: str, session_context: dict = None) -> OrchestratorResponse:
        """
        Original pipeline: regex intent parsing → flat RAG → if-elif layout → voice.
        Kept as fallback when PIPELINE_V2=0.
        """
        start_time = time.time()

        # Merge session context
        if session_context:
            self.context.update(session_context)

        # 2A: Parse intent
        intent = self._parse_intent(transcript)

        # Short-circuit: out-of-scope and conversation skip RAG entirely.
        # layout_json=None means "keep current dashboard" — don't wipe widgets.
        if intent.type == "out_of_scope":
            processing_time = int((time.time() - start_time) * 1000)
            logger.info(f"Out-of-scope query: '{transcript[:80]}' — returning scope message, keeping layout")
            return OrchestratorResponse(
                voice_response=OUT_OF_SCOPE_MESSAGE,
                layout_json=None,
                context_update=self._update_context(intent, []),
                intent=intent,
                rag_results=[],
                processing_time_ms=processing_time,
                filler_text="",
            )

        if intent.type == "conversation":
            voice_response = self._generate_conversation_response(intent)
            processing_time = int((time.time() - start_time) * 1000)
            logger.info(f"Conversation: '{transcript[:80]}' → '{voice_response[:80]}'")
            return OrchestratorResponse(
                voice_response=voice_response,
                layout_json=None,
                context_update=self._update_context(intent, []),
                intent=intent,
                rag_results=[],
                processing_time_ms=processing_time,
                filler_text="",
            )

        if intent.type == "greeting":
            voice_response = self._generate_greeting()
            processing_time = int((time.time() - start_time) * 1000)
            logger.info(f"Greeting: '{transcript[:80]}' → '{voice_response[:80]}', keeping layout")
            return OrchestratorResponse(
                voice_response=voice_response,
                layout_json=None,
                context_update=self._update_context(intent, []),
                intent=intent,
                rag_results=[],
                processing_time_ms=processing_time,
                filler_text="",
            )

        # Generate filler based on intent
        filler = self._generate_filler(intent)

        # 2B: Execute parallel RAG queries
        rag_results = self._execute_rag_queries(intent, transcript)

        # 2C: Generate response and layout
        voice_response = self._generate_response(intent, rag_results)
        layout_json = self._generate_layout(intent, rag_results)
        context_update = self._update_context(intent, rag_results)

        processing_time = int((time.time() - start_time) * 1000)

        return OrchestratorResponse(
            voice_response=voice_response,
            layout_json=layout_json,
            context_update=context_update,
            intent=intent,
            rag_results=rag_results,
            processing_time_ms=processing_time,
            filler_text=filler,
        )

    def _parse_intent(self, transcript: str) -> Intent:
        """
        2A: Parse intent from transcript.

        Currently uses keyword matching. Future: upgrade to phi-3 or similar.
        """
        text_lower = transcript.lower()

        # Detect intent type
        intent_type = self._detect_intent_type(text_lower)

        # Detect relevant domains
        domains = self._detect_domains(text_lower)

        # Extract entities
        entities = self._extract_entities(text_lower)

        # Drill-down pattern: "tell me more about X" → treat as industrial query
        drill_down_match = re.search(r"tell me more about\s+(.+)", text_lower)
        if drill_down_match and not domains:
            domains = ["industrial"]
            intent_type = "query"

        # Scope guard: if no domain matched and not a greeting,
        # classify as conversation or out_of_scope
        if intent_type != "greeting" and not domains:
            if self._is_conversation(text_lower):
                intent_type = "conversation"
            else:
                intent_type = "out_of_scope"

        # Calculate confidence based on matches
        if intent_type in ("out_of_scope", "conversation"):
            confidence = 0.9
        else:
            confidence = min(1.0, len(domains) * 0.3 + (0.4 if entities else 0.2))

        return Intent(
            type=intent_type,
            domains=domains,
            entities=entities,
            confidence=confidence,
            raw_text=transcript,
        )

    def _is_conversation(self, text: str) -> bool:
        """Check if text matches casual conversation patterns."""
        for pattern in CONVERSATION_PATTERNS:
            if re.search(pattern, text):
                return True
        return False

    def _detect_intent_type(self, text: str) -> str:
        """Detect the type of user intent."""
        # Query patterns
        query_patterns = [
            r"\b(what|what's|whats|how|how's|hows|show|tell|get|check|status|current)\b",
            r"\?$",
        ]

        # Action patterns
        action_patterns = [
            r"\b(start|stop|turn|set|adjust|change|update|create|delete|add|remove)\b",
        ]

        # Greeting patterns
        greeting_patterns = [
            r"\b(hello|hi|hey|good morning|good afternoon|good evening)\b",
        ]

        # Check patterns — order matters.
        # If text matches both greeting AND query/action, prefer query/action
        # (e.g. "Hello, show me transformer status" is a query, not a greeting).
        is_greeting = any(re.search(p, text) for p in greeting_patterns)
        is_action = any(re.search(p, text) for p in action_patterns)
        is_query = any(re.search(p, text) for p in query_patterns)

        if is_action:
            return "action"
        if is_query:
            return "query"
        if is_greeting:
            return "greeting"

        return "query"  # Default to query

    def _detect_domains(self, text: str) -> list:
        """Detect which domains are relevant to the query."""
        detected = []

        for domain, keywords in DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    if domain not in detected:
                        detected.append(domain)
                    break

        # Return empty list when no domain keywords match.
        # _parse_intent() uses empty domains to classify as conversation or out_of_scope.
        return detected

    def _extract_entities(self, text: str) -> dict:
        """Extract named entities from the text."""
        entities = {}

        # Extract numbers
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
        if numbers:
            entities["numbers"] = numbers

        # Extract device references (pump 1, motor 3, transformer 2, etc.)
        device_refs = re.findall(
            r'\b(pump|motor|sensor|device|machine|transformer|generator|chiller|compressor)\s*(\d+)\b',
            text,
        )
        if device_refs:
            entities["devices"] = [f"{d[0]}_{d[1]}" for d in device_refs]

        # Extract time references
        time_refs = re.findall(r'\b(today|yesterday|last\s+\w+|this\s+\w+|past\s+\d+\s+\w+)\b', text)
        if time_refs:
            entities["time"] = time_refs

        return entities

    def _generate_filler(self, intent: Intent) -> str:
        """Generate appropriate filler text based on intent."""
        import random

        if intent.type == "greeting":
            fillers = FILLER_TEMPLATES["greeting"]
            return random.choice(fillers)

        # No filler needed for instant responses
        if intent.type in ("out_of_scope", "conversation"):
            return ""

        if "alerts" in intent.domains:
            fillers = FILLER_TEMPLATES["checking"]
        elif intent.type == "action":
            fillers = FILLER_TEMPLATES["processing"]
        else:
            fillers = FILLER_TEMPLATES["fetching"]

        return random.choice(fillers)

    def _execute_rag_queries(self, intent: Intent, transcript: str) -> list:
        """
        2B: Execute parallel RAG queries for relevant domains.
        """
        results = []

        # Submit queries in parallel
        futures = {}
        for domain in intent.domains:
            future = self.executor.submit(
                self._query_rag_pipeline,
                domain,
                transcript,
                intent.entities,
            )
            futures[future] = domain

        # Collect results
        for future in as_completed(futures, timeout=30.0):
            domain = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(RAGResult(
                    domain=domain,
                    success=False,
                    error=str(e),
                ))

        return results

    def _query_rag_pipeline(self, domain: str, query: str, entities: dict) -> RAGResult:
        """Query a specific RAG pipeline."""
        start_time = time.time()

        try:
            # Use the real RAG pipeline for industrial, alerts, and tasks domains
            if domain in ["industrial", "alerts", "tasks"]:
                rag_pipeline = get_rag_pipeline()

                query_lower = query.lower()
                include_alerts = domain == "alerts" or "alert" in query_lower or "warning" in query_lower or "fault" in query_lower
                include_maintenance = (
                    "maintenance" in query_lower or "repair" in query_lower or
                    "service" in query_lower or "inspection" in query_lower
                )
                include_shift_logs = (
                    "shift" in query_lower or "handover" in query_lower or
                    "supervisor" in query_lower or "last night" in query_lower
                )
                include_work_orders = (
                    domain == "tasks" or "work order" in query_lower or
                    "task" in query_lower or "pending" in query_lower or
                    "overdue" in query_lower
                )

                rag_response = rag_pipeline.query(
                    question=query,
                    n_results=8,
                    include_alerts=include_alerts,
                    include_maintenance=include_maintenance,
                    include_documents=True,
                    include_work_orders=include_work_orders,
                    include_shift_logs=include_shift_logs,
                )

                # Convert RAG response to structured data
                data = self._parse_rag_response(rag_response, domain, query)

                # Fetch energy time-series data if query mentions energy/power/load
                energy_keywords = ["energy", "power", "load", "consumption", "kwh", "kw", "voltage", "trend", "graph", "chart"]
                if any(kw in query_lower for kw in energy_keywords):
                    try:
                        energy_data = rag_pipeline.query_energy_sql(days=30)
                        if energy_data:
                            data["energy_timeseries"] = energy_data
                    except Exception as e:
                        logger.warning(f"Energy SQL query failed: {e}")

            # Keep stub data for domains not yet implemented in RAG
            elif domain == "supply":
                data = self._get_supply_stub_data(query, entities)
            elif domain == "people":
                data = self._get_people_stub_data(query, entities)
            else:
                data = {}

            execution_time = int((time.time() - start_time) * 1000)

            return RAGResult(
                domain=domain,
                success=True,
                data=data,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            logger.error(f"RAG query failed for domain {domain}: {e}")
            execution_time = int((time.time() - start_time) * 1000)

            # Fallback to stub data on error
            if domain == "industrial":
                data = self._get_industrial_stub_data(query, entities)
            elif domain == "alerts":
                data = self._get_alerts_stub_data(query, entities)
            else:
                data = {}

            return RAGResult(
                domain=domain,
                success=True,  # Still return success with fallback data
                data=data,
                execution_time_ms=execution_time,
            )

    def _parse_rag_response(self, rag_response: RAGResponse, domain: str, query: str = "") -> dict:
        """
        Parse RAG response into structured data for the orchestrator.

        Categorizes all retrieved docs by type so _generate_layout can pick
        the right widget for each data category.
        """
        data = {
            "llm_response": rag_response.llm_response,
            "sources": rag_response.sources,
            "summary": rag_response.llm_response,
        }

        # Categorize all retrieved docs by their source collection
        equipment_list = []
        alert_list = []
        maintenance_list = []
        work_order_list = []
        shift_log_list = []
        operational_doc_list = []

        for doc in rag_response.retrieved_docs:
            doc_id = doc.id or ""
            doc_type = doc.metadata.get("equipment_type", "")

            if doc_id.startswith("wo_"):
                # Work order document
                work_order_list.append({
                    "id": doc.metadata.get("wo_id", doc_id),
                    "equipment_id": doc.metadata.get("equipment_id", ""),
                    "equipment_name": doc.metadata.get("equipment_name", ""),
                    "work_type": doc.metadata.get("work_type", ""),
                    "priority": doc.metadata.get("priority", "medium"),
                    "status": doc.metadata.get("status", "open"),
                    "content": doc.content,
                    "relevance_score": doc.score,
                })
            elif doc_id.startswith("shift_"):
                # Shift log document
                shift_log_list.append({
                    "id": doc.metadata.get("log_id", doc_id),
                    "shift_date": doc.metadata.get("shift_date", ""),
                    "shift_name": doc.metadata.get("shift_name", ""),
                    "supervisor": doc.metadata.get("supervisor", ""),
                    "content": doc.content,
                    "relevance_score": doc.score,
                })
            elif doc_id.startswith("opdoc_"):
                # Operational document (SOP, inspection, etc.)
                operational_doc_list.append({
                    "id": doc.metadata.get("doc_id", doc_id),
                    "doc_type": doc.metadata.get("doc_type", ""),
                    "title": doc.metadata.get("title", ""),
                    "equipment_type": doc.metadata.get("equipment_type", ""),
                    "content": doc.content,
                    "relevance_score": doc.score,
                })
            elif doc_id.startswith("maint_"):
                # Maintenance record
                maintenance_list.append({
                    "id": doc_id,
                    "equipment_id": doc.metadata.get("equipment_id", ""),
                    "equipment_name": doc.metadata.get("equipment_name", ""),
                    "maintenance_type": doc.metadata.get("maintenance_type", ""),
                    "content": doc.content,
                    "relevance_score": doc.score,
                })
            elif doc_id.startswith("alert_"):
                # Alert document
                alert_list.append({
                    "id": doc_id,
                    "severity": doc.metadata.get("severity", "info"),
                    "source": doc.metadata.get("equipment_name", "Unknown"),
                    "equipment_id": doc.metadata.get("equipment_id", ""),
                    "message": doc.content,
                    "acknowledged": doc.metadata.get("acknowledged", False),
                    "resolved": doc.metadata.get("resolved", False),
                    "relevance_score": doc.score,
                })
            elif doc_type and doc_type not in ["", "unknown"]:
                # Equipment document
                equipment_list.append({
                    "id": doc.metadata.get("equipment_id", doc.id),
                    "name": doc.metadata.get("name", "Unknown"),
                    "type": doc_type,
                    "status": doc.metadata.get("status", "unknown"),
                    "health": doc.metadata.get("health_score", 0),
                    "location": doc.metadata.get("location", ""),
                    "building": doc.metadata.get("building", ""),
                    "criticality": doc.metadata.get("criticality", "medium"),
                    "content": doc.content,
                    "relevance_score": doc.score,
                })

        if equipment_list:
            data["devices"] = equipment_list
        if alert_list:
            data["alerts"] = alert_list
            data["alert_count"] = len(alert_list)
        if maintenance_list:
            data["maintenance"] = maintenance_list
        if work_order_list:
            data["work_orders"] = work_order_list
        if shift_log_list:
            data["shift_logs"] = shift_log_list
        if operational_doc_list:
            data["operational_docs"] = operational_doc_list

        logger.info(
            f"RAG parsed: {len(equipment_list)} equipment, {len(alert_list)} alerts, "
            f"{len(maintenance_list)} maintenance, {len(work_order_list)} work orders, "
            f"{len(shift_log_list)} shift logs, {len(operational_doc_list)} op docs"
        )

        return data

    def _get_industrial_stub_data(self, query: str, entities: dict) -> dict:
        """Stub data for industrial domain."""
        return {
            "metrics": [
                {"name": "grid_voltage", "value": 238.4, "unit": "V", "status": "normal"},
                {"name": "total_power", "value": 1247.8, "unit": "kW", "status": "normal"},
            ],
            "devices": [
                {"id": "pump_1", "name": "Pump 1", "status": "running", "health": 95},
                {"id": "pump_2", "name": "Pump 2", "status": "running", "health": 92},
                {"id": "pump_3", "name": "Pump 3", "status": "warning", "health": 78},
                {"id": "pump_4", "name": "Pump 4", "status": "running", "health": 88},
            ],
            "summary": "3 of 4 pumps running normally. Pump 3 showing elevated temperature.",
        }

    def _get_alerts_stub_data(self, query: str, entities: dict) -> dict:
        """Stub data for alerts domain."""
        return {
            "alerts": [
                {
                    "id": "alert_1",
                    "severity": "warning",
                    "source": "pump_3",
                    "message": "Temperature elevated to 78°C (threshold: 75°C)",
                    "timestamp": "2026-01-28T10:30:00Z",
                    "acknowledged": False,
                },
            ],
            "summary": "1 active warning alert for Pump 3 temperature.",
            "count": 1,
        }

    def _get_supply_stub_data(self, query: str, entities: dict) -> dict:
        """Stub data for supply domain."""
        return {
            "inventory": [
                {"item": "Bearings SKF-6205", "quantity": 24, "reorder_point": 20, "status": "ok"},
                {"item": "Seals CR-12345", "quantity": 8, "reorder_point": 15, "status": "low"},
            ],
            "pending_orders": 3,
            "summary": "8 items below reorder point. 3 purchase orders pending.",
        }

    def _get_people_stub_data(self, query: str, entities: dict) -> dict:
        """Stub data for people domain."""
        return {
            "on_shift": 45,
            "absent": 3,
            "upcoming_shifts": [
                {"shift": "Evening", "start": "14:00", "staff": 42},
            ],
            "summary": "45 staff on current shift. 3 absences today.",
        }

    def _get_tasks_stub_data(self, query: str, entities: dict) -> dict:
        """Stub data for tasks domain."""
        return {
            "pending": 12,
            "in_progress": 5,
            "due_today": 3,
            "overdue": 1,
            "summary": "12 pending tasks, 3 due today, 1 overdue.",
        }

    def _generate_response(self, intent: Intent, rag_results: list) -> str:
        """
        2C: Generate natural language response for Layer 1 TTS.

        Uses the LLM response from RAG pipeline when available.
        """
        if intent.type == "greeting":
            # Check context for time-appropriate greeting
            return self._generate_greeting()

        if not rag_results:
            return RESPONSE_TEMPLATES["no_data"]

        # Check for LLM-generated responses first (from RAG pipeline)
        for result in rag_results:
            if result.success and "llm_response" in result.data:
                llm_response = result.data["llm_response"]
                # Check if LLM response is valid (not an error message)
                if llm_response and not llm_response.startswith("[LLM"):
                    logger.info(f"Using LLM response from RAG for domain: {result.domain}")
                    return llm_response

        # Fallback: combine summaries from all domains
        response_parts = []

        for result in rag_results:
            if not result.success:
                continue

            if "summary" in result.data:
                response_parts.append(result.data["summary"])

        if response_parts:
            return " ".join(response_parts)

        return RESPONSE_TEMPLATES["no_data"]

    def _generate_greeting(self) -> str:
        """Generate context-appropriate greeting."""
        import datetime

        hour = datetime.datetime.now().hour

        if hour < 12:
            greeting = "Good morning!"
        elif hour < 17:
            greeting = "Good afternoon!"
        else:
            greeting = "Good evening!"

        # Add proactive question
        return f"{greeting} How can I help you with operations today?"

    def _generate_conversation_response(self, intent: Intent) -> str:
        """Generate a natural response for casual conversation (no RAG needed)."""
        text = intent.raw_text.lower()

        if re.search(r"\b(thank|thanks|appreciate)\b", text):
            return "You're welcome! Let me know if you need anything else about operations."

        if re.search(r"\b(how are you|how're you|how do you do|how have you been)\b", text):
            return "I'm running well, thank you! How can I help with operations today?"

        if re.search(r"\b(what can you do|what do you do|help me|can you help)\b", text):
            return (
                "I can help you with equipment monitoring, alert management, "
                "maintenance tracking, supply chain status, workforce management, "
                "and task tracking. Just ask me anything about your operations!"
            )

        if re.search(r"\b(who are you|what are you|your name|are you a robot|are you ai|are you real)\b", text):
            return "I'm your Command Center operations assistant. I help monitor and manage industrial operations."

        if re.search(r"\b(bye|goodbye|good night|see you|take care)\b", text):
            return "Talk to you later! I'll be here if you need anything."

        if re.search(r"\b(ok|okay|got it|understood|sure|nice|awesome|great|cool)\b", text):
            return "Sounds good. Let me know if you need anything."

        if re.search(r"\b(never mind|nevermind|forget it|no problem|you're welcome)\b", text):
            return "No worries. I'm here whenever you need me."

        return "I'm here to help with operations. What would you like to know?"

    # ================================================================
    # Widget Data Formatters — structure RAG data into widget-expected shapes
    # ================================================================

    def _format_kpi(self, label: str, value, unit: str = "", state: str = "normal", variant: str = "kpi_live-standard") -> dict:
        """Format data for KPI widget. Matches fixtureData.ts kpi variants."""
        return {
            "scenario": "kpi",
            "fixture": variant,
            "data_override": {
                "demoData": {
                    "label": label,
                    "value": str(value),
                    "unit": unit,
                    "state": state,
                }
            },
        }

    def _format_alert(self, alert: dict) -> dict:
        """Format a raw alert dict into the alerts widget expected shape."""
        severity = alert.get("severity", "info")
        # Parse structured fields from the content string
        content = alert.get("message", alert.get("content", ""))
        source = alert.get("source", "Unknown")

        # Extract parameter/value/threshold from content if available
        # Content format: "Alert: {msg} | Equipment: {name} | Severity: {sev} | Parameter: {p} | Value: {v} {u} | Threshold: {t} {u}"
        evidence = {}
        for segment in content.split("|"):
            segment = segment.strip()
            if segment.startswith("Value:"):
                parts = segment.replace("Value:", "").strip().split()
                if parts:
                    evidence["value"] = parts[0]
                    evidence["unit"] = parts[1] if len(parts) > 1 else ""
            elif segment.startswith("Parameter:"):
                evidence["label"] = segment.replace("Parameter:", "").strip()
            elif segment.startswith("Threshold:"):
                evidence["threshold"] = segment.replace("Threshold:", "").strip()

        # Pick fixture variant based on severity
        fixture_map = {
            "critical": "modal-ups-battery-critical",
            "high": "toast-power-factor-critical-low",
            "warning": "banner-energy-peak-threshold-exceeded",
            "medium": "badge-ahu-01-high-temperature",
            "low": "card-dg-02-started-successfully",
            "info": "card-dg-02-started-successfully",
            "success": "card-dg-02-started-successfully",
        }
        fixture = fixture_map.get(severity, "banner-energy-peak-threshold-exceeded")

        # Variant mapping
        variant_map = {
            "critical": "modal",
            "high": "toast",
            "warning": "banner",
            "medium": "badge",
            "low": "card",
            "info": "card",
            "success": "card",
        }

        return {
            "variant": variant_map.get(severity, "banner"),
            "data": {
                "id": alert.get("id", "ALT-000"),
                "title": evidence.get("label", source),
                "message": content.split("|")[0].replace("Alert:", "").strip() if "|" in content else content[:120],
                "severity": severity,
                "category": "Equipment",
                "source": source,
                "state": "acknowledged" if alert.get("acknowledged") else "new",
                "evidence": {
                    "label": evidence.get("label", "Value"),
                    "value": evidence.get("value", "—"),
                    "unit": evidence.get("unit", ""),
                    "trend": "up" if severity in ("critical", "high", "warning") else "stable",
                },
                "threshold": evidence.get("threshold", "N/A"),
                "actions": [
                    {"label": "Acknowledge", "intent": "ack", "type": "primary"},
                    {"label": "View Details", "intent": "open", "type": "secondary"},
                ],
            },
        }

    def _format_trend_from_energy(self, energy_data: list) -> dict:
        """Format energy SQL time-series data for the trend widget."""
        if not energy_data:
            return {}

        # Group by meter and pick first meter with data
        meters = {}
        for row in energy_data:
            mid = row.get("meter_id", "unknown")
            if mid not in meters:
                meters[mid] = {"name": row.get("meter_name", mid), "points": []}
            meters[mid]["points"].append(row)

        # Use first meter for single-line trend
        first_meter = list(meters.values())[0]
        points = sorted(first_meter["points"], key=lambda p: p.get("timestamp", ""))

        time_series = []
        for p in points[-20:]:  # Last 20 data points
            ts = p.get("timestamp", "")
            # Extract time portion (HH:MM)
            time_str = ts[11:16] if len(ts) > 16 else ts[-5:]
            time_series.append({
                "time": time_str,
                "value": round(float(p.get("power_kw", 0)), 1),
            })

        return {
            "demoData": {
                "label": f"{first_meter['name']} Power (kW)",
                "timeRange": "Recent readings",
                "unit": "kW",
                "timeSeries": time_series,
            }
        }

    def _format_trend_from_devices(self, devices: list) -> dict:
        """Format device health scores as a trend-like time series."""
        time_series = []
        for i, dev in enumerate(devices[:20]):
            time_series.append({
                "time": dev.get("name", f"Device {i+1}")[:10],
                "value": float(dev.get("health", 0)),
            })
        return {
            "demoData": {
                "label": "Equipment Health Scores",
                "timeRange": "Current snapshot",
                "unit": "%",
                "timeSeries": time_series,
            }
        }

    def _format_category_bar(self, items: list, title: str, value_key: str = "value",
                              category_key: str = "category", variant: str = "VERTICAL") -> dict:
        """Format data for category-bar widget."""
        data_points = []
        for item in items[:12]:
            data_points.append({
                "category": str(item.get(category_key, item.get("name", "?"))),
                "value": item.get(value_key, 0),
            })
        return {
            "config": {
                "variant": variant,
                "title": title,
                "description": "",
                "dataKeys": ["value"],
                "colors": ["#262626"],
                "layout": "horizontal",
            },
            "data": data_points,
            "enableBrush": False,
        }

    def _format_composition_donut(self, items: list) -> dict:
        """Format data for composition donut/pie widget."""
        return [{"name": item.get("name", "?"), "value": item.get("value", 0)} for item in items[:6]]

    def _format_comparison(self, label: str, unit: str, label_a: str, value_a, label_b: str, value_b) -> dict:
        """Format data for comparison widget (side-by-side)."""
        va = float(value_a) if value_a else 0
        vb = float(value_b) if value_b else 0
        delta = round(va - vb, 2)
        delta_pct = round((delta / vb * 100) if vb else 0, 1)
        return {
            "demoData": {
                "label": label,
                "unit": unit,
                "labelA": label_a,
                "valueA": va,
                "labelB": label_b,
                "valueB": vb,
                "delta": delta,
                "deltaPct": delta_pct,
            }
        }

    def _format_matrix_heatmap(self, devices: list) -> dict:
        """Format device health data as a heatmap (equipment × status metrics)."""
        rows = []
        cells = []
        cols = [
            {"id": "health", "label": "Health %"},
            {"id": "status", "label": "Status Score"},
        ]
        status_scores = {"running": 100, "standby": 70, "maintenance": 50, "warning": 30, "fault": 10, "stopped": 0, "offline": 0}

        for dev in devices[:8]:
            row_id = dev.get("id", "?")
            rows.append({"id": row_id, "label": dev.get("name", row_id)[:12]})
            cells.append({"rowId": row_id, "colId": "health", "value": dev.get("health", 0)})
            cells.append({"rowId": row_id, "colId": "status", "value": status_scores.get(dev.get("status", ""), 50)})

        return {
            "spec": {
                "variant": "VALUE_HEATMAP",
                "demoData": {
                    "label": "Equipment Health Matrix",
                    "dataset": {
                        "min": 0,
                        "max": 100,
                        "unit": "%",
                        "rows": rows,
                        "cols": cols,
                        "cells": cells,
                    }
                },
            }
        }

    def _format_eventlog(self, items: list, log_type: str = "maintenance") -> dict:
        """Format maintenance/work orders/shift logs as event log stream data."""
        events = []
        for item in items[:15]:
            events.append({
                "id": item.get("id", ""),
                "type": log_type,
                "content": item.get("content", "")[:200],
                "source": item.get("equipment_name", item.get("supervisor", "")),
                "priority": item.get("priority", "medium"),
                "status": item.get("status", ""),
            })
        return {"events": events}

    def _format_timeline(self, items: list) -> dict:
        """Format maintenance/work order history as timeline data."""
        entries = []
        for item in items[:10]:
            entries.append({
                "id": item.get("id", ""),
                "label": item.get("equipment_name", item.get("title", "")),
                "description": item.get("content", "")[:100],
                "status": item.get("status", ""),
                "type": item.get("work_type", item.get("maintenance_type", "")),
            })
        return {"entries": entries}

    def _format_trend_multi_line(self, energy_data: list) -> dict:
        """Format energy data for trend-multi-line widget (multiple series)."""
        if not energy_data:
            return {}

        # Group by meter
        meters = {}
        for row in energy_data:
            mid = row.get("meter_id", "unknown")
            if mid not in meters:
                meters[mid] = {"name": row.get("meter_name", mid), "points": []}
            meters[mid]["points"].append(row)

        if len(meters) < 2:
            return {}

        # Build series config
        colors = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]
        series = []
        for i, (mid, info) in enumerate(list(meters.items())[:6]):
            series.append({
                "id": mid,
                "label": info["name"],
                "source": mid,
                "colorToken": colors[i % len(colors)],
                "lineStyle": "solid",
                "yAxis": "left",
                "unit": "kW",
            })

        # Build data points — align by timestamp
        timestamps = set()
        for info in meters.values():
            for p in info["points"]:
                timestamps.add(p.get("timestamp", ""))
        timestamps = sorted(timestamps)[-30:]  # Last 30 points

        data = []
        for ts in timestamps:
            point = {"timestamp": ts}
            for mid, info in meters.items():
                val = next((p.get("power_kw", 0) for p in info["points"] if p.get("timestamp") == ts), 0)
                point[mid] = round(float(val), 1)
            data.append(point)

        return {
            "id": "multi_meter_comparison",
            "name": "Power Sources Comparison",
            "description": "Multi-meter power readings over time",
            "timeRange": "P1D",
            "granularity": "15m",
            "series": series,
            "data": data,
        }

    def _format_trends_cumulative(self, energy_data: list) -> dict:
        """Format energy data for trends-cumulative widget."""
        if not energy_data:
            return {}

        # Group by meter, pick first
        meters = {}
        for row in energy_data:
            mid = row.get("meter_id", "unknown")
            if mid not in meters:
                meters[mid] = {"name": row.get("meter_name", mid), "points": []}
            meters[mid]["points"].append(row)

        first_meter = list(meters.values())[0]
        points = sorted(first_meter["points"], key=lambda p: p.get("timestamp", ""))[-30:]

        cumulative = 0
        data = []
        for p in points:
            raw = round(float(p.get("power_kw", 0)), 1)
            cumulative += raw * 0.25  # 15-min intervals → kWh
            data.append({
                "x": p.get("timestamp", ""),
                "EB_raw": raw,
                "EB_cumulative": round(cumulative, 1),
            })

        return {
            "config": {
                "title": f"Cumulative Energy — {first_meter['name']}",
                "subtitle": "Today",
                "variant": "V1",
                "mode": "cumulative",
                "series": [{
                    "id": "EB",
                    "label": first_meter["name"],
                    "unit": "kWh",
                    "color": "#3b82f6",
                }],
            },
            "data": data,
            "timeRange": "1D",
        }

    def _format_distribution(self, items: list, title: str, variant: str = "DIST_LOAD_BY_ASSET") -> dict:
        """Format data for the distribution widget."""
        series = []
        colors = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899",
                  "#06b6d4", "#84cc16", "#f97316", "#6366f1", "#14b8a6", "#e11d48"]
        total = sum(item.get("value", 0) for item in items[:12])

        for i, item in enumerate(items[:12]):
            val = item.get("value", 0)
            series.append({
                "label": item.get("name", item.get("label", item.get("category", f"Item {i+1}"))),
                "value": val,
                "percentage": round((val / total * 100) if total else 0, 1),
                "color": colors[i % len(colors)],
            })

        # Pick representation based on variant
        rep_map = {
            "DIST_ENERGY_SOURCE_SHARE": ("Donut", "1:1", "proportional"),
            "DIST_LOAD_BY_ASSET": ("Horizontal Bar", "3:1", "rank_emphasis"),
            "DIST_CONSUMPTION_BY_CATEGORY": ("Pie", "1:1", "proportional"),
            "DIST_CONSUMPTION_BY_SHIFT": ("Grouped Bar", "16:9", "comparative"),
            "DIST_DOWNTIME_TOP_CONTRIBUTORS": ("Pareto Bar", "16:9", "rank_emphasis"),
        }
        rep, aspect, encoding = rep_map.get(variant, ("Horizontal Bar", "3:1", "rank_emphasis"))

        return {
            "coreWidget": "Distribution",
            "variant": variant,
            "representation": rep,
            "encoding": encoding,
            "layout": {"aspectRatio": aspect, "zones": ["chart", "legend"]},
            "visual": {"colorMapping": encoding},
            "distributionRules": {"topN": len(series), "groupRemaining": False},
            "demoData": {
                "total": total,
                "unit": "kW",
                "series": series,
            },
        }

    def _format_flow_sankey(self, devices: list, energy_data: list = None) -> dict:
        """Format energy/equipment data for flow-sankey widget."""
        nodes = []
        links = []

        if energy_data:
            # Build source → bus → load flow from energy meters
            meters = {}
            for row in energy_data:
                mid = row.get("meter_id", "unknown")
                if mid not in meters:
                    meters[mid] = {"name": row.get("meter_name", mid), "total": 0, "count": 0}
                meters[mid]["total"] += float(row.get("power_kw", 0))
                meters[mid]["count"] += 1

            # Create source nodes
            total_power = 0
            source_colors = {"grid": "#3b82f6", "solar": "#10b981", "dg": "#f59e0b", "eb": "#3b82f6"}
            for mid, info in list(meters.items())[:6]:
                avg_kw = round(info["total"] / max(info["count"], 1), 1)
                total_power += avg_kw
                color = "#6366f1"
                for key, c in source_colors.items():
                    if key in mid.lower() or key in info["name"].lower():
                        color = c
                        break
                nodes.append({"id": mid, "label": info["name"], "type": "source", "value": avg_kw, "color": color})

            # Bus node
            nodes.append({"id": "main_bus", "label": "Main Bus", "type": "bus", "value": total_power, "color": "#737373"})
            for n in nodes[:-1]:
                links.append({"source": n["id"], "target": "main_bus", "value": n["value"]})

        elif devices:
            # Build from equipment types
            type_groups = {}
            for dev in devices[:12]:
                t = dev.get("type", "other").replace("_", " ").title()
                if t not in type_groups:
                    type_groups[t] = {"devices": [], "total_health": 0}
                type_groups[t]["devices"].append(dev)
                type_groups[t]["total_health"] += dev.get("health", 50)

            nodes.append({"id": "plant", "label": "Plant", "type": "source", "value": len(devices), "color": "#3b82f6"})
            for t, group in type_groups.items():
                tid = t.lower().replace(" ", "_")
                nodes.append({"id": tid, "label": t, "type": "load", "value": len(group["devices"]), "color": "#10b981"})
                links.append({"source": "plant", "target": tid, "value": len(group["devices"])})

        return {
            "coreWidget": "FLOW / SANKEY",
            "variant": "FLOW_SANKEY_STANDARD",
            "purpose": "Energy flow from sources to loads",
            "representation": "Classic Left-to-Right Sankey",
            "demoData": {"nodes": nodes, "links": links},
        }

    def _format_pulseview(self, devices: list, alerts: list) -> dict:
        """Format data for pulseview widget — operational pulse overview."""
        running = sum(1 for d in devices if d.get("status") == "running")
        warning = sum(1 for d in devices if d.get("status") in ("warning", "maintenance"))
        critical_alerts = sum(1 for a in alerts if a.get("severity") in ("critical", "high"))
        return {
            "totalDevices": len(devices),
            "running": running,
            "warning": warning,
            "criticalAlerts": critical_alerts,
        }

    def _format_peoplehexgrid(self, people_data: dict) -> dict:
        """Format data for peoplehexgrid widget."""
        return {
            "onShift": people_data.get("on_shift", 0),
            "absent": people_data.get("absent", 0),
            "shifts": people_data.get("upcoming_shifts", []),
        }

    def _format_peoplenetwork(self, people_data: dict) -> dict:
        """Format data for peoplenetwork widget."""
        return {
            "onShift": people_data.get("on_shift", 0),
            "absent": people_data.get("absent", 0),
        }

    def _generate_heading(self, intent: Intent, rag_results: list) -> str:
        """
        Generate a concise dashboard heading — the gist of what the user asked.

        Examples:
          "Show transformer 1 status" → "Transformer 1 Status"
          "Compare transformer 1 vs 2" → "Comparison: Transformer 1 vs Transformer 2"
          "Any alerts?" → "Active Alerts"
          "Show me power distribution" → "Power Distribution"
        """
        text = intent.raw_text.strip()
        text_lower = text.lower()

        # Comparison queries
        compare_match = re.search(
            r'\b(?:compar\w*|versus|vs\.?|difference|between)\b',
            text_lower,
        )
        if compare_match:
            # Extract the compared entities
            # "compare transformer 1 and transformer 2" → "Transformer 1 vs Transformer 2"
            entities = re.findall(
                r'((?:transformer|pump|motor|chiller|generator|device|sensor)\s*\d*)',
                text_lower,
            )
            if len(entities) >= 2:
                return f"Comparison: {entities[0].title()} vs {entities[1].title()}"
            return "Comparison Overview"

        # Alert queries
        if "alerts" in intent.domains and len(intent.domains) == 1:
            alert_count = 0
            for r in rag_results:
                if r.domain == "alerts" and r.success:
                    alert_count = r.data.get("count", len(r.data.get("alerts", [])))
            if alert_count:
                return f"Active Alerts ({alert_count})"
            return "Alerts Overview"

        # Device-specific queries
        device_entities = intent.entities.get("devices", [])
        if device_entities:
            names = [d.replace("_", " ").title() for d in device_entities[:3]]
            return " & ".join(names) + " Status"

        # Domain-based fallback
        domain_headings = {
            "industrial": "Equipment Overview",
            "supply": "Supply Chain",
            "people": "Workforce",
            "tasks": "Tasks & Projects",
        }
        if intent.domains:
            # Use first domain as primary
            primary = intent.domains[0]
            heading = domain_headings.get(primary, primary.title())

            # Enhance with keywords from the query
            for keyword in ["status", "health", "performance", "usage", "capacity", "trend", "history"]:
                if keyword in text_lower:
                    heading += f" — {keyword.title()}"
                    break

            return heading

        return "Dashboard"

    def _generate_layout(self, intent: Intent, rag_results: list) -> dict:
        """
        2C: Generate layout JSON for Layer 3 (Blob).

        Intelligently selects widget scenarios based on:
        1. Query intent type (status, comparison, trend, alert, maintenance, etc.)
        2. Data shape from RAG (what collections returned results)
        3. Entity targeting (specific device → hero, general → multi-widget)

        Maps to all 23 widget scenarios with correct data_override shapes.
        """
        heading = self._generate_heading(intent, rag_results)
        widgets = []
        query_lower = intent.raw_text.lower()

        # Detect query characteristics
        target_entities = set(intent.entities.get("devices", []))
        is_comparison = bool(re.search(
            r'\b(?:compar\w*|versus|vs\.?|difference|between)\b', query_lower))
        is_trend_query = bool(re.search(
            r'\b(?:trend|graph|chart|over time|history|historical|last \d+|past \d+)\b', query_lower))
        is_distribution_query = bool(re.search(
            r'\b(?:distribut\w*|breakdown|composition|split|share|proportion|pie|donut)\b', query_lower))
        is_maintenance_query = bool(re.search(
            r'\b(?:maintenan\w*|repair|service|inspection|overhaul|parts|replaced)\b', query_lower))
        is_shift_query = bool(re.search(
            r'\b(?:shift|handover|supervisor|last night|morning|evening|night)\b', query_lower))
        is_work_order_query = bool(re.search(
            r'\b(?:work order|task|pending|overdue|assigned|open ticket|wo-|wo_)\b', query_lower))
        is_energy_query = bool(re.search(
            r'\b(?:energy|power|load|consumption|kwh|kw|voltage|current|electrical)\b', query_lower))
        is_health_query = bool(re.search(
            r'\b(?:health|condition|status|overview|dashboard|summary)\b', query_lower))
        is_flow_query = bool(re.search(
            r'\b(?:flow|sankey|where does|goes to|feeds|source.?to|losses|energy balance)\b', query_lower))
        is_cumulative_query = bool(re.search(
            r'\b(?:cumulative|total today|accumulated|running total|daily total|how much.*so far)\b', query_lower))
        is_multi_source_query = bool(re.search(
            r'\b(?:eb vs|dg vs|solar vs|grid vs|sources|by source|phase|phases|multi.?meter)\b', query_lower))
        is_pq_query = bool(re.search(
            r'\b(?:power quality|harmonic|thd|power factor|sag|swell|dip|voltage dip|pf penalty)\b', query_lower))
        is_hvac_query = bool(re.search(
            r'\b(?:hvac|ahu|chiller|cooling|comfort|setpoint|zone temp|air handling)\b', query_lower))
        is_ups_dg_query = bool(re.search(
            r'\b(?:ups|battery|runtime|dg|diesel|transfer|generator|backup power|amf)\b', query_lower))
        is_top_consumers_query = bool(re.search(
            r'\b(?:top consumers?|biggest load|highest load|most energy|ranking|top \d+|worst performers?)\b', query_lower))

        logger.info(
            f"Layout generation — query: '{query_lower[:60]}' | "
            f"domains: {intent.domains} | entities: {target_entities} | "
            f"comparison={is_comparison} trend={is_trend_query} "
            f"distribution={is_distribution_query} maintenance={is_maintenance_query} "
            f"shift={is_shift_query} work_order={is_work_order_query} energy={is_energy_query} "
            f"flow={is_flow_query} cumulative={is_cumulative_query} multi_source={is_multi_source_query} "
            f"pq={is_pq_query} hvac={is_hvac_query} ups_dg={is_ups_dg_query}"
        )

        # ── Determine which enrichment scenarios are relevant to this query ──
        relevant_enrichments: set[str] = set()
        if is_energy_query or is_cumulative_query or is_multi_source_query:
            relevant_enrichments.update(["trend", "trend-multi-line", "trends-cumulative",
                                          "distribution", "composition", "flow-sankey"])
        if is_maintenance_query or is_work_order_query:
            relevant_enrichments.update(["timeline", "eventlogstream", "alerts"])
        if is_shift_query:
            relevant_enrichments.update(["eventlogstream", "timeline"])
        if is_health_query or target_entities:
            relevant_enrichments.update(["matrix-heatmap", "comparison", "category-bar", "alerts"])
        if is_hvac_query or is_ups_dg_query:
            relevant_enrichments.update(["trend", "comparison", "alerts", "category-bar"])
        if is_pq_query:
            relevant_enrichments.update(["trend", "trend-multi-line", "distribution"])
        if is_flow_query:
            relevant_enrichments.update(["distribution", "composition"])
        if is_top_consumers_query:
            relevant_enrichments.update(["category-bar", "distribution", "trend", "trends-cumulative"])
        if is_comparison:
            relevant_enrichments.update(["comparison", "trend-multi-line"])
        if is_trend_query:
            relevant_enrichments.update(["trend", "trends-cumulative", "trend-multi-line"])
        if is_distribution_query:
            relevant_enrichments.update(["distribution", "composition", "category-bar"])
        # Broad queries (no specific flags) → allow all enrichments
        if not any([is_energy_query, is_maintenance_query, is_shift_query, is_work_order_query,
                    is_health_query, is_hvac_query, is_ups_dg_query, is_pq_query, is_flow_query,
                    is_cumulative_query, is_multi_source_query, is_top_consumers_query,
                    is_comparison, is_trend_query, is_distribution_query]):
            relevant_enrichments.update(["trend", "trend-multi-line", "trends-cumulative",
                                          "distribution", "composition", "category-bar",
                                          "timeline", "eventlogstream", "matrix-heatmap",
                                          "comparison", "flow-sankey", "alerts"])

        # Collect all data from all RAG results
        all_devices = []
        all_alerts = []
        all_maintenance = []
        all_work_orders = []
        all_shift_logs = []
        all_op_docs = []
        energy_timeseries = []
        supply_data = {}
        people_data = {}

        for result in rag_results:
            if not result.success:
                continue
            d = result.data
            all_devices.extend(d.get("devices", []))
            all_alerts.extend(d.get("alerts", []))
            all_maintenance.extend(d.get("maintenance", []))
            all_work_orders.extend(d.get("work_orders", []))
            all_shift_logs.extend(d.get("shift_logs", []))
            all_op_docs.extend(d.get("operational_docs", []))
            if d.get("energy_timeseries"):
                energy_timeseries = d["energy_timeseries"]
            if result.domain == "supply":
                supply_data = d
            if result.domain == "people":
                people_data = d

        # Sort devices by relevance
        all_devices.sort(key=lambda d: d.get("relevance_score", 0), reverse=True)
        all_alerts.sort(key=lambda a: (
            {"critical": 0, "high": 1, "warning": 2, "medium": 3, "low": 4, "info": 5}.get(
                a.get("severity", "info"), 5),
            -a.get("relevance_score", 0)))

        # ── 1. COMPARISON QUERY → comparison + trend-multi-line + KPIs ──
        if is_comparison and len(all_devices) >= 2:
            d1, d2 = all_devices[0], all_devices[1]
            widgets.append({
                "scenario": "comparison",
                "fixture": "side_by_side_visual-plain-values",
                "relevance": 0.98,
                "size": "hero",
                "position": "top-center",
                "data_override": self._format_comparison(
                    label="Equipment Health",
                    unit="%",
                    label_a=d1.get("name", "Device A"),
                    value_a=d1.get("health", 0),
                    label_b=d2.get("name", "Device B"),
                    value_b=d2.get("health", 0),
                ),
            })
            # Multi-line trend if we have energy data for both
            if energy_timeseries:
                multi_data = self._format_trend_multi_line(energy_timeseries)
                if multi_data:
                    widgets.append({
                        "scenario": "trend-multi-line",
                        "fixture": "power-sources-stacked",
                        "relevance": 0.88,
                        "size": "expanded",
                        "position": "middle-left",
                        "data_override": multi_data,
                    })
            # Device KPIs
            for i, dev in enumerate(all_devices[:4]):
                status = dev.get("status", "normal")
                kpi_variant = "kpi_live-standard"
                if status in ("fault", "critical"):
                    kpi_variant = "kpi_alert-critical-state"
                elif status in ("warning", "maintenance"):
                    kpi_variant = "kpi_alert-warning-state"
                kpi = self._format_kpi(
                    label=dev.get("name", f"Device {i+1}"),
                    value=f"{dev.get('health', '—')}",
                    unit="%",
                    state=status,
                    variant=kpi_variant,
                )
                widgets.append({
                    **kpi,
                    "relevance": 0.85 - (i * 0.05),
                    "size": "compact",
                    "position": None,
                })

        # ── 2. FLOW / SANKEY QUERY → flow-sankey + distribution + KPIs ──
        elif is_flow_query:
            sankey_data = self._format_flow_sankey(all_devices, energy_timeseries)
            sankey_data["_query_context"] = query_lower  # pass query for fixture_selector
            if sankey_data.get("demoData", {}).get("nodes"):
                widgets.append({
                    "scenario": "flow-sankey",
                    "fixture": "flow_sankey_standard-classic-left-to-right-sankey",
                    "relevance": 0.97,
                    "size": "hero",
                    "position": "top-center",
                    "data_override": sankey_data,
                })
            # Supporting distribution
            if all_devices:
                type_counts = {}
                for dev in all_devices:
                    t = dev.get("type", "other").replace("_", " ").title()
                    type_counts[t] = type_counts.get(t, 0) + 1
                dist_items = [{"name": t, "value": c} for t, c in type_counts.items()]
                widgets.append({
                    "scenario": "distribution",
                    "fixture": "dist_energy_source_share-donut",
                    "relevance": 0.82,
                    "size": "expanded",
                    "position": "middle-right",
                    "data_override": self._format_distribution(dist_items, "Equipment Distribution",
                                                                "DIST_ENERGY_SOURCE_SHARE"),
                })
            # Energy KPIs
            if energy_timeseries:
                latest = energy_timeseries[-1] if energy_timeseries else {}
                widgets.append({
                    **self._format_kpi("Current Load", str(round(float(latest.get("power_kw", 0)), 1)), "kW", "normal"),
                    "relevance": 0.78,
                    "size": "compact",
                    "position": None,
                })

        # ── 3. MULTI-SOURCE ENERGY → trend-multi-line + distribution + cumulative ──
        elif is_multi_source_query and energy_timeseries:
            multi_data = self._format_trend_multi_line(energy_timeseries)
            if multi_data:
                widgets.append({
                    "scenario": "trend-multi-line",
                    "fixture": "power-sources-stacked",
                    "relevance": 0.96,
                    "size": "hero",
                    "position": "top-center",
                    "data_override": multi_data,
                })
            # Source distribution as donut
            meters = {}
            for row in energy_timeseries:
                mid = row.get("meter_id", "unknown")
                if mid not in meters:
                    meters[mid] = {"name": row.get("meter_name", mid), "total": 0, "count": 0}
                meters[mid]["total"] += float(row.get("power_kw", 0))
                meters[mid]["count"] += 1
            dist_items = [{"name": info["name"], "value": round(info["total"] / max(info["count"], 1), 1)}
                         for info in meters.values()]
            if dist_items:
                widgets.append({
                    "scenario": "distribution",
                    "fixture": "dist_energy_source_share-donut",
                    "relevance": 0.85,
                    "size": "expanded",
                    "position": "middle-left",
                    "data_override": self._format_distribution(dist_items, "Energy by Source",
                                                                "DIST_ENERGY_SOURCE_SHARE"),
                })
            # Cumulative view
            cum_data = self._format_trends_cumulative(energy_timeseries)
            if cum_data:
                widgets.append({
                    "scenario": "trends-cumulative",
                    "fixture": "energy-consumption",
                    "relevance": 0.78,
                    "size": "expanded",
                    "position": "middle-right",
                    "data_override": cum_data,
                })

        # ── 4. CUMULATIVE ENERGY → trends-cumulative + trend + KPI ──
        elif is_cumulative_query and energy_timeseries:
            cum_data = self._format_trends_cumulative(energy_timeseries)
            if cum_data:
                widgets.append({
                    "scenario": "trends-cumulative",
                    "fixture": "energy-consumption",
                    "relevance": 0.96,
                    "size": "hero",
                    "position": "top-center",
                    "data_override": cum_data,
                })
            # Supporting trend
            trend_data = self._format_trend_from_energy(energy_timeseries)
            if trend_data:
                widgets.append({
                    "scenario": "trend",
                    "fixture": "trend_live-line",
                    "relevance": 0.82,
                    "size": "expanded",
                    "position": "middle-left",
                    "data_override": trend_data,
                })
            # Total consumption KPI
            total_kwh = sum(float(r.get("power_kw", 0)) * 0.25 for r in energy_timeseries)
            widgets.append({
                **self._format_kpi("Total Energy", str(round(total_kwh, 1)), "kWh", "normal",
                                   "kpi_accumulated-daily-total"),
                "relevance": 0.80,
                "size": "compact",
                "position": None,
            })

        # ── 5. TOP CONSUMERS / RANKING → distribution + category-bar + KPI ──
        elif is_top_consumers_query:
            if all_devices:
                # Rank by health (inverse — lower health = more concerning)
                ranked = sorted(all_devices, key=lambda d: d.get("health", 100))
                dist_items = [{"name": d.get("name", "?"), "value": d.get("health", 0)} for d in ranked[:10]]
                widgets.append({
                    "scenario": "distribution",
                    "fixture": "dist_load_by_asset-horizontal-bar",
                    "relevance": 0.95,
                    "size": "hero",
                    "position": "top-center",
                    "data_override": self._format_distribution(dist_items, "Equipment by Health",
                                                                "DIST_LOAD_BY_ASSET"),
                })
                # Category bar by type
                type_counts = {}
                for dev in all_devices:
                    t = dev.get("type", "other").replace("_", " ").title()
                    type_counts[t] = type_counts.get(t, 0) + 1
                bar_items = [{"category": t, "value": c} for t, c in type_counts.items()]
                widgets.append({
                    "scenario": "category-bar",
                    "fixture": "oee-by-machine",
                    "relevance": 0.80,
                    "size": "expanded",
                    "position": "middle-left",
                    "data_override": self._format_category_bar(bar_items, "Equipment by Type"),
                })
            if energy_timeseries:
                # Top meters by avg load
                meters = {}
                for row in energy_timeseries:
                    mid = row.get("meter_id", "unknown")
                    if mid not in meters:
                        meters[mid] = {"name": row.get("meter_name", mid), "total": 0, "count": 0}
                    meters[mid]["total"] += float(row.get("power_kw", 0))
                    meters[mid]["count"] += 1
                ranked_meters = sorted(meters.values(), key=lambda m: m["total"] / max(m["count"], 1), reverse=True)
                for i, m in enumerate(ranked_meters[:3]):
                    avg = round(m["total"] / max(m["count"], 1), 1)
                    widgets.append({
                        **self._format_kpi(m["name"], str(avg), "kW", "normal"),
                        "relevance": 0.75 - (i * 0.05),
                        "size": "compact",
                        "position": None,
                    })

        # ── 6. HVAC QUERY → trend-multi-line + KPI + matrix-heatmap ──
        elif is_hvac_query:
            hvac_devices = [d for d in all_devices if d.get("type", "").lower() in
                           ("ahu", "chiller", "cooling_tower", "hvac", "air_handling_unit")]
            if not hvac_devices:
                hvac_devices = all_devices[:6]

            # Multi-line trend for HVAC metrics
            if energy_timeseries:
                multi_data = self._format_trend_multi_line(energy_timeseries)
                if multi_data:
                    widgets.append({
                        "scenario": "trend-multi-line",
                        "fixture": "hvac-performance",
                        "relevance": 0.95,
                        "size": "hero",
                        "position": "top-center",
                        "data_override": multi_data,
                    })

            # HVAC device KPIs
            for i, dev in enumerate(hvac_devices[:4]):
                status = dev.get("status", "normal")
                kpi_variant = "kpi_alert-warning-state" if status in ("warning", "maintenance") else "kpi_live-standard"
                widgets.append({
                    **self._format_kpi(dev.get("name", f"HVAC {i+1}"), f"{dev.get('health', '—')}%", "",
                                       status, kpi_variant),
                    "relevance": 0.82 - (i * 0.04),
                    "size": "compact",
                    "position": None,
                })

            # Heatmap: zone/device health
            if len(hvac_devices) >= 3:
                widgets.append({
                    "scenario": "matrix-heatmap",
                    "fixture": "status-matrix",
                    "relevance": 0.75,
                    "size": "expanded",
                    "position": "middle-right",
                    "data_override": self._format_matrix_heatmap(hvac_devices),
                })

        # ── 7. UPS / DG / BACKUP POWER → edgedevicepanel + trend-multi-line + KPI ──
        elif is_ups_dg_query:
            backup_devices = [d for d in all_devices if d.get("type", "").lower() in
                             ("ups", "diesel_generator", "amf_panel", "generator")]
            if not backup_devices:
                backup_devices = all_devices[:6]

            # Device panel for UPS/DG
            if len(backup_devices) >= 2:
                widgets.append({
                    "scenario": "edgedevicepanel",
                    "fixture": "default-render",
                    "relevance": 0.95,
                    "size": "hero",
                    "position": "top-center",
                    "data_override": {"devices": backup_devices[:8]},
                })

            # Trend for backup power metrics
            if energy_timeseries:
                multi_data = self._format_trend_multi_line(energy_timeseries)
                if multi_data:
                    widgets.append({
                        "scenario": "trend-multi-line",
                        "fixture": "ups-health-dual-axis",
                        "relevance": 0.85,
                        "size": "expanded",
                        "position": "middle-left",
                        "data_override": multi_data,
                    })

            # KPIs per backup device
            for i, dev in enumerate(backup_devices[:3]):
                status = dev.get("status", "normal")
                kpi_variant = "kpi_alert-critical-state" if status in ("fault", "critical") else "kpi_live-standard"
                widgets.append({
                    **self._format_kpi(dev.get("name", f"UPS/DG {i+1}"), f"{dev.get('health', '—')}%", "",
                                       status, kpi_variant),
                    "relevance": 0.80 - (i * 0.05),
                    "size": "compact",
                    "position": None,
                })

            # Transfer event log if available
            if all_maintenance or all_shift_logs:
                log_items = all_maintenance[:5] or all_shift_logs[:5]
                widgets.append({
                    "scenario": "eventlogstream",
                    "fixture": "chronological-timeline",
                    "relevance": 0.70,
                    "size": "expanded",
                    "position": "middle-right",
                    "data_override": self._format_eventlog(log_items, "transfer_event"),
                })

        # ── 8. POWER QUALITY → matrix-heatmap + trend + alerts ──
        elif is_pq_query:
            # Heatmap for PQ metrics across devices
            if all_devices and len(all_devices) >= 2:
                widgets.append({
                    "scenario": "matrix-heatmap",
                    "fixture": "value-heatmap",
                    "relevance": 0.95,
                    "size": "hero",
                    "position": "top-center",
                    "data_override": self._format_matrix_heatmap(all_devices),
                })

            # Trend for PQ data
            if energy_timeseries:
                trend_data = self._format_trend_from_energy(energy_timeseries)
                if trend_data:
                    widgets.append({
                        "scenario": "trend",
                        "fixture": "trend_alert_context-line-threshold",
                        "relevance": 0.88,
                        "size": "expanded",
                        "position": "middle-left",
                        "data_override": trend_data,
                    })

            # PQ alerts
            if all_alerts:
                formatted_alerts = [self._format_alert(a) for a in all_alerts[:3]]
                if formatted_alerts:
                    widgets.append({
                        "scenario": "alerts",
                        "fixture": "toast-power-factor-critical-low",
                        "relevance": 0.82,
                        "size": "expanded",
                        "position": "middle-right",
                        "data_override": formatted_alerts[0],
                    })

            # PQ KPIs
            if all_devices:
                avg_health = round(sum(d.get("health", 0) for d in all_devices) / len(all_devices))
                pq_state = "warning" if avg_health < 80 else "normal"
                widgets.append({
                    **self._format_kpi("Avg Power Quality", f"{avg_health}", "%", pq_state,
                                       "kpi_alert-warning-state" if pq_state == "warning" else "kpi_live-standard"),
                    "relevance": 0.78,
                    "size": "compact",
                    "position": None,
                })

        # ── 9. MAINTENANCE QUERY → eventlog + timeline + composition + KPIs ──
        elif is_maintenance_query or (all_maintenance and not all_devices):
            if all_maintenance:
                widgets.append({
                    "scenario": "eventlogstream",
                    "fixture": "chronological-timeline",
                    "relevance": 0.95,
                    "size": "hero",
                    "position": "top-center",
                    "data_override": self._format_eventlog(all_maintenance, "maintenance"),
                })
                widgets.append({
                    **self._format_kpi("Maintenance Records", str(len(all_maintenance)), "", "normal",
                                       "kpi_accumulated-daily-total"),
                    "relevance": 0.80,
                    "size": "compact",
                    "position": None,
                })

            if all_work_orders:
                widgets.append({
                    "scenario": "timeline",
                    "fixture": "linear-incident-timeline",
                    "relevance": 0.88,
                    "size": "expanded",
                    "position": "middle-left",
                    "data_override": self._format_timeline(all_work_orders),
                })

            # Maintenance type distribution (composition)
            if all_maintenance:
                type_counts = {}
                for m in all_maintenance:
                    t = m.get("maintenance_type", "other").replace("_", " ").title()
                    type_counts[t] = type_counts.get(t, 0) + 1
                donut_items = [{"name": t, "value": c} for t, c in type_counts.items()]
                widgets.append({
                    "scenario": "composition",
                    "fixture": "donut_pie",
                    "relevance": 0.75,
                    "size": "expanded",
                    "position": "middle-right",
                    "data_override": {
                        "demoData": self._format_composition_donut(donut_items),
                    },
                })

        # ── 10. SHIFT LOG QUERY → eventlog + timeline + KPI ──
        elif is_shift_query or (all_shift_logs and not all_devices):
            if all_shift_logs:
                widgets.append({
                    "scenario": "eventlogstream",
                    "fixture": "chronological-timeline",
                    "relevance": 0.95,
                    "size": "hero",
                    "position": "top-center",
                    "data_override": self._format_eventlog(all_shift_logs, "shift_log"),
                })
                widgets.append({
                    "scenario": "timeline",
                    "fixture": "multi-lane-shift-schedule",
                    "relevance": 0.82,
                    "size": "expanded",
                    "position": "middle-left",
                    "data_override": self._format_timeline(all_shift_logs),
                })
                widgets.append({
                    **self._format_kpi("Shift Logs", str(len(all_shift_logs)), "", "normal"),
                    "relevance": 0.75,
                    "size": "compact",
                    "position": None,
                })

        # ── 11. WORK ORDER QUERY → eventlog + timeline + distribution ──
        elif is_work_order_query or (all_work_orders and "tasks" in intent.domains):
            if all_work_orders:
                widgets.append({
                    "scenario": "eventlogstream",
                    "fixture": "tabular-log-view",
                    "relevance": 0.95,
                    "size": "hero",
                    "position": "top-center",
                    "data_override": self._format_eventlog(all_work_orders, "work_order"),
                })
                widgets.append({
                    "scenario": "timeline",
                    "fixture": "linear-incident-timeline",
                    "relevance": 0.85,
                    "size": "expanded",
                    "position": "middle-left",
                    "data_override": self._format_timeline(all_work_orders),
                })
                # Priority distribution
                priority_counts = {}
                for wo in all_work_orders:
                    p = wo.get("priority", "medium")
                    priority_counts[p] = priority_counts.get(p, 0) + 1
                dist_items = [{"name": p.title(), "value": c} for p, c in priority_counts.items()]
                if len(dist_items) > 1:
                    widgets.append({
                        "scenario": "distribution",
                        "fixture": "dist_consumption_by_category-pie",
                        "relevance": 0.78,
                        "size": "expanded",
                        "position": "middle-right",
                        "data_override": self._format_distribution(dist_items, "Work Orders by Priority",
                                                                    "DIST_CONSUMPTION_BY_CATEGORY"),
                    })
                # Status KPIs
                open_count = sum(1 for wo in all_work_orders if wo.get("status") in ("open", "in_progress"))
                widgets.append({
                    **self._format_kpi("Open Work Orders", str(open_count), "",
                                       "warning" if open_count > 5 else "normal",
                                       "kpi_alert-warning-state" if open_count > 5 else "kpi_live-standard"),
                    "relevance": 0.76,
                    "size": "compact",
                    "position": None,
                })

        # ── 12. TREND / ENERGY QUERY → trend + trend-multi-line + category-bar ──
        elif (is_trend_query or is_energy_query) and not is_distribution_query:
            # Primary: trend chart from energy time-series
            if energy_timeseries:
                trend_data = self._format_trend_from_energy(energy_timeseries)
                if trend_data:
                    widgets.append({
                        "scenario": "trend",
                        "fixture": "trend_live-line",
                        "relevance": 0.96,
                        "size": "hero",
                        "position": "top-center",
                        "data_override": trend_data,
                    })
                # Multi-line if multiple meters
                multi_data = self._format_trend_multi_line(energy_timeseries)
                if multi_data:
                    widgets.append({
                        "scenario": "trend-multi-line",
                        "fixture": "power-sources-stacked",
                        "relevance": 0.85,
                        "size": "expanded",
                        "position": "middle-left",
                        "data_override": multi_data,
                    })

            # If we have devices, show health as secondary trend
            if all_devices and not energy_timeseries:
                trend_data = self._format_trend_from_devices(all_devices)
                widgets.append({
                    "scenario": "trend",
                    "fixture": "trend_live-line",
                    "relevance": 0.94,
                    "size": "hero",
                    "position": "top-center",
                    "data_override": trend_data,
                })

            # Category bar: equipment health by type
            if all_devices:
                type_health = {}
                for dev in all_devices:
                    t = dev.get("type", "other")
                    if t not in type_health:
                        type_health[t] = []
                    type_health[t].append(dev.get("health", 0))
                bar_items = [
                    {"category": t.replace("_", " ").title(), "value": round(sum(h) / len(h))}
                    for t, h in type_health.items()
                ]
                widgets.append({
                    "scenario": "category-bar",
                    "fixture": "oee-by-machine",
                    "relevance": 0.75,
                    "size": "expanded",
                    "position": "middle-right",
                    "data_override": self._format_category_bar(
                        bar_items, "Avg Health by Equipment Type"),
                })

            # Supporting KPIs
            if all_devices:
                running = sum(1 for d in all_devices if d.get("status") == "running")
                total = len(all_devices)
                widgets.append({
                    **self._format_kpi("Equipment Online", f"{running}/{total}", "", "normal"),
                    "relevance": 0.7,
                    "size": "compact",
                    "position": None,
                })

        # ── 13. DISTRIBUTION QUERY → distribution + composition + category-bar ──
        elif is_distribution_query:
            if all_devices:
                # Distribution by status (donut)
                status_counts = {}
                for dev in all_devices:
                    s = dev.get("status", "unknown")
                    status_counts[s] = status_counts.get(s, 0) + 1
                dist_items = [{"name": s.title(), "value": c} for s, c in status_counts.items()]
                widgets.append({
                    "scenario": "distribution",
                    "fixture": "dist_energy_source_share-donut",
                    "relevance": 0.95,
                    "size": "hero",
                    "position": "top-center",
                    "data_override": self._format_distribution(dist_items, "Equipment by Status",
                                                                "DIST_ENERGY_SOURCE_SHARE"),
                })

                # Composition by type (stacked bar)
                type_counts = {}
                for dev in all_devices:
                    t = dev.get("type", "other").replace("_", " ").title()
                    type_counts[t] = type_counts.get(t, 0) + 1
                donut_items = [{"name": t, "value": c} for t, c in type_counts.items()]
                widgets.append({
                    "scenario": "composition",
                    "fixture": "donut_pie",
                    "relevance": 0.85,
                    "size": "expanded",
                    "position": "middle-left",
                    "data_override": {
                        "demoData": self._format_composition_donut(donut_items),
                    },
                })

                # Category bar detail
                bar_items = [{"category": t, "value": c} for t, c in type_counts.items()]
                widgets.append({
                    "scenario": "category-bar",
                    "fixture": "oee-by-machine",
                    "relevance": 0.78,
                    "size": "expanded",
                    "position": "middle-right",
                    "data_override": self._format_category_bar(bar_items, "Equipment by Type"),
                })

        # ── 14. ALERT-FOCUSED QUERY → alerts + distribution + matrix-heatmap + KPI ──
        elif "alerts" in intent.domains:
            if all_alerts:
                formatted_alerts = [self._format_alert(a) for a in all_alerts[:5]]
                top_severity = all_alerts[0].get("severity", "info")
                fixture_map = {
                    "critical": "modal-ups-battery-critical",
                    "high": "toast-power-factor-critical-low",
                    "warning": "banner-energy-peak-threshold-exceeded",
                }
                fixture = fixture_map.get(top_severity, "banner-energy-peak-threshold-exceeded")

                widgets.append({
                    "scenario": "alerts",
                    "fixture": fixture,
                    "relevance": 0.96,
                    "size": "hero",
                    "position": "top-center",
                    "data_override": formatted_alerts[0] if formatted_alerts else {},
                })

                # Additional alerts as expanded cards
                for i, fa in enumerate(formatted_alerts[1:4]):
                    widgets.append({
                        "scenario": "alerts",
                        "fixture": "badge-ahu-01-high-temperature",
                        "relevance": 0.88 - (i * 0.05),
                        "size": "normal",
                        "position": None,
                        "data_override": fa,
                    })

                # Alert count KPI
                alert_kpi_variant = "kpi_alert-critical-state" if top_severity in ("critical", "high") else "kpi_alert-warning-state"
                widgets.append({
                    **self._format_kpi("Active Alerts", str(len(all_alerts)), "", top_severity, alert_kpi_variant),
                    "relevance": 0.92,
                    "size": "compact",
                    "position": None,
                })

                # Severity distribution
                sev_counts = {}
                for a in all_alerts:
                    s = a.get("severity", "info")
                    sev_counts[s] = sev_counts.get(s, 0) + 1
                if len(sev_counts) > 1:
                    dist_items = [{"name": s.title(), "value": c} for s, c in sev_counts.items()]
                    widgets.append({
                        "scenario": "distribution",
                        "fixture": "dist_consumption_by_category-pie",
                        "relevance": 0.75,
                        "size": "expanded",
                        "position": "middle-left",
                        "data_override": self._format_distribution(dist_items, "Alerts by Severity",
                                                                    "DIST_CONSUMPTION_BY_CATEGORY"),
                    })

                # Affected equipment heatmap
                if all_devices and len(all_devices) >= 3:
                    widgets.append({
                        "scenario": "matrix-heatmap",
                        "fixture": "status-matrix",
                        "relevance": 0.72,
                        "size": "expanded",
                        "position": "middle-right",
                        "data_override": self._format_matrix_heatmap(all_devices),
                    })

        # ── 15. GENERAL EQUIPMENT / STATUS QUERY (default) ──
        else:
            # Device KPIs for targeted devices
            if all_devices:
                sorted_devices = all_devices[:6]
                for i, device in enumerate(sorted_devices):
                    device_id = device.get("id", "")
                    device_name = device.get("name", f"Device {i+1}")
                    name_lower = device_name.lower().replace(" ", "_").replace("-", "_")
                    status = device.get("status", "normal")

                    is_target = (
                        device_id in target_entities or
                        name_lower in target_entities or
                        any(t in name_lower or name_lower in t for t in target_entities)
                    )

                    # Pick KPI variant based on status
                    if status in ("fault", "critical"):
                        kpi_variant = "kpi_alert-critical-state"
                    elif status in ("warning", "maintenance"):
                        kpi_variant = "kpi_alert-warning-state"
                    elif status == "running":
                        kpi_variant = "kpi_live-standard"
                    elif status in ("standby", "stopped", "offline"):
                        kpi_variant = "kpi_status-offline"
                    else:
                        kpi_variant = "kpi_live-standard"

                    if is_target:
                        size = "hero"
                        relevance = 0.98
                    elif target_entities:
                        size = "compact"
                        relevance = 0.5 - (i * 0.05)
                    else:
                        size = "compact"
                        relevance = 0.90 - (i * 0.04)

                    kpi = self._format_kpi(
                        label=device_name,
                        value=f"{device.get('health', '—')}%",
                        unit=device.get("type", "").replace("_", " ").title(),
                        state=status,
                        variant=kpi_variant,
                    )
                    widgets.append({
                        **kpi,
                        "relevance": relevance,
                        "size": size,
                        "position": None,
                    })

                # If many devices and no specific target → device panel
                if len(all_devices) > 4 and not target_entities:
                    widgets.append({
                        "scenario": "edgedevicepanel",
                        "fixture": "default-render",
                        "relevance": 0.85,
                        "size": "hero",
                        "position": "top-center",
                        "data_override": {"devices": all_devices[:12]},
                    })

                # Health heatmap if enough devices
                if len(all_devices) >= 3 and is_health_query:
                    widgets.append({
                        "scenario": "matrix-heatmap",
                        "fixture": "value-heatmap",
                        "relevance": 0.78,
                        "size": "expanded",
                        "position": "middle-right",
                        "data_override": self._format_matrix_heatmap(all_devices),
                    })

            # Trend chart from energy data
            if energy_timeseries:
                trend_data = self._format_trend_from_energy(energy_timeseries)
                if trend_data:
                    widgets.append({
                        "scenario": "trend",
                        "fixture": "trend_live-line",
                        "relevance": 0.75,
                        "size": "expanded",
                        "position": "middle-left",
                        "data_override": trend_data,
                    })

            # Alerts sidebar if any
            if all_alerts:
                formatted_alerts = [self._format_alert(a) for a in all_alerts[:3]]
                if formatted_alerts:
                    top_severity = all_alerts[0].get("severity", "info")
                    widgets.append({
                        "scenario": "alerts",
                        "fixture": "banner-energy-peak-threshold-exceeded",
                        "relevance": 0.82,
                        "size": "expanded",
                        "position": "middle-right",
                        "data_override": formatted_alerts[0],
                    })
                    widgets.append({
                        **self._format_kpi(
                            "Active Alerts", str(len(all_alerts)), "", top_severity,
                            "kpi_alert-critical-state" if top_severity in ("critical", "high") else "kpi_alert-warning-state"),
                        "relevance": 0.80,
                        "size": "compact",
                        "position": None,
                    })

            # Maintenance/work order context if available
            if all_work_orders and not is_maintenance_query:
                open_count = sum(1 for wo in all_work_orders if wo.get("status") in ("open", "in_progress"))
                if open_count > 0:
                    widgets.append({
                        **self._format_kpi("Open Work Orders", str(open_count), "", "normal",
                                           "kpi_accumulated-daily-total"),
                        "relevance": 0.60,
                        "size": "compact",
                        "position": None,
                    })

        # ══════════════════════════════════════════════════════════════
        # CONTEXTUAL ENRICHMENT — fill empty scenario slots with
        # supporting widgets from whatever data the RAG returned
        # ══════════════════════════════════════════════════════════════
        used_scenarios = {w["scenario"] for w in widgets}

        # Alerts enrichment — if we have alerts and no alerts widget yet
        if all_alerts and "alerts" not in used_scenarios and "alerts" in relevant_enrichments:
            top = all_alerts[0]
            widgets.append({
                "scenario": "alerts",
                "fixture": "banner-energy-peak-threshold-exceeded",
                "relevance": 0.72,
                "size": "normal",
                "position": None,
                "data_override": self._format_alert(top),
            })

        # Trend enrichment — energy consumption area chart
        if energy_timeseries and "trend" not in used_scenarios and "trend" in relevant_enrichments:
            trend_data = self._format_trend_from_energy(energy_timeseries)
            if trend_data:
                if isinstance(trend_data.get("demoData"), dict):
                    trend_data["demoData"]["label"] = "energy consumption"
                widgets.append({
                    "scenario": "trend",
                    "fixture": "trend_live-area",
                    "relevance": 0.68,
                    "size": "expanded",
                    "position": None,
                    "data_override": trend_data,
                })

        # Trend-multi-line enrichment — multi-source power
        if energy_timeseries and "trend-multi-line" not in used_scenarios and "trend-multi-line" in relevant_enrichments:
            multi_data = self._format_trend_multi_line(energy_timeseries)
            if multi_data:
                if isinstance(multi_data.get("demoData"), dict):
                    multi_data["demoData"]["label"] = "power source comparison"
                widgets.append({
                    "scenario": "trend-multi-line",
                    "fixture": "power-sources-stacked",
                    "relevance": 0.65,
                    "size": "expanded",
                    "position": None,
                    "data_override": multi_data,
                })

        # Distribution enrichment — load by asset (horizontal bar)
        if all_devices and "distribution" not in used_scenarios and "distribution" in relevant_enrichments:
            type_counts = {}
            for dev in all_devices:
                t = dev.get("type", "other").replace("_", " ").title()
                type_counts[t] = type_counts.get(t, 0) + 1
            if len(type_counts) > 1:
                dist_items = [{"name": t, "value": c} for t, c in type_counts.items()]
                dist_data = self._format_distribution(
                    dist_items, "Load by Asset Type", "DIST_LOAD_BY_ASSET")
                if isinstance(dist_data.get("demoData"), dict):
                    dist_data["demoData"]["label"] = "load by asset"
                widgets.append({
                    "scenario": "distribution",
                    "fixture": "dist_load_by_asset-horizontal-bar",
                    "relevance": 0.63,
                    "size": "normal",
                    "position": None,
                    "data_override": dist_data,
                })

        # Composition enrichment — status breakdown as treemap or donut
        if all_devices and "composition" not in used_scenarios and "composition" in relevant_enrichments:
            status_counts = {}
            for dev in all_devices:
                s = dev.get("status", "unknown").title()
                status_counts[s] = status_counts.get(s, 0) + 1
            if len(status_counts) > 1:
                items = [{"name": s, "value": c} for s, c in status_counts.items()]
                comp_data = {"demoData": self._format_composition_donut(items), "label": "status breakdown"}
                widgets.append({
                    "scenario": "composition",
                    "fixture": "donut_pie",
                    "relevance": 0.58,
                    "size": "normal",
                    "position": None,
                    "data_override": comp_data,
                })

        # Category-bar enrichment — OEE / health by equipment type
        if all_devices and "category-bar" not in used_scenarios and "category-bar" in relevant_enrichments:
            type_health = {}
            for dev in all_devices:
                t = dev.get("type", "other").replace("_", " ").title()
                if t not in type_health:
                    type_health[t] = []
                type_health[t].append(dev.get("health", 0))
            if type_health:
                bar_items = [{"category": t, "value": round(sum(h) / len(h))}
                             for t, h in type_health.items()]
                bar_data = self._format_category_bar(bar_items, "Health by Equipment Type")
                if isinstance(bar_data.get("demoData"), dict):
                    bar_data["demoData"]["label"] = "equipment availability"
                widgets.append({
                    "scenario": "category-bar",
                    "fixture": "oee-by-machine",
                    "relevance": 0.56,
                    "size": "normal",
                    "position": None,
                    "data_override": bar_data,
                })

        # Timeline enrichment — machine state or shift schedule
        if (all_maintenance or all_work_orders) and "timeline" not in used_scenarios and "timeline" in relevant_enrichments:
            items = all_work_orders[:5] or all_maintenance[:5]
            tl_data = self._format_timeline(items)
            if isinstance(tl_data, dict) and isinstance(tl_data.get("demoData"), dict):
                tl_data["demoData"]["label"] = "machine state history" if all_maintenance else "work order timeline"
            widgets.append({
                "scenario": "timeline",
                "fixture": "machine-state-timeline",
                "relevance": 0.55,
                "size": "normal",
                "position": None,
                "data_override": tl_data,
            })

        # EventLogStream enrichment — tabular or grouped view
        if (all_maintenance or all_shift_logs or all_work_orders) and "eventlogstream" not in used_scenarios and "eventlogstream" in relevant_enrichments:
            items = all_maintenance[:5] or all_shift_logs[:5] or all_work_orders[:5]
            log_type = "maintenance" if all_maintenance else ("shift_log" if all_shift_logs else "work_order")
            el_data = self._format_eventlog(items, log_type)
            if isinstance(el_data, dict) and isinstance(el_data.get("demoData"), dict):
                el_data["demoData"]["label"] = "maintenance log" if all_maintenance else "shift log"
            widgets.append({
                "scenario": "eventlogstream",
                "fixture": "tabular-log-view",
                "relevance": 0.53,
                "size": "normal",
                "position": None,
                "data_override": el_data,
            })

        # Matrix-heatmap enrichment — status matrix for device health
        if all_devices and len(all_devices) >= 3 and "matrix-heatmap" not in used_scenarios and "matrix-heatmap" in relevant_enrichments:
            hm_data = self._format_matrix_heatmap(all_devices)
            if isinstance(hm_data, dict) and isinstance(hm_data.get("demoData"), dict):
                hm_data["demoData"]["label"] = "equipment health status"
            widgets.append({
                "scenario": "matrix-heatmap",
                "fixture": "status-matrix",
                "relevance": 0.52,
                "size": "normal",
                "position": None,
                "data_override": hm_data,
            })

        # Trends-cumulative enrichment — running total energy
        if energy_timeseries and "trends-cumulative" not in used_scenarios and "trends-cumulative" in relevant_enrichments:
            cum_data = self._format_trends_cumulative(energy_timeseries)
            if cum_data:
                if isinstance(cum_data.get("demoData"), dict):
                    cum_data["demoData"]["label"] = "cumulative energy consumption"
                widgets.append({
                    "scenario": "trends-cumulative",
                    "fixture": "energy-consumption",
                    "relevance": 0.50,
                    "size": "normal",
                    "position": None,
                    "data_override": cum_data,
                })

        # Comparison enrichment — deviation bar for device health
        if all_devices and len(all_devices) >= 2 and "comparison" not in used_scenarios and "comparison" in relevant_enrichments:
            d1, d2 = all_devices[0], all_devices[1]
            comp_data = self._format_comparison(
                label="Equipment Health", unit="%",
                label_a=d1.get("name", "A"), value_a=d1.get("health", 0),
                label_b=d2.get("name", "B"), value_b=d2.get("health", 0))
            if isinstance(comp_data.get("demoData"), dict):
                comp_data["demoData"]["label"] = "equipment deviation"
            widgets.append({
                "scenario": "comparison",
                "fixture": "delta_bar_visual-deviation-bar",
                "relevance": 0.48,
                "size": "normal",
                "position": None,
                "data_override": comp_data,
            })

        # Flow-sankey enrichment — energy flow if enough data
        if all_devices and len(all_devices) >= 3 and "flow-sankey" not in used_scenarios and "flow-sankey" in relevant_enrichments:
            sankey_data = self._format_flow_sankey(all_devices, energy_timeseries)
            sankey_data["_query_context"] = query_lower
            if sankey_data.get("demoData", {}).get("nodes"):
                widgets.append({
                    "scenario": "flow-sankey",
                    "fixture": "flow_sankey_standard-classic-left-to-right-sankey",
                    "relevance": 0.46,
                    "size": "expanded",
                    "position": None,
                    "data_override": sankey_data,
                })

        # ── Enrichment cap: max 5 enrichment widgets ──
        primary_widgets = [w for w in widgets if w["relevance"] >= 0.75]
        enrichment_widgets = [w for w in widgets if w["relevance"] < 0.75]
        if len(enrichment_widgets) > 5:
            enrichment_widgets.sort(key=lambda w: w["relevance"], reverse=True)
            keep_ids = {id(w) for w in enrichment_widgets[:5]}
            widgets = [w for w in widgets if w["relevance"] >= 0.75 or id(w) in keep_ids]

        # ══════════════════════════════════════════════════════════════
        # CROSS-DOMAIN WIDGETS — appended regardless of primary query
        # ══════════════════════════════════════════════════════════════

        # ── SUPPLY DOMAIN → supplychainglobe + distribution + KPIs ──
        if supply_data:
            inventory = supply_data.get("inventory", [])
            # Supply chain globe for logistics overview
            widgets.append({
                "scenario": "supplychainglobe",
                "fixture": "default-render",
                "relevance": 0.68,
                "size": "expanded",
                "position": "bottom",
                "data_override": supply_data,
            })
            if inventory:
                dist_items = [{"name": item.get("item", ""), "value": item.get("quantity", 0)} for item in inventory]
                widgets.append({
                    "scenario": "distribution",
                    "fixture": "dist_load_by_asset-horizontal-bar",
                    "relevance": 0.62,
                    "size": "expanded",
                    "position": "bottom",
                    "data_override": self._format_distribution(dist_items, "Inventory Levels",
                                                                "DIST_LOAD_BY_ASSET"),
                })
            widgets.append({
                **self._format_kpi("Pending Orders", str(supply_data.get("pending_orders", 0)), "", "normal"),
                "relevance": 0.55,
                "size": "compact",
                "position": None,
            })
            low_stock = sum(1 for i in inventory if i.get("status") == "low")
            if low_stock > 0:
                widgets.append({
                    **self._format_kpi("Low Stock Items", str(low_stock), "",
                                       "warning", "kpi_alert-warning-state"),
                    "relevance": 0.58,
                    "size": "compact",
                    "position": None,
                })

        # ── PEOPLE DOMAIN → peoplehexgrid + peoplenetwork + peopleview + KPIs ──
        if people_data:
            on_shift = people_data.get("on_shift", 0)
            absent = people_data.get("absent", 0)

            # Hex grid for team visualization
            widgets.append({
                "scenario": "peoplehexgrid",
                "fixture": "default-render",
                "relevance": 0.68,
                "size": "expanded",
                "position": "bottom",
                "data_override": self._format_peoplehexgrid(people_data),
            })
            # Network view for relationships
            widgets.append({
                "scenario": "peoplenetwork",
                "fixture": "default-render",
                "relevance": 0.62,
                "size": "expanded",
                "position": "bottom",
                "data_override": self._format_peoplenetwork(people_data),
            })
            # People directory
            widgets.append({
                "scenario": "peopleview",
                "fixture": "default-render",
                "relevance": 0.58,
                "size": "expanded",
                "position": "bottom",
                "data_override": people_data,
            })
            widgets.append({
                **self._format_kpi("On Shift", str(on_shift), "staff", "normal"),
                "relevance": 0.55,
                "size": "compact",
                "position": None,
            })
            if absent > 0:
                widgets.append({
                    **self._format_kpi("Absent Today", str(absent), "", "warning" if absent > 5 else "normal",
                                       "kpi_alert-warning-state" if absent > 5 else "kpi_live-standard"),
                    "relevance": 0.52,
                    "size": "compact",
                    "position": None,
                })

        # ── KPI CAP: max 3 KPIs per layout (run BEFORE overall cap) ──
        # Sort first so we keep the highest-relevance KPIs
        widgets.sort(key=lambda w: w["relevance"], reverse=True)
        kpi_indices = [i for i, w in enumerate(widgets) if w["scenario"] == "kpi"]
        if len(kpi_indices) > 3:
            excess = kpi_indices[3:]
            excess_kpis = [widgets[i] for i in excess]
            for i in sorted(excess, reverse=True):
                widgets.pop(i)

            # Aggregate excess into a category-bar if we have data and no category-bar yet
            has_catbar = any(w["scenario"] == "category-bar" for w in widgets)
            if excess_kpis and not has_catbar:
                bar_items = []
                for ek in excess_kpis:
                    demo = (ek.get("data_override") or {}).get("demoData", {})
                    if isinstance(demo, dict):
                        label = demo.get("label", "Item")
                        val_str = str(demo.get("value", "0")).replace("%", "").replace(",", "")
                        try:
                            val = float(val_str)
                        except (ValueError, TypeError):
                            val = 0
                        bar_items.append({"category": label, "value": val})
                if bar_items:
                    widgets.append({
                        "scenario": "category-bar",
                        "fixture": "efficiency-deviation",
                        "relevance": 0.65,
                        "size": "expanded",
                        "position": None,
                        "data_override": self._format_category_bar(bar_items, "Additional Metrics"),
                    })

        # Cap at 10 widgets (after KPI cap, so enrichment widgets survive)
        widgets.sort(key=lambda w: w["relevance"], reverse=True)
        widgets = widgets[:10]

        # ── Fixture selection: use FixtureSelector for context-aware variety ──
        fixture_sel = FixtureSelector()
        for w in widgets:
            # Let the selector pick the best fixture based on data context + diversity
            w["fixture"] = fixture_sel.select(w["scenario"], w.get("data_override") or {})

        # Inject heightHint from SCENARIO_HEIGHT_HINTS (default: "medium")
        for w in widgets:
            if "heightHint" not in w:
                w["heightHint"] = SCENARIO_HEIGHT_HINTS.get(w["scenario"], "medium")

        # Size-heightHint coherence: prevent awkward narrow+tall widgets
        for w in widgets:
            hint = w.get("heightHint", "medium")
            size = w.get("size", "normal")
            if size == "hero":
                continue
            # x-tall widgets need at least 6 columns for proper aspect ratio
            if hint == "x-tall" and size in ("normal", "compact"):
                w["size"] = "expanded"
            # tall chart-type widgets with only 4 cols look cramped
            if hint == "tall" and size == "normal" and w.get("scenario") in (
                "composition", "category-bar", "flow-sankey", "matrix-heatmap",
                "comparison", "timeline", "eventlogstream",
            ):
                w["size"] = "expanded"

        logger.info(
            f"Layout result: {len(widgets)} widgets — "
            f"{[w['scenario'] for w in widgets]}"
        )

        return {
            "heading": heading,
            "widgets": widgets,
            "transitions": {},
        }


    def _update_context(self, intent: Intent, rag_results: list) -> dict:
        """Update conversation context based on current query."""
        return {
            "last_intent": intent.type,
            "last_domains": intent.domains,
            "last_query": intent.raw_text,
            "timestamp": time.time(),
        }

    def get_proactive_trigger(self, system_context: dict) -> Optional[str]:
        """
        Generate a proactive question based on system context.

        Called by Layer 1 when context is pushed from backend.
        """
        # Check for alerts
        if system_context.get("active_alerts", 0) > 0:
            return f"I noticed there are {system_context['active_alerts']} active alerts. Would you like me to summarize them?"

        # Check for shift start
        if system_context.get("shift_start"):
            return "Good morning! A new shift is starting. Would you like a status update on production?"

        # Check for anomalies
        if system_context.get("anomalies"):
            return "I've detected some unusual patterns in the data. Want me to explain what I'm seeing?"

        return None


# Singleton instance
_orchestrator = None


def get_orchestrator() -> Layer2Orchestrator:
    """Get or create the orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Layer2Orchestrator()
    return _orchestrator
