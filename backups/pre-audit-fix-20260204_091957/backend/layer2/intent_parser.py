"""
LLM-based Intent Parser for Pipeline v2.

Replaces the regex-based _parse_intent() in orchestrator.py with a small LLM
(8B) that classifies user messages into structured intents with entity extraction,
characteristic detection, and urgency assessment.

Falls back to regex-based parsing if the LLM is unavailable.
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from layer2.rag_pipeline import get_rag_pipeline

logger = logging.getLogger(__name__)

# ── Intent types ──

INTENT_TYPES = [
    "query",              # asking for information, status, data, comparisons, trends
    "action_reminder",    # set a reminder, schedule something, alert me later
    "action_message",     # send a message, notify someone, contact a team
    "action_control",     # start/stop equipment, set parameters, adjust values
    "action_task",        # create work order, assign task, update ticket
    "conversation",       # greeting, thanks, small talk, meta-questions
    "out_of_scope",       # not related to industrial operations
]

# ── Characteristics that the LLM should detect ──

CHARACTERISTICS = [
    "comparison",       # comparing two or more entities
    "trend",            # over time, historical, time series
    "distribution",     # breakdown, composition, share, proportion
    "maintenance",      # repairs, inspections, service records
    "shift",            # shift handover, supervisor, night/morning
    "work_orders",      # tasks, tickets, pending, overdue
    "energy",           # power, load, consumption, kWh, voltage
    "health_status",    # condition, status, overview, dashboard
    "flow_sankey",      # energy balance, losses, source-to-destination
    "cumulative",       # running total, accumulated, daily total
    "multi_source",     # EB vs DG vs solar, by source, phases
    "power_quality",    # harmonic, THD, power factor, sag, swell
    "hvac",             # AHU, chiller, cooling, comfort, zone temp
    "ups_dg",           # UPS, battery, diesel generator, backup power
    "top_consumers",    # biggest load, ranking, worst performers
    "alerts",           # alarms, warnings, critical, threshold breach
    "people",           # employees, attendance, leave, scheduling
    "supply_chain",     # inventory, vendors, procurement, shipments
]

# ── System prompt for the LLM ──

SYSTEM_PROMPT = """You are an intent classifier for an industrial operations command center.
You classify user messages and extract structured information.
You MUST respond with ONLY valid JSON, no explanation or markdown."""

PARSE_PROMPT_TEMPLATE = """Classify this user message for an industrial command center.

Intent types:
- "query": asking for information, status, data, comparisons, trends
- "action_reminder": set a reminder, schedule something
- "action_message": send a message, notify someone
- "action_control": start/stop equipment, set parameters
- "action_task": create work order, assign task
- "conversation": greeting, thanks, small talk, meta-questions
- "out_of_scope": not related to industrial operations

Characteristics (select ALL that apply from this list):
comparison, trend, distribution, maintenance, shift, work_orders, energy,
health_status, flow_sankey, cumulative, multi_source, power_quality,
hvac, ups_dg, top_consumers, alerts, people, supply_chain

Domains (select all that apply):
industrial, supply, people, tasks, alerts

Respond with this JSON structure:
{{
  "type": "<intent_type>",
  "domains": ["<domain1>", ...],
  "entities": {{
    "devices": ["<device name or ID>", ...],
    "numbers": ["<number>", ...],
    "time": ["<time reference>", ...]
  }},
  "parameters": {{}},
  "urgency": "low|normal|high|critical",
  "primary_characteristic": "<main characteristic or null>",
  "secondary_characteristics": ["<char1>", ...],
  "confidence": 0.0
}}

User message: "{transcript}"

JSON:"""


@dataclass
class ParsedIntent:
    """Structured intent from LLM or regex parsing."""
    type: str = "query"
    domains: list = field(default_factory=list)
    entities: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    urgency: str = "normal"
    primary_characteristic: Optional[str] = None
    secondary_characteristics: list = field(default_factory=list)
    confidence: float = 0.0
    raw_text: str = ""
    parse_method: str = "llm"  # "llm" or "regex"


class IntentParser:
    """LLM-based intent parser with regex fallback."""

    def __init__(self):
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            self._pipeline = get_rag_pipeline()
        return self._pipeline

    def parse(self, transcript: str) -> ParsedIntent:
        """Parse intent from transcript using 8B LLM, with regex fallback."""
        # Try LLM first
        try:
            result = self._parse_with_llm(transcript)
            if result is not None:
                return result
        except Exception as e:
            logger.warning(f"LLM intent parsing failed: {e}")

        # Fallback to regex
        logger.info("Falling back to regex intent parsing")
        return self._parse_with_regex(transcript)

    def _parse_with_llm(self, transcript: str) -> Optional[ParsedIntent]:
        """Parse intent using the fast LLM model."""
        llm = self.pipeline.llm_fast
        prompt = PARSE_PROMPT_TEMPLATE.format(transcript=transcript)

        data = llm.generate_json(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=512,
        )

        if data is None:
            return None

        # Validate and normalize
        intent_type = data.get("type", "query")
        if intent_type not in INTENT_TYPES:
            intent_type = "query"

        primary_char = data.get("primary_characteristic")
        if primary_char and primary_char not in CHARACTERISTICS:
            primary_char = None

        secondary = [c for c in data.get("secondary_characteristics", []) if c in CHARACTERISTICS]

        domains = data.get("domains", [])
        valid_domains = {"industrial", "supply", "people", "tasks", "alerts"}
        domains = [d for d in domains if d in valid_domains]

        entities = data.get("entities", {})
        # Normalize entity structure
        normalized_entities = {}
        if entities.get("devices"):
            normalized_entities["devices"] = entities["devices"]
        if entities.get("numbers"):
            normalized_entities["numbers"] = entities["numbers"]
        if entities.get("time"):
            normalized_entities["time"] = entities["time"]

        return ParsedIntent(
            type=intent_type,
            domains=domains,
            entities=normalized_entities,
            parameters=data.get("parameters", {}),
            urgency=data.get("urgency", "normal"),
            primary_characteristic=primary_char,
            secondary_characteristics=secondary,
            confidence=min(1.0, max(0.0, data.get("confidence", 0.8))),
            raw_text=transcript,
            parse_method="llm",
        )

    # ── Regex fallback (mirrors old orchestrator._parse_intent logic) ──

    def _parse_with_regex(self, transcript: str) -> ParsedIntent:
        """Regex-based intent parsing — fallback when LLM is unavailable."""
        text = transcript.lower()

        intent_type = self._detect_intent_type_regex(text)
        domains = self._detect_domains_regex(text)
        entities = self._extract_entities_regex(text)

        # Drill-down pattern
        if re.search(r"tell me more about\s+", text) and not domains:
            domains = ["industrial"]
            intent_type = "query"

        # Scope guard
        if intent_type not in ("greeting", "conversation") and not domains:
            if self._is_conversation_regex(text):
                intent_type = "conversation"
            else:
                intent_type = "out_of_scope"

        # Detect characteristics via regex
        primary, secondary = self._detect_characteristics_regex(text)

        confidence = 0.6  # lower confidence for regex
        if domains:
            confidence = min(1.0, len(domains) * 0.25 + (0.3 if entities else 0.15))

        return ParsedIntent(
            type=intent_type,
            domains=domains,
            entities=entities,
            urgency="normal",
            primary_characteristic=primary,
            secondary_characteristics=secondary,
            confidence=confidence,
            raw_text=transcript,
            parse_method="regex",
        )

    def _detect_intent_type_regex(self, text: str) -> str:
        greeting_pats = [r"\b(hello|hi|hey|good morning|good afternoon|good evening)\b"]
        action_pats = [r"\b(start|stop|turn|set|adjust|change|update|create|delete|add|remove)\b"]
        query_pats = [r"\b(what|what's|whats|how|how's|hows|show|tell|get|check|status|current)\b", r"\?$"]

        # Action subtypes
        reminder_pats = [r"\b(remind|reminder|schedule|alert me|notify me at|wake me)\b"]
        message_pats = [r"\b(send|message|tell .+ that|notify .+ about|email|sms)\b"]
        control_pats = [r"\b(start|stop|turn on|turn off|set .*to|adjust|ramp|increase|decrease)\b"]
        task_pats = [r"\b(create .* order|assign|open ticket|raise .* request|work order)\b"]

        is_greeting = any(re.search(p, text) for p in greeting_pats)
        is_action = any(re.search(p, text) for p in action_pats)
        is_query = any(re.search(p, text) for p in query_pats)

        if any(re.search(p, text) for p in reminder_pats):
            return "action_reminder"
        if any(re.search(p, text) for p in task_pats):
            return "action_task"
        if any(re.search(p, text) for p in message_pats):
            return "action_message"
        if any(re.search(p, text) for p in control_pats) and is_action:
            return "action_control"
        if is_action:
            return "action_control"
        if is_query:
            return "query"
        if is_greeting:
            return "greeting"
        return "query"

    def _detect_domains_regex(self, text: str) -> list:
        DOMAIN_KEYWORDS = {
            "industrial": [
                "pump", "motor", "temperature", "pressure", "voltage", "current",
                "power", "energy", "device", "sensor", "meter", "transformer",
                "generator", "panel", "ups", "chiller", "ahu", "cooling",
                "compressor", "hvac", "consumption", "load", "kva", "kw",
                "harmonic", "thd", "power factor", "water", "nitrogen",
            ],
            "supply": [
                "inventory", "stock", "supplier", "vendor", "purchase",
                "procurement", "shipment", "delivery", "warehouse", "logistics",
            ],
            "people": [
                "employee", "worker", "staff", "team", "shift", "schedule",
                "attendance", "leave", "hr", "training", "safety", "overtime",
                "technician", "operator", "supervisor",
            ],
            "tasks": [
                "task", "project", "milestone", "deadline", "work order",
                "ticket", "issue", "priority", "pending", "overdue", "backlog",
            ],
            "alerts": [
                "alert", "alarm", "warning", "critical", "urgent",
                "notification", "fault", "failure", "anomaly", "threshold",
                "trip", "tripped",
            ],
        }
        detected = []
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                detected.append(domain)
        return detected

    def _extract_entities_regex(self, text: str) -> dict:
        entities = {}
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
        if numbers:
            entities["numbers"] = numbers
        device_refs = re.findall(
            r'\b(pump|motor|sensor|device|transformer|generator|chiller|compressor)\s*(\d+)\b',
            text,
        )
        if device_refs:
            entities["devices"] = [f"{d[0]}_{d[1]}" for d in device_refs]
        time_refs = re.findall(r'\b(today|yesterday|last\s+\w+|this\s+\w+|past\s+\d+\s+\w+)\b', text)
        if time_refs:
            entities["time"] = time_refs
        return entities

    def _is_conversation_regex(self, text: str) -> bool:
        patterns = [
            r"\b(how are you|how're you|what's up|whats up)\b",
            r"\b(thank you|thanks|appreciate)\b",
            r"\b(what can you do|what do you do|who are you)\b",
            r"\b(bye|goodbye|good night|see you)\b",
            r"\b(nice|awesome|great|cool|ok|okay|got it)\b",
            r"\b(tell me a joke|are you a robot|are you ai)\b",
        ]
        return any(re.search(p, text) for p in patterns)

    def _detect_characteristics_regex(self, text: str) -> tuple[Optional[str], list]:
        """Detect primary and secondary characteristics via regex."""
        char_patterns = {
            "comparison": r'\b(?:compar\w*|versus|vs\.?|difference|between)\b',
            "trend": r'\b(?:trend|graph|chart|over time|history|historical|last \d+|past \d+)\b',
            "distribution": r'\b(?:distribut\w*|breakdown|composition|split|share|proportion|pie|donut)\b',
            "maintenance": r'\b(?:maintenan\w*|repair|service|inspection|overhaul|parts|replaced)\b',
            "shift": r'\b(?:shift|handover|supervisor|last night|morning shift|evening shift|night shift)\b',
            "work_orders": r'\b(?:work order|task|pending|overdue|assigned|open ticket)\b',
            "energy": r'\b(?:energy|power|load|consumption|kwh|kw|voltage|current|electrical)\b',
            "health_status": r'\b(?:health|condition|status|overview|dashboard|summary)\b',
            "flow_sankey": r'\b(?:flow|sankey|where does|goes to|feeds|source.?to|losses|energy balance)\b',
            "cumulative": r'\b(?:cumulative|total today|accumulated|running total|daily total|how much.*so far)\b',
            "multi_source": r'\b(?:eb vs|dg vs|solar vs|grid vs|sources|by source|phase|phases|multi.?meter)\b',
            "power_quality": r'\b(?:power quality|harmonic|thd|power factor|sag|swell|dip|voltage dip)\b',
            "hvac": r'\b(?:hvac|ahu|chiller|cooling|comfort|setpoint|zone temp|air handling)\b',
            "ups_dg": r'\b(?:ups|battery|runtime|dg|diesel|transfer|generator|backup power|amf)\b',
            "top_consumers": r'\b(?:top consumers?|biggest load|highest load|most energy|ranking|top \d+)\b',
            "alerts": r'\b(?:alert|alarm|warning|critical|fault|threshold|breach)\b',
        }

        matched = []
        for char, pattern in char_patterns.items():
            if re.search(pattern, text):
                matched.append(char)

        if not matched:
            return None, []

        primary = matched[0]
        secondary = matched[1:]
        return primary, secondary
