#!/usr/bin/env python3
"""
Claude-as-Teacher: Capture real thinking traces for distillation.

Runs prompts through the Claude Code CLI with the SAME system prompt
that the production model (Ollama) uses. This is critical — the training
data must match the inference-time prompt.

Two capture modes:
  1. Thinking traces: Mined from session transcripts at ~/.claude/projects/
     These contain Claude's REAL internal reasoning (thinking blocks).
     Training format: <think>reasoning</think>answer

  2. Direct answer: When no thinking blocks are found (simple queries),
     the answer alone is used as SFT target.
     Training format: answer (no <think> tags)

Both modes produce valid SFT data. The model learns WHEN to think
(complex queries) and when to answer directly (simple queries).

The system prompt comes from widget_selector.py (SYSTEM_PROMPT + FAST_SELECT_PROMPT),
which is the same prompt Ollama receives during inference.
"""

import subprocess
import json
import glob
import os
import sys
import time
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# PRODUCTION SYSTEM PROMPT — shared with widget_selector.py
# This MUST match what Ollama sees at inference time.
# ═══════════════════════════════════════════════════════════════════════════════

def get_production_system_prompt() -> str:
    """Load the system prompt from widget_selector.py.

    Tries to import from the backend module first. If running standalone
    (e.g., from claude-rl-agent/src/), falls back to reading the file directly.
    """
    # Try direct import (works when backend is on sys.path)
    try:
        from layer2.widget_selector import SYSTEM_PROMPT
        return SYSTEM_PROMPT
    except ImportError:
        pass

    # Fallback: read from file
    widget_selector_path = (
        Path(__file__).resolve().parent.parent.parent
        / "backend" / "layer2" / "widget_selector.py"
    )
    if widget_selector_path.exists():
        content = widget_selector_path.read_text()
        # Extract SYSTEM_PROMPT between triple quotes
        marker = 'SYSTEM_PROMPT = """'
        start = content.find(marker)
        if start != -1:
            start += len(marker)
            end = content.find('"""', start)
            if end != -1:
                return content[start:end].strip()

    # Last resort: hardcoded copy
    return (
        "You are a dashboard widget selector for an industrial operations command center.\n"
        "You select the best combination of widgets to build an informative dashboard for the user's query.\n"
        "You MUST respond with ONLY valid JSON, no explanation or markdown."
    )


def get_production_select_prompt() -> Optional[str]:
    """Load the FAST_SELECT_PROMPT template from widget_selector.py.

    Returns the raw template string with {placeholders} intact,
    or None if not available.
    """
    try:
        from layer2.widget_selector import FAST_SELECT_PROMPT
        return FAST_SELECT_PROMPT
    except ImportError:
        pass

    widget_selector_path = (
        Path(__file__).resolve().parent.parent.parent
        / "backend" / "layer2" / "widget_selector.py"
    )
    if widget_selector_path.exists():
        content = widget_selector_path.read_text()
        marker = "FAST_SELECT_PROMPT = '''"
        start = content.find(marker)
        if start != -1:
            start += len(marker)
            end = content.find("'''", start)
            if end != -1:
                return content[start:end]

    return None


def build_full_prompt(
    query: str,
    intent_type: str = "query",
    domains: str = "industrial",
    entities: str = "none",
    entity_hint: str = "",
    primary_char: str = "general query",
    secondary_chars: str = "none",
    data_summary: str = "No pre-fetched data available.",
    widget_count: int = 8,
    use_evolver: bool = False,
) -> tuple:
    """Build the EXACT same prompt that Ollama receives in production.

    This mirrors widget_selector.build_production_prompt() so Claude gets
    identical input during trace capture. Prompt parity is CRITICAL —
    the training data is only valid if Claude sees the same prompt as Ollama.

    Tries to import build_production_prompt() directly (works when backend
    is on sys.path). Falls back to building it locally from file reads.

    Returns:
        (system_prompt, formatted_user_prompt, prompt_version)
    """
    # Best path: import directly from backend
    try:
        from layer2.widget_selector import build_production_prompt
        return build_production_prompt(
            query=query, intent_type=intent_type, domains=domains,
            entities=entities, entity_hint=entity_hint,
            primary_char=primary_char, secondary_chars=secondary_chars,
            data_summary=data_summary, widget_count=widget_count,
            use_evolver=use_evolver,
        )
    except ImportError:
        pass

    # Fallback: build it locally from file reads
    system_prompt = get_production_system_prompt()
    prompt_template = get_production_select_prompt()
    if not prompt_template:
        logger.warning("Could not load FAST_SELECT_PROMPT, using raw query")
        return system_prompt, query, "raw"

    # Load the widget catalog text
    catalog_text = _get_catalog_text_standalone()

    user_prompt = prompt_template.format(
        catalog=catalog_text,
        query=query,
        intent_type=intent_type,
        domains=domains,
        entities=entities,
        primary_char=primary_char,
        secondary_chars=secondary_chars,
        data_summary=data_summary,
        widget_count=widget_count,
        entity_hint=entity_hint,
    )

    return system_prompt, user_prompt, "static"


def _get_catalog_text_standalone() -> str:
    """Load widget catalog text when running standalone (no backend imports).

    Mirrors widget_catalog.get_catalog_prompt_text().
    """
    try:
        from layer2.widget_catalog import get_catalog_prompt_text
        return get_catalog_prompt_text()
    except ImportError:
        pass

    # Fallback: read and parse widget_catalog.py
    catalog_path = (
        Path(__file__).resolve().parent.parent.parent
        / "backend" / "layer2" / "widget_catalog.py"
    )
    if not catalog_path.exists():
        return "(widget catalog not available)"

    try:
        # Execute widget_catalog.py in isolated namespace to get WIDGET_CATALOG
        namespace = {}
        exec(compile(catalog_path.read_text(), str(catalog_path), "exec"), namespace)
        widget_catalog = namespace.get("WIDGET_CATALOG", [])
        lines = []
        for w in widget_catalog:
            if w.get("scenario") in ("helpview", "pulseview"):
                continue
            sizes = ", ".join(w.get("sizes", []))
            good = ", ".join(w.get("good_for", []))
            lines.append(
                f'- {w["scenario"]} (height={w["height_units"]}, sizes=[{sizes}]): '
                f'{w["description"]} Good for: {good}'
            )
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"Failed to parse widget_catalog.py: {e}")
        return "(widget catalog parse error)"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClaudeThinking:
    """Claude's captured reasoning process — the training data."""
    thinking_text: str          # The raw thinking chain (THE BRAIN)
    answer_text: str            # The final answer
    stop_reason: str            # "success", "error", "timeout"
    input_tokens: int
    output_tokens: int
    model: str
    time_s: float
    has_thinking: bool = False  # Whether real thinking was captured

    def to_sft_target(self) -> str:
        """Format as SFT training target.

        Two modes:
          - With thinking: <think>reasoning</think>answer
          - Direct answer: just the answer (model learns not all queries need thinking)
        """
        if self.thinking_text:
            return f"<think>\n{self.thinking_text}\n</think>\n{self.answer_text}"
        return self.answer_text

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ClaudeThinking":
        # Handle old format without has_thinking field
        if "has_thinking" not in d:
            d["has_thinking"] = bool(d.get("thinking_text", ""))
        return cls(**d)


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSCRIPT MINING — extract real thinking from Claude Code session files
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_thinking_from_transcript(transcript_path: str) -> tuple:
    """Read a session transcript and extract thinking + text blocks.

    Claude Code stores full API responses in JSONL transcripts at
    ~/.claude/projects/<project>/session_id.jsonl. Each assistant
    message has content blocks that may include:
      - {"type": "thinking", "thinking": "..."} — Claude's internal reasoning
      - {"type": "text", "text": "..."} — the visible answer

    Returns:
        (thinking_text, answer_text) — all thinking blocks joined, all text blocks joined
    """
    thinking_parts = []
    text_parts = []

    with open(transcript_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = data.get("message", {})
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "thinking":
                    thinking_parts.append(block["thinking"])
                elif block.get("type") == "text" and block.get("text"):
                    text_parts.append(block["text"])

    return "\n".join(thinking_parts), "\n".join(text_parts)


def _find_transcript(session_id: str) -> Optional[str]:
    """Find the session transcript file for a given session ID.

    Claude Code stores transcripts at:
    ~/.claude/projects/<project-slug>/session_id.jsonl
    """
    pattern = os.path.expanduser(f"~/.claude/projects/*/{session_id}.jsonl")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


# ═══════════════════════════════════════════════════════════════════════════════
# CLAUDE TEACHER — capture thinking + direct answers
# ═══════════════════════════════════════════════════════════════════════════════

class ClaudeTeacher:
    """Captures Claude's reasoning for distillation training.

    Uses the SAME system prompt as production (widget_selector.py).
    Runs prompts through Claude Code CLI, then mines thinking blocks
    from session transcripts.

    Produces two types of training data:
      1. Thinking traces: <think>real_reasoning</think>answer
      2. Direct answers: just the answer (when query is too simple for thinking)

    Both are valid SFT targets — the model learns when to think and when not to.

    Usage:
        # Build the same prompt Ollama gets, send it to Claude
        system, prompt, version = build_full_prompt("What is pump 001 vibration?")
        teacher = ClaudeTeacher()
        result = teacher.think(prompt, system_prompt=system)
    """

    def __init__(
        self,
        cli_path: str = "claude",
        model: str = "sonnet",
        timeout: int = 120,
        cwd: str = None,
    ):
        self.cli_path = cli_path
        self.model = model
        self.timeout = timeout
        # Run from a neutral dir to avoid workspace trust prompts
        self.cwd = cwd or "/tmp"

    def think(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> ClaudeThinking:
        """Send prompt to Claude CLI, mine thinking from session transcript.

        Args:
            prompt: The prompt to send (should be the full production prompt
                    built by build_full_prompt() for prompt parity).
            system_prompt: System prompt (passed via --append-system-prompt).

        Returns:
            ClaudeThinking with thinking + answer. has_thinking=True if
            thinking blocks were found in the session transcript.
        """
        start = time.time()

        cmd = [
            self.cli_path, "-p",
            "--output-format", "json",
            "--model", self.model,
            prompt,
        ]

        if system_prompt:
            cmd.extend(["--append-system-prompt", system_prompt])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.cwd,
            )
            elapsed = time.time() - start

            if result.returncode != 0:
                logger.error(f"Claude CLI error (rc={result.returncode}): {result.stderr[:200]}")
                return ClaudeThinking(
                    thinking_text="",
                    answer_text=f"[CLI Error: rc={result.returncode}]",
                    stop_reason="error",
                    input_tokens=0, output_tokens=0,
                    model=self.model, time_s=elapsed,
                    has_thinking=False,
                )

            # Parse the JSON result
            data = json.loads(result.stdout)
            answer_text = data.get("result", "")
            stop_reason = data.get("subtype", "success")
            session_id = data.get("session_id", "")

            # Extract token usage
            usage = data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            # Detect actual model used
            model_usage = data.get("modelUsage", {})
            actual_model = list(model_usage.keys())[0] if model_usage else self.model

            # Mine the thinking blocks from the session transcript
            thinking_text = ""
            has_thinking = False
            if session_id:
                transcript = _find_transcript(session_id)
                if transcript:
                    thinking_text, transcript_answer = _extract_thinking_from_transcript(transcript)
                    has_thinking = bool(thinking_text and len(thinking_text) > 20)
                    # Use transcript answer if available (more complete)
                    if transcript_answer:
                        answer_text = transcript_answer
                    logger.debug(f"Transcript: {transcript} ({len(thinking_text)} chars thinking)")
                else:
                    logger.warning(f"No transcript found for session {session_id}")

            result_obj = ClaudeThinking(
                thinking_text=thinking_text,
                answer_text=answer_text,
                stop_reason=stop_reason,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=actual_model,
                time_s=elapsed,
                has_thinking=has_thinking,
            )

            mode = "THINKING" if has_thinking else "DIRECT"
            logger.info(
                f"[{mode}] Claude {elapsed:.1f}s | "
                f"thinking: {len(thinking_text)} chars | "
                f"answer: {len(answer_text)} chars | "
                f"tokens: {input_tokens}in/{output_tokens}out"
            )

            return result_obj

        except subprocess.TimeoutExpired:
            logger.error(f"Claude CLI timeout after {self.timeout}s")
            return ClaudeThinking(
                thinking_text="", answer_text="[ERROR: Timeout]",
                stop_reason="timeout", input_tokens=0, output_tokens=0,
                model=self.model, time_s=float(self.timeout),
                has_thinking=False,
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse CLI JSON output: {e}")
            return ClaudeThinking(
                thinking_text="", answer_text=f"[JSON Parse Error: {e}]",
                stop_reason="error", input_tokens=0, output_tokens=0,
                model=self.model, time_s=time.time() - start,
                has_thinking=False,
            )
        except Exception as e:
            logger.error(f"Claude CLI error: {e}")
            return ClaudeThinking(
                thinking_text="", answer_text=f"[ERROR: {e}]",
                stop_reason="error", input_tokens=0, output_tokens=0,
                model=self.model, time_s=time.time() - start,
                has_thinking=False,
            )

