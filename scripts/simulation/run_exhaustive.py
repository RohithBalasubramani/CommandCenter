#!/usr/bin/env python3
"""
Exhaustive Widget Test Data Generator

For each question in the bank:
  1. Run through the real pipeline → get natural widget selection
  2. For each natural widget, clone it across ALL fixture variants of that scenario
  3. For each missing scenario, create a synthetic widget and clone across all fixtures
  4. Store everything in a structured JSON for the rating UI

Usage:
    python run_exhaustive.py                              # All questions
    python run_exhaustive.py --category energy             # One category
    python run_exhaustive.py --question q001               # One question
    python run_exhaustive.py --natural-only                # Skip forced scenarios
    python run_exhaustive.py --parallel 4                  # Concurrent pipeline calls
    python run_exhaustive.py --resume <run_id>             # Resume partial run
"""

import argparse
import copy
import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: 'requests' library required. Install with: pip install requests")
    sys.exit(1)

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
QUESTION_BANK = SCRIPT_DIR / "question_bank_exhaustive.json"
RESULTS_DIR = SCRIPT_DIR / "results"
FRONTEND_PUBLIC = PROJECT_ROOT / "frontend" / "public" / "simulation"

DEFAULT_BACKEND = "http://localhost:8100"
ORCHESTRATE_ENDPOINT = "/api/layer2/orchestrate/"
HEALTH_ENDPOINT = "/api/layer2/rag/industrial/health/"


# ── Fixture variant catalog (mirrors llm_fixture_selector.py) ──

FIXTURE_DESCRIPTIONS = {
    "kpi": {
        "kpi_alert-critical-state": "Red/urgent KPI card with pulsing alert",
        "kpi_alert-warning-state": "Amber warning KPI card",
        "kpi_status-offline": "Greyed-out offline indicator",
        "kpi_live-high-contrast": "High-contrast live value card",
        "kpi_status-badge": "Status badge showing operational state",
        "kpi_lifecycle-dark-mode-gauge": "Circular gauge dial",
        "kpi_accumulated-daily-total": "Counter-style daily total",
        "kpi_lifecycle-progress-bar": "Progress bar for lifecycle",
        "kpi_live-standard": "Default numeric display",
    },
    "alerts": {
        "modal-ups-battery-critical": "Full modal alert for critical situations",
        "toast-power-factor-critical-low": "Toast notification for warnings",
        "card-dg-02-started-successfully": "Success card for resolved events",
        "badge-ahu-01-high-temperature": "Badge-style alert for info notices",
        "banner-energy-peak-threshold-exceeded": "Banner alert for peak events",
    },
    "trend": {
        "trend_alert_context-line-threshold": "Line chart with threshold markers",
        "trend_live-area": "Filled area chart",
        "trend_phased-rgb-phase-line": "RGB phase lines for 3-phase data",
        "trend_pattern-heatmap": "Heatmap pattern for cyclic patterns",
        "trend_standard-step-line": "Step line chart for discrete states",
        "trend_live-line": "Standard live line chart",
    },
    "trend-multi-line": {
        "power-sources-stacked": "Stacked area showing power sources",
        "main-lt-phases-current": "Multi-line for phase currents",
        "ups-health-dual-axis": "Dual-axis chart for UPS metrics",
        "power-quality": "Multi-line for power quality",
        "hvac-performance": "Multi-line for HVAC performance",
        "energy-demand": "Multi-line energy demand chart",
    },
    "trends-cumulative": {
        "instantaneous-power": "Live instantaneous power with cumulative",
        "source-mix": "Cumulative by source",
        "performance-vs-baseline": "Actual vs target comparison",
        "cost-vs-budget": "Cost tracking against budget",
        "batch-production": "Batch production cumulative tracking",
        "energy-consumption": "Standard energy consumption cumulative",
    },
    "distribution": {
        "dist_energy_source_share-donut": "Donut chart energy source share",
        "dist_energy_source_share-100-stacked-bar": "100% stacked bar breakdown",
        "dist_load_by_asset-horizontal-bar": "Horizontal bar load by asset",
        "dist_consumption_by_category-pie": "Pie chart consumption by category",
        "dist_consumption_by_shift-grouped-bar": "Grouped bar by shift",
        "dist_downtime_top_contributors-pareto-bar": "Pareto bar top contributors",
    },
    "comparison": {
        "waterfall_visual-loss-analysis": "Waterfall chart loss analysis",
        "grouped_bar_visual-phase-comparison": "Grouped bars phase comparison",
        "delta_bar_visual-deviation-bar": "Delta deviation bar",
        "small_multiples_visual-temp-grid": "Small multiples temperature grid",
        "composition_split_visual-load-type": "Split composition load type",
        "side_by_side_visual-plain-values": "Side-by-side plain values",
    },
    "composition": {
        "stacked_area": "Stacked area composition over time",
        "donut_pie": "Donut/pie share snapshot",
        "waterfall": "Waterfall gain/loss breakdown",
        "treemap": "Treemap hierarchical breakdown",
        "stacked_bar": "Stacked bar category composition",
    },
    "flow-sankey": {
        "flow_sankey_energy_balance-sankey-with-explicit-loss-branches-dropping-out": "Sankey with loss branches",
        "flow_sankey_multi_source-many-to-one-flow-diagram": "Many-to-one flow diagram",
        "flow_sankey_layered-multi-stage-hierarchical-flow": "Multi-stage hierarchical flow",
        "flow_sankey_time_sliced-sankey-with-time-scrubberplayer": "Time-sliced sankey with scrubber",
        "flow_sankey_standard-classic-left-to-right-sankey": "Classic left-to-right sankey",
    },
    "matrix-heatmap": {
        "correlation-matrix": "Correlation matrix",
        "calendar-heatmap": "Calendar heatmap",
        "status-matrix": "Status matrix equipment grid",
        "density-matrix": "Density matrix",
        "value-heatmap": "Value heatmap",
    },
    "timeline": {
        "machine-state-timeline": "Machine state bars over time",
        "multi-lane-shift-schedule": "Multi-lane shift schedule",
        "forensic-annotated-view": "Forensic annotated timeline",
        "log-density-burst-analysis": "Log density burst analysis",
        "linear-incident-timeline": "Linear incident timeline",
    },
    "eventlogstream": {
        "tabular-log-view": "Tabular log maintenance table",
        "correlation-stack": "Correlation stack grouped events",
        "grouped-by-asset": "Grouped by asset events",
        "compact-card-feed": "Compact card feed",
        "chronological-timeline": "Chronological timeline stream",
    },
    "category-bar": {
        "oee-by-machine": "OEE by machine bars",
        "downtime-duration": "Downtime duration ranking",
        "production-states": "Production states status bars",
        "shift-comparison": "Shift comparison bars",
        "efficiency-deviation": "Efficiency deviation bars",
    },
}

SINGLE_VARIANT_SCENARIOS = {
    "chatstream": "default-render",
    "edgedevicepanel": "default-render",
    "peoplehexgrid": "default-render",
    "peoplenetwork": "default-render",
    "peopleview": "default-render",
    "supplychainglobe": "default-render",
}

# ── Demo shapes from widget_schemas.py (for synthetic widgets) ──

DEMO_SHAPES = {
    "kpi": {
        "demoData": {
            "label": "Metric Name",
            "value": "42",
            "unit": "kW",
            "state": "normal",
        }
    },
    "alerts": {
        "demoData": {
            "id": "ALT-001",
            "title": "Parameter Alert",
            "message": "Value exceeded threshold",
            "severity": "warning",
            "category": "Equipment",
            "source": "Device",
            "state": "new",
            "evidence": {"label": "Value", "value": "95", "unit": "%", "trend": "up"},
            "threshold": "90%",
            "actions": [],
        }
    },
    "comparison": {
        "demoData": {
            "label": "Metric Comparison",
            "unit": "%",
            "labelA": "Entity A",
            "valueA": 92,
            "labelB": "Entity B",
            "valueB": 87,
            "delta": 5,
            "deltaPct": 5.7,
        }
    },
    "trend": {
        "demoData": {
            "label": "Metric Trend",
            "unit": "kW",
            "timeSeries": [
                {"time": "00:00", "value": 42}, {"time": "04:00", "value": 45},
                {"time": "08:00", "value": 65}, {"time": "12:00", "value": 78},
                {"time": "16:00", "value": 72}, {"time": "20:00", "value": 55},
                {"time": "24:00", "value": 48},
            ],
            "timeRange": "last_24h",
        }
    },
    "trend-multi-line": {
        "demoData": {
            "label": "Multi-Metric Trend",
            "unit": "kW",
            "series": [
                {"name": "Source A", "timeSeries": [
                    {"time": "00:00", "value": 30}, {"time": "06:00", "value": 45},
                    {"time": "12:00", "value": 60}, {"time": "18:00", "value": 40},
                ]},
                {"name": "Source B", "timeSeries": [
                    {"time": "00:00", "value": 20}, {"time": "06:00", "value": 35},
                    {"time": "12:00", "value": 50}, {"time": "18:00", "value": 30},
                ]},
            ],
        }
    },
    "trends-cumulative": {
        "config": {
            "title": "Cumulative Trend",
            "subtitle": "",
            "variant": "V1",
            "mode": "cumulative",
            "series": [{"id": "S1", "label": "Energy", "unit": "kWh", "color": "#2563eb"}],
        },
        "data": [
            {"x": "2026-01-31T00:00:00Z", "S1_raw": 10, "S1_cumulative": 10},
            {"x": "2026-01-31T04:00:00Z", "S1_raw": 15, "S1_cumulative": 25},
            {"x": "2026-01-31T08:00:00Z", "S1_raw": 25, "S1_cumulative": 50},
            {"x": "2026-01-31T12:00:00Z", "S1_raw": 30, "S1_cumulative": 80},
            {"x": "2026-01-31T16:00:00Z", "S1_raw": 20, "S1_cumulative": 100},
            {"x": "2026-01-31T20:00:00Z", "S1_raw": 15, "S1_cumulative": 115},
        ],
    },
    "distribution": {
        "demoData": {
            "total": 1000,
            "unit": "kW",
            "series": [
                {"label": "HVAC", "value": 350},
                {"label": "Production", "value": 400},
                {"label": "Lighting", "value": 150},
                {"label": "Utilities", "value": 100},
            ],
        }
    },
    "composition": {
        "demoData": {
            "label": "Load Composition",
            "unit": "kW",
            "categories": ["Morning", "Evening", "Night"],
            "series": [
                {"name": "HVAC", "values": [120, 180, 100]},
                {"name": "Production", "values": [200, 250, 80]},
                {"name": "Lighting", "values": [80, 90, 60]},
            ],
        }
    },
    "category-bar": {
        "demoData": {
            "label": "Equipment Ranking",
            "unit": "kW",
            "categories": ["Transformer 1", "Chiller 1", "Pump 1", "AHU 1", "Motor 1"],
            "values": [450, 320, 180, 150, 120],
        }
    },
    "timeline": {
        "demoData": {
            "title": "Event Timeline",
            "range": {"start": "2026-01-01", "end": "2026-01-31"},
            "events": [
                {"time": "2026-01-05", "label": "Preventive Maintenance", "type": "maintenance"},
                {"time": "2026-01-12", "label": "Alarm: High Temperature", "type": "alert"},
                {"time": "2026-01-20", "label": "Inspection Completed", "type": "inspection"},
                {"time": "2026-01-28", "label": "Work Order Created", "type": "work_order"},
            ],
        }
    },
    "flow-sankey": {
        "demoData": {
            "label": "Energy Flow",
            "unit": "kW",
            "nodes": [
                {"id": "grid", "label": "Grid"},
                {"id": "solar", "label": "Solar"},
                {"id": "main_panel", "label": "Main Panel"},
                {"id": "hvac", "label": "HVAC"},
                {"id": "production", "label": "Production"},
                {"id": "lighting", "label": "Lighting"},
                {"id": "loss", "label": "Loss"},
            ],
            "links": [
                {"source": "grid", "target": "main_panel", "value": 600},
                {"source": "solar", "target": "main_panel", "value": 200},
                {"source": "main_panel", "target": "hvac", "value": 300},
                {"source": "main_panel", "target": "production", "value": 350},
                {"source": "main_panel", "target": "lighting", "value": 100},
                {"source": "main_panel", "target": "loss", "value": 50},
            ],
        }
    },
    "matrix-heatmap": {
        "demoData": {
            "label": "Equipment Health Matrix",
            "rows": ["Transformer 1", "Transformer 2", "Chiller 1", "Pump 1", "AHU 1"],
            "cols": ["Health", "Load", "Temperature", "Vibration"],
            "dataset": [
                [0.72, 0.85, 0.65, 0.90],
                [0.91, 0.60, 0.80, 0.95],
                [0.85, 0.70, 0.55, 0.88],
                [0.68, 0.92, 0.75, 0.40],
                [0.95, 0.45, 0.82, 0.78],
            ],
        }
    },
    "eventlogstream": {
        "demoData": {
            "title": "Event Log",
            "events": [
                {"timestamp": "2026-01-31T10:00:00Z", "type": "warning", "message": "High temperature on Transformer 1", "source": "Transformer 1"},
                {"timestamp": "2026-01-31T09:30:00Z", "type": "info", "message": "Maintenance completed on Pump 1", "source": "Pump 1"},
                {"timestamp": "2026-01-31T09:00:00Z", "type": "error", "message": "Power factor below threshold", "source": "Main Panel"},
                {"timestamp": "2026-01-31T08:30:00Z", "type": "info", "message": "Shift handover completed", "source": "System"},
            ],
        }
    },
    "edgedevicepanel": {
        "demoData": {
            "device": {
                "name": "Transformer 1",
                "id": "TF-001",
                "type": "transformer",
                "status": "running",
                "health": 94,
            },
            "readings": [],
            "alerts": [],
        }
    },
}


# ── Helpers ──

def load_questions(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def filter_questions(questions, category=None, question_id=None):
    if question_id:
        return [q for q in questions if q["id"] == question_id]
    if category:
        return [q for q in questions if q["category"] == category]
    return questions


def check_backend(backend_url: str):
    try:
        resp = requests.get(f"{backend_url}{HEALTH_ENDPOINT}", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def make_entry_id(counter: list[int]) -> str:
    counter[0] += 1
    return f"e{counter[0]:05d}"


def extract_entities_from_question(question_text: str) -> list[str]:
    """Simple regex-based entity extraction from question text."""
    import re
    entities = []
    # Match patterns like "transformer 1", "AHU 2", "pump 1", "chiller 3"
    patterns = re.findall(
        r'\b(transformer|pump|motor|chiller|ahu|dg|ups|panel|meter|compressor|generator)\s*(\d+)\b',
        question_text.lower()
    )
    for name, num in patterns:
        entities.append(f"{name.title()} {num}")
    # Named equipment
    named = re.findall(r'\b(main panel|main lt|cell lt|amf|ht-\w+)\b', question_text.lower())
    entities.extend([n.title() for n in named])
    return entities or ["Equipment"]


def build_synthetic_widget(scenario: str, question: dict, size: str = "normal") -> dict:
    """Build a synthetic widget entry for a missing scenario using demo_shape."""
    demo = copy.deepcopy(DEMO_SHAPES.get(scenario, {"demoData": {}}))
    entities = extract_entities_from_question(question["question"])
    q_text = question["question"]

    # Customize labels from question context
    if "demoData" in demo:
        dd = demo["demoData"]
        if "label" in dd:
            dd["label"] = q_text[:60]
        # Replace generic entity names with question entities
        if "labelA" in dd and len(entities) >= 1:
            dd["labelA"] = entities[0]
        if "labelB" in dd and len(entities) >= 2:
            dd["labelB"] = entities[1] if len(entities) > 1 else "Other"
        if "source" in dd and entities:
            dd["source"] = entities[0]
        if "device" in dd and isinstance(dd["device"], dict) and entities:
            dd["device"]["name"] = entities[0]
        # Add query context for fixture selection
        dd["_query_context"] = q_text.lower()
    elif "config" in demo:
        # trends-cumulative format
        demo["config"]["title"] = q_text[:60]

    return {
        "scenario": scenario,
        "fixture": "",  # will be set per-variant
        "size": size,
        "data_override": demo,
    }


def run_pipeline_question(question: dict, backend_url: str, timeout: int = 120) -> dict | None:
    """Send question through the orchestrator pipeline, return raw response."""
    session_id = str(uuid.uuid4())
    try:
        resp = requests.post(
            f"{backend_url}{ORCHESTRATE_ENDPOINT}",
            json={"transcript": question["question"], "session_id": session_id, "context": {}},
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"  ERROR: HTTP {resp.status_code} for {question['id']}")
            return None
    except requests.Timeout:
        print(f"  TIMEOUT: {question['id']}")
        return None
    except requests.ConnectionError:
        print(f"  CONNECTION ERROR: {question['id']}")
        return None
    except Exception as e:
        print(f"  ERROR: {question['id']}: {e}")
        return None


def expand_question(
    question: dict,
    pipeline_response: dict | None,
    counter: list[int],
    natural_only: bool = False,
) -> list[dict]:
    """Expand a single question into all entry variants."""
    entries = []
    layout = (pipeline_response or {}).get("layout_json") or {}
    natural_widgets = layout.get("widgets", [])
    heading = layout.get("heading", "Dashboard")
    intent = (pipeline_response or {}).get("intent", {})
    voice = (pipeline_response or {}).get("voice_response", "")
    proc_ms = (pipeline_response or {}).get("processing_time_ms", 0)

    pipeline_meta = {
        "intent": intent,
        "voice_response": voice[:500],
        "processing_time_ms": proc_ms,
    }

    used_scenarios = set()

    # ── Mode 1: Natural widgets → expand to all fixture variants ──
    for i, widget in enumerate(natural_widgets):
        scenario = widget.get("scenario", "")
        fixture_used = widget.get("fixture", "")
        size = widget.get("size", "normal")
        data_override = widget.get("data_override", {})
        used_scenarios.add(scenario)

        variants = FIXTURE_DESCRIPTIONS.get(scenario, {})
        if not variants:
            # Single-variant scenario — store as-is
            entries.append({
                "entry_id": make_entry_id(counter),
                "question_id": question["id"],
                "question": question["question"],
                "category": question["category"],
                "scenario": scenario,
                "fixture": fixture_used or SINGLE_VARIANT_SCENARIOS.get(scenario, "default-render"),
                "size": size,
                "natural": True,
                "forced": False,
                "pipeline_selected_fixture": fixture_used,
                "widget_index": i,
                "data_override": data_override,
                "layout_context": {
                    "heading": heading,
                    "total_widgets": len(natural_widgets),
                    "position": i,
                },
                "pipeline_meta": pipeline_meta,
                "rating": None,
                "tags": [],
                "notes": "",
            })
        else:
            # Multi-variant — create one entry per fixture
            for fixture_slug in variants:
                entries.append({
                    "entry_id": make_entry_id(counter),
                    "question_id": question["id"],
                    "question": question["question"],
                    "category": question["category"],
                    "scenario": scenario,
                    "fixture": fixture_slug,
                    "size": size,
                    "natural": True,
                    "forced": False,
                    "pipeline_selected_fixture": fixture_used,
                    "widget_index": i,
                    "data_override": data_override,
                    "layout_context": {
                        "heading": heading,
                        "total_widgets": len(natural_widgets),
                        "position": i,
                    },
                    "pipeline_meta": pipeline_meta,
                    "rating": None,
                    "tags": [],
                    "notes": "",
                })

    # ── Mode 2: Force missing scenarios ──
    if not natural_only:
        all_multi = set(FIXTURE_DESCRIPTIONS.keys())
        missing = all_multi - used_scenarios

        # Also include force_scenarios from question bank
        force = set(question.get("force_scenarios", []))
        missing = missing | (force - used_scenarios)

        for scenario in sorted(missing):
            variants = FIXTURE_DESCRIPTIONS.get(scenario, {})
            if not variants:
                # Single-variant forced scenario
                synthetic = build_synthetic_widget(scenario, question)
                synthetic["fixture"] = SINGLE_VARIANT_SCENARIOS.get(scenario, "default-render")
                entries.append({
                    "entry_id": make_entry_id(counter),
                    "question_id": question["id"],
                    "question": question["question"],
                    "category": question["category"],
                    "scenario": scenario,
                    "fixture": synthetic["fixture"],
                    "size": "normal",
                    "natural": False,
                    "forced": True,
                    "pipeline_selected_fixture": None,
                    "widget_index": -1,
                    "data_override": synthetic["data_override"],
                    "layout_context": {
                        "heading": heading,
                        "total_widgets": len(natural_widgets),
                        "position": -1,
                    },
                    "pipeline_meta": pipeline_meta,
                    "rating": None,
                    "tags": [],
                    "notes": "",
                })
            else:
                synthetic = build_synthetic_widget(scenario, question)
                for fixture_slug in variants:
                    entries.append({
                        "entry_id": make_entry_id(counter),
                        "question_id": question["id"],
                        "question": question["question"],
                        "category": question["category"],
                        "scenario": scenario,
                        "fixture": fixture_slug,
                        "size": "normal",
                        "natural": False,
                        "forced": True,
                        "pipeline_selected_fixture": None,
                        "widget_index": -1,
                        "data_override": synthetic["data_override"],
                        "layout_context": {
                            "heading": heading,
                            "total_widgets": len(natural_widgets),
                            "position": -1,
                        },
                        "pipeline_meta": pipeline_meta,
                        "rating": None,
                        "tags": [],
                        "notes": "",
                    })

    return entries


def generate_summary(entries: list[dict], questions_run: int, failed: int) -> dict:
    """Summary statistics for the run."""
    by_scenario = {}
    by_category = {}
    natural_count = 0
    forced_count = 0

    for e in entries:
        s = e["scenario"]
        by_scenario[s] = by_scenario.get(s, 0) + 1
        c = e["category"]
        by_category[c] = by_category.get(c, 0) + 1
        if e["natural"]:
            natural_count += 1
        else:
            forced_count += 1

    return {
        "total_entries": len(entries),
        "natural_entries": natural_count,
        "forced_entries": forced_count,
        "questions_run": questions_run,
        "questions_failed": failed,
        "by_scenario": dict(sorted(by_scenario.items(), key=lambda x: -x[1])),
        "by_category": dict(sorted(by_category.items(), key=lambda x: -x[1])),
        "unique_scenarios": len(by_scenario),
        "unique_categories": len(by_category),
    }


def main():
    parser = argparse.ArgumentParser(description="Exhaustive Widget Test Data Generator")
    parser.add_argument("--category", type=str, help="Run only questions in this category")
    parser.add_argument("--question", type=str, help="Run a single question by ID")
    parser.add_argument("--natural-only", action="store_true", help="Skip forced scenario injection")
    parser.add_argument("--parallel", type=int, default=1, help="Parallel pipeline workers")
    parser.add_argument("--backend", type=str, default=DEFAULT_BACKEND, help="Backend URL")
    parser.add_argument("--timeout", type=int, default=120, help="Per-question timeout (sec)")
    parser.add_argument("--resume", type=str, help="Resume a partial run by ID")
    parser.add_argument("--dry-run", action="store_true", help="List questions without running")
    parser.add_argument("--question-bank", type=str, help="Custom question bank JSON path")
    args = parser.parse_args()

    # Load questions
    qb_path = Path(args.question_bank) if args.question_bank else QUESTION_BANK
    if not qb_path.exists():
        print(f"Error: Question bank not found at {qb_path}")
        sys.exit(1)

    questions = load_questions(qb_path)
    questions = filter_questions(questions, args.category, args.question)

    if not questions:
        print("No questions matched the filter.")
        sys.exit(1)

    print(f"Exhaustive Simulation: {len(questions)} question(s)")
    print(f"Mode: {'natural only' if args.natural_only else 'natural + forced'}")

    if args.dry_run:
        for q in questions:
            print(f"  [{q['id']}] ({q['category']}) {q['question']}")
        # Estimate entries
        avg_natural = 7 * 6  # ~7 widgets × ~6 variants
        avg_forced = 6 * 5.5  # ~6 missing × ~5.5 variants
        total = len(questions) * (avg_natural + (0 if args.natural_only else avg_forced))
        print(f"\nEstimated entries: ~{int(total)}")
        return

    # Check resume
    existing_entries = []
    completed_ids = set()
    if args.resume:
        resume_path = RESULTS_DIR / args.resume / "exhaustive_data.json"
        if resume_path.exists():
            with open(resume_path) as f:
                existing = json.load(f)
            existing_entries = existing.get("entries", [])
            completed_ids = {e["question_id"] for e in existing_entries}
            print(f"Resuming: {len(completed_ids)} questions already done, {len(existing_entries)} entries loaded")
            questions = [q for q in questions if q["id"] not in completed_ids]
            if not questions:
                print("All questions already completed!")
                return

    # Check backend
    print(f"Backend: {args.backend}")
    health = check_backend(args.backend)
    if health is None:
        print("Error: Backend not reachable.")
        sys.exit(1)
    print(f"  Status: {health.get('status', '?')}, Equipment: {health.get('equipment_count', 0)}")
    print()

    # Run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_id = args.resume or f"exhaustive_{timestamp}"
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    counter = [len(existing_entries)]  # mutable counter for entry IDs
    all_entries = list(existing_entries)
    failed_count = 0
    start_time = time.time()

    def process_one(q):
        resp = run_pipeline_question(q, args.backend, args.timeout)
        return q, resp

    if args.parallel > 1:
        print(f"Running with {args.parallel} parallel workers...")
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(process_one, q): q for q in questions}
            for i, future in enumerate(as_completed(futures), 1):
                q, resp = future.result()
                if resp is None:
                    failed_count += 1
                    print(f"  [{i}/{len(questions)}] {q['id']} FAILED")
                    continue
                entries = expand_question(q, resp, counter, args.natural_only)
                all_entries.extend(entries)
                print(f"  [{i}/{len(questions)}] {q['id']} → {len(entries)} entries")

                # Incremental save every 10 questions
                if i % 10 == 0:
                    _save(run_dir, run_id, all_entries, questions, failed_count, start_time)
    else:
        print("Running sequentially...")
        for i, q in enumerate(questions, 1):
            resp = run_pipeline_question(q, args.backend, args.timeout)
            if resp is None:
                failed_count += 1
                print(f"  [{i}/{len(questions)}] {q['id']} FAILED")
                continue
            entries = expand_question(q, resp, counter, args.natural_only)
            all_entries.extend(entries)
            print(f"  [{i}/{len(questions)}] {q['id']} → {len(entries)} entries")

            # Incremental save every 5 questions
            if i % 5 == 0:
                _save(run_dir, run_id, all_entries, questions, failed_count, start_time)

    # Final save
    _save(run_dir, run_id, all_entries, questions, failed_count, start_time)

    # Copy to frontend
    FRONTEND_PUBLIC.mkdir(parents=True, exist_ok=True)
    frontend_path = FRONTEND_PUBLIC / "exhaustive_data.json"
    with open(frontend_path, "w") as f:
        json.dump({"run_id": run_id, "entries": all_entries}, f)
    print(f"\nFrontend copy: {frontend_path}")

    # Summary
    summary = generate_summary(all_entries, len(questions), failed_count)
    elapsed = int(time.time() - start_time)
    print(f"\n{'='*60}")
    print(f"Run: {run_id}")
    print(f"Total entries: {summary['total_entries']} (natural: {summary['natural_entries']}, forced: {summary['forced_entries']})")
    print(f"Questions: {summary['questions_run']} run, {summary['questions_failed']} failed")
    print(f"Scenarios covered: {summary['unique_scenarios']}")
    print(f"Elapsed: {elapsed}s")
    print(f"\nPer scenario:")
    for s, c in list(summary["by_scenario"].items())[:15]:
        print(f"  {s}: {c}")
    print(f"{'='*60}")


def _save(run_dir, run_id, entries, questions, failed, start_time):
    """Incremental save to disk."""
    data = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "question_bank": str(QUESTION_BANK.name),
        "total_questions": len(questions),
        "failed_questions": failed,
        "entries": entries,
    }
    out_path = run_dir / "exhaustive_data.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=1)

    summary = generate_summary(entries, len(questions), failed)
    sum_path = run_dir / "summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
