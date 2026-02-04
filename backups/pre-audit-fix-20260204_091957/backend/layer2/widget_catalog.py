"""
Widget Catalog — describes all available widget scenarios for the LLM widget selector.

Each entry provides:
- scenario: the widget type identifier
- description: human-readable explanation of what the widget shows
- good_for: keywords/phrases indicating when to use this widget
- sizing: recommended sizes and height hints
- height_units: cost in the height budget system (short=1, medium=2, tall=3, x-tall=4)
"""

WIDGET_CATALOG = [
    {
        "scenario": "kpi",
        "description": "Single metric display — shows one number with label, unit, and optional state (warning/critical). Best for key metrics, live readings, status indicators.",
        "good_for": ["single metric", "status", "live reading", "count", "percentage", "quick glance", "health score"],
        "sizes": ["compact", "normal"],
        "height_units": 1,
    },
    {
        "scenario": "alerts",
        "description": "Alert notification panel — shows active alerts with severity, source equipment, and timestamp. Best for showing current warnings or critical issues.",
        "good_for": ["alerts", "alarms", "warnings", "critical issues", "faults", "threshold breaches", "anomalies"],
        "sizes": ["normal", "expanded"],
        "height_units": 2,
    },
    {
        "scenario": "comparison",
        "description": "Side-by-side comparison of two values with delta indicator. Best for comparing two devices, before/after, actual vs target.",
        "good_for": ["comparison", "vs", "delta", "deviation", "difference", "before after", "actual vs target"],
        "sizes": ["expanded", "hero"],
        "height_units": 2,
    },
    {
        "scenario": "trend",
        "description": "Time series line/area chart for a single metric. Shows how a value changes over time with optional threshold line.",
        "good_for": ["trend", "history", "over time", "time series", "monitoring", "trajectory", "last 24 hours"],
        "sizes": ["expanded", "hero"],
        "height_units": 3,
    },
    {
        "scenario": "trend-multi-line",
        "description": "Multi-line time series chart — overlays 2-4 metrics on the same time axis for correlation analysis.",
        "good_for": ["correlation", "multi-metric trend", "overlay", "compare over time", "multiple parameters"],
        "sizes": ["expanded", "hero"],
        "height_units": 3,
    },
    {
        "scenario": "trends-cumulative",
        "description": "Stacked area or cumulative line chart — shows running totals or accumulated values over time.",
        "good_for": ["cumulative", "total today", "accumulated", "running total", "daily total", "so far"],
        "sizes": ["expanded", "hero"],
        "height_units": 3,
    },
    {
        "scenario": "distribution",
        "description": "Pie/donut or proportional chart showing how a total breaks down into parts.",
        "good_for": ["distribution", "breakdown", "share", "proportion", "split", "by type", "by category"],
        "sizes": ["normal", "expanded"],
        "height_units": 3,
    },
    {
        "scenario": "composition",
        "description": "Stacked bar or grouped composition chart — shows how a metric is composed of sub-parts over categories.",
        "good_for": ["composition", "stacked", "grouped breakdown", "by source", "contribution"],
        "sizes": ["expanded", "hero"],
        "height_units": 3,
    },
    {
        "scenario": "category-bar",
        "description": "Horizontal or vertical bar chart comparing a metric across categories (equipment types, zones, floors).",
        "good_for": ["ranking", "top consumers", "by zone", "by floor", "bar chart", "category comparison"],
        "sizes": ["expanded", "hero"],
        "height_units": 3,
    },
    {
        "scenario": "timeline",
        "description": "Horizontal timeline showing events in chronological order — maintenance events, shift changes, milestones.",
        "good_for": ["timeline", "events", "chronological", "history", "schedule", "milestones", "maintenance log"],
        "sizes": ["expanded", "hero"],
        "height_units": 3,
    },
    {
        "scenario": "flow-sankey",
        "description": "Sankey diagram showing flow from sources to destinations — energy flows, material flows, loss analysis.",
        "good_for": ["flow", "sankey", "energy balance", "losses", "source to destination", "where does it go"],
        "sizes": ["hero"],
        "height_units": 4,
    },
    {
        "scenario": "matrix-heatmap",
        "description": "Color-coded matrix/heatmap — shows values across two dimensions (e.g., equipment × parameter, hour × day).",
        "good_for": ["heatmap", "matrix", "cross-tabulation", "pattern", "correlation matrix", "all equipment"],
        "sizes": ["hero"],
        "height_units": 4,
    },
    {
        "scenario": "eventlogstream",
        "description": "Scrollable event log — real-time feed of system events, maintenance logs, operator actions.",
        "good_for": ["event log", "log stream", "recent events", "audit trail", "what happened", "shift log"],
        "sizes": ["expanded", "hero"],
        "height_units": 4,
    },
    {
        "scenario": "edgedevicepanel",
        "description": "Detailed single-device panel — shows all parameters, readings, and status for one specific piece of equipment.",
        "good_for": ["device detail", "single device", "all parameters", "equipment panel", "deep dive", "specific equipment"],
        "sizes": ["hero"],
        "height_units": 4,
    },
    {
        "scenario": "chatstream",
        "description": "Conversational AI stream — shows ongoing conversation context, useful for complex multi-turn queries.",
        "good_for": ["conversation", "chat", "follow-up", "context"],
        "sizes": ["expanded"],
        "height_units": 4,
    },
    {
        "scenario": "peopleview",
        "description": "Workforce overview — attendance, shift roster, team assignments, leave status.",
        "good_for": ["people", "workforce", "attendance", "shift roster", "who is on shift", "team"],
        "sizes": ["expanded", "hero"],
        "height_units": 3,
    },
    {
        "scenario": "peoplehexgrid",
        "description": "Hex grid visualization of personnel — shows team members spatially by zone or department.",
        "good_for": ["team layout", "zone staffing", "department view", "who is where"],
        "sizes": ["expanded", "hero"],
        "height_units": 3,
    },
    {
        "scenario": "peoplenetwork",
        "description": "Network graph of people relationships — reporting lines, team collaboration, communication patterns.",
        "good_for": ["org chart", "reporting", "team network", "collaboration"],
        "sizes": ["expanded", "hero"],
        "height_units": 3,
    },
    {
        "scenario": "supplychainglobe",
        "description": "3D globe visualization of supply chain — shows vendor locations, shipment routes, delivery status.",
        "good_for": ["supply chain", "vendors", "shipments", "global", "logistics", "delivery tracking"],
        "sizes": ["hero"],
        "height_units": 3,
    },
]

# Quick lookup: scenario name → catalog entry
CATALOG_BY_SCENARIO = {w["scenario"]: w for w in WIDGET_CATALOG}

# All valid scenario names
VALID_SCENARIOS = set(CATALOG_BY_SCENARIO.keys())


def get_catalog_prompt_text() -> str:
    """Format the widget catalog as a text block for inclusion in LLM prompts."""
    lines = []
    for w in WIDGET_CATALOG:
        # Skip banned scenarios
        if w["scenario"] in ("helpview", "pulseview"):
            continue
        sizes = ", ".join(w["sizes"])
        good = ", ".join(w["good_for"])
        lines.append(
            f'- {w["scenario"]} (height={w["height_units"]}, sizes=[{sizes}]): '
            f'{w["description"]} Good for: {good}'
        )
    return "\n".join(lines)
