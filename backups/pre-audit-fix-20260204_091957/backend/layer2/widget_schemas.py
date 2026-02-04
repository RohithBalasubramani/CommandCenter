"""
Widget Data Schemas — per-widget data contracts for Pipeline v2.

Each widget scenario declares:
- required: fields that MUST be present in the data_override.demoData
- optional: fields that improve the widget but aren't mandatory
- rag_strategy: which data collection strategy to use
- default_collections: which ChromaDB collections to search

These schemas drive the data_collector.py — each widget gets exactly the data
it needs, formatted to match the frontend fixtureData.ts demoData shapes.
"""

WIDGET_SCHEMAS = {
    "kpi": {
        "description": "Single metric KPI",
        "required": ["label", "value", "unit"],
        "optional": ["state", "period", "max", "context"],
        "rag_strategy": "single_metric",
        "default_collections": ["equipment"],
        "demo_shape": {
            "demoData": {
                "label": "Metric Name",
                "value": "42",
                "unit": "kW",
                "state": "normal",  # normal | warning | critical
            }
        },
    },
    "alerts": {
        "description": "Alert notification panel",
        "required": ["id", "title", "message", "severity", "source"],
        "optional": ["evidence", "threshold", "actions", "assignee", "timestamp", "state"],
        "rag_strategy": "alert_query",
        "default_collections": ["alerts"],
        "demo_shape": {
            "demoData": {
                "id": "ALT-001",
                "title": "Parameter Name",
                "message": "Alert description",
                "severity": "warning",
                "category": "Equipment",
                "source": "Device Name",
                "state": "new",
                "evidence": {
                    "label": "Value",
                    "value": "95",
                    "unit": "%",
                    "trend": "up",
                },
                "threshold": "90%",
                "actions": [],
            }
        },
    },
    "comparison": {
        "description": "Side-by-side comparison",
        "required": ["label", "unit", "labelA", "valueA", "labelB", "valueB"],
        "optional": ["delta", "deltaPct"],
        "rag_strategy": "multi_entity_metric",
        "default_collections": ["equipment"],
        "demo_shape": {
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
    },
    "trend": {
        "description": "Time series line/area chart",
        "required": ["label", "unit", "timeSeries"],
        "optional": ["timeRange", "threshold"],
        "rag_strategy": "time_series",
        "default_collections": ["equipment"],
        "demo_shape": {
            "demoData": {
                "label": "Metric Trend",
                "unit": "kW",
                "timeSeries": [{"time": "2026-01-31T00:00:00Z", "value": 42}],
                "timeRange": "last_24h",
            }
        },
    },
    "trend-multi-line": {
        "description": "Multi-line time series chart",
        "required": ["label", "unit", "series"],
        "optional": ["timeRange"],
        "rag_strategy": "multi_time_series",
        "default_collections": ["equipment"],
        "demo_shape": {
            "demoData": {
                "label": "Multi-Metric Trend",
                "unit": "kW",
                "series": [
                    {"name": "Metric A", "timeSeries": [{"time": "...", "value": 0}]},
                    {"name": "Metric B", "timeSeries": [{"time": "...", "value": 0}]},
                ],
            }
        },
    },
    "trends-cumulative": {
        "description": "Stacked area / cumulative chart",
        "required": ["config", "data"],
        "optional": [],
        "rag_strategy": "cumulative_time_series",
        "default_collections": ["equipment"],
        "demo_shape": {
            "config": {
                "title": "Cumulative Trend",
                "subtitle": "",
                "variant": "V1",
                "mode": "cumulative",
                "series": [{"id": "S1", "label": "Value", "unit": "kWh", "color": "#2563eb"}],
            },
            "data": [{"x": "2026-01-31T00:00:00Z", "S1_raw": 0, "S1_cumulative": 0}],
        },
    },
    "distribution": {
        "description": "Proportional breakdown chart",
        "required": ["total", "unit", "series"],
        "optional": [],
        "rag_strategy": "aggregation",
        "default_collections": ["equipment"],
        "demo_shape": {
            "demoData": {
                "total": 1000,
                "unit": "kW",
                "series": [
                    {"label": "Category A", "value": 400},
                    {"label": "Category B", "value": 350},
                    {"label": "Category C", "value": 250},
                ],
            }
        },
    },
    "composition": {
        "description": "Stacked bar / grouped composition",
        "required": ["label", "unit", "categories", "series"],
        "optional": [],
        "rag_strategy": "aggregation",
        "default_collections": ["equipment"],
        "demo_shape": {
            "demoData": {
                "label": "Composition",
                "unit": "kW",
                "categories": ["Cat A", "Cat B"],
                "series": [
                    {"name": "Group 1", "values": [100, 200]},
                    {"name": "Group 2", "values": [150, 180]},
                ],
            }
        },
    },
    "category-bar": {
        "description": "Bar chart across categories",
        "required": ["label", "unit", "categories", "values"],
        "optional": ["orientation"],
        "rag_strategy": "aggregation",
        "default_collections": ["equipment"],
        "demo_shape": {
            "demoData": {
                "label": "Category Comparison",
                "unit": "kW",
                "categories": ["Item 1", "Item 2", "Item 3"],
                "values": [100, 80, 60],
            }
        },
    },
    "timeline": {
        "description": "Horizontal timeline of events",
        "required": ["title", "range", "events"],
        "optional": ["lanes"],
        "rag_strategy": "events_in_range",
        "default_collections": ["maintenance", "shift_logs"],
        "demo_shape": {
            "demoData": {
                "title": "Event Timeline",
                "range": {"start": "2026-01-01", "end": "2026-01-31"},
                "events": [
                    {"time": "2026-01-15", "label": "Event 1", "type": "maintenance"},
                ],
            }
        },
    },
    "flow-sankey": {
        "description": "Sankey flow diagram",
        "required": ["label", "nodes", "links"],
        "optional": ["unit"],
        "rag_strategy": "flow_analysis",
        "default_collections": ["equipment"],
        "demo_shape": {
            "demoData": {
                "label": "Energy Flow",
                "unit": "kW",
                "nodes": [{"id": "source", "label": "Grid"}],
                "links": [{"source": "source", "target": "dest", "value": 100}],
            }
        },
    },
    "matrix-heatmap": {
        "description": "Matrix / heatmap visualization",
        "required": ["label", "dataset"],
        "optional": ["unit", "rows", "cols"],
        "rag_strategy": "cross_tabulation",
        "default_collections": ["equipment"],
        "demo_shape": {
            "demoData": {
                "label": "Health Matrix",
                "rows": ["Equipment 1", "Equipment 2"],
                "cols": ["Metric A", "Metric B"],
                "dataset": [[0.9, 0.85], [0.7, 0.92]],
            }
        },
    },
    "eventlogstream": {
        "description": "Scrollable event log / log stream",
        "required": ["events"],
        "optional": ["title", "filters"],
        "rag_strategy": "events_in_range",
        "default_collections": ["maintenance", "shift_logs", "work_orders"],
        "demo_shape": {
            "demoData": {
                "title": "Event Log",
                "events": [
                    {"timestamp": "2026-01-31T10:00:00Z", "type": "info", "message": "Event description", "source": "System"},
                ],
            }
        },
    },
    "edgedevicepanel": {
        "description": "Detailed single-device panel",
        "required": ["device"],
        "optional": ["readings", "alerts", "maintenance"],
        "rag_strategy": "single_entity_deep",
        "default_collections": ["equipment", "alerts", "maintenance"],
        "demo_shape": {
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
    },
    "chatstream": {
        "description": "Conversational AI stream",
        "required": ["messages"],
        "optional": [],
        "rag_strategy": "none",
        "default_collections": [],
        "demo_shape": {
            "demoData": {
                "messages": [],
            }
        },
    },
    "helpview": {
        "description": "Help and capabilities display",
        "required": [],
        "optional": [],
        "rag_strategy": "none",
        "default_collections": [],
        "demo_shape": {"demoData": {}},
    },
    "peopleview": {
        "description": "Workforce overview",
        "required": ["roster"],
        "optional": ["shifts", "attendance"],
        "rag_strategy": "people_query",
        "default_collections": ["shift_logs"],
        "demo_shape": {
            "demoData": {
                "roster": [],
                "shifts": [],
            }
        },
    },
    "peoplehexgrid": {
        "description": "Personnel hex grid",
        "required": ["people"],
        "optional": ["zones"],
        "rag_strategy": "people_query",
        "default_collections": ["shift_logs"],
        "demo_shape": {"demoData": {"people": [], "zones": []}},
    },
    "peoplenetwork": {
        "description": "People network graph",
        "required": ["nodes", "edges"],
        "optional": [],
        "rag_strategy": "people_query",
        "default_collections": ["shift_logs"],
        "demo_shape": {"demoData": {"nodes": [], "edges": []}},
    },
    "supplychainglobe": {
        "description": "Supply chain globe",
        "required": ["locations", "routes"],
        "optional": [],
        "rag_strategy": "supply_query",
        "default_collections": ["work_orders"],
        "demo_shape": {"demoData": {"locations": [], "routes": []}},
    },
    "pulseview": {
        "description": "Real-time pulse view",
        "required": [],
        "optional": ["signals"],
        "rag_strategy": "none",
        "default_collections": [],
        "demo_shape": {"demoData": {}},
    },
}
