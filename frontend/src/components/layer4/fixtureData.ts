// Auto-generated from Widgets/scenarios.sqlite3
// Maps scenario slug → fixture slug → fixture data (visual config + demo data)

// Fixture data shapes are heterogeneous — fields vary across scenarios.
// Using a loose index signature to accommodate all variants.
export type FixtureData = Record<string, unknown>;

export interface ScenarioMeta {
  name: string;
  defaultFixture: string;
  variants: Record<string, FixtureData>;
}

export const FIXTURES: Record<string, ScenarioMeta> = {
  "alerts": {
    "name": "Alerts",
    "defaultFixture": "banner-energy-peak-threshold-exceeded",
    "variants": {
      "banner-energy-peak-threshold-exceeded": {
        "variant": "banner",
        "data": {
          "id": "ALT-001",
          "title": "Energy Peak Threshold Exceeded",
          "message": "Main feeder consumption exceeded 1200kW.",
          "severity": "warning",
          "category": "Energy",
          "source": "Main LT Panel",
          "timestamp": "2026-01-26T14:08:23.263Z",
          "state": "escalated",
          "evidence": {
            "label": "Load",
            "value": "1,240",
            "unit": "kW",
            "trend": "up"
          },
          "threshold": "1200 kW",
          "triggerCondition": "Load > 1200kW for 15m",
          "occurrenceCount": 3,
          "actions": [
            {
              "label": "Acknowledge",
              "intent": "ack",
              "type": "primary"
            },
            {
              "label": "Load Shedding",
              "intent": "open",
              "type": "secondary"
            }
          ]
        }
      },
      "toast-power-factor-critical-low": {
        "variant": "toast",
        "data": {
          "id": "ALT-002",
          "title": "Power Factor Critical Low",
          "message": "PF dropped to 0.72. Capacitor bank 3 failed.",
          "severity": "critical",
          "category": "Machine",
          "source": "CapBank-03",
          "timestamp": "2026-01-26T13:55:23.263Z",
          "state": "new",
          "evidence": {
            "label": "PF",
            "value": "0.72",
            "unit": "\u03c6",
            "trend": "down"
          },
          "threshold": "0.85",
          "triggerCondition": "PF < 0.85 instantaneous",
          "occurrenceCount": 1,
          "autoActionEnabled": true,
          "actions": [
            {
              "label": "Reset Bank",
              "intent": "drilldown",
              "type": "primary",
              "requiresConfirm": true
            },
            {
              "label": "Assign",
              "intent": "assign",
              "type": "ghost"
            }
          ]
        }
      },
      "card-dg-02-started-successfully": {
        "variant": "card",
        "data": {
          "id": "ALT-003",
          "title": "DG-02 Started Successfully",
          "message": "Auto-start sequence completed. Synchronization active.",
          "severity": "success",
          "category": "Energy",
          "source": "DG-02",
          "timestamp": "2026-01-26T13:25:23.263Z",
          "state": "auto_resolved",
          "threshold": "N/A",
          "triggerCondition": "Grid Fail Detected",
          "actions": [
            {
              "label": "View Logs",
              "intent": "open",
              "type": "ghost"
            }
          ]
        }
      },
      "badge-ahu-01-high-temperature": {
        "variant": "badge",
        "data": {
          "id": "ALT-004",
          "title": "AHU-01 High Temperature",
          "message": "Return air temp variance +4\u00b0C above setpoint.",
          "severity": "warning",
          "category": "Maintenance",
          "source": "AHU-01",
          "timestamp": "2026-01-26T12:10:23.263Z",
          "state": "acknowledged",
          "assignee": {
            "name": "Dave M.",
            "initials": "DM"
          },
          "evidence": {
            "label": "Temp",
            "value": "26",
            "unit": "\u00b0C",
            "trend": "up"
          },
          "threshold": "22\u00b0C \u00b1 2\u00b0C",
          "actions": [
            {
              "label": "Snooze 1h",
              "intent": "snooze",
              "type": "ghost"
            },
            {
              "label": "Resolve",
              "intent": "resolve",
              "type": "secondary"
            }
          ]
        }
      },
      "modal-ups-battery-critical": {
        "variant": "modal",
        "data": {
          "id": "ALT-005",
          "title": "UPS Battery Critical",
          "message": "Battery string B voltage deviation > 5%. Risk of failure.",
          "severity": "critical",
          "category": "Safety",
          "source": "UPS-Main",
          "timestamp": "2026-01-26T14:05:23.263Z",
          "state": "new",
          "tags": [
            "Anomaly",
            "Risk"
          ],
          "threshold": "5% Deviation",
          "triggerCondition": "Voltage variance > 5%",
          "actions": [
            {
              "label": "Emergency SOP",
              "intent": "open",
              "type": "primary"
            }
          ]
        }
      }
    }
  },
  "category-bar": {
    "name": "Category Bar",
    "defaultFixture": "oee-by-machine",
    "variants": {
      "oee-by-machine": {
        "config": {
          "variant": "VERTICAL",
          "title": "OEE By Machine",
          "description": "Overall Equipment Effectiveness across production lines.",
          "dataKeys": [
            "value"
          ],
          "colors": [
            "#262626"
          ],
          "layout": "horizontal"
        },
        "data": [
          {
            "category": "M-101",
            "value": 59
          },
          {
            "category": "M-102",
            "value": 65
          },
          {
            "category": "M-103",
            "value": 106
          },
          {
            "category": "M-104",
            "value": 34
          },
          {
            "category": "M-105",
            "value": 70
          },
          {
            "category": "M-106",
            "value": 34
          }
        ],
        "enableBrush": false
      },
      "downtime-duration": {
        "config": {
          "variant": "HORIZONTAL",
          "title": "Downtime Duration",
          "description": "Total downtime minutes per machine.",
          "dataKeys": [
            "value"
          ],
          "colors": [
            "#2563eb"
          ],
          "layout": "vertical"
        },
        "data": [
          {
            "category": "M-101",
            "value": 81
          },
          {
            "category": "M-102",
            "value": 51
          },
          {
            "category": "M-103",
            "value": 86
          },
          {
            "category": "M-104",
            "value": 91
          },
          {
            "category": "M-105",
            "value": 100
          },
          {
            "category": "M-106",
            "value": 90
          }
        ],
        "enableBrush": false
      },
      "production-states": {
        "config": {
          "variant": "STACKED",
          "title": "Production States",
          "description": "Distribution of machine states (Run/Idle/Down).",
          "dataKeys": [
            "value",
            "value2",
            "value3"
          ],
          "colors": [
            "#16a34a",
            "#d4d4d4",
            "#ef4444"
          ],
          "stacked": true,
          "layout": "horizontal"
        },
        "data": [
          {
            "category": "M-101",
            "value": 56,
            "value2": 18,
            "value3": 1
          },
          {
            "category": "M-102",
            "value": 97,
            "value2": 19,
            "value3": 8
          },
          {
            "category": "M-103",
            "value": 87,
            "value2": 15,
            "value3": 3
          },
          {
            "category": "M-104",
            "value": 92,
            "value2": 9,
            "value3": 9
          },
          {
            "category": "M-105",
            "value": 74,
            "value2": 0,
            "value3": 2
          },
          {
            "category": "M-106",
            "value": 75,
            "value2": 12,
            "value3": 1
          }
        ],
        "enableBrush": false
      },
      "shift-comparison": {
        "config": {
          "variant": "GROUPED",
          "title": "Shift Comparison",
          "description": "Output comparison between shifts.",
          "dataKeys": [
            "value",
            "value2"
          ],
          "colors": [
            "#171717",
            "#a3a3a3"
          ],
          "layout": "horizontal"
        },
        "data": [
          {
            "category": "M-101",
            "value": 661,
            "value2": 593
          },
          {
            "category": "M-102",
            "value": 638,
            "value2": 612
          },
          {
            "category": "M-103",
            "value": 1420,
            "value2": 501
          },
          {
            "category": "M-104",
            "value": 927,
            "value2": 1415
          },
          {
            "category": "M-105",
            "value": 1470,
            "value2": 550
          }
        ],
        "enableBrush": false
      },
      "efficiency-deviation": {
        "config": {
          "variant": "DIVERGING",
          "title": "Efficiency Deviation",
          "description": "Deviation from standard cycle time.",
          "dataKeys": [
            "value"
          ],
          "colors": [
            "#525252"
          ],
          "layout": "horizontal"
        },
        "data": [
          {
            "category": "M-101",
            "value": -6
          },
          {
            "category": "M-102",
            "value": 16
          },
          {
            "category": "M-103",
            "value": 8
          },
          {
            "category": "M-104",
            "value": 10
          },
          {
            "category": "M-105",
            "value": -17
          },
          {
            "category": "M-106",
            "value": -11
          }
        ],
        "enableBrush": false
      }
    }
  },
  "chatstream": {
    "name": "ChatStream",
    "defaultFixture": "default-render",
    "variants": {
      "default-render": {}
    }
  },
  "comparison": {
    "name": "Comparison",
    "defaultFixture": "side_by_side_visual-plain-values",
    "variants": {
      "side_by_side_visual-plain-values": {
        "coreWidget": "COMPARISON",
        "variant": "SIDE_BY_SIDE_VISUAL",
        "representation": "Plain Values",
        "encoding": "Text+Arrow",
        "layout": {
          "padding": "p-4",
          "radius": "rounded-none",
          "zones": "row"
        },
        "visual": {
          "background": "bg-transparent",
          "border": "border-none",
          "theme": "dark",
          "colors": {
            "text": "text-neutral-900",
            "label": "text-neutral-500",
            "good": "text-emerald-600",
            "critical": "text-red-600"
          }
        },
        "states": {
          "default": ""
        },
        "interactions": {
          "hoverTooltip": "Frequency deviation"
        },
        "demoData": {
          "label": "Grid Frequency",
          "unit": "Hz",
          "labelA": "Source A",
          "valueA": 50.02,
          "labelB": "Source B",
          "valueB": 49.98,
          "delta": 0.04,
          "deltaPct": 0.08
        }
      },
      "delta_bar_visual-deviation-bar": {
        "coreWidget": "COMPARISON",
        "variant": "DELTA_BAR_VISUAL",
        "representation": "Deviation Bar",
        "encoding": "Bar",
        "layout": {
          "padding": "p-4",
          "radius": "rounded-none",
          "zones": "stack"
        },
        "visual": {
          "background": "bg-transparent",
          "border": "border-none",
          "theme": "dark",
          "colors": {
            "text": "text-neutral-900",
            "label": "text-neutral-500",
            "good": "bg-emerald-600",
            "critical": "bg-amber-500"
          }
        },
        "states": {
          "default": ""
        },
        "interactions": {
          "hoverTooltip": "PF Lag"
        },
        "demoData": {
          "label": "Power Factor",
          "unit": "PF",
          "labelA": "Actual",
          "valueA": 0.92,
          "labelB": "Target",
          "valueB": 0.98,
          "delta": -0.06,
          "deltaPct": -6.1
        }
      },
      "grouped_bar_visual-phase-comparison": {
        "coreWidget": "COMPARISON",
        "variant": "GROUPED_BAR_VISUAL",
        "representation": "Phase Comparison",
        "encoding": "GroupedBar",
        "layout": {
          "padding": "p-4",
          "radius": "rounded-none",
          "zones": "stack"
        },
        "visual": {
          "background": "bg-transparent",
          "border": "border-none",
          "theme": "dark",
          "colors": {
            "text": "text-neutral-900",
            "label": "text-neutral-500",
            "highlight": "bg-blue-600"
          }
        },
        "states": {
          "default": ""
        },
        "interactions": {
          "hoverTooltip": "Amps per phase"
        },
        "demoData": {
          "label": "Phase Current Balance",
          "unit": "A",
          "labelA": "T-1h",
          "labelB": "Now",
          "items": [
            {
              "label": "L1",
              "valueA": 124,
              "valueB": 138
            },
            {
              "label": "L2",
              "valueA": 122,
              "valueB": 125
            },
            {
              "label": "L3",
              "valueA": 118,
              "valueB": 140
            }
          ]
        }
      },
      "waterfall_visual-loss-analysis": {
        "coreWidget": "COMPARISON",
        "variant": "WATERFALL_VISUAL",
        "representation": "Loss Analysis",
        "encoding": "Waterfall",
        "layout": {
          "padding": "p-4",
          "radius": "rounded-none",
          "zones": "stack"
        },
        "visual": {
          "background": "bg-transparent",
          "border": "border-none",
          "theme": "dark",
          "colors": {
            "text": "text-neutral-900",
            "label": "text-neutral-500",
            "good": "bg-emerald-500",
            "critical": "bg-red-500",
            "highlight": "bg-neutral-800"
          }
        },
        "states": {
          "default": ""
        },
        "interactions": {
          "hoverTooltip": "Efficiency steps"
        },
        "demoData": {
          "label": "Motor Efficiency Loss",
          "unit": "%",
          "valueA": 98,
          "valueB": 92.5,
          "items": [
            {
              "label": "Rated",
              "valueA": 98
            },
            {
              "label": "Heat",
              "diff": -3.2
            },
            {
              "label": "Friction",
              "diff": -1.1
            },
            {
              "label": "Harmonics",
              "diff": -1.2
            },
            {
              "label": "Actual",
              "valueA": 92.5
            }
          ]
        }
      },
      "small_multiples_visual-temp-grid": {
        "coreWidget": "COMPARISON",
        "variant": "SMALL_MULTIPLES_VISUAL",
        "representation": "Temp Grid",
        "encoding": "MicroCharts",
        "layout": {
          "padding": "p-2",
          "radius": "rounded-none",
          "zones": "grid"
        },
        "visual": {
          "background": "bg-transparent",
          "border": "border-none",
          "theme": "dark",
          "colors": {
            "text": "text-neutral-900",
            "label": "text-neutral-500"
          }
        },
        "states": {
          "default": ""
        },
        "interactions": {
          "clickSelect": "Machine details"
        },
        "demoData": {
          "label": "Bearing Temps",
          "unit": "\u00b0C",
          "items": [
            {
              "label": "M1-Drve",
              "valueA": 65,
              "valueB": 72,
              "diff": 7
            },
            {
              "label": "M1-Load",
              "valueA": 62,
              "valueB": 61,
              "diff": -1
            },
            {
              "label": "M2-Drve",
              "valueA": 70,
              "valueB": 85,
              "diff": 15
            },
            {
              "label": "M2-Load",
              "valueA": 68,
              "valueB": 69,
              "diff": 1
            }
          ]
        }
      },
      "composition_split_visual-load-type": {
        "coreWidget": "COMPARISON",
        "variant": "COMPOSITION_SPLIT_VISUAL",
        "representation": "Load Type",
        "encoding": "StackedBar",
        "layout": {
          "padding": "p-4",
          "radius": "rounded-none",
          "zones": "stack"
        },
        "visual": {
          "background": "bg-transparent",
          "border": "border-none",
          "theme": "dark",
          "colors": {
            "text": "text-neutral-900",
            "label": "text-neutral-500"
          }
        },
        "states": {
          "default": ""
        },
        "interactions": {
          "hoverTooltip": "kVAR vs kW"
        },
        "demoData": {
          "label": "Load Composition",
          "unit": "%",
          "labelA": "Line 1",
          "labelB": "Line 2",
          "composition": [
            {
              "label": "Resistive",
              "valueA": 70,
              "valueB": 60,
              "color": "bg-blue-600"
            },
            {
              "label": "Inductive",
              "valueA": 25,
              "valueB": 35,
              "color": "bg-amber-500"
            },
            {
              "label": "Capacitive",
              "valueA": 5,
              "valueB": 5,
              "color": "bg-emerald-500"
            }
          ]
        }
      }
    }
  },
  "composition": {
    "name": "Composition",
    "defaultFixture": "stacked_bar",
    "variants": {
      "stacked_bar": {
        "coreWidget": "COMPOSITION",
        "variant": "STACKED_BAR",
        "representation": "VERTICAL_STACKED_COLUMN",
        "encoding": {
          "x": "time_bucket | category",
          "y": "value (sum)",
          "color": "series_id",
          "stackId": "series_id"
        },
        "layout": {
          "minHeight": 320,
          "padding": "normal",
          "responsive": true
        },
        "visual": {
          "palette": [
            "#171717",
            "#525252",
            "#a3a3a3"
          ],
          "bg": "bg-white",
          "border": "border-gray-100"
        },
        "states": {
          "default": {
            "visual": {
              "opacity": 1
            },
            "behavior": {
              "clickable": true,
              "tooltip": true
            }
          },
          "hover": {
            "visual": {
              "opacity": 1,
              "shadow": "md"
            },
            "behavior": {
              "clickable": true,
              "tooltip": true
            }
          },
          "selected": {
            "visual": {
              "stroke": "#2563eb",
              "strokeWidth": 2
            },
            "behavior": {
              "clickable": true,
              "tooltip": true
            }
          },
          "active": {
            "visual": {
              "opacity": 1
            },
            "behavior": {
              "clickable": true,
              "tooltip": true
            }
          },
          "disabled": {
            "visual": {
              "opacity": 0.5,
              "cursor": "not-allowed"
            },
            "behavior": {
              "clickable": false,
              "tooltip": false
            }
          },
          "loading": {
            "visual": {
              "opacity": 0.3
            },
            "behavior": {
              "clickable": false,
              "tooltip": false
            }
          },
          "empty": {
            "visual": {
              "opacity": 0
            },
            "behavior": {
              "clickable": false,
              "tooltip": false
            }
          },
          "dataGap": {
            "visual": {
              "fill": "url(#diagonal-stripe)"
            },
            "behavior": {
              "clickable": false,
              "tooltip": true
            }
          }
        },
        "interactionPolicy": {
          "selection": {
            "enabled": true,
            "mode": "single",
            "persistent": true,
            "detailsDrawer": true
          },
          "hover": {
            "tooltip": true,
            "highlightSegment": true
          },
          "legendInteraction": {
            "toggleVisibility": true,
            "isolateSegment": false
          },
          "keyboardSupport": {
            "enabled": true,
            "tabNavigable": true
          }
        },
        "tooltip": {
          "show": true,
          "position": "auto",
          "content": "detailed"
        },
        "constraints": {
          "maxCategories": 20,
          "maxSeries": 8
        },
        "demoData": [
          {
            "name": "Shift 1",
            "Solar": 4000,
            "Grid": 2400,
            "Diesel": 2400
          },
          {
            "name": "Shift 2",
            "Solar": 3000,
            "Grid": 1398,
            "Diesel": 2210
          },
          {
            "name": "Shift 3",
            "Solar": 2000,
            "Grid": 9800,
            "Diesel": 2290
          }
        ]
      },
      "stacked_area": {
        "coreWidget": "COMPOSITION",
        "variant": "STACKED_AREA",
        "representation": "TIME_SERIES_AREA",
        "encoding": {
          "x": "timestamp",
          "y": "value (cumulative)",
          "color": "category"
        },
        "layout": {
          "minHeight": 300,
          "padding": "tight",
          "responsive": true
        },
        "visual": {
          "palette": [
            "#2563eb",
            "#60a5fa",
            "#93c5fd"
          ],
          "bg": "bg-white",
          "border": "border-gray-100"
        },
        "states": {
          "default": {
            "visual": {
              "opacity": 0.9
            },
            "behavior": {
              "clickable": false,
              "tooltip": true
            }
          },
          "hover": {
            "visual": {
              "opacity": 1
            },
            "behavior": {
              "clickable": false,
              "tooltip": true
            }
          },
          "selected": {
            "visual": {
              "strokeWidth": 2
            },
            "behavior": {
              "clickable": true,
              "tooltip": true
            }
          },
          "active": {
            "visual": {
              "opacity": 1
            },
            "behavior": {
              "clickable": false,
              "tooltip": true
            }
          },
          "disabled": {
            "visual": {
              "opacity": 0.2
            },
            "behavior": {
              "clickable": false,
              "tooltip": false
            }
          },
          "loading": {
            "visual": {
              "opacity": 0.1
            },
            "behavior": {
              "clickable": false,
              "tooltip": false
            }
          },
          "empty": {
            "visual": {
              "opacity": 0
            },
            "behavior": {
              "clickable": false,
              "tooltip": false
            }
          },
          "dataGap": {
            "visual": {
              "strokeDasharray": "4 4"
            },
            "behavior": {
              "clickable": false,
              "tooltip": true
            }
          }
        },
        "interactionPolicy": {
          "selection": {
            "enabled": true,
            "mode": "single",
            "persistent": false,
            "detailsDrawer": false
          },
          "hover": {
            "tooltip": true,
            "highlightSegment": false
          },
          "legendInteraction": {
            "toggleVisibility": true,
            "isolateSegment": true
          },
          "keyboardSupport": {
            "enabled": true,
            "tabNavigable": true
          }
        },
        "tooltip": {
          "show": true,
          "position": "auto",
          "content": "standard"
        },
        "constraints": {
          "minDataPoints": 2,
          "maxSeries": 5
        },
        "demoData": [
          {
            "time": "08:00",
            "Running": 65,
            "Idle": 20,
            "Down": 15
          },
          {
            "time": "09:00",
            "Running": 70,
            "Idle": 25,
            "Down": 5
          },
          {
            "time": "10:00",
            "Running": 60,
            "Idle": 30,
            "Down": 10
          },
          {
            "time": "11:00",
            "Running": 75,
            "Idle": 15,
            "Down": 10
          },
          {
            "time": "12:00",
            "Running": 80,
            "Idle": 10,
            "Down": 10
          }
        ]
      },
      "donut_pie": {
        "coreWidget": "COMPOSITION",
        "variant": "DONUT_PIE",
        "representation": "RADIAL_DONUT",
        "encoding": {
          "theta": "value",
          "color": "category",
          "label": "percentage"
        },
        "layout": {
          "minHeight": 250,
          "padding": "normal",
          "responsive": true
        },
        "visual": {
          "palette": [
            "#16a34a",
            "#eab308",
            "#dc2626",
            "#a3a3a3"
          ],
          "bg": "bg-white",
          "border": "border-gray-100"
        },
        "states": {
          "default": {
            "visual": {
              "stroke": "#fff",
              "strokeWidth": 2
            },
            "behavior": {
              "clickable": true,
              "tooltip": true
            }
          },
          "hover": {
            "visual": {
              "scale": "1.05",
              "shadow": "lg"
            },
            "behavior": {
              "clickable": true,
              "tooltip": true
            }
          },
          "selected": {
            "visual": {
              "stroke": "#171717",
              "strokeWidth": 3
            },
            "behavior": {
              "clickable": true,
              "tooltip": true
            }
          },
          "active": {
            "visual": {
              "opacity": 1
            },
            "behavior": {
              "clickable": true,
              "tooltip": true
            }
          },
          "disabled": {
            "visual": {
              "opacity": 0.3
            },
            "behavior": {
              "clickable": false,
              "tooltip": false
            }
          },
          "loading": {
            "visual": {
              "opacity": 0.1
            },
            "behavior": {
              "clickable": false,
              "tooltip": false
            }
          },
          "empty": {
            "visual": {
              "fill": "#f5f5f5"
            },
            "behavior": {
              "clickable": false,
              "tooltip": false
            }
          },
          "dataGap": {
            "visual": {},
            "behavior": {
              "clickable": false,
              "tooltip": false
            }
          }
        },
        "interactionPolicy": {
          "selection": {
            "enabled": true,
            "mode": "single",
            "persistent": true,
            "detailsDrawer": true
          },
          "hover": {
            "tooltip": true,
            "highlightSegment": true
          },
          "legendInteraction": {
            "toggleVisibility": true,
            "isolateSegment": true
          },
          "keyboardSupport": {
            "enabled": true,
            "tabNavigable": true
          }
        },
        "tooltip": {
          "show": true,
          "position": "auto",
          "content": "detailed"
        },
        "constraints": {
          "maxSegments": 5,
          "innerRadiusRatio": 0.6
        },
        "demoData": [
          {
            "name": "Running",
            "value": 450
          },
          {
            "name": "Idle",
            "value": 300
          },
          {
            "name": "Faulted",
            "value": 120
          },
          {
            "name": "Offline",
            "value": 80
          }
        ]
      },
      "waterfall": {
        "coreWidget": "COMPOSITION",
        "variant": "WATERFALL",
        "representation": "WATERFALL_BRIDGE",
        "encoding": {
          "x": "step",
          "y": "delta",
          "color": "type (positive/negative/total)"
        },
        "layout": {
          "minHeight": 350,
          "padding": "normal",
          "responsive": true
        },
        "visual": {
          "palette": [
            "#171717",
            "#16a34a",
            "#dc2626",
            "#2563eb"
          ],
          "bg": "bg-white",
          "border": "border-gray-100"
        },
        "states": {
          "default": {
            "visual": {},
            "behavior": {
              "clickable": true,
              "tooltip": true
            }
          },
          "hover": {
            "visual": {
              "opacity": 0.8
            },
            "behavior": {
              "clickable": true,
              "tooltip": true
            }
          },
          "selected": {
            "visual": {
              "stroke": "#000",
              "strokeWidth": 2
            },
            "behavior": {
              "clickable": true,
              "tooltip": true
            }
          },
          "active": {
            "visual": {
              "opacity": 1
            },
            "behavior": {
              "clickable": true,
              "tooltip": true
            }
          },
          "disabled": {
            "visual": {
              "opacity": 0.5
            },
            "behavior": {
              "clickable": false,
              "tooltip": false
            }
          },
          "loading": {
            "visual": {
              "opacity": 0.3
            },
            "behavior": {
              "clickable": false,
              "tooltip": false
            }
          },
          "empty": {
            "visual": {
              "opacity": 0
            },
            "behavior": {
              "clickable": false,
              "tooltip": false
            }
          },
          "dataGap": {
            "visual": {},
            "behavior": {
              "clickable": false,
              "tooltip": false
            }
          }
        },
        "interactionPolicy": {
          "selection": {
            "enabled": true,
            "mode": "single",
            "persistent": false,
            "detailsDrawer": true
          },
          "hover": {
            "tooltip": true,
            "highlightSegment": true
          },
          "legendInteraction": {
            "toggleVisibility": false,
            "isolateSegment": false
          },
          "keyboardSupport": {
            "enabled": true,
            "tabNavigable": true
          }
        },
        "tooltip": {
          "show": true,
          "position": "top",
          "content": "detailed"
        },
        "constraints": {
          "maxSteps": 12
        },
        "demoData": [
          {
            "name": "Gross Revenue",
            "value": 5000,
            "type": "start"
          },
          {
            "name": "Material Cost",
            "value": -1200,
            "type": "loss"
          },
          {
            "name": "Labor Cost",
            "value": -800,
            "type": "loss"
          },
          {
            "name": "Efficiency Gain",
            "value": 400,
            "type": "gain"
          },
          {
            "name": "Net Profit",
            "value": 3400,
            "type": "end"
          }
        ]
      },
      "treemap": {
        "coreWidget": "COMPOSITION",
        "variant": "TREEMAP",
        "representation": "RECTANGULAR_PARTITION",
        "encoding": {
          "size": "value",
          "color": "category",
          "label": "name"
        },
        "layout": {
          "minHeight": 400,
          "padding": "tight",
          "responsive": true
        },
        "visual": {
          "palette": [
            "#171717",
            "#404040",
            "#737373",
            "#a3a3a3",
            "#d4d4d4"
          ],
          "bg": "bg-white",
          "border": "border-gray-100"
        },
        "states": {
          "default": {
            "visual": {
              "stroke": "#fff",
              "strokeWidth": 1
            },
            "behavior": {
              "clickable": true,
              "tooltip": true
            }
          },
          "hover": {
            "visual": {
              "shadow": "inner",
              "strokeWidth": 2
            },
            "behavior": {
              "clickable": true,
              "tooltip": true
            }
          },
          "selected": {
            "visual": {
              "stroke": "#2563eb",
              "strokeWidth": 3
            },
            "behavior": {
              "clickable": true,
              "tooltip": true
            }
          },
          "active": {
            "visual": {
              "opacity": 1
            },
            "behavior": {
              "clickable": true,
              "tooltip": true
            }
          },
          "disabled": {
            "visual": {
              "opacity": 0.5,
              "grayscale": 1
            },
            "behavior": {
              "clickable": false,
              "tooltip": false
            }
          },
          "loading": {
            "visual": {
              "opacity": 0.2
            },
            "behavior": {
              "clickable": false,
              "tooltip": false
            }
          },
          "empty": {
            "visual": {
              "opacity": 0
            },
            "behavior": {
              "clickable": false,
              "tooltip": false
            }
          },
          "dataGap": {
            "visual": {},
            "behavior": {
              "clickable": false,
              "tooltip": false
            }
          }
        },
        "interactionPolicy": {
          "selection": {
            "enabled": true,
            "mode": "single",
            "persistent": true,
            "detailsDrawer": true
          },
          "hover": {
            "tooltip": true,
            "highlightSegment": true
          },
          "legendInteraction": {
            "toggleVisibility": false,
            "isolateSegment": false
          },
          "keyboardSupport": {
            "enabled": true,
            "tabNavigable": true
          }
        },
        "tooltip": {
          "show": true,
          "position": "auto",
          "content": "standard"
        },
        "constraints": {
          "minArea": 50,
          "maxDepth": 2
        },
        "demoData": [
          {
            "name": "Production",
            "size": 1200,
            "category": "Ops"
          },
          {
            "name": "Maintenance",
            "size": 800,
            "category": "Ops"
          },
          {
            "name": "Logistics",
            "size": 450,
            "category": "Supply"
          },
          {
            "name": "Quality",
            "size": 300,
            "category": "Ops"
          },
          {
            "name": "Facilities",
            "size": 200,
            "category": "Admin"
          },
          {
            "name": "Safety",
            "size": 100,
            "category": "Admin"
          }
        ]
      }
    }
  },
  "distribution": {
    "name": "Distribution",
    "defaultFixture": "dist_energy_source_share-donut",
    "variants": {
      "dist_energy_source_share-donut": {
        "coreWidget": "Distribution",
        "variant": "DIST_ENERGY_SOURCE_SHARE",
        "representation": "Donut",
        "encoding": "Proportional",
        "semanticMetadata": {
          "primaryQuestion": "Where is our energy coming from right now?",
          "intentType": "Operational Awareness",
          "decisionSupport": "Determine if renewable integration targets are being met.",
          "misuseGuard": "Do not use to analyze total consumption volume changes."
        },
        "layout": {
          "aspectRatio": "1:1",
          "zones": [
            "header",
            "chart",
            "legend",
            "footer"
          ]
        },
        "visual": {
          "innerRadius": "60%",
          "outerRadius": "100%",
          "colorMapping": "semantic_source",
          "legendStyling": "bottom_horizontal"
        },
        "states": {
          "default": "All segments visible",
          "hover": "Highlight hovered segment, dim others (opacity 0.3)",
          "active": "Selected segment stroke emphasized",
          "disabled": "Grayscale, opacity 0.5",
          "empty": "Gray ring with 'No Data' label",
          "loading": "Animated skeleton ring",
          "warning_skew": "Alert if 'Grid' > 80%"
        },
        "interactions": {
          "hover": "Show tooltip with Value (kW) + Percentage (%)",
          "click": "Filter dashboard by Source",
          "legendToggle": "Hide/Show Source"
        },
        "distributionRules": {
          "topN": null,
          "groupRemaining": false,
          "sumTo100": true
        },
        "demoData": {
          "total": 1250,
          "unit": "kW",
          "series": [
            {
              "label": "Grid",
              "value": 750,
              "percentage": 60,
              "color": "#171717"
            },
            {
              "label": "Solar",
              "value": 300,
              "percentage": 24,
              "color": "#16a34a"
            },
            {
              "label": "Diesel",
              "value": 200,
              "percentage": 16,
              "color": "#d97706"
            }
          ]
        }
      },
      "dist_energy_source_share-100-stacked-bar": {
        "coreWidget": "Distribution",
        "variant": "DIST_ENERGY_SOURCE_SHARE",
        "representation": "100% Stacked Bar",
        "encoding": "Proportional",
        "semanticMetadata": {
          "primaryQuestion": "What is the ratio of energy sources?",
          "intentType": "Operational Awareness",
          "decisionSupport": "Quickly assess renewable penetration.",
          "misuseGuard": "Do not use for absolute consumption values."
        },
        "layout": {
          "aspectRatio": "3:1",
          "zones": [
            "header",
            "chart",
            "legend"
          ]
        },
        "visual": {
          "orientation": "horizontal",
          "colorMapping": "semantic_source"
        },
        "states": {
          "default": "Stacked bar visible",
          "hover": "Tooltip on segment",
          "active": "Segment outline",
          "empty": "Gray bar",
          "loading": "Pulse animation",
          "warning_skew": "None"
        },
        "interactions": {
          "hover": "Show tooltip with Percentage (%)",
          "click": "Filter dashboard by Source",
          "legendToggle": "Hide/Show Source"
        },
        "distributionRules": {
          "topN": null,
          "groupRemaining": false,
          "sumTo100": true
        },
        "demoData": {
          "total": 100,
          "unit": "%",
          "series": [
            {
              "label": "Grid",
              "value": 60,
              "percentage": 60,
              "color": "#171717"
            },
            {
              "label": "Solar",
              "value": 24,
              "percentage": 24,
              "color": "#16a34a"
            },
            {
              "label": "Diesel",
              "value": 16,
              "percentage": 16,
              "color": "#d97706"
            }
          ]
        }
      },
      "dist_load_by_asset-horizontal-bar": {
        "coreWidget": "Distribution",
        "variant": "DIST_LOAD_BY_ASSET",
        "representation": "Horizontal Bar",
        "encoding": "Ranked",
        "semanticMetadata": {
          "primaryQuestion": "Which assets are consuming the most power?",
          "intentType": "Diagnostic",
          "decisionSupport": "Identify high-load assets for load shedding.",
          "misuseGuard": "Do not use to compare assets with different capacity ratings."
        },
        "layout": {
          "aspectRatio": "4:3",
          "zones": [
            "header",
            "chart",
            "tooltip"
          ]
        },
        "visual": {
          "barSize": 24,
          "grid": false,
          "colorMapping": "rank_emphasis"
        },
        "states": {
          "default": "Ordered descending",
          "hover": "Bar brightness increase",
          "active": "Bar selected",
          "empty": "Empty state illustration",
          "loading": "Skeleton bars",
          "warning_skew": "Alert if Top 1 > 50% Total"
        },
        "interactions": {
          "hover": "Show absolute value + % of total",
          "click": "Open Asset Details",
          "legendToggle": "N/A"
        },
        "distributionRules": {
          "topN": 5,
          "groupRemaining": true,
          "sort": "descending"
        },
        "demoData": {
          "total": 4200,
          "unit": "kWh",
          "series": [
            {
              "label": "Compressor A",
              "value": 1200,
              "percentage": 28.5,
              "color": "#171717"
            },
            {
              "label": "Chiller Unit 2",
              "value": 980,
              "percentage": 23.3,
              "color": "#404040"
            },
            {
              "label": "Conveyor Main",
              "value": 850,
              "percentage": 20.2,
              "color": "#525252"
            },
            {
              "label": "Pump Stn 4",
              "value": 600,
              "percentage": 14.2,
              "color": "#737373"
            },
            {
              "label": "Others",
              "value": 570,
              "percentage": 13.5,
              "color": "#a3a3a3"
            }
          ]
        }
      },
      "dist_consumption_by_category-pie": {
        "coreWidget": "Distribution",
        "variant": "DIST_CONSUMPTION_BY_CATEGORY",
        "representation": "Pie",
        "encoding": "Proportional",
        "semanticMetadata": {
          "primaryQuestion": "How is consumption distributed across categories?",
          "intentType": "Operational Awareness",
          "decisionSupport": "Cost allocation and budgeting.",
          "misuseGuard": "Do not use for real-time anomaly detection."
        },
        "layout": {
          "aspectRatio": "1:1",
          "zones": [
            "header",
            "chart",
            "legend"
          ]
        },
        "visual": {
          "innerRadius": "0%",
          "outerRadius": "100%",
          "colorMapping": "categorical_distinct"
        },
        "states": {
          "default": "Segments visible",
          "hover": "Pop segment",
          "active": "Isolate category",
          "empty": "Gray circle",
          "loading": "Skeleton circle",
          "warning_skew": "None"
        },
        "interactions": {
          "hover": "Show value + %",
          "click": "Filter by Category",
          "legendToggle": "Hide/Show Category"
        },
        "distributionRules": {
          "topN": 6,
          "groupRemaining": true,
          "sumTo100": true
        },
        "demoData": {
          "total": 100,
          "unit": "%",
          "series": [
            {
              "label": "HVAC",
              "value": 45,
              "percentage": 45,
              "color": "#2563eb"
            },
            {
              "label": "Production",
              "value": 35,
              "percentage": 35,
              "color": "#3b82f6"
            },
            {
              "label": "Lighting",
              "value": 15,
              "percentage": 15,
              "color": "#60a5fa"
            },
            {
              "label": "IT/Admin",
              "value": 5,
              "percentage": 5,
              "color": "#93c5fd"
            }
          ]
        }
      },
      "dist_consumption_by_shift-grouped-bar": {
        "coreWidget": "Distribution",
        "variant": "DIST_CONSUMPTION_BY_SHIFT",
        "representation": "Grouped Bar",
        "encoding": "Comparative",
        "semanticMetadata": {
          "primaryQuestion": "How does consumption compare across shifts?",
          "intentType": "Optimization",
          "decisionSupport": "Shift balancing and target adjustment.",
          "misuseGuard": "Not a trend chart; discrete buckets only."
        },
        "layout": {
          "aspectRatio": "16:9",
          "zones": [
            "header",
            "chart",
            "legend"
          ]
        },
        "visual": {
          "barSize": 32,
          "colorMapping": "actual_vs_target"
        },
        "states": {
          "default": "Grouped by shift",
          "hover": "Highlight group",
          "active": "Filter shift",
          "empty": "No bars",
          "loading": "Skeleton bars",
          "warning_skew": "Alert if Actual > Target + 10%"
        },
        "interactions": {
          "hover": "Show Actual vs Target delta",
          "click": "Drill into Shift details",
          "legendToggle": "Hide Actual/Target"
        },
        "distributionRules": {
          "topN": null,
          "groupRemaining": false,
          "sort": "chronological"
        },
        "demoData": {
          "categories": [
            "Shift 1",
            "Shift 2",
            "Shift 3"
          ],
          "unit": "kWh",
          "series": [
            {
              "name": "Actual",
              "data": [
                450,
                480,
                320
              ],
              "color": "#171717"
            },
            {
              "name": "Target",
              "data": [
                400,
                400,
                300
              ],
              "color": "#d4d4d4"
            }
          ]
        }
      },
      "dist_downtime_top_contributors-pareto-bar": {
        "coreWidget": "Distribution",
        "variant": "DIST_DOWNTIME_TOP_CONTRIBUTORS",
        "representation": "Pareto Bar",
        "encoding": "Ranked",
        "semanticMetadata": {
          "primaryQuestion": "What are the top contributors to downtime?",
          "intentType": "Diagnostic",
          "decisionSupport": "Prioritize maintenance tasks (80/20 rule).",
          "misuseGuard": "Do not use for minor operational stops."
        },
        "layout": {
          "aspectRatio": "4:3",
          "zones": [
            "header",
            "chart"
          ]
        },
        "visual": {
          "secondaryAxis": true,
          "colorMapping": "alert_severity"
        },
        "states": {
          "default": "Bars descending + Line ascending",
          "hover": "Show individual + cumulative %",
          "active": "Drill down",
          "empty": "No downtime recorded",
          "loading": "Skeleton",
          "warning_skew": "None"
        },
        "interactions": {
          "hover": "Value + Cumulative %",
          "click": "Open Root Cause Analysis",
          "legendToggle": "N/A"
        },
        "distributionRules": {
          "topN": 10,
          "groupRemaining": true,
          "sort": "descending"
        },
        "demoData": {
          "unit": "min",
          "series": [
            {
              "label": "Jam",
              "value": 120,
              "cumulative": 40,
              "color": "#ef4444"
            },
            {
              "label": "Overheat",
              "value": 90,
              "cumulative": 70,
              "color": "#ef4444"
            },
            {
              "label": "Sensor",
              "value": 45,
              "cumulative": 85,
              "color": "#ef4444"
            },
            {
              "label": "Power",
              "value": 30,
              "cumulative": 95,
              "color": "#ef4444"
            },
            {
              "label": "Other",
              "value": 15,
              "cumulative": 100,
              "color": "#a3a3a3"
            }
          ]
        }
      }
    }
  },
  "edgedevicepanel": {
    "name": "EdgeDevicePanel",
    "defaultFixture": "default-render",
    "variants": {
      "default-render": {}
    }
  },
  "eventlogstream": {
    "name": "EventLogStream",
    "defaultFixture": "chronological-timeline",
    "variants": {
      "chronological-timeline": {
        "variation": "Chronological Timeline"
      },
      "compact-card-feed": {
        "variation": "Compact Card Feed"
      },
      "tabular-log-view": {
        "variation": "Tabular Log View"
      },
      "correlation-stack": {
        "variation": "Correlation Stack"
      },
      "grouped-by-asset": {
        "variation": "Grouped by Asset"
      }
    }
  },
  "flow-sankey": {
    "name": "Flow (Sankey)",
    "defaultFixture": "flow_sankey_standard-classic-left-to-right-sankey",
    "variants": {
      "flow_sankey_standard-classic-left-to-right-sankey": {
        "coreWidget": "FLOW / SANKEY",
        "variant": "FLOW_SANKEY_STANDARD",
        "purpose": "Understand direction, source \u2192 sink energy distribution",
        "representation": "Classic Left-to-Right Sankey",
        "encoding": {
          "Band Width": "Power Magnitude (kW)",
          "Color": "Source Type (Grid/Solar/DG)",
          "Opacity": "Connection Strength"
        },
        "states": {
          "default": "Clean flow paths",
          "hover": "Highlight entire path (Source->Sink)",
          "selected": "Dim unrelated nodes"
        },
        "interactions": [
          "Hover node for total kW",
          "Click link for % contribution"
        ],
        "demoData": {
          "nodes": [
            {
              "id": "src1",
              "label": "Grid",
              "type": "source",
              "value": 450,
              "color": "#2563eb"
            },
            {
              "id": "src2",
              "label": "Solar",
              "type": "source",
              "value": 120,
              "color": "#16a34a"
            },
            {
              "id": "bus",
              "label": "Main Bus",
              "type": "bus",
              "value": 570,
              "color": "#404040"
            },
            {
              "id": "load1",
              "label": "Machinery",
              "type": "load",
              "value": 300,
              "color": "#171717"
            },
            {
              "id": "load2",
              "label": "HVAC",
              "type": "load",
              "value": 200,
              "color": "#525252"
            },
            {
              "id": "load3",
              "label": "Lighting",
              "type": "load",
              "value": 70,
              "color": "#737373"
            }
          ],
          "links": [
            {
              "source": "src1",
              "target": "bus",
              "value": 450
            },
            {
              "source": "src2",
              "target": "bus",
              "value": 120
            },
            {
              "source": "bus",
              "target": "load1",
              "value": 300
            },
            {
              "source": "bus",
              "target": "load2",
              "value": 200
            },
            {
              "source": "bus",
              "target": "load3",
              "value": 70
            }
          ]
        }
      },
      "flow_sankey_energy_balance-sankey-with-explicit-loss-branches-dropping-out": {
        "coreWidget": "FLOW / SANKEY",
        "variant": "FLOW_SANKEY_ENERGY_BALANCE",
        "purpose": "Visualize Input vs Useful Output vs Losses",
        "representation": "Sankey with explicit 'Loss' branches dropping out",
        "encoding": {
          "Main Path": "Useful Energy",
          "Diverging Path": "Losses (Red/Hatched)",
          "Node Size": "Efficiency Point"
        },
        "states": {
          "default": "Show losses in red",
          "warning": "Pulse high-loss paths"
        },
        "interactions": [
          "Toggle Loss visibility",
          "Hover for efficiency %"
        ],
        "demoData": {
          "inflow": 1000,
          "outflow": 850,
          "losses": 150,
          "lossBreakdown": [
            {
              "label": "Heat Loss",
              "value": 80
            },
            {
              "label": "Transmission",
              "value": 50
            },
            {
              "label": "Idle",
              "value": 20
            }
          ]
        }
      },
      "flow_sankey_multi_source-many-to-one-flow-diagram": {
        "coreWidget": "FLOW / SANKEY",
        "variant": "FLOW_SANKEY_MULTI_SOURCE",
        "purpose": "Compare multiple generation sources feeding a common load",
        "representation": "Many-to-One Flow Diagram",
        "encoding": {
          "Input Color": "Source Category",
          "Input Height": "Generation Capacity",
          "Merge Point": "Total System Load"
        },
        "states": {
          "default": "Vibrant source colors",
          "inactive": "Greyed out (0 kW sources)"
        },
        "interactions": [
          "Click source to isolate contribution"
        ],
        "demoData": {
          "sources": [
            {
              "label": "Grid Incomer A",
              "value": 800,
              "color": "#2563eb"
            },
            {
              "label": "Grid Incomer B",
              "value": 600,
              "color": "#2563eb"
            },
            {
              "label": "Rooftop Solar",
              "value": 300,
              "color": "#16a34a"
            },
            {
              "label": "Diesel Gen 1",
              "value": 0,
              "color": "#d97706"
            }
          ],
          "totalLoad": 1700
        }
      },
      "flow_sankey_layered-multi-stage-hierarchical-flow": {
        "coreWidget": "FLOW / SANKEY",
        "variant": "FLOW_SANKEY_LAYERED",
        "purpose": "Drill down from plant level to asset level",
        "representation": "Multi-stage hierarchical flow",
        "encoding": {
          "Column": "Hierarchy Level",
          "Splits": "Distribution count"
        },
        "states": {
          "default": "Collapsed high-level view",
          "expanded": "Detailed asset view"
        },
        "interactions": [
          "Click column to expand/collapse"
        ],
        "demoData": {
          "levels": [
            {
              "name": "Utility",
              "value": 2500
            },
            {
              "name": "Feeders",
              "count": 4
            },
            {
              "name": "Sub-Panels",
              "count": 12
            },
            {
              "name": "Assets",
              "count": 45
            }
          ]
        }
      },
      "flow_sankey_time_sliced-sankey-with-time-scrubberplayer": {
        "coreWidget": "FLOW / SANKEY",
        "variant": "FLOW_SANKEY_TIME_SLICED",
        "purpose": "Observe flow changes over time (Day/Shift)",
        "representation": "Sankey with Time Scrubber/Player",
        "encoding": {
          "Animation": "Change in magnitude",
          "Tweening": "Smooth flow transition"
        },
        "states": {
          "playing": "Continuous animation",
          "paused": "Static snapshot"
        },
        "interactions": [
          "Drag timeline",
          "Play/Pause"
        ],
        "demoData": {
          "frames": [
            {
              "time": "08:00",
              "grid": 200,
              "solar": 50
            },
            {
              "time": "10:00",
              "grid": 300,
              "solar": 200
            },
            {
              "time": "12:00",
              "grid": 250,
              "solar": 450
            },
            {
              "time": "14:00",
              "grid": 300,
              "solar": 400
            }
          ]
        }
      }
    }
  },
  "kpi": {
    "name": "KPI",
    "defaultFixture": "kpi_live-standard",
    "variants": {
      "kpi_live-standard": {
        "coreWidget": "KPI",
        "variant": "KPI_LIVE",
        "representation": "Standard",
        "encoding": "Text",
        "layout": {
          "padding": "p-6",
          "radius": "rounded-xl",
          "zones": "flex-col"
        },
        "visual": {
          "background": "bg-white",
          "border": "border-gray-100",
          "theme": "light"
        },
        "states": {
          "default": "text-neutral-900"
        },
        "demoData": {
          "label": "Grid Voltage",
          "value": "238.4",
          "unit": "V"
        }
      },
      "kpi_live-high-contrast": {
        "coreWidget": "KPI",
        "variant": "KPI_LIVE",
        "representation": "High Contrast",
        "encoding": "Text",
        "layout": {
          "padding": "p-6",
          "radius": "rounded-xl",
          "zones": "flex-col"
        },
        "visual": {
          "background": "bg-neutral-900",
          "border": "border-neutral-800",
          "theme": "dark"
        },
        "states": {
          "default": "text-white"
        },
        "demoData": {
          "label": "Active Power",
          "value": "482",
          "unit": "kW"
        }
      },
      "kpi_alert-warning-state": {
        "coreWidget": "KPI",
        "variant": "KPI_ALERT",
        "representation": "Warning State",
        "encoding": "Icon+Text",
        "layout": {
          "padding": "p-6",
          "radius": "rounded-xl",
          "zones": "flex-row"
        },
        "visual": {
          "background": "bg-amber-50",
          "border": "border-amber-100",
          "theme": "light"
        },
        "states": {
          "warning": "text-amber-700"
        },
        "demoData": {
          "label": "Hydraulic Pressure",
          "value": "840",
          "unit": "PSI",
          "state": "warning"
        }
      },
      "kpi_alert-critical-state": {
        "coreWidget": "KPI",
        "variant": "KPI_ALERT",
        "representation": "Critical State",
        "encoding": "Icon+Text",
        "layout": {
          "padding": "p-6",
          "radius": "rounded-xl",
          "zones": "flex-row"
        },
        "visual": {
          "background": "bg-red-50",
          "border": "border-red-100",
          "theme": "light"
        },
        "states": {
          "critical": "text-red-700"
        },
        "demoData": {
          "label": "Core Temp",
          "value": "92.4",
          "unit": "\u00b0C",
          "state": "critical"
        }
      },
      "kpi_accumulated-daily-total": {
        "coreWidget": "KPI",
        "variant": "KPI_ACCUMULATED",
        "representation": "Daily Total",
        "encoding": "Text+Subtext",
        "layout": {
          "padding": "p-6",
          "radius": "rounded-xl",
          "zones": "stack"
        },
        "visual": {
          "background": "bg-white",
          "border": "border-gray-100",
          "theme": "light"
        },
        "states": {
          "default": "text-neutral-900"
        },
        "demoData": {
          "label": "Water Usage",
          "value": "1,240",
          "unit": "L",
          "period": "Since 00:00"
        }
      },
      "kpi_lifecycle-progress-bar": {
        "coreWidget": "KPI",
        "variant": "KPI_LIFECYCLE",
        "representation": "Progress Bar",
        "encoding": "Bar",
        "layout": {
          "padding": "p-6",
          "radius": "rounded-xl",
          "zones": "stack"
        },
        "visual": {
          "background": "bg-white",
          "border": "border-gray-100",
          "theme": "light"
        },
        "states": {
          "default": "text-blue-600"
        },
        "demoData": {
          "label": "Filter Integrity",
          "value": 85,
          "max": 100,
          "unit": "%"
        }
      },
      "kpi_lifecycle-dark-mode-gauge": {
        "coreWidget": "KPI",
        "variant": "KPI_LIFECYCLE",
        "representation": "Dark Mode Gauge",
        "encoding": "Bar",
        "layout": {
          "padding": "p-6",
          "radius": "rounded-xl",
          "zones": "stack"
        },
        "visual": {
          "background": "bg-neutral-900",
          "border": "border-neutral-800",
          "theme": "dark"
        },
        "states": {
          "default": "text-green-500"
        },
        "demoData": {
          "label": "Battery Level",
          "value": 42,
          "max": 100,
          "unit": "%"
        }
      },
      "kpi_status-badge": {
        "coreWidget": "KPI",
        "variant": "KPI_STATUS",
        "representation": "Badge",
        "encoding": "Status Dot",
        "layout": {
          "padding": "p-6",
          "radius": "rounded-xl",
          "zones": "flex-between"
        },
        "visual": {
          "background": "bg-white",
          "border": "border-gray-100",
          "theme": "light"
        },
        "states": {
          "default": "text-neutral-900"
        },
        "demoData": {
          "label": "Main Pump A",
          "status": "Operational",
          "statusColor": "bg-green-500"
        }
      },
      "kpi_status-offline": {
        "coreWidget": "KPI",
        "variant": "KPI_STATUS",
        "representation": "Offline",
        "encoding": "Status Dot",
        "layout": {
          "padding": "p-6",
          "radius": "rounded-xl",
          "zones": "flex-between"
        },
        "visual": {
          "background": "bg-neutral-50",
          "border": "border-neutral-200",
          "theme": "light"
        },
        "states": {
          "default": "text-neutral-500"
        },
        "demoData": {
          "label": "Backup Generator",
          "status": "Standby",
          "statusColor": "bg-neutral-400"
        }
      }
    }
  },
  "matrix-heatmap": {
    "name": "Matrix / Heatmap",
    "defaultFixture": "value-heatmap",
    "variants": {
      "value-heatmap": {
        "spec": {
          "variant": "VALUE_HEATMAP",
          "demoData": {
            "label": "Line Efficiency (Shift vs Machine)",
            "timestamp": "2023-11-01T14:30:00Z",
            "dataset": {
              "min": 60,
              "max": 100,
              "unit": "%",
              "rows": [
                {
                  "id": "m1",
                  "label": "CNC-01"
                },
                {
                  "id": "m2",
                  "label": "CNC-02"
                },
                {
                  "id": "m3",
                  "label": "CNC-03"
                },
                {
                  "id": "m4",
                  "label": "CNC-04"
                },
                {
                  "id": "m5",
                  "label": "PRESS-A"
                },
                {
                  "id": "m6",
                  "label": "PRESS-B"
                }
              ],
              "cols": [
                {
                  "id": "s1",
                  "label": "Shift 1"
                },
                {
                  "id": "s2",
                  "label": "Shift 2"
                },
                {
                  "id": "s3",
                  "label": "Shift 3"
                }
              ],
              "cells": [
                {
                  "rowId": "m1",
                  "colId": "s1",
                  "value": 98
                },
                {
                  "rowId": "m1",
                  "colId": "s2",
                  "value": 95
                },
                {
                  "rowId": "m1",
                  "colId": "s3",
                  "value": 88
                },
                {
                  "rowId": "m2",
                  "colId": "s1",
                  "value": 92
                },
                {
                  "rowId": "m2",
                  "colId": "s2",
                  "value": 65
                },
                {
                  "rowId": "m2",
                  "colId": "s3",
                  "value": 70
                },
                {
                  "rowId": "m3",
                  "colId": "s1",
                  "value": 85
                },
                {
                  "rowId": "m3",
                  "colId": "s2",
                  "value": 88
                },
                {
                  "rowId": "m3",
                  "colId": "s3",
                  "value": 90
                },
                {
                  "rowId": "m4",
                  "colId": "s1",
                  "value": 45
                },
                {
                  "rowId": "m4",
                  "colId": "s2",
                  "value": 50
                },
                {
                  "rowId": "m4",
                  "colId": "s3",
                  "value": 0
                },
                {
                  "rowId": "m5",
                  "colId": "s1",
                  "value": 99
                },
                {
                  "rowId": "m5",
                  "colId": "s2",
                  "value": 98
                },
                {
                  "rowId": "m5",
                  "colId": "s3",
                  "value": 99
                },
                {
                  "rowId": "m6",
                  "colId": "s1",
                  "value": 78
                },
                {
                  "rowId": "m6",
                  "colId": "s2",
                  "value": 82
                },
                {
                  "rowId": "m6",
                  "colId": "s3",
                  "value": 80
                }
              ]
            }
          }
        },
        "state": {
          "isLoading": false,
          "isEmpty": false,
          "isError": false
        }
      },
      "correlation-matrix": {
        "spec": {
          "variant": "CORRELATION_MATRIX",
          "demoData": {
            "label": "Sensor Correlation Analysis",
            "timestamp": "2023-11-01T14:30:00Z",
            "dataset": {
              "min": -1,
              "max": 1,
              "unit": "r",
              "rows": [
                {
                  "id": "vib",
                  "label": "Vib"
                },
                {
                  "id": "tmp",
                  "label": "Temp"
                },
                {
                  "id": "spd",
                  "label": "RPM"
                },
                {
                  "id": "prs",
                  "label": "Pres"
                },
                {
                  "id": "cur",
                  "label": "Amp"
                }
              ],
              "cols": [
                {
                  "id": "vib",
                  "label": "Vib"
                },
                {
                  "id": "tmp",
                  "label": "Temp"
                },
                {
                  "id": "spd",
                  "label": "RPM"
                },
                {
                  "id": "prs",
                  "label": "Pres"
                },
                {
                  "id": "cur",
                  "label": "Amp"
                }
              ],
              "cells": [
                {
                  "rowId": "vib",
                  "colId": "vib",
                  "value": 1
                },
                {
                  "rowId": "vib",
                  "colId": "tmp",
                  "value": 0.85
                },
                {
                  "rowId": "vib",
                  "colId": "spd",
                  "value": 0.92
                },
                {
                  "rowId": "vib",
                  "colId": "prs",
                  "value": -0.1
                },
                {
                  "rowId": "vib",
                  "colId": "cur",
                  "value": 0.4
                },
                {
                  "rowId": "tmp",
                  "colId": "vib",
                  "value": 0.85
                },
                {
                  "rowId": "tmp",
                  "colId": "tmp",
                  "value": 1
                },
                {
                  "rowId": "tmp",
                  "colId": "spd",
                  "value": 0.7
                },
                {
                  "rowId": "tmp",
                  "colId": "prs",
                  "value": 0.2
                },
                {
                  "rowId": "tmp",
                  "colId": "cur",
                  "value": 0.5
                },
                {
                  "rowId": "spd",
                  "colId": "vib",
                  "value": 0.92
                },
                {
                  "rowId": "spd",
                  "colId": "tmp",
                  "value": 0.7
                },
                {
                  "rowId": "spd",
                  "colId": "spd",
                  "value": 1
                },
                {
                  "rowId": "spd",
                  "colId": "prs",
                  "value": -0.05
                },
                {
                  "rowId": "spd",
                  "colId": "cur",
                  "value": 0.8
                },
                {
                  "rowId": "prs",
                  "colId": "vib",
                  "value": -0.1
                },
                {
                  "rowId": "prs",
                  "colId": "tmp",
                  "value": 0.2
                },
                {
                  "rowId": "prs",
                  "colId": "spd",
                  "value": -0.05
                },
                {
                  "rowId": "prs",
                  "colId": "prs",
                  "value": 1
                },
                {
                  "rowId": "prs",
                  "colId": "cur",
                  "value": 0.1
                },
                {
                  "rowId": "cur",
                  "colId": "vib",
                  "value": 0.4
                },
                {
                  "rowId": "cur",
                  "colId": "tmp",
                  "value": 0.5
                },
                {
                  "rowId": "cur",
                  "colId": "spd",
                  "value": 0.8
                },
                {
                  "rowId": "cur",
                  "colId": "prs",
                  "value": 0.1
                },
                {
                  "rowId": "cur",
                  "colId": "cur",
                  "value": 1
                }
              ]
            }
          }
        },
        "state": {
          "isLoading": false,
          "isEmpty": false,
          "isError": false
        }
      },
      "calendar-heatmap": {
        "spec": {
          "variant": "CALENDAR_HEATMAP",
          "demoData": {
            "label": "October Availability",
            "timestamp": "2023-10-31T23:59:00Z",
            "dataset": {
              "min": 0,
              "max": 24,
              "unit": "hrs",
              "rows": [
                {
                  "id": "w1",
                  "label": "W40"
                },
                {
                  "id": "w2",
                  "label": "W41"
                },
                {
                  "id": "w3",
                  "label": "W42"
                },
                {
                  "id": "w4",
                  "label": "W43"
                }
              ],
              "cols": [
                {
                  "id": "mon",
                  "label": "M"
                },
                {
                  "id": "tue",
                  "label": "T"
                },
                {
                  "id": "wed",
                  "label": "W"
                },
                {
                  "id": "thu",
                  "label": "T"
                },
                {
                  "id": "fri",
                  "label": "F"
                },
                {
                  "id": "sat",
                  "label": "S"
                },
                {
                  "id": "sun",
                  "label": "S"
                }
              ],
              "cells": [
                {
                  "rowId": "w1",
                  "colId": "mon",
                  "value": 24
                },
                {
                  "rowId": "w1",
                  "colId": "tue",
                  "value": 23.5
                },
                {
                  "rowId": "w1",
                  "colId": "wed",
                  "value": 24
                },
                {
                  "rowId": "w1",
                  "colId": "thu",
                  "value": 18
                },
                {
                  "rowId": "w1",
                  "colId": "fri",
                  "value": 24
                },
                {
                  "rowId": "w1",
                  "colId": "sat",
                  "value": 24
                },
                {
                  "rowId": "w1",
                  "colId": "sun",
                  "value": 12
                },
                {
                  "rowId": "w2",
                  "colId": "mon",
                  "value": 24
                },
                {
                  "rowId": "w2",
                  "colId": "tue",
                  "value": 24
                },
                {
                  "rowId": "w2",
                  "colId": "wed",
                  "value": 10
                },
                {
                  "rowId": "w2",
                  "colId": "thu",
                  "value": 0
                },
                {
                  "rowId": "w2",
                  "colId": "fri",
                  "value": 22
                },
                {
                  "rowId": "w2",
                  "colId": "sat",
                  "value": 24
                },
                {
                  "rowId": "w2",
                  "colId": "sun",
                  "value": 24
                },
                {
                  "rowId": "w3",
                  "colId": "mon",
                  "value": 24
                },
                {
                  "rowId": "w3",
                  "colId": "tue",
                  "value": 24
                },
                {
                  "rowId": "w3",
                  "colId": "wed",
                  "value": 23
                },
                {
                  "rowId": "w3",
                  "colId": "thu",
                  "value": 24
                },
                {
                  "rowId": "w3",
                  "colId": "fri",
                  "value": 24
                },
                {
                  "rowId": "w3",
                  "colId": "sat",
                  "value": 16
                },
                {
                  "rowId": "w3",
                  "colId": "sun",
                  "value": 24
                },
                {
                  "rowId": "w4",
                  "colId": "mon",
                  "value": 22
                },
                {
                  "rowId": "w4",
                  "colId": "tue",
                  "value": 24
                },
                {
                  "rowId": "w4",
                  "colId": "wed",
                  "value": 24
                },
                {
                  "rowId": "w4",
                  "colId": "thu",
                  "value": 24
                },
                {
                  "rowId": "w4",
                  "colId": "fri",
                  "value": 24
                },
                {
                  "rowId": "w4",
                  "colId": "sat",
                  "value": 24
                },
                {
                  "rowId": "w4",
                  "colId": "sun",
                  "value": 24
                }
              ]
            }
          }
        },
        "state": {
          "isLoading": false,
          "isEmpty": false,
          "isError": false
        }
      },
      "status-matrix": {
        "spec": {
          "variant": "STATUS_MATRIX",
          "demoData": {
            "label": "Fleet Battery Status",
            "timestamp": "2023-11-01T14:30:00Z",
            "dataset": {
              "min": 0,
              "max": 3,
              "rows": [
                {
                  "id": "r1",
                  "label": "Zone A"
                },
                {
                  "id": "r2",
                  "label": "Zone B"
                },
                {
                  "id": "r3",
                  "label": "Zone C"
                }
              ],
              "cols": [
                {
                  "id": "c1",
                  "label": "U-01"
                },
                {
                  "id": "c2",
                  "label": "U-02"
                },
                {
                  "id": "c3",
                  "label": "U-03"
                },
                {
                  "id": "c4",
                  "label": "U-04"
                },
                {
                  "id": "c5",
                  "label": "U-05"
                }
              ],
              "cells": [
                {
                  "rowId": "r1",
                  "colId": "c1",
                  "value": 0,
                  "severity": "normal"
                },
                {
                  "rowId": "r1",
                  "colId": "c2",
                  "value": 0,
                  "severity": "normal"
                },
                {
                  "rowId": "r1",
                  "colId": "c3",
                  "value": 1,
                  "severity": "warning"
                },
                {
                  "rowId": "r1",
                  "colId": "c4",
                  "value": 0,
                  "severity": "normal"
                },
                {
                  "rowId": "r1",
                  "colId": "c5",
                  "value": 0,
                  "severity": "normal"
                },
                {
                  "rowId": "r2",
                  "colId": "c1",
                  "value": 0,
                  "severity": "normal"
                },
                {
                  "rowId": "r2",
                  "colId": "c2",
                  "value": 2,
                  "severity": "critical"
                },
                {
                  "rowId": "r2",
                  "colId": "c3",
                  "value": 2,
                  "severity": "critical"
                },
                {
                  "rowId": "r2",
                  "colId": "c4",
                  "value": 0,
                  "severity": "normal"
                },
                {
                  "rowId": "r2",
                  "colId": "c5",
                  "value": 3,
                  "severity": "unknown"
                },
                {
                  "rowId": "r3",
                  "colId": "c1",
                  "value": 0,
                  "severity": "normal"
                },
                {
                  "rowId": "r3",
                  "colId": "c2",
                  "value": 0,
                  "severity": "normal"
                },
                {
                  "rowId": "r3",
                  "colId": "c3",
                  "value": 0,
                  "severity": "normal"
                },
                {
                  "rowId": "r3",
                  "colId": "c4",
                  "value": 1,
                  "severity": "warning"
                },
                {
                  "rowId": "r3",
                  "colId": "c5",
                  "value": 0,
                  "severity": "normal"
                }
              ]
            }
          }
        },
        "state": {
          "isLoading": false,
          "isEmpty": false,
          "isError": false
        }
      },
      "density-matrix": {
        "spec": {
          "variant": "DENSITY_MATRIX",
          "demoData": {
            "label": "Surface Defect Density",
            "timestamp": "2023-11-01T14:30:00Z",
            "dataset": {
              "min": 0,
              "max": 50,
              "unit": "defects",
              "rows": [
                {
                  "id": "y4",
                  "label": "80-100"
                },
                {
                  "id": "y3",
                  "label": "60-80"
                },
                {
                  "id": "y2",
                  "label": "40-60"
                },
                {
                  "id": "y1",
                  "label": "20-40"
                },
                {
                  "id": "y0",
                  "label": "0-20"
                }
              ],
              "cols": [
                {
                  "id": "x0",
                  "label": "0-20"
                },
                {
                  "id": "x1",
                  "label": "20-40"
                },
                {
                  "id": "x2",
                  "label": "40-60"
                },
                {
                  "id": "x3",
                  "label": "60-80"
                },
                {
                  "id": "x4",
                  "label": "80-100"
                }
              ],
              "cells": [
                {
                  "rowId": "y0",
                  "colId": "x0",
                  "value": 2
                },
                {
                  "rowId": "y0",
                  "colId": "x1",
                  "value": 5
                },
                {
                  "rowId": "y0",
                  "colId": "x2",
                  "value": 1
                },
                {
                  "rowId": "y0",
                  "colId": "x3",
                  "value": 0
                },
                {
                  "rowId": "y0",
                  "colId": "x4",
                  "value": 0
                },
                {
                  "rowId": "y1",
                  "colId": "x0",
                  "value": 1
                },
                {
                  "rowId": "y1",
                  "colId": "x1",
                  "value": 8
                },
                {
                  "rowId": "y1",
                  "colId": "x2",
                  "value": 3
                },
                {
                  "rowId": "y1",
                  "colId": "x3",
                  "value": 1
                },
                {
                  "rowId": "y1",
                  "colId": "x4",
                  "value": 0
                },
                {
                  "rowId": "y2",
                  "colId": "x0",
                  "value": 0
                },
                {
                  "rowId": "y2",
                  "colId": "x1",
                  "value": 12
                },
                {
                  "rowId": "y2",
                  "colId": "x2",
                  "value": 45
                },
                {
                  "rowId": "y2",
                  "colId": "x3",
                  "value": 38
                },
                {
                  "rowId": "y2",
                  "colId": "x4",
                  "value": 5
                },
                {
                  "rowId": "y3",
                  "colId": "x0",
                  "value": 2
                },
                {
                  "rowId": "y3",
                  "colId": "x1",
                  "value": 6
                },
                {
                  "rowId": "y3",
                  "colId": "x2",
                  "value": 30
                },
                {
                  "rowId": "y3",
                  "colId": "x3",
                  "value": 25
                },
                {
                  "rowId": "y3",
                  "colId": "x4",
                  "value": 4
                },
                {
                  "rowId": "y4",
                  "colId": "x0",
                  "value": 0
                },
                {
                  "rowId": "y4",
                  "colId": "x1",
                  "value": 1
                },
                {
                  "rowId": "y4",
                  "colId": "x2",
                  "value": 8
                },
                {
                  "rowId": "y4",
                  "colId": "x3",
                  "value": 5
                },
                {
                  "rowId": "y4",
                  "colId": "x4",
                  "value": 0
                }
              ]
            }
          }
        },
        "state": {
          "isLoading": false,
          "isEmpty": false,
          "isError": false
        }
      }
    }
  },
  "peoplehexgrid": {
    "name": "PeopleHexGrid",
    "defaultFixture": "default-render",
    "variants": {
      "default-render": {
        "teams": [
          { "id": "t1", "label": "Module A", "status": "full", "coverage": 98 },
          { "id": "t2", "label": "Module B", "status": "gap", "coverage": 82 },
          { "id": "t3", "label": "Module C", "status": "full", "coverage": 95 },
          { "id": "t4", "label": "Logistics", "status": "full", "coverage": 100 },
          { "id": "t5", "label": "QA Shift", "status": "crit", "coverage": 64 },
          { "id": "t6", "label": "Maintenance", "status": "gap", "coverage": 78 },
          { "id": "t7", "label": "Safety", "status": "full", "coverage": 100 },
          { "id": "t8", "label": "Pack", "status": "full", "coverage": 92 },
          { "id": "t9", "label": "Clean Rm", "status": "gap", "coverage": 88 },
          { "id": "t10", "label": "Dispatch", "status": "full", "coverage": 96 }
        ],
        "accent": "#a855f7"
      }
    }
  },
  "peoplenetwork": {
    "name": "PeopleNetwork",
    "defaultFixture": "default-render",
    "variants": {
      "default-render": {}
    }
  },
  "peopleview": {
    "name": "PeopleView",
    "defaultFixture": "default-render",
    "variants": {
      "default-render": {}
    }
  },
  "supplychainglobe": {
    "name": "SupplyChainGlobe",
    "defaultFixture": "default-render",
    "variants": {
      "default-render": {
        "lanes": [
          { "id": "l1", "from": { "lat": 17.385, "lng": 78.486 }, "to": { "lat": 51.507, "lng": -0.127 }, "status": "live" },
          { "id": "l2", "from": { "lat": 17.385, "lng": 78.486 }, "to": { "lat": 35.689, "lng": 139.691 }, "status": "live" },
          { "id": "l3", "from": { "lat": 22.396, "lng": 114.109 }, "to": { "lat": 17.385, "lng": 78.486 }, "status": "held" },
          { "id": "l4", "from": { "lat": 37.774, "lng": -122.419 }, "to": { "lat": 17.385, "lng": 78.486 }, "status": "delivered" },
          { "id": "l5", "from": { "lat": 1.352, "lng": 103.819 }, "to": { "lat": 48.856, "lng": 2.352 }, "status": "live" },
          { "id": "l6", "from": { "lat": 25.276, "lng": 55.296 }, "to": { "lat": 17.385, "lng": 78.486 }, "status": "delivered" }
        ],
        "accent": "#22d3ee"
      }
    }
  },
  "timeline": {
    "name": "Timeline",
    "defaultFixture": "linear-incident-timeline",
    "variants": {
      "linear-incident-timeline": {
        "id": "tl-1",
        "title": "Linear Incident Timeline",
        "description": "Discrete alarms and system events visualized on a single temporal axis.",
        "variant": "linear",
        "range": {
          "start": "08:00",
          "end": "16:00"
        },
        "events": [
          {
            "id": "e1",
            "startTime": "08:30",
            "label": "System Start",
            "status": "success",
            "icon": "check"
          },
          {
            "id": "e2",
            "startTime": "10:15",
            "label": "Pressure Warning",
            "status": "warning",
            "icon": "alert"
          },
          {
            "id": "e3",
            "startTime": "11:45",
            "label": "Pump Failure",
            "status": "critical",
            "icon": "error"
          },
          {
            "id": "e4",
            "startTime": "12:30",
            "label": "Manual Override",
            "status": "neutral",
            "icon": "zap"
          },
          {
            "id": "e5",
            "startTime": "14:00",
            "label": "System Restored",
            "status": "success",
            "icon": "check"
          }
        ]
      },
      "machine-state-timeline": {
        "id": "tl-2",
        "title": "Machine State Timeline",
        "description": "Continuous status monitoring (Run/Idle/Fault) for a single asset.",
        "variant": "status",
        "range": {
          "start": "09:00",
          "end": "13:00"
        },
        "events": [
          {
            "id": "s1",
            "startTime": "09:00",
            "endTime": "10:30",
            "label": "Running",
            "status": "success"
          },
          {
            "id": "s2",
            "startTime": "10:30",
            "endTime": "10:45",
            "label": "Idle",
            "status": "idle"
          },
          {
            "id": "s3",
            "startTime": "10:45",
            "endTime": "11:15",
            "label": "Running",
            "status": "success"
          },
          {
            "id": "s4",
            "startTime": "11:15",
            "endTime": "12:00",
            "label": "Fault",
            "status": "critical"
          },
          {
            "id": "s5",
            "startTime": "12:00",
            "endTime": "13:00",
            "label": "Maintenance",
            "status": "warning"
          }
        ]
      },
      "multi-lane-shift-schedule": {
        "id": "tl-3",
        "title": "Multi-Lane Shift Schedule",
        "description": "Operational comparison across multiple production lines.",
        "variant": "multilane",
        "range": {
          "start": "06:00",
          "end": "14:00"
        },
        "lanes": [
          {
            "id": "L1",
            "label": "Line A"
          },
          {
            "id": "L2",
            "label": "Line B"
          },
          {
            "id": "L3",
            "label": "Line C"
          }
        ],
        "events": [
          {
            "id": "la1",
            "laneId": "L1",
            "startTime": "06:00",
            "endTime": "10:00",
            "label": "Production",
            "status": "success"
          },
          {
            "id": "la2",
            "laneId": "L1",
            "startTime": "10:00",
            "endTime": "10:30",
            "label": "Break",
            "status": "idle"
          },
          {
            "id": "la3",
            "laneId": "L1",
            "startTime": "10:30",
            "endTime": "14:00",
            "label": "Production",
            "status": "success"
          },
          {
            "id": "lb1",
            "laneId": "L2",
            "startTime": "06:00",
            "endTime": "08:00",
            "label": "Setup",
            "status": "warning"
          },
          {
            "id": "lb2",
            "laneId": "L2",
            "startTime": "08:00",
            "endTime": "12:00",
            "label": "Production",
            "status": "success"
          },
          {
            "id": "lb3",
            "laneId": "L2",
            "startTime": "12:00",
            "endTime": "14:00",
            "label": "Cleaning",
            "status": "neutral"
          },
          {
            "id": "lc1",
            "laneId": "L3",
            "startTime": "06:00",
            "endTime": "09:15",
            "label": "Down",
            "status": "critical"
          },
          {
            "id": "lc2",
            "laneId": "L3",
            "startTime": "09:15",
            "endTime": "14:00",
            "label": "Production",
            "status": "success"
          }
        ]
      },
      "forensic-annotated-view": {
        "id": "tl-4",
        "title": "Forensic Annotated View",
        "description": "Layered visualization for Root Cause Analysis (RCA) with human annotations overlaying machine state.",
        "variant": "forensic",
        "range": {
          "start": "2023-11-15T09:30:00",
          "end": "2023-11-15T11:00:00"
        },
        "stateSegments": [
          {
            "id": "s1",
            "start": "2023-11-15T09:30:00",
            "end": "2023-11-15T09:45:00",
            "status": "success",
            "label": "RUN"
          },
          {
            "id": "s2",
            "start": "2023-11-15T09:45:00",
            "end": "2023-11-15T09:55:00",
            "status": "warning",
            "label": "UNSTABLE"
          },
          {
            "id": "s3",
            "start": "2023-11-15T09:55:00",
            "end": "2023-11-15T10:30:00",
            "status": "critical",
            "label": "TRIP"
          },
          {
            "id": "s4",
            "start": "2023-11-15T10:30:00",
            "end": "2023-11-15T11:00:00",
            "status": "maintenance",
            "label": "INVESTIGATION"
          }
        ],
        "events": [
          {
            "id": "e1",
            "startTime": "2023-11-15T09:45:10",
            "label": "Vibration Alert",
            "status": "warning",
            "icon": "alert"
          },
          {
            "id": "e2",
            "startTime": "2023-11-15T09:55:02",
            "label": "E-STOP Trigger",
            "status": "critical",
            "icon": "error"
          }
        ],
        "annotations": [
          {
            "id": "a1",
            "time": "2023-11-15T09:50:00",
            "type": "operator_note",
            "label": "Loud noise reported",
            "author": "Operator A"
          },
          {
            "id": "a2",
            "time": "2023-11-15T10:15:00",
            "type": "rca_finding",
            "label": "Bearing Seizure Confirmed",
            "author": "Lead Eng"
          },
          {
            "id": "a3",
            "time": "2023-11-15T10:45:00",
            "type": "action",
            "label": "Parts Ordered",
            "author": "System"
          }
        ],
        "incidentWindow": {
          "start": "2023-11-15T09:45:00",
          "end": "2023-11-15T10:30:00",
          "label": "Incident #4492"
        }
      },
      "log-density-burst-analysis": {
        "id": "tl-5",
        "title": "Log Density & Burst Analysis",
        "description": "High-frequency event visualization with heat-density mapping and clustering.",
        "variant": "dense",
        "range": {
          "start": "08:00:00",
          "end": "08:30:00"
        },
        "events": [
          {
            "id": "d1",
            "startTime": "08:02:00",
            "label": "Heartbeat",
            "status": "neutral"
          },
          {
            "id": "d2",
            "startTime": "08:08:30",
            "label": "Auth Check",
            "status": "success"
          },
          {
            "id": "d3",
            "startTime": "08:12:15",
            "label": "Latency Spike",
            "status": "warning"
          },
          {
            "id": "d_burst",
            "startTime": "08:15:10",
            "label": "Packet Loss",
            "status": "critical",
            "clusterCount": 42,
            "severityBreakdown": {
              "critical": 24,
              "warning": 12,
              "info": 6
            },
            "burstDetails": {
              "span": "3.2s",
              "dominantType": "Packet Loss",
              "correlationId": "NET-449"
            }
          },
          {
            "id": "d4",
            "startTime": "08:22:00",
            "label": "Retry",
            "status": "warning"
          },
          {
            "id": "d5",
            "startTime": "08:28:00",
            "label": "Connected",
            "status": "success"
          }
        ]
      }
    }
  },
  "trend": {
    "name": "Trend",
    "defaultFixture": "trend_live-line",
    "variants": {
      "trend_live-line": {
        "coreWidget": "Trend",
        "variant": "TREND_LIVE",
        "representation": "Line",
        "encoding": "Continuous Value",
        "layout": {
          "aspectRatio": "16:9",
          "padding": "p-4",
          "radius": "rounded-xl",
          "zones": "header-chart"
        },
        "visual": {
          "background": "bg-white",
          "border": "border-gray-100",
          "theme": "light",
          "gridlines": "stroke-gray-100 stroke-dasharray-4",
          "axisStyling": "text-xs text-gray-400",
          "colors": [
            "#2563eb"
          ]
        },
        "states": {
          "default": "opacity-100",
          "loading": "animate-pulse"
        },
        "interactions": {
          "hover": "tooltip",
          "click": "openTimeline",
          "pan": "enabled"
        },
        "tooltip": {
          "showTimestamp": true,
          "showValue": true,
          "showUnit": true
        },
        "demoData": {
          "label": "Grid Frequency (Hz)",
          "timeRange": "Last 10 min",
          "unit": "Hz",
          "timeSeries": [
            {
              "time": "10:30",
              "value": 50.0461065798708
            },
            {
              "time": "10:31",
              "value": 49.90095575522087
            },
            {
              "time": "10:32",
              "value": 49.91084510608234
            },
            {
              "time": "10:33",
              "value": 49.987915336742496
            },
            {
              "time": "10:34",
              "value": 49.956816535239426
            },
            {
              "time": "10:35",
              "value": 49.92199707367458
            },
            {
              "time": "10:36",
              "value": 49.93686846453948
            },
            {
              "time": "10:37",
              "value": 49.97787352359903
            },
            {
              "time": "10:38",
              "value": 49.96604399905409
            },
            {
              "time": "10:39",
              "value": 49.98375264071542
            },
            {
              "time": "10:40",
              "value": 50.06375639272408
            },
            {
              "time": "10:41",
              "value": 50.05390976664452
            },
            {
              "time": "10:42",
              "value": 49.90270927175201
            },
            {
              "time": "10:43",
              "value": 50.09243393392763
            },
            {
              "time": "10:44",
              "value": 50.05759969322143
            },
            {
              "time": "10:45",
              "value": 50.065655897173755
            },
            {
              "time": "10:46",
              "value": 49.99045251444593
            },
            {
              "time": "10:47",
              "value": 50.013512592625545
            },
            {
              "time": "10:48",
              "value": 49.91755943195498
            },
            {
              "time": "10:49",
              "value": 49.93489853549759
            }
          ]
        }
      },
      "trend_live-area": {
        "coreWidget": "Trend",
        "variant": "TREND_LIVE",
        "representation": "Area",
        "encoding": "Continuous Value",
        "layout": {
          "aspectRatio": "16:9",
          "padding": "p-4",
          "radius": "rounded-xl",
          "zones": "header-chart"
        },
        "visual": {
          "background": "bg-neutral-900",
          "border": "border-neutral-800",
          "theme": "dark",
          "gridlines": "stroke-neutral-800",
          "axisStyling": "text-xs text-neutral-500",
          "colors": [
            "#10b981"
          ]
        },
        "states": {
          "default": "opacity-100"
        },
        "interactions": {
          "hover": "tooltip"
        },
        "tooltip": {
          "showTimestamp": true,
          "showValue": true,
          "showUnit": true
        },
        "demoData": {
          "label": "Live Load (kW)",
          "timeRange": "Last 10 min",
          "unit": "kW",
          "timeSeries": [
            {
              "time": "10:30",
              "value": 426.6334134699687
            },
            {
              "time": "10:31",
              "value": 448.0362934535094
            },
            {
              "time": "10:32",
              "value": 434.8242109828984
            },
            {
              "time": "10:33",
              "value": 434.3859479061838
            },
            {
              "time": "10:34",
              "value": 446.7331036417136
            },
            {
              "time": "10:35",
              "value": 459.1426923001811
            },
            {
              "time": "10:36",
              "value": 458.8469906133787
            },
            {
              "time": "10:37",
              "value": 424.0500706571146
            },
            {
              "time": "10:38",
              "value": 460.16952589030757
            },
            {
              "time": "10:39",
              "value": 462.5039768343818
            },
            {
              "time": "10:40",
              "value": 425.6164445577721
            },
            {
              "time": "10:41",
              "value": 453.5399951529945
            },
            {
              "time": "10:42",
              "value": 462.54246611754036
            },
            {
              "time": "10:43",
              "value": 444.5263639833649
            },
            {
              "time": "10:44",
              "value": 428.73850903004717
            },
            {
              "time": "10:45",
              "value": 444.9367963176666
            },
            {
              "time": "10:46",
              "value": 458.2635090725685
            },
            {
              "time": "10:47",
              "value": 432.2993738096513
            },
            {
              "time": "10:48",
              "value": 447.79198557493964
            },
            {
              "time": "10:49",
              "value": 439.52977076406205
            }
          ]
        }
      },
      "trend_standard-step-line": {
        "coreWidget": "Trend",
        "variant": "TREND_STANDARD",
        "representation": "Step Line",
        "encoding": "Continuous Value",
        "layout": {
          "aspectRatio": "4:3",
          "padding": "p-6",
          "radius": "rounded-xl",
          "zones": "header-chart-footer"
        },
        "visual": {
          "background": "bg-white",
          "border": "border-gray-100",
          "theme": "light",
          "gridlines": "stroke-gray-100",
          "axisStyling": "text-xs text-gray-400",
          "colors": [
            "#6366f1"
          ]
        },
        "interactions": {
          "hover": "tooltip",
          "zoom": "enabled"
        },
        "tooltip": {
          "showTimestamp": true,
          "showValue": true
        },
        "demoData": {
          "label": "Pump State Logic",
          "timeRange": "Last 24h",
          "unit": "State",
          "timeSeries": [
            {
              "time": "0:00",
              "value": 0
            },
            {
              "time": "2:00",
              "value": 0
            },
            {
              "time": "4:00",
              "value": 0
            },
            {
              "time": "6:00",
              "value": 0
            },
            {
              "time": "8:00",
              "value": 1
            },
            {
              "time": "10:00",
              "value": 1
            },
            {
              "time": "12:00",
              "value": 1
            },
            {
              "time": "14:00",
              "value": 1
            },
            {
              "time": "16:00",
              "value": 0
            },
            {
              "time": "18:00",
              "value": 0
            },
            {
              "time": "20:00",
              "value": 0
            },
            {
              "time": "22:00",
              "value": 0
            }
          ]
        }
      },
      "trend_phased-rgb-phase-line": {
        "coreWidget": "Trend",
        "variant": "TREND_PHASED",
        "representation": "RGB Phase Line",
        "encoding": "Multi-series",
        "layout": {
          "aspectRatio": "16:9",
          "padding": "p-4",
          "radius": "rounded-xl",
          "zones": "header-chart-legend"
        },
        "visual": {
          "background": "bg-neutral-950",
          "border": "border-neutral-900",
          "theme": "dark",
          "gridlines": "stroke-neutral-800",
          "axisStyling": "text-xs text-neutral-500",
          "colors": [
            "#ef4444",
            "#eab308",
            "#3b82f6"
          ]
        },
        "interactions": {
          "hover": "tooltip",
          "toggleSeries": true
        },
        "tooltip": {
          "showTimestamp": true,
          "showValue": true
        },
        "demoData": {
          "label": "Phase Voltages (V)",
          "timeRange": "Realtime",
          "unit": "V",
          "timeSeries": [
            {
              "time": "T-15s",
              "r": 233.40349570840513,
              "y": 232.61141176906122,
              "b": 235.74678551580482
            },
            {
              "time": "T-14s",
              "r": 232.1074451323688,
              "y": 229.35483000759467,
              "b": 236.83427694354253
            },
            {
              "time": "T-13s",
              "r": 234.8932995096129,
              "y": 232.10489134287695,
              "b": 233.5008255198097
            },
            {
              "time": "T-12s",
              "r": 234.24382107264069,
              "y": 229.39429320344843,
              "b": 236.47184233663168
            },
            {
              "time": "T-11s",
              "r": 230.4403541728748,
              "y": 230.2948262258466,
              "b": 232.54606644436856
            },
            {
              "time": "T-10s",
              "r": 232.85341977276153,
              "y": 229.94327111024572,
              "b": 235.09736993152114
            },
            {
              "time": "T-9s",
              "r": 234.86684686077228,
              "y": 230.22625746182902,
              "b": 234.77228833439978
            },
            {
              "time": "T-8s",
              "r": 232.73402420796546,
              "y": 231.11555182924837,
              "b": 236.0806119250385
            },
            {
              "time": "T-7s",
              "r": 232.54070925872833,
              "y": 229.18523802222623,
              "b": 235.7870827302402
            },
            {
              "time": "T-6s",
              "r": 230.27161833045693,
              "y": 230.92292496576073,
              "b": 234.98625290548776
            },
            {
              "time": "T-5s",
              "r": 232.18005084167476,
              "y": 228.29864786237462,
              "b": 234.5350832220268
            },
            {
              "time": "T-4s",
              "r": 230.14899464587046,
              "y": 232.67783773763733,
              "b": 233.75236981734432
            },
            {
              "time": "T-3s",
              "r": 234.56209979876752,
              "y": 231.4105724811129,
              "b": 233.61733930411845
            },
            {
              "time": "T-2s",
              "r": 232.4743798442209,
              "y": 230.44646985371858,
              "b": 235.50271623731385
            },
            {
              "time": "T-1s",
              "r": 233.10638565825627,
              "y": 228.22514292937677,
              "b": 234.56844168306674
            }
          ]
        }
      },
      "trend_alert_context-line-threshold": {
        "coreWidget": "Trend",
        "variant": "TREND_ALERT_CONTEXT",
        "representation": "Line + Threshold",
        "encoding": "Continuous Value",
        "layout": {
          "aspectRatio": "16:9",
          "padding": "p-4",
          "radius": "rounded-xl",
          "zones": "header-chart"
        },
        "visual": {
          "background": "bg-white",
          "border": "border-red-100",
          "theme": "light",
          "gridlines": "stroke-gray-100",
          "axisStyling": "text-xs text-gray-400",
          "colors": [
            "#ef4444"
          ]
        },
        "states": {
          "critical": "border-red-500 bg-red-50"
        },
        "interactions": {
          "hover": "tooltip",
          "click": "openAnalysis"
        },
        "tooltip": {
          "showTimestamp": true,
          "showThreshold": true
        },
        "demoData": {
          "label": "Compressor Temp (\u00b0C)",
          "timeRange": "Last Hour",
          "unit": "\u00b0C",
          "timeSeries": [
            {
              "time": "0:00",
              "value": 61.12324517574248,
              "threshold": 85
            },
            {
              "time": "1:00",
              "value": 64.75566113898833,
              "threshold": 85
            },
            {
              "time": "2:00",
              "value": 64.43610491500537,
              "threshold": 85
            },
            {
              "time": "3:00",
              "value": 70.93953669360218,
              "threshold": 85
            },
            {
              "time": "4:00",
              "value": 72.05759402795087,
              "threshold": 85
            },
            {
              "time": "5:00",
              "value": 73.50116223121155,
              "threshold": 85
            },
            {
              "time": "6:00",
              "value": 73.85775467165588,
              "threshold": 85
            },
            {
              "time": "7:00",
              "value": 76.14634925530214,
              "threshold": 85
            },
            {
              "time": "8:00",
              "value": 80.0591841870426,
              "threshold": 85
            },
            {
              "time": "9:00",
              "value": 79.77220044785756,
              "threshold": 85
            },
            {
              "time": "10:00",
              "value": 81.78169417734648,
              "threshold": 85
            },
            {
              "time": "11:00",
              "value": 85.60241933249375,
              "threshold": 85
            },
            {
              "time": "12:00",
              "value": 87.10450404354525,
              "threshold": 85
            },
            {
              "time": "13:00",
              "value": 86.60384942313414,
              "threshold": 85
            },
            {
              "time": "14:00",
              "value": 89.12281961691089,
              "threshold": 85
            },
            {
              "time": "15:00",
              "value": 91.58046452209099,
              "threshold": 85
            },
            {
              "time": "16:00",
              "value": 92.9066410548781,
              "threshold": 85
            },
            {
              "time": "17:00",
              "value": 94.90248663143817,
              "threshold": 85
            },
            {
              "time": "18:00",
              "value": 97.37082354734262,
              "threshold": 85
            },
            {
              "time": "19:00",
              "value": 99.68461516115799,
              "threshold": 85
            }
          ]
        }
      },
      "trend_pattern-heatmap": {
        "coreWidget": "Trend",
        "variant": "TREND_PATTERN",
        "representation": "Heatmap",
        "encoding": "Density",
        "layout": {
          "aspectRatio": "2:1",
          "padding": "p-6",
          "radius": "rounded-xl",
          "zones": "header-chart"
        },
        "visual": {
          "background": "bg-white",
          "border": "border-gray-100",
          "theme": "light",
          "colors": [
            "#f3f4f6",
            "#93c5fd",
            "#3b82f6",
            "#1d4ed8"
          ]
        },
        "interactions": {
          "hover": "tooltip"
        },
        "tooltip": {
          "showValue": true
        },
        "demoData": {
          "label": "Peak Load Distribution",
          "timeRange": "Last 7 Days",
          "unit": "%",
          "buckets": [
            {
              "id": 0,
              "intensity": 0.18958212477311465,
              "label": "Mon"
            },
            {
              "id": 1,
              "intensity": 0.2558145707985282,
              "label": ""
            },
            {
              "id": 2,
              "intensity": 0.2863123681137876,
              "label": ""
            },
            {
              "id": 3,
              "intensity": 0.46297436892141364,
              "label": ""
            },
            {
              "id": 4,
              "intensity": 0.7257710127205887,
              "label": ""
            },
            {
              "id": 5,
              "intensity": 0.7951666139291578,
              "label": ""
            },
            {
              "id": 6,
              "intensity": 0.9426214113258389,
              "label": "Mon"
            },
            {
              "id": 7,
              "intensity": 0.06852092376563257,
              "label": ""
            },
            {
              "id": 8,
              "intensity": 0.36644801359725165,
              "label": ""
            },
            {
              "id": 9,
              "intensity": 0.36787987612542294,
              "label": ""
            },
            {
              "id": 10,
              "intensity": 0.8832321027177235,
              "label": ""
            },
            {
              "id": 11,
              "intensity": 0.06974895947007975,
              "label": ""
            },
            {
              "id": 12,
              "intensity": 0.12699017825521852,
              "label": "Mon"
            },
            {
              "id": 13,
              "intensity": 0.8615201627070626,
              "label": ""
            },
            {
              "id": 14,
              "intensity": 0.510414897762606,
              "label": ""
            },
            {
              "id": 15,
              "intensity": 0.6567436410231264,
              "label": ""
            },
            {
              "id": 16,
              "intensity": 0.9157248441591859,
              "label": ""
            },
            {
              "id": 17,
              "intensity": 0.29652253436134246,
              "label": ""
            },
            {
              "id": 18,
              "intensity": 0.4886875625409923,
              "label": "Mon"
            },
            {
              "id": 19,
              "intensity": 0.8690127024997372,
              "label": ""
            },
            {
              "id": 20,
              "intensity": 0.38383981640480935,
              "label": ""
            },
            {
              "id": 21,
              "intensity": 0.3845912060933323,
              "label": ""
            },
            {
              "id": 22,
              "intensity": 0.7933217040861369,
              "label": ""
            },
            {
              "id": 23,
              "intensity": 0.9736409560490247,
              "label": ""
            },
            {
              "id": 24,
              "intensity": 0.036520389401334175,
              "label": "Mon"
            },
            {
              "id": 25,
              "intensity": 0.8037485172315018,
              "label": ""
            },
            {
              "id": 26,
              "intensity": 0.5531210047231871,
              "label": ""
            },
            {
              "id": 27,
              "intensity": 0.5681051990038046,
              "label": ""
            },
            {
              "id": 28,
              "intensity": 0.8963173597640164,
              "label": ""
            },
            {
              "id": 29,
              "intensity": 0.8333328521388035,
              "label": ""
            },
            {
              "id": 30,
              "intensity": 0.9356152860639437,
              "label": "Mon"
            },
            {
              "id": 31,
              "intensity": 0.5164185459237474,
              "label": ""
            },
            {
              "id": 32,
              "intensity": 0.40932512445174907,
              "label": ""
            },
            {
              "id": 33,
              "intensity": 0.9178450135697962,
              "label": ""
            },
            {
              "id": 34,
              "intensity": 0.670383830380924,
              "label": ""
            },
            {
              "id": 35,
              "intensity": 0.36451019170824117,
              "label": ""
            },
            {
              "id": 36,
              "intensity": 0.49212664887218716,
              "label": "Mon"
            },
            {
              "id": 37,
              "intensity": 0.23800636653576634,
              "label": ""
            },
            {
              "id": 38,
              "intensity": 0.8481194488986474,
              "label": ""
            },
            {
              "id": 39,
              "intensity": 0.6883782547854154,
              "label": ""
            },
            {
              "id": 40,
              "intensity": 0.3976100743423616,
              "label": ""
            },
            {
              "id": 41,
              "intensity": 0.8126632003014744,
              "label": ""
            }
          ]
        }
      }
    }
  },
  "trend-multi-line": {
    "name": "Trend Multi-line",
    "defaultFixture": "power-sources-stacked",
    "variants": {
      "power-sources-stacked": {
        "id": "power_sources",
        "name": "Power Sources (Stacked)",
        "description": "Comparison of Grid vs Generator vs Solar contribution.",
        "timeRange": "P1D",
        "granularity": "15m",
        "series": [
          {
            "id": "grid",
            "label": "Grid (EB)",
            "source": "Meter-01",
            "colorToken": "#2563eb",
            "lineStyle": "solid",
            "yAxis": "left",
            "unit": "kW"
          },
          {
            "id": "solar",
            "label": "Solar PV",
            "source": "Inv-01",
            "colorToken": "#16a34a",
            "lineStyle": "solid",
            "yAxis": "left",
            "unit": "kW"
          },
          {
            "id": "dg",
            "label": "Diesel Gen",
            "source": "DG-01",
            "colorToken": "#d97706",
            "lineStyle": "solid",
            "yAxis": "left",
            "unit": "kW"
          }
        ],
        "data": [
          {
            "timestamp": "2026-01-26T08:10:23.260Z",
            "grid": 460.07,
            "solar": 105.95,
            "dg": 9.63
          },
          {
            "timestamp": "2026-01-26T08:25:23.260Z",
            "grid": 452.07,
            "solar": 115.71,
            "dg": -0.16
          },
          {
            "timestamp": "2026-01-26T08:40:23.260Z",
            "grid": 441.39,
            "solar": 98.71,
            "dg": -0.87
          },
          {
            "timestamp": "2026-01-26T08:55:23.260Z",
            "grid": 478.01,
            "solar": 112.04,
            "dg": -18.51
          },
          {
            "timestamp": "2026-01-26T09:10:23.260Z",
            "grid": 457.54,
            "solar": 142.39,
            "dg": 3.09
          },
          {
            "timestamp": "2026-01-26T09:25:23.260Z",
            "grid": 462.27,
            "solar": 101.87,
            "dg": 18.61
          },
          {
            "timestamp": "2026-01-26T09:40:23.260Z",
            "grid": 480.36,
            "solar": 112.48,
            "dg": 13.66
          },
          {
            "timestamp": "2026-01-26T09:55:23.260Z",
            "grid": 490.44,
            "solar": 124.32,
            "dg": -22.93
          },
          {
            "timestamp": "2026-01-26T10:10:23.260Z",
            "grid": 496.89,
            "solar": 135.99,
            "dg": -23.94
          },
          {
            "timestamp": "2026-01-26T10:25:23.260Z",
            "grid": 499.26,
            "solar": 138.56,
            "dg": 15.88
          },
          {
            "timestamp": "2026-01-26T10:40:23.260Z",
            "grid": 499.16,
            "solar": 137.65,
            "dg": 15.48
          },
          {
            "timestamp": "2026-01-26T10:55:23.260Z",
            "grid": 502.99,
            "solar": 121.47,
            "dg": 20.14
          },
          {
            "timestamp": "2026-01-26T11:10:23.260Z",
            "grid": 468.62,
            "solar": 121.97,
            "dg": 10.11
          },
          {
            "timestamp": "2026-01-26T11:25:23.260Z",
            "grid": 491.29,
            "solar": 133.85,
            "dg": -17.42
          },
          {
            "timestamp": "2026-01-26T11:40:23.260Z",
            "grid": 485.27,
            "solar": 142.41,
            "dg": 23.3
          },
          {
            "timestamp": "2026-01-26T11:55:23.260Z",
            "grid": 515.12,
            "solar": 134.02,
            "dg": -18.36
          },
          {
            "timestamp": "2026-01-26T12:10:23.260Z",
            "grid": 507.9,
            "solar": 124.51,
            "dg": 2.58
          },
          {
            "timestamp": "2026-01-26T12:25:23.260Z",
            "grid": 513.76,
            "solar": 118.13,
            "dg": -24.04
          },
          {
            "timestamp": "2026-01-26T12:40:23.260Z",
            "grid": 477.58,
            "solar": 113.9,
            "dg": 20.47
          },
          {
            "timestamp": "2026-01-26T12:55:23.260Z",
            "grid": 509.84,
            "solar": 114.37,
            "dg": 18.91
          },
          {
            "timestamp": "2026-01-26T13:10:23.260Z",
            "grid": 494.68,
            "solar": 128.65,
            "dg": -4.03
          },
          {
            "timestamp": "2026-01-26T13:25:23.260Z",
            "grid": 485.48,
            "solar": 121.5,
            "dg": 17.07
          },
          {
            "timestamp": "2026-01-26T13:40:23.260Z",
            "grid": 465.27,
            "solar": 120.88,
            "dg": -18.5
          },
          {
            "timestamp": "2026-01-26T13:55:23.260Z",
            "grid": 488.01,
            "solar": 104.42,
            "dg": -10.38
          }
        ]
      },
      "main-lt-phases-current": {
        "id": "lt_phases",
        "name": "Main LT Phases (Current)",
        "description": "Three-phase current balancing monitoring.",
        "timeRange": "PT1H",
        "granularity": "1m",
        "thresholds": [
          {
            "value": 800,
            "label": "Max Rated",
            "color": "#ef4444",
            "type": "max"
          }
        ],
        "series": [
          {
            "id": "r_phase",
            "label": "R-Phase",
            "source": "Main-LT",
            "colorToken": "#ef4444",
            "lineStyle": "solid",
            "yAxis": "left",
            "unit": "A"
          },
          {
            "id": "y_phase",
            "label": "Y-Phase",
            "source": "Main-LT",
            "colorToken": "#eab308",
            "lineStyle": "solid",
            "yAxis": "left",
            "unit": "A"
          },
          {
            "id": "b_phase",
            "label": "B-Phase",
            "source": "Main-LT",
            "colorToken": "#3b82f6",
            "lineStyle": "solid",
            "yAxis": "left",
            "unit": "A"
          }
        ],
        "data": [
          {
            "timestamp": "2026-01-25T23:10:23.260Z",
            "r_phase": 648.87,
            "y_phase": 647.27,
            "b_phase": 656.55
          },
          {
            "timestamp": "2026-01-25T23:25:23.260Z",
            "r_phase": 653.63,
            "y_phase": 657.66,
            "b_phase": 673.61
          },
          {
            "timestamp": "2026-01-25T23:40:23.260Z",
            "r_phase": 668.06,
            "y_phase": 660.94,
            "b_phase": 674.57
          },
          {
            "timestamp": "2026-01-25T23:55:23.260Z",
            "r_phase": 673.74,
            "y_phase": 670.91,
            "b_phase": 672.12
          },
          {
            "timestamp": "2026-01-26T00:10:23.260Z",
            "r_phase": 679.22,
            "y_phase": 677.13,
            "b_phase": 682.72
          },
          {
            "timestamp": "2026-01-26T00:25:23.260Z",
            "r_phase": 687.83,
            "y_phase": 673.55,
            "b_phase": 694.22
          },
          {
            "timestamp": "2026-01-26T00:40:23.260Z",
            "r_phase": 687.68,
            "y_phase": 678.57,
            "b_phase": 694.12
          },
          {
            "timestamp": "2026-01-26T00:55:23.260Z",
            "r_phase": 692.18,
            "y_phase": 686.35,
            "b_phase": 696.82
          },
          {
            "timestamp": "2026-01-26T01:10:23.260Z",
            "r_phase": 702.38,
            "y_phase": 691.46,
            "b_phase": 701.18
          },
          {
            "timestamp": "2026-01-26T01:25:23.260Z",
            "r_phase": 700.62,
            "y_phase": 692.24,
            "b_phase": 709.2
          },
          {
            "timestamp": "2026-01-26T01:40:23.260Z",
            "r_phase": 702.09,
            "y_phase": 700.29,
            "b_phase": 709.11
          },
          {
            "timestamp": "2026-01-26T01:55:23.260Z",
            "r_phase": 707.85,
            "y_phase": 707.08,
            "b_phase": 712.19
          },
          {
            "timestamp": "2026-01-26T02:10:23.260Z",
            "r_phase": 716.68,
            "y_phase": 701.68,
            "b_phase": 728.08
          },
          {
            "timestamp": "2026-01-26T02:25:23.260Z",
            "r_phase": 719.76,
            "y_phase": 706.43,
            "b_phase": 729.45
          },
          {
            "timestamp": "2026-01-26T02:40:23.260Z",
            "r_phase": 721.31,
            "y_phase": 714.65,
            "b_phase": 731.28
          },
          {
            "timestamp": "2026-01-26T02:55:23.260Z",
            "r_phase": 715.28,
            "y_phase": 702.64,
            "b_phase": 722.99
          },
          {
            "timestamp": "2026-01-26T03:10:23.260Z",
            "r_phase": 721.81,
            "y_phase": 713.36,
            "b_phase": 728.04
          },
          {
            "timestamp": "2026-01-26T03:25:23.260Z",
            "r_phase": 717.92,
            "y_phase": 703.79,
            "b_phase": 728.6
          },
          {
            "timestamp": "2026-01-26T03:40:23.260Z",
            "r_phase": 713.5,
            "y_phase": 712.05,
            "b_phase": 728.37
          },
          {
            "timestamp": "2026-01-26T03:55:23.260Z",
            "r_phase": 714.25,
            "y_phase": 709.25,
            "b_phase": 727.65
          },
          {
            "timestamp": "2026-01-26T04:10:23.260Z",
            "r_phase": 711.7,
            "y_phase": 699.9,
            "b_phase": 723.57
          },
          {
            "timestamp": "2026-01-26T04:25:23.260Z",
            "r_phase": 712.44,
            "y_phase": 694.5,
            "b_phase": 722.4
          },
          {
            "timestamp": "2026-01-26T04:40:23.260Z",
            "r_phase": 697.05,
            "y_phase": 691.06,
            "b_phase": 717.49
          },
          {
            "timestamp": "2026-01-26T04:55:23.260Z",
            "r_phase": 695.38,
            "y_phase": 692.58,
            "b_phase": 708.83
          },
          {
            "timestamp": "2026-01-26T05:10:23.260Z",
            "r_phase": 694.17,
            "y_phase": 682.5,
            "b_phase": 698.08
          },
          {
            "timestamp": "2026-01-26T05:25:23.260Z",
            "r_phase": 696.19,
            "y_phase": 680.2,
            "b_phase": 697
          },
          {
            "timestamp": "2026-01-26T05:40:23.260Z",
            "r_phase": 681.62,
            "y_phase": 684.02,
            "b_phase": 692.6
          },
          {
            "timestamp": "2026-01-26T05:55:23.260Z",
            "r_phase": 674.66,
            "y_phase": 672.74,
            "b_phase": 685.16
          },
          {
            "timestamp": "2026-01-26T06:10:23.260Z",
            "r_phase": 667.04,
            "y_phase": 671.4,
            "b_phase": 689.3
          },
          {
            "timestamp": "2026-01-26T06:25:23.260Z",
            "r_phase": 664.1,
            "y_phase": 657.48,
            "b_phase": 675.3
          },
          {
            "timestamp": "2026-01-26T06:40:23.260Z",
            "r_phase": 653.63,
            "y_phase": 649.78,
            "b_phase": 671
          },
          {
            "timestamp": "2026-01-26T06:55:23.260Z",
            "r_phase": 647.59,
            "y_phase": 653.47,
            "b_phase": 666.9
          },
          {
            "timestamp": "2026-01-26T07:10:23.260Z",
            "r_phase": 650.76,
            "y_phase": 633.98,
            "b_phase": 649.03
          },
          {
            "timestamp": "2026-01-26T07:25:23.260Z",
            "r_phase": 644.21,
            "y_phase": 637.64,
            "b_phase": 649.97
          },
          {
            "timestamp": "2026-01-26T07:40:23.260Z",
            "r_phase": 637.94,
            "y_phase": 626.08,
            "b_phase": 646.19
          },
          {
            "timestamp": "2026-01-26T07:55:23.260Z",
            "r_phase": 623.17,
            "y_phase": 627.53,
            "b_phase": 641.22
          },
          {
            "timestamp": "2026-01-26T08:10:23.260Z",
            "r_phase": 614.3,
            "y_phase": 609.73,
            "b_phase": 628.13
          },
          {
            "timestamp": "2026-01-26T08:25:23.260Z",
            "r_phase": 621.57,
            "y_phase": 607.62,
            "b_phase": 631.53
          },
          {
            "timestamp": "2026-01-26T08:40:23.260Z",
            "r_phase": 615.14,
            "y_phase": 612.03,
            "b_phase": 624.48
          },
          {
            "timestamp": "2026-01-26T08:55:23.260Z",
            "r_phase": 610.33,
            "y_phase": 597.97,
            "b_phase": 609.13
          },
          {
            "timestamp": "2026-01-26T09:10:23.260Z",
            "r_phase": 594.13,
            "y_phase": 592.43,
            "b_phase": 609.23
          },
          {
            "timestamp": "2026-01-26T09:25:23.260Z",
            "r_phase": 594.3,
            "y_phase": 586.83,
            "b_phase": 611.93
          },
          {
            "timestamp": "2026-01-26T09:40:23.260Z",
            "r_phase": 594.18,
            "y_phase": 593.58,
            "b_phase": 606.07
          },
          {
            "timestamp": "2026-01-26T09:55:23.260Z",
            "r_phase": 584.59,
            "y_phase": 587.56,
            "b_phase": 605.21
          },
          {
            "timestamp": "2026-01-26T10:10:23.260Z",
            "r_phase": 586.21,
            "y_phase": 586.59,
            "b_phase": 594.95
          },
          {
            "timestamp": "2026-01-26T10:25:23.260Z",
            "r_phase": 581.05,
            "y_phase": 581.68,
            "b_phase": 595.18
          },
          {
            "timestamp": "2026-01-26T10:40:23.260Z",
            "r_phase": 585.17,
            "y_phase": 577.3,
            "b_phase": 594.07
          },
          {
            "timestamp": "2026-01-26T10:55:23.260Z",
            "r_phase": 585.09,
            "y_phase": 586.78,
            "b_phase": 596.09
          },
          {
            "timestamp": "2026-01-26T11:10:23.260Z",
            "r_phase": 582.05,
            "y_phase": 575.8,
            "b_phase": 599.55
          },
          {
            "timestamp": "2026-01-26T11:25:23.260Z",
            "r_phase": 591.02,
            "y_phase": 582.66,
            "b_phase": 594.08
          },
          {
            "timestamp": "2026-01-26T11:40:23.260Z",
            "r_phase": 582.34,
            "y_phase": 577.45,
            "b_phase": 594.14
          },
          {
            "timestamp": "2026-01-26T11:55:23.260Z",
            "r_phase": 595.05,
            "y_phase": 586.95,
            "b_phase": 603.77
          },
          {
            "timestamp": "2026-01-26T12:10:23.260Z",
            "r_phase": 591.64,
            "y_phase": 588.25,
            "b_phase": 598.96
          },
          {
            "timestamp": "2026-01-26T12:25:23.260Z",
            "r_phase": 592.85,
            "y_phase": 593.09,
            "b_phase": 603.27
          },
          {
            "timestamp": "2026-01-26T12:40:23.260Z",
            "r_phase": 595.16,
            "y_phase": 589.57,
            "b_phase": 607.44
          },
          {
            "timestamp": "2026-01-26T12:55:23.260Z",
            "r_phase": 608.3,
            "y_phase": 603.55,
            "b_phase": 612.08
          },
          {
            "timestamp": "2026-01-26T13:10:23.260Z",
            "r_phase": 614.6,
            "y_phase": 609.86,
            "b_phase": 623.91
          },
          {
            "timestamp": "2026-01-26T13:25:23.260Z",
            "r_phase": 607.73,
            "y_phase": 613.81,
            "b_phase": 627.73
          },
          {
            "timestamp": "2026-01-26T13:40:23.260Z",
            "r_phase": 618.95,
            "y_phase": 610.74,
            "b_phase": 627.84
          },
          {
            "timestamp": "2026-01-26T13:55:23.260Z",
            "r_phase": 631.72,
            "y_phase": 621.05,
            "b_phase": 642.35
          }
        ]
      },
      "ups-health-dual-axis": {
        "id": "ups_health",
        "name": "UPS Health (Dual Axis)",
        "description": "Battery charge percentage vs Output Load.",
        "timeRange": "P1D",
        "granularity": "1h",
        "series": [
          {
            "id": "batt_pct",
            "label": "Battery %",
            "source": "UPS-A",
            "colorToken": "#10b981",
            "lineStyle": "solid",
            "yAxis": "right",
            "unit": "%"
          },
          {
            "id": "load_kw",
            "label": "Output Load",
            "source": "UPS-A",
            "colorToken": "#6366f1",
            "lineStyle": "solid",
            "yAxis": "left",
            "unit": "kW"
          }
        ],
        "data": [
          {
            "timestamp": "2026-01-26T08:10:23.260Z",
            "batt_pct": 94.66,
            "load_kw": 43.45
          },
          {
            "timestamp": "2026-01-26T08:25:23.260Z",
            "batt_pct": 97.27,
            "load_kw": 47.44
          },
          {
            "timestamp": "2026-01-26T08:40:23.260Z",
            "batt_pct": 97.32,
            "load_kw": 45.21
          },
          {
            "timestamp": "2026-01-26T08:55:23.260Z",
            "batt_pct": 99.26,
            "load_kw": 47.91
          },
          {
            "timestamp": "2026-01-26T09:10:23.260Z",
            "batt_pct": 98.94,
            "load_kw": 48.69
          },
          {
            "timestamp": "2026-01-26T09:25:23.260Z",
            "batt_pct": 97.78,
            "load_kw": 46.68
          },
          {
            "timestamp": "2026-01-26T09:40:23.260Z",
            "batt_pct": 102.54,
            "load_kw": 48.89
          },
          {
            "timestamp": "2026-01-26T09:55:23.260Z",
            "batt_pct": 100.06,
            "load_kw": 48.53
          },
          {
            "timestamp": "2026-01-26T10:10:23.260Z",
            "batt_pct": 99.91,
            "load_kw": 47.09
          },
          {
            "timestamp": "2026-01-26T10:25:23.260Z",
            "batt_pct": 104.71,
            "load_kw": 46.39
          },
          {
            "timestamp": "2026-01-26T10:40:23.260Z",
            "batt_pct": 104.62,
            "load_kw": 50.58
          },
          {
            "timestamp": "2026-01-26T10:55:23.260Z",
            "batt_pct": 101,
            "load_kw": 51.07
          },
          {
            "timestamp": "2026-01-26T11:10:23.260Z",
            "batt_pct": 102.94,
            "load_kw": 50.98
          },
          {
            "timestamp": "2026-01-26T11:25:23.260Z",
            "batt_pct": 104.67,
            "load_kw": 46.91
          },
          {
            "timestamp": "2026-01-26T11:40:23.260Z",
            "batt_pct": 103.68,
            "load_kw": 47.17
          },
          {
            "timestamp": "2026-01-26T11:55:23.260Z",
            "batt_pct": 106.15,
            "load_kw": 51.77
          },
          {
            "timestamp": "2026-01-26T12:10:23.260Z",
            "batt_pct": 102.75,
            "load_kw": 50.27
          },
          {
            "timestamp": "2026-01-26T12:25:23.260Z",
            "batt_pct": 102.81,
            "load_kw": 50.98
          },
          {
            "timestamp": "2026-01-26T12:40:23.260Z",
            "batt_pct": 103.11,
            "load_kw": 47.28
          },
          {
            "timestamp": "2026-01-26T12:55:23.260Z",
            "batt_pct": 102.9,
            "load_kw": 47.02
          },
          {
            "timestamp": "2026-01-26T13:10:23.260Z",
            "batt_pct": 101.63,
            "load_kw": 46.8
          },
          {
            "timestamp": "2026-01-26T13:25:23.260Z",
            "batt_pct": 101.44,
            "load_kw": 49.94
          },
          {
            "timestamp": "2026-01-26T13:40:23.260Z",
            "batt_pct": 104.84,
            "load_kw": 47.5
          },
          {
            "timestamp": "2026-01-26T13:55:23.260Z",
            "batt_pct": 103.72,
            "load_kw": 48.05
          }
        ]
      },
      "power-quality": {
        "id": "pq_metrics",
        "name": "Power Quality",
        "description": "Frequency stability and Power Factor.",
        "timeRange": "PT4H",
        "granularity": "5m",
        "series": [
          {
            "id": "freq",
            "label": "Frequency",
            "source": "Inc-01",
            "colorToken": "#8b5cf6",
            "lineStyle": "solid",
            "yAxis": "left",
            "unit": "Hz"
          },
          {
            "id": "pf",
            "label": "Power Factor",
            "source": "Inc-01",
            "colorToken": "#ec4899",
            "lineStyle": "dashed",
            "yAxis": "right",
            "unit": "pf"
          }
        ],
        "data": [
          {
            "timestamp": "2026-01-26T02:10:23.260Z",
            "freq": 50.02,
            "pf": 0.96
          },
          {
            "timestamp": "2026-01-26T02:25:23.260Z",
            "freq": 50.55,
            "pf": 0.95
          },
          {
            "timestamp": "2026-01-26T02:40:23.260Z",
            "freq": 51.02,
            "pf": 0.96
          },
          {
            "timestamp": "2026-01-26T02:55:23.260Z",
            "freq": 51.5,
            "pf": 1.04
          },
          {
            "timestamp": "2026-01-26T03:10:23.260Z",
            "freq": 51.91,
            "pf": 0.99
          },
          {
            "timestamp": "2026-01-26T03:25:23.260Z",
            "freq": 52.41,
            "pf": 0.99
          },
          {
            "timestamp": "2026-01-26T03:40:23.260Z",
            "freq": 52.83,
            "pf": 1.05
          },
          {
            "timestamp": "2026-01-26T03:55:23.260Z",
            "freq": 53.23,
            "pf": 1.09
          },
          {
            "timestamp": "2026-01-26T04:10:23.260Z",
            "freq": 53.64,
            "pf": 1.05
          },
          {
            "timestamp": "2026-01-26T04:25:23.260Z",
            "freq": 53.92,
            "pf": 1.02
          },
          {
            "timestamp": "2026-01-26T04:40:23.260Z",
            "freq": 54.22,
            "pf": 1.08
          },
          {
            "timestamp": "2026-01-26T04:55:23.260Z",
            "freq": 54.49,
            "pf": 1.03
          },
          {
            "timestamp": "2026-01-26T05:10:23.260Z",
            "freq": 54.7,
            "pf": 1.08
          },
          {
            "timestamp": "2026-01-26T05:25:23.260Z",
            "freq": 54.86,
            "pf": 1.07
          },
          {
            "timestamp": "2026-01-26T05:40:23.260Z",
            "freq": 54.97,
            "pf": 1.08
          },
          {
            "timestamp": "2026-01-26T05:55:23.260Z",
            "freq": 54.99,
            "pf": 1.11
          },
          {
            "timestamp": "2026-01-26T06:10:23.260Z",
            "freq": 55.01,
            "pf": 1.09
          },
          {
            "timestamp": "2026-01-26T06:25:23.260Z",
            "freq": 54.94,
            "pf": 1.09
          },
          {
            "timestamp": "2026-01-26T06:40:23.260Z",
            "freq": 54.9,
            "pf": 1.07
          },
          {
            "timestamp": "2026-01-26T06:55:23.260Z",
            "freq": 54.77,
            "pf": 1.12
          },
          {
            "timestamp": "2026-01-26T07:10:23.260Z",
            "freq": 54.58,
            "pf": 1.02
          },
          {
            "timestamp": "2026-01-26T07:25:23.260Z",
            "freq": 54.36,
            "pf": 1.02
          },
          {
            "timestamp": "2026-01-26T07:40:23.260Z",
            "freq": 54.05,
            "pf": 1.05
          },
          {
            "timestamp": "2026-01-26T07:55:23.260Z",
            "freq": 53.77,
            "pf": 1.05
          },
          {
            "timestamp": "2026-01-26T08:10:23.260Z",
            "freq": 53.35,
            "pf": 1.08
          },
          {
            "timestamp": "2026-01-26T08:25:23.260Z",
            "freq": 53.01,
            "pf": 1.05
          },
          {
            "timestamp": "2026-01-26T08:40:23.260Z",
            "freq": 52.61,
            "pf": 1.06
          },
          {
            "timestamp": "2026-01-26T08:55:23.260Z",
            "freq": 52.14,
            "pf": 1.06
          },
          {
            "timestamp": "2026-01-26T09:10:23.260Z",
            "freq": 51.69,
            "pf": 1.04
          },
          {
            "timestamp": "2026-01-26T09:25:23.260Z",
            "freq": 51.21,
            "pf": 1
          },
          {
            "timestamp": "2026-01-26T09:40:23.260Z",
            "freq": 50.66,
            "pf": 0.95
          },
          {
            "timestamp": "2026-01-26T09:55:23.260Z",
            "freq": 50.25,
            "pf": 0.94
          },
          {
            "timestamp": "2026-01-26T10:10:23.260Z",
            "freq": 49.68,
            "pf": 1.01
          },
          {
            "timestamp": "2026-01-26T10:25:23.260Z",
            "freq": 49.2,
            "pf": 0.95
          },
          {
            "timestamp": "2026-01-26T10:40:23.260Z",
            "freq": 48.7,
            "pf": 0.91
          },
          {
            "timestamp": "2026-01-26T10:55:23.260Z",
            "freq": 48.24,
            "pf": 0.92
          },
          {
            "timestamp": "2026-01-26T11:10:23.260Z",
            "freq": 47.76,
            "pf": 0.97
          },
          {
            "timestamp": "2026-01-26T11:25:23.260Z",
            "freq": 47.31,
            "pf": 0.91
          },
          {
            "timestamp": "2026-01-26T11:40:23.260Z",
            "freq": 46.94,
            "pf": 0.91
          },
          {
            "timestamp": "2026-01-26T11:55:23.260Z",
            "freq": 46.55,
            "pf": 0.88
          },
          {
            "timestamp": "2026-01-26T12:10:23.260Z",
            "freq": 46.2,
            "pf": 0.9
          },
          {
            "timestamp": "2026-01-26T12:25:23.260Z",
            "freq": 45.91,
            "pf": 0.93
          },
          {
            "timestamp": "2026-01-26T12:40:23.260Z",
            "freq": 45.66,
            "pf": 0.92
          },
          {
            "timestamp": "2026-01-26T12:55:23.260Z",
            "freq": 45.41,
            "pf": 0.93
          },
          {
            "timestamp": "2026-01-26T13:10:23.260Z",
            "freq": 45.2,
            "pf": 0.92
          },
          {
            "timestamp": "2026-01-26T13:25:23.260Z",
            "freq": 45.08,
            "pf": 0.88
          },
          {
            "timestamp": "2026-01-26T13:40:23.260Z",
            "freq": 45.01,
            "pf": 0.93
          },
          {
            "timestamp": "2026-01-26T13:55:23.260Z",
            "freq": 45.03,
            "pf": 0.9
          }
        ]
      },
      "hvac-performance": {
        "id": "hvac_perf",
        "name": "HVAC Performance",
        "description": "Supply vs Return temperature delta.",
        "timeRange": "P12H",
        "granularity": "30m",
        "series": [
          {
            "id": "temp_supp",
            "label": "Supply Temp",
            "source": "AHU-2",
            "colorToken": "#3b82f6",
            "lineStyle": "solid",
            "yAxis": "left",
            "unit": "\u00b0C"
          },
          {
            "id": "temp_ret",
            "label": "Return Temp",
            "source": "AHU-2",
            "colorToken": "#ef4444",
            "lineStyle": "solid",
            "yAxis": "left",
            "unit": "\u00b0C"
          },
          {
            "id": "humidity",
            "label": "Humidity",
            "source": "AHU-2",
            "colorToken": "#06b6d4",
            "lineStyle": "dotted",
            "yAxis": "right",
            "unit": "%RH"
          }
        ],
        "data": [
          {
            "timestamp": "2026-01-26T08:10:23.260Z",
            "temp_supp": 17.93,
            "temp_ret": 24.58,
            "humidity": 54.48
          },
          {
            "timestamp": "2026-01-26T08:25:23.260Z",
            "temp_supp": 18.81,
            "temp_ret": 24.91,
            "humidity": 54.94
          },
          {
            "timestamp": "2026-01-26T08:40:23.260Z",
            "temp_supp": 18.6,
            "temp_ret": 25.04,
            "humidity": 55.44
          },
          {
            "timestamp": "2026-01-26T08:55:23.260Z",
            "temp_supp": 17.86,
            "temp_ret": 24.02,
            "humidity": 56.01
          },
          {
            "timestamp": "2026-01-26T09:10:23.260Z",
            "temp_supp": 18.01,
            "temp_ret": 24.84,
            "humidity": 57.67
          },
          {
            "timestamp": "2026-01-26T09:25:23.260Z",
            "temp_supp": 18.08,
            "temp_ret": 25.21,
            "humidity": 57.82
          },
          {
            "timestamp": "2026-01-26T09:40:23.260Z",
            "temp_supp": 19.53,
            "temp_ret": 26,
            "humidity": 59.04
          },
          {
            "timestamp": "2026-01-26T09:55:23.260Z",
            "temp_supp": 18.31,
            "temp_ret": 25.81,
            "humidity": 59.03
          },
          {
            "timestamp": "2026-01-26T10:10:23.260Z",
            "temp_supp": 18.96,
            "temp_ret": 25.08,
            "humidity": 59.17
          },
          {
            "timestamp": "2026-01-26T10:25:23.260Z",
            "temp_supp": 20.27,
            "temp_ret": 25.35,
            "humidity": 58.31
          },
          {
            "timestamp": "2026-01-26T10:40:23.260Z",
            "temp_supp": 20.43,
            "temp_ret": 25.27,
            "humidity": 59.4
          },
          {
            "timestamp": "2026-01-26T10:55:23.260Z",
            "temp_supp": 20.06,
            "temp_ret": 25.45,
            "humidity": 60.71
          },
          {
            "timestamp": "2026-01-26T11:10:23.260Z",
            "temp_supp": 19.71,
            "temp_ret": 25.73,
            "humidity": 60.56
          },
          {
            "timestamp": "2026-01-26T11:25:23.260Z",
            "temp_supp": 19.58,
            "temp_ret": 26.02,
            "humidity": 59.68
          },
          {
            "timestamp": "2026-01-26T11:40:23.260Z",
            "temp_supp": 19.03,
            "temp_ret": 25.97,
            "humidity": 60.62
          },
          {
            "timestamp": "2026-01-26T11:55:23.260Z",
            "temp_supp": 19.21,
            "temp_ret": 26.42,
            "humidity": 60.49
          },
          {
            "timestamp": "2026-01-26T12:10:23.260Z",
            "temp_supp": 19.76,
            "temp_ret": 26.33,
            "humidity": 60.7
          },
          {
            "timestamp": "2026-01-26T12:25:23.260Z",
            "temp_supp": 18.82,
            "temp_ret": 27.05,
            "humidity": 59.6
          },
          {
            "timestamp": "2026-01-26T12:40:23.260Z",
            "temp_supp": 19.56,
            "temp_ret": 26.12,
            "humidity": 60.74
          },
          {
            "timestamp": "2026-01-26T12:55:23.260Z",
            "temp_supp": 19.83,
            "temp_ret": 25.88,
            "humidity": 60.91
          },
          {
            "timestamp": "2026-01-26T13:10:23.260Z",
            "temp_supp": 20.32,
            "temp_ret": 26.7,
            "humidity": 60.25
          },
          {
            "timestamp": "2026-01-26T13:25:23.260Z",
            "temp_supp": 20.06,
            "temp_ret": 26.5,
            "humidity": 59.4
          },
          {
            "timestamp": "2026-01-26T13:40:23.260Z",
            "temp_supp": 18.56,
            "temp_ret": 26.43,
            "humidity": 59.16
          },
          {
            "timestamp": "2026-01-26T13:55:23.260Z",
            "temp_supp": 19.04,
            "temp_ret": 25.5,
            "humidity": 59.65
          }
        ]
      },
      "energy-demand": {
        "id": "energy_demand",
        "name": "Energy & Demand",
        "description": "Cumulative consumption vs Instantaneous demand.",
        "timeRange": "P1D",
        "granularity": "1h",
        "series": [
          {
            "id": "kwh",
            "label": "Energy (Cumulative)",
            "source": "Meter-Main",
            "colorToken": "#14b8a6",
            "lineStyle": "solid",
            "yAxis": "left",
            "unit": "kWh"
          },
          {
            "id": "kw",
            "label": "Demand (Inst)",
            "source": "Meter-Main",
            "colorToken": "#f59e0b",
            "lineStyle": "solid",
            "yAxis": "right",
            "unit": "kW"
          }
        ],
        "data": [
          {
            "timestamp": "2026-01-26T08:10:23.260Z",
            "kwh": 1042.1221349965122,
            "kw": 330.5
          },
          {
            "timestamp": "2026-01-26T08:25:23.260Z",
            "kwh": 1388.7643974698385,
            "kw": 357.34
          },
          {
            "timestamp": "2026-01-26T08:40:23.260Z",
            "kwh": 1700.5644564328372,
            "kw": 376.54
          },
          {
            "timestamp": "2026-01-26T08:55:23.260Z",
            "kwh": 2084.279581417656,
            "kw": 364.53
          },
          {
            "timestamp": "2026-01-26T09:10:23.260Z",
            "kwh": 2432.4584486732933,
            "kw": 371.92
          },
          {
            "timestamp": "2026-01-26T09:25:23.260Z",
            "kwh": 2760.103161495448,
            "kw": 379.33
          },
          {
            "timestamp": "2026-01-26T09:40:23.260Z",
            "kwh": 3146.313538977644,
            "kw": 389
          },
          {
            "timestamp": "2026-01-26T09:55:23.260Z",
            "kwh": 3451.4778962601704,
            "kw": 389.36
          },
          {
            "timestamp": "2026-01-26T10:10:23.260Z",
            "kwh": 3803.6118088478743,
            "kw": 363.99
          },
          {
            "timestamp": "2026-01-26T10:25:23.260Z",
            "kwh": 4179.109427353734,
            "kw": 393.52
          },
          {
            "timestamp": "2026-01-26T10:40:23.260Z",
            "kwh": 4502.852568261638,
            "kw": 363.09
          },
          {
            "timestamp": "2026-01-26T10:55:23.260Z",
            "kwh": 4854.7174305589415,
            "kw": 361.83
          },
          {
            "timestamp": "2026-01-26T11:10:23.260Z",
            "kwh": 5242.788349410541,
            "kw": 379.18
          },
          {
            "timestamp": "2026-01-26T11:25:23.260Z",
            "kwh": 5598.03576000902,
            "kw": 365.7
          },
          {
            "timestamp": "2026-01-26T11:40:23.260Z",
            "kwh": 5901.958746577082,
            "kw": 379.97
          },
          {
            "timestamp": "2026-01-26T11:55:23.260Z",
            "kwh": 6280.930139908247,
            "kw": 377.9
          },
          {
            "timestamp": "2026-01-26T12:10:23.260Z",
            "kwh": 6609.897277666401,
            "kw": 379.14
          },
          {
            "timestamp": "2026-01-26T12:25:23.260Z",
            "kwh": 6978.016192547646,
            "kw": 387.77
          },
          {
            "timestamp": "2026-01-26T12:40:23.260Z",
            "kwh": 7337.389797035052,
            "kw": 381.27
          },
          {
            "timestamp": "2026-01-26T12:55:23.260Z",
            "kwh": 7676.179013915651,
            "kw": 367.53
          },
          {
            "timestamp": "2026-01-26T13:10:23.260Z",
            "kwh": 8019.12979689869,
            "kw": 396.78
          },
          {
            "timestamp": "2026-01-26T13:25:23.260Z",
            "kwh": 8353.832417456568,
            "kw": 375.01
          },
          {
            "timestamp": "2026-01-26T13:40:23.260Z",
            "kwh": 8742.256865887988,
            "kw": 384.54
          },
          {
            "timestamp": "2026-01-26T13:55:23.260Z",
            "kwh": 9084.385141632174,
            "kw": 381.1
          }
        ]
      }
    }
  },
  "trends-cumulative": {
    "name": "Trends Cumulative",
    "defaultFixture": "energy-consumption",
    "variants": {
      "energy-consumption": {
        "config": {
          "title": "Energy Consumption",
          "subtitle": "Plant A \u2022 Utilities \u2022 Today",
          "variant": "V1",
          "mode": "cumulative",
          "series": [
            {
              "id": "EB",
              "label": "Grid Power",
              "unit": "kWh",
              "color": "#2563eb"
            }
          ]
        },
        "data": [
          {
            "x": "2026-01-25T18:30:00.000Z",
            "EB_raw": 27.65,
            "EB_cumulative": 6.91
          },
          {
            "x": "2026-01-25T18:45:00.000Z",
            "EB_raw": 28.24,
            "EB_cumulative": 13.97
          },
          {
            "x": "2026-01-25T19:00:00.000Z",
            "EB_raw": 28.11,
            "EB_cumulative": 21
          },
          {
            "x": "2026-01-25T19:15:00.000Z",
            "EB_raw": 27.45,
            "EB_cumulative": 27.86
          },
          {
            "x": "2026-01-25T19:30:00.000Z",
            "EB_raw": 22.63,
            "EB_cumulative": 33.52
          },
          {
            "x": "2026-01-25T19:45:00.000Z",
            "EB_raw": 23.08,
            "EB_cumulative": 39.29
          },
          {
            "x": "2026-01-25T20:00:00.000Z",
            "EB_raw": 26.24,
            "EB_cumulative": 45.85
          },
          {
            "x": "2026-01-25T20:15:00.000Z",
            "EB_raw": 23.25,
            "EB_cumulative": 51.66
          },
          {
            "x": "2026-01-25T20:30:00.000Z",
            "EB_raw": 28.87,
            "EB_cumulative": 58.88
          },
          {
            "x": "2026-01-25T20:45:00.000Z",
            "EB_raw": 21.28,
            "EB_cumulative": 64.2
          },
          {
            "x": "2026-01-25T21:00:00.000Z",
            "EB_raw": 28.81,
            "EB_cumulative": 71.41
          },
          {
            "x": "2026-01-25T21:15:00.000Z",
            "EB_raw": 22.41,
            "EB_cumulative": 77.01
          },
          {
            "x": "2026-01-25T21:30:00.000Z",
            "EB_raw": 25.77,
            "EB_cumulative": 83.45
          },
          {
            "x": "2026-01-25T21:45:00.000Z",
            "EB_raw": 28.07,
            "EB_cumulative": 90.47
          },
          {
            "x": "2026-01-25T22:00:00.000Z",
            "EB_raw": 29.9,
            "EB_cumulative": 97.94
          },
          {
            "x": "2026-01-25T22:15:00.000Z",
            "EB_raw": 26.73,
            "EB_cumulative": 104.62
          },
          {
            "x": "2026-01-25T22:30:00.000Z",
            "EB_raw": 21.68,
            "EB_cumulative": 110.05
          },
          {
            "x": "2026-01-25T22:45:00.000Z",
            "EB_raw": 22.11,
            "EB_cumulative": 115.57
          },
          {
            "x": "2026-01-25T23:00:00.000Z",
            "EB_raw": 28.34,
            "EB_cumulative": 122.66
          },
          {
            "x": "2026-01-25T23:15:00.000Z",
            "EB_raw": 25.15,
            "EB_cumulative": 128.95
          },
          {
            "x": "2026-01-25T23:30:00.000Z",
            "EB_raw": 25.79,
            "EB_cumulative": 135.39
          },
          {
            "x": "2026-01-25T23:45:00.000Z",
            "EB_raw": 21.95,
            "EB_cumulative": 140.88
          },
          {
            "x": "2026-01-26T00:00:00.000Z",
            "EB_raw": 27.28,
            "EB_cumulative": 147.7
          },
          {
            "x": "2026-01-26T00:15:00.000Z",
            "EB_raw": 26,
            "EB_cumulative": 154.2
          },
          {
            "x": "2026-01-26T00:30:00.000Z",
            "EB_raw": 27.88,
            "EB_cumulative": 161.17
          },
          {
            "x": "2026-01-26T00:45:00.000Z",
            "EB_raw": 22.68,
            "EB_cumulative": 166.84
          },
          {
            "x": "2026-01-26T01:00:00.000Z",
            "EB_raw": 24.66,
            "EB_cumulative": 173.01
          },
          {
            "x": "2026-01-26T01:15:00.000Z",
            "EB_raw": 27.33,
            "EB_cumulative": 179.84
          },
          {
            "x": "2026-01-26T01:30:00.000Z",
            "EB_raw": 29.26,
            "EB_cumulative": 187.15
          },
          {
            "x": "2026-01-26T01:45:00.000Z",
            "EB_raw": 25.25,
            "EB_cumulative": 193.47
          },
          {
            "x": "2026-01-26T02:00:00.000Z",
            "EB_raw": 27.03,
            "EB_cumulative": 200.22
          },
          {
            "x": "2026-01-26T02:15:00.000Z",
            "EB_raw": 25.54,
            "EB_cumulative": 206.61
          },
          {
            "x": "2026-01-26T02:30:00.000Z",
            "EB_raw": 30.44,
            "EB_cumulative": 214.22
          },
          {
            "x": "2026-01-26T02:45:00.000Z",
            "EB_raw": 100.15,
            "EB_cumulative": 239.25
          },
          {
            "x": "2026-01-26T03:00:00.000Z",
            "EB_raw": 103.46,
            "EB_cumulative": 265.12
          },
          {
            "x": "2026-01-26T03:15:00.000Z",
            "EB_raw": 102.07,
            "EB_cumulative": 290.64
          },
          {
            "x": "2026-01-26T03:30:00.000Z",
            "EB_raw": 102.23,
            "EB_cumulative": 316.2
          },
          {
            "x": "2026-01-26T03:45:00.000Z",
            "EB_raw": 101.89,
            "EB_cumulative": 341.67
          },
          {
            "x": "2026-01-26T04:00:00.000Z",
            "EB_raw": 108.66,
            "EB_cumulative": 368.83
          },
          {
            "x": "2026-01-26T04:15:00.000Z",
            "EB_raw": 107.12,
            "EB_cumulative": 395.61
          },
          {
            "x": "2026-01-26T04:30:00.000Z",
            "EB_raw": 101.9,
            "EB_cumulative": 421.09
          },
          {
            "x": "2026-01-26T04:45:00.000Z",
            "EB_raw": 105.98,
            "EB_cumulative": 447.58
          },
          {
            "x": "2026-01-26T05:00:00.000Z",
            "EB_raw": 104.22,
            "EB_cumulative": 473.64
          },
          {
            "x": "2026-01-26T05:15:00.000Z",
            "EB_raw": 105.37,
            "EB_cumulative": 499.98
          },
          {
            "x": "2026-01-26T05:30:00.000Z",
            "EB_raw": 107.83,
            "EB_cumulative": 526.94
          },
          {
            "x": "2026-01-26T05:45:00.000Z",
            "EB_raw": 100.46,
            "EB_cumulative": 552.05
          },
          {
            "x": "2026-01-26T06:00:00.000Z",
            "EB_raw": 103.56,
            "EB_cumulative": 577.94
          },
          {
            "x": "2026-01-26T06:15:00.000Z",
            "EB_raw": 101.95,
            "EB_cumulative": 603.43
          },
          {
            "x": "2026-01-26T06:30:00.000Z",
            "EB_raw": 109.65,
            "EB_cumulative": 630.84
          },
          {
            "x": "2026-01-26T06:45:00.000Z",
            "EB_raw": 105.35,
            "EB_cumulative": 657.17
          },
          {
            "x": "2026-01-26T07:00:00.000Z",
            "EB_raw": 105.12,
            "EB_cumulative": 683.45
          },
          {
            "x": "2026-01-26T07:15:00.000Z",
            "EB_raw": 107.04,
            "EB_cumulative": 710.21
          },
          {
            "x": "2026-01-26T07:30:00.000Z",
            "EB_raw": 105.67,
            "EB_cumulative": 736.63
          },
          {
            "x": "2026-01-26T07:45:00.000Z",
            "EB_raw": 102.62,
            "EB_cumulative": 762.29
          },
          {
            "x": "2026-01-26T08:00:00.000Z",
            "EB_raw": 103.79,
            "EB_cumulative": 788.23
          },
          {
            "x": "2026-01-26T08:15:00.000Z",
            "EB_raw": 100.59,
            "EB_cumulative": 813.38
          },
          {
            "x": "2026-01-26T08:30:00.000Z",
            "EB_raw": 101.24,
            "EB_cumulative": 838.69
          },
          {
            "x": "2026-01-26T08:45:00.000Z",
            "EB_raw": 101.66,
            "EB_cumulative": 864.1
          },
          {
            "x": "2026-01-26T09:00:00.000Z",
            "EB_raw": 106.94,
            "EB_cumulative": 890.84
          },
          {
            "x": "2026-01-26T09:15:00.000Z",
            "EB_raw": 106.38,
            "EB_cumulative": 917.43
          },
          {
            "x": "2026-01-26T09:30:00.000Z",
            "EB_raw": 103.45,
            "EB_cumulative": 943.3
          },
          {
            "x": "2026-01-26T09:45:00.000Z",
            "EB_raw": 102.67,
            "EB_cumulative": 968.96
          },
          {
            "x": "2026-01-26T10:00:00.000Z",
            "EB_raw": 105.28,
            "EB_cumulative": 995.28
          },
          {
            "x": "2026-01-26T10:15:00.000Z",
            "EB_raw": 105.62,
            "EB_cumulative": 1021.69
          },
          {
            "x": "2026-01-26T10:30:00.000Z",
            "EB_raw": 109.14,
            "EB_cumulative": 1048.97
          },
          {
            "x": "2026-01-26T10:45:00.000Z",
            "EB_raw": 108.4,
            "EB_cumulative": 1076.07
          },
          {
            "x": "2026-01-26T11:00:00.000Z",
            "EB_raw": 107.92,
            "EB_cumulative": 1103.05
          },
          {
            "x": "2026-01-26T11:15:00.000Z",
            "EB_raw": 102.21,
            "EB_cumulative": 1128.6
          },
          {
            "x": "2026-01-26T11:30:00.000Z",
            "EB_raw": 105.29,
            "EB_cumulative": 1154.93
          },
          {
            "x": "2026-01-26T11:45:00.000Z",
            "EB_raw": 106.96,
            "EB_cumulative": 1181.67
          },
          {
            "x": "2026-01-26T12:00:00.000Z",
            "EB_raw": 100.46,
            "EB_cumulative": 1206.78
          },
          {
            "x": "2026-01-26T12:15:00.000Z",
            "EB_raw": 101.78,
            "EB_cumulative": 1232.22
          },
          {
            "x": "2026-01-26T12:30:00.000Z",
            "EB_raw": 28.43,
            "EB_cumulative": 1239.33
          },
          {
            "x": "2026-01-26T12:45:00.000Z",
            "EB_raw": 20.23,
            "EB_cumulative": 1244.39
          },
          {
            "x": "2026-01-26T13:00:00.000Z",
            "EB_raw": 28.21,
            "EB_cumulative": 1251.44
          },
          {
            "x": "2026-01-26T13:15:00.000Z",
            "EB_raw": 29.99,
            "EB_cumulative": 1258.94
          },
          {
            "x": "2026-01-26T13:30:00.000Z",
            "EB_raw": 21.93,
            "EB_cumulative": 1264.42
          },
          {
            "x": "2026-01-26T13:45:00.000Z",
            "EB_raw": 28.33,
            "EB_cumulative": 1271.51
          },
          {
            "x": "2026-01-26T14:00:00.000Z",
            "EB_raw": 42.43,
            "EB_cumulative": 1282.11
          }
        ],
        "timeRange": "1D",
        "selectedDate": "2026-01-26T14:10:23.261Z"
      },
      "instantaneous-power": {
        "config": {
          "title": "Instantaneous Power",
          "subtitle": "Main Incomer \u2022 Last 24 Hours",
          "variant": "V2",
          "mode": "area",
          "series": [
            {
              "id": "EB",
              "label": "Power Draw",
              "unit": "kW",
              "color": "#171717"
            }
          ]
        },
        "data": [
          {
            "x": "2026-01-25T18:30:00.000Z",
            "EB_raw": 24.44,
            "EB_cumulative": 6.11
          },
          {
            "x": "2026-01-25T18:45:00.000Z",
            "EB_raw": 20.2,
            "EB_cumulative": 11.16
          },
          {
            "x": "2026-01-25T19:00:00.000Z",
            "EB_raw": 29.64,
            "EB_cumulative": 18.57
          },
          {
            "x": "2026-01-25T19:15:00.000Z",
            "EB_raw": 22.09,
            "EB_cumulative": 24.09
          },
          {
            "x": "2026-01-25T19:30:00.000Z",
            "EB_raw": 22.39,
            "EB_cumulative": 29.69
          },
          {
            "x": "2026-01-25T19:45:00.000Z",
            "EB_raw": 20.2,
            "EB_cumulative": 34.74
          },
          {
            "x": "2026-01-25T20:00:00.000Z",
            "EB_raw": 26.62,
            "EB_cumulative": 41.39
          },
          {
            "x": "2026-01-25T20:15:00.000Z",
            "EB_raw": 25.36,
            "EB_cumulative": 47.73
          },
          {
            "x": "2026-01-25T20:30:00.000Z",
            "EB_raw": 22.79,
            "EB_cumulative": 53.43
          },
          {
            "x": "2026-01-25T20:45:00.000Z",
            "EB_raw": 27.74,
            "EB_cumulative": 60.36
          },
          {
            "x": "2026-01-25T21:00:00.000Z",
            "EB_raw": 21.63,
            "EB_cumulative": 65.77
          },
          {
            "x": "2026-01-25T21:15:00.000Z",
            "EB_raw": 21.72,
            "EB_cumulative": 71.2
          },
          {
            "x": "2026-01-25T21:30:00.000Z",
            "EB_raw": 24.03,
            "EB_cumulative": 77.21
          },
          {
            "x": "2026-01-25T21:45:00.000Z",
            "EB_raw": 20.16,
            "EB_cumulative": 82.25
          },
          {
            "x": "2026-01-25T22:00:00.000Z",
            "EB_raw": 28.61,
            "EB_cumulative": 89.4
          },
          {
            "x": "2026-01-25T22:15:00.000Z",
            "EB_raw": 29.29,
            "EB_cumulative": 96.72
          },
          {
            "x": "2026-01-25T22:30:00.000Z",
            "EB_raw": 26.75,
            "EB_cumulative": 103.41
          },
          {
            "x": "2026-01-25T22:45:00.000Z",
            "EB_raw": 26.26,
            "EB_cumulative": 109.97
          },
          {
            "x": "2026-01-25T23:00:00.000Z",
            "EB_raw": 21.7,
            "EB_cumulative": 115.4
          },
          {
            "x": "2026-01-25T23:15:00.000Z",
            "EB_raw": 23.51,
            "EB_cumulative": 121.27
          },
          {
            "x": "2026-01-25T23:30:00.000Z",
            "EB_raw": 20.12,
            "EB_cumulative": 126.31
          },
          {
            "x": "2026-01-25T23:45:00.000Z",
            "EB_raw": 28.62,
            "EB_cumulative": 133.46
          },
          {
            "x": "2026-01-26T00:00:00.000Z",
            "EB_raw": 22.27,
            "EB_cumulative": 139.03
          },
          {
            "x": "2026-01-26T00:15:00.000Z",
            "EB_raw": 23.18,
            "EB_cumulative": 144.82
          },
          {
            "x": "2026-01-26T00:30:00.000Z",
            "EB_raw": 22.45,
            "EB_cumulative": 150.44
          },
          {
            "x": "2026-01-26T00:45:00.000Z",
            "EB_raw": 26.28,
            "EB_cumulative": 157.01
          },
          {
            "x": "2026-01-26T01:00:00.000Z",
            "EB_raw": 27.15,
            "EB_cumulative": 163.79
          },
          {
            "x": "2026-01-26T01:15:00.000Z",
            "EB_raw": 21.74,
            "EB_cumulative": 169.23
          },
          {
            "x": "2026-01-26T01:30:00.000Z",
            "EB_raw": 26.05,
            "EB_cumulative": 175.74
          },
          {
            "x": "2026-01-26T01:45:00.000Z",
            "EB_raw": 26.43,
            "EB_cumulative": 182.35
          },
          {
            "x": "2026-01-26T02:00:00.000Z",
            "EB_raw": 29.43,
            "EB_cumulative": 189.71
          },
          {
            "x": "2026-01-26T02:15:00.000Z",
            "EB_raw": 29.91,
            "EB_cumulative": 197.18
          },
          {
            "x": "2026-01-26T02:30:00.000Z",
            "EB_raw": 24.1,
            "EB_cumulative": 203.21
          },
          {
            "x": "2026-01-26T02:45:00.000Z",
            "EB_raw": 106.95,
            "EB_cumulative": 229.95
          },
          {
            "x": "2026-01-26T03:00:00.000Z",
            "EB_raw": 103.39,
            "EB_cumulative": 255.79
          },
          {
            "x": "2026-01-26T03:15:00.000Z",
            "EB_raw": 104.68,
            "EB_cumulative": 281.97
          },
          {
            "x": "2026-01-26T03:30:00.000Z",
            "EB_raw": 101,
            "EB_cumulative": 307.22
          },
          {
            "x": "2026-01-26T03:45:00.000Z",
            "EB_raw": 101.37,
            "EB_cumulative": 332.56
          },
          {
            "x": "2026-01-26T04:00:00.000Z",
            "EB_raw": 105.78,
            "EB_cumulative": 359
          },
          {
            "x": "2026-01-26T04:15:00.000Z",
            "EB_raw": 103.9,
            "EB_cumulative": 384.98
          },
          {
            "x": "2026-01-26T04:30:00.000Z",
            "EB_raw": 109.24,
            "EB_cumulative": 412.29
          },
          {
            "x": "2026-01-26T04:45:00.000Z",
            "EB_raw": 109.83,
            "EB_cumulative": 439.75
          },
          {
            "x": "2026-01-26T05:00:00.000Z",
            "EB_raw": 101.25,
            "EB_cumulative": 465.06
          },
          {
            "x": "2026-01-26T05:15:00.000Z",
            "EB_raw": 109.76,
            "EB_cumulative": 492.5
          },
          {
            "x": "2026-01-26T05:30:00.000Z",
            "EB_raw": 103.01,
            "EB_cumulative": 518.25
          },
          {
            "x": "2026-01-26T05:45:00.000Z",
            "EB_raw": 101.3,
            "EB_cumulative": 543.57
          },
          {
            "x": "2026-01-26T06:00:00.000Z",
            "EB_raw": 104.12,
            "EB_cumulative": 569.6
          },
          {
            "x": "2026-01-26T06:15:00.000Z",
            "EB_raw": 106.97,
            "EB_cumulative": 596.35
          },
          {
            "x": "2026-01-26T06:30:00.000Z",
            "EB_raw": 105.22,
            "EB_cumulative": 622.65
          },
          {
            "x": "2026-01-26T06:45:00.000Z",
            "EB_raw": 102.45,
            "EB_cumulative": 648.27
          },
          {
            "x": "2026-01-26T07:00:00.000Z",
            "EB_raw": 102.27,
            "EB_cumulative": 673.83
          },
          {
            "x": "2026-01-26T07:15:00.000Z",
            "EB_raw": 102.67,
            "EB_cumulative": 699.5
          },
          {
            "x": "2026-01-26T07:30:00.000Z",
            "EB_raw": 102.66,
            "EB_cumulative": 725.16
          },
          {
            "x": "2026-01-26T07:45:00.000Z",
            "EB_raw": 107.17,
            "EB_cumulative": 751.96
          },
          {
            "x": "2026-01-26T08:00:00.000Z",
            "EB_raw": 104.74,
            "EB_cumulative": 778.14
          },
          {
            "x": "2026-01-26T08:15:00.000Z",
            "EB_raw": 106.45,
            "EB_cumulative": 804.76
          },
          {
            "x": "2026-01-26T08:30:00.000Z",
            "EB_raw": 102.58,
            "EB_cumulative": 830.4
          },
          {
            "x": "2026-01-26T08:45:00.000Z",
            "EB_raw": 103.1,
            "EB_cumulative": 856.18
          },
          {
            "x": "2026-01-26T09:00:00.000Z",
            "EB_raw": 102.69,
            "EB_cumulative": 881.85
          },
          {
            "x": "2026-01-26T09:15:00.000Z",
            "EB_raw": 104.15,
            "EB_cumulative": 907.88
          },
          {
            "x": "2026-01-26T09:30:00.000Z",
            "EB_raw": 105.38,
            "EB_cumulative": 934.23
          },
          {
            "x": "2026-01-26T09:45:00.000Z",
            "EB_raw": 107.25,
            "EB_cumulative": 961.04
          },
          {
            "x": "2026-01-26T10:00:00.000Z",
            "EB_raw": 109.49,
            "EB_cumulative": 988.41
          },
          {
            "x": "2026-01-26T10:15:00.000Z",
            "EB_raw": 109.06,
            "EB_cumulative": 1015.68
          },
          {
            "x": "2026-01-26T10:30:00.000Z",
            "EB_raw": 104.87,
            "EB_cumulative": 1041.9
          },
          {
            "x": "2026-01-26T10:45:00.000Z",
            "EB_raw": 105.89,
            "EB_cumulative": 1068.37
          },
          {
            "x": "2026-01-26T11:00:00.000Z",
            "EB_raw": 107.81,
            "EB_cumulative": 1095.32
          },
          {
            "x": "2026-01-26T11:15:00.000Z",
            "EB_raw": 105.35,
            "EB_cumulative": 1121.66
          },
          {
            "x": "2026-01-26T11:30:00.000Z",
            "EB_raw": 152.64,
            "EB_cumulative": 1159.82
          },
          {
            "x": "2026-01-26T11:45:00.000Z",
            "EB_raw": 103.47,
            "EB_cumulative": 1185.69
          },
          {
            "x": "2026-01-26T12:00:00.000Z",
            "EB_raw": 161.17,
            "EB_cumulative": 1225.98
          },
          {
            "x": "2026-01-26T12:15:00.000Z",
            "EB_raw": 101.44,
            "EB_cumulative": 1251.34
          },
          {
            "x": "2026-01-26T12:30:00.000Z",
            "EB_raw": 27.81,
            "EB_cumulative": 1258.29
          },
          {
            "x": "2026-01-26T12:45:00.000Z",
            "EB_raw": 44.67,
            "EB_cumulative": 1269.46
          },
          {
            "x": "2026-01-26T13:00:00.000Z",
            "EB_raw": 26.19,
            "EB_cumulative": 1276
          },
          {
            "x": "2026-01-26T13:15:00.000Z",
            "EB_raw": 27.83,
            "EB_cumulative": 1282.96
          },
          {
            "x": "2026-01-26T13:30:00.000Z",
            "EB_raw": 23.21,
            "EB_cumulative": 1288.76
          },
          {
            "x": "2026-01-26T13:45:00.000Z",
            "EB_raw": 24.18,
            "EB_cumulative": 1294.81
          },
          {
            "x": "2026-01-26T14:00:00.000Z",
            "EB_raw": 29.76,
            "EB_cumulative": 1302.25
          }
        ],
        "timeRange": "1D",
        "selectedDate": "2026-01-26T14:10:23.261Z"
      },
      "source-mix": {
        "config": {
          "title": "Source Mix",
          "subtitle": "Grid vs Solar vs DG \u2022 Today",
          "variant": "V3",
          "mode": "area",
          "stacked": true,
          "series": [
            {
              "id": "EB",
              "label": "Grid",
              "unit": "kW",
              "color": "#2563eb"
            },
            {
              "id": "Solar",
              "label": "Solar",
              "unit": "kW",
              "color": "#16a34a"
            },
            {
              "id": "DG",
              "label": "Diesel Gen",
              "unit": "kW",
              "color": "#d97706"
            }
          ]
        },
        "data": [
          {
            "x": "2026-01-25T18:30:00.000Z",
            "EB_raw": 28.44,
            "EB_cumulative": 7.11,
            "Solar_raw": 0,
            "Solar_cumulative": 0,
            "DG_raw": 47.67,
            "DG_cumulative": 11.92
          },
          {
            "x": "2026-01-25T18:45:00.000Z",
            "EB_raw": 25.01,
            "EB_cumulative": 13.36,
            "Solar_raw": 0,
            "Solar_cumulative": 0,
            "DG_raw": 42.22,
            "DG_cumulative": 22.47
          },
          {
            "x": "2026-01-25T19:00:00.000Z",
            "EB_raw": 23.48,
            "EB_cumulative": 19.23,
            "Solar_raw": 0,
            "Solar_cumulative": 0,
            "DG_raw": 77.38,
            "DG_cumulative": 41.82
          },
          {
            "x": "2026-01-25T19:15:00.000Z",
            "EB_raw": 39.42,
            "EB_cumulative": 29.09,
            "Solar_raw": 0,
            "Solar_cumulative": 0,
            "DG_raw": 69.07,
            "DG_cumulative": 59.08
          },
          {
            "x": "2026-01-25T19:30:00.000Z",
            "EB_raw": 26.26,
            "EB_cumulative": 35.65,
            "Solar_raw": 0,
            "Solar_cumulative": 0,
            "DG_raw": 87.82,
            "DG_cumulative": 81.04
          },
          {
            "x": "2026-01-25T19:45:00.000Z",
            "EB_raw": 27.46,
            "EB_cumulative": 42.52,
            "Solar_raw": 0,
            "Solar_cumulative": 0,
            "DG_raw": 50.74,
            "DG_cumulative": 93.73
          },
          {
            "x": "2026-01-25T20:00:00.000Z",
            "EB_raw": 23.27,
            "EB_cumulative": 48.33,
            "Solar_raw": 0,
            "Solar_cumulative": 0,
            "DG_raw": 97.91,
            "DG_cumulative": 118.2
          },
          {
            "x": "2026-01-25T20:15:00.000Z",
            "EB_raw": 27.49,
            "EB_cumulative": 55.2,
            "Solar_raw": 0,
            "Solar_cumulative": 0,
            "DG_raw": 88.14,
            "DG_cumulative": 140.24
          },
          {
            "x": "2026-01-25T20:30:00.000Z",
            "EB_raw": 24.1,
            "EB_cumulative": 61.23,
            "Solar_raw": 0,
            "Solar_cumulative": 0,
            "DG_raw": 53.86,
            "DG_cumulative": 153.7
          },
          {
            "x": "2026-01-25T20:45:00.000Z",
            "EB_raw": 21.81,
            "EB_cumulative": 66.68,
            "Solar_raw": 0,
            "Solar_cumulative": 0,
            "DG_raw": 50.1,
            "DG_cumulative": 166.23
          },
          {
            "x": "2026-01-25T21:00:00.000Z",
            "EB_raw": 28.8,
            "EB_cumulative": 73.88,
            "Solar_raw": 0,
            "Solar_cumulative": 0,
            "DG_raw": 50.22,
            "DG_cumulative": 178.79
          },
          {
            "x": "2026-01-25T21:15:00.000Z",
            "EB_raw": 23.62,
            "EB_cumulative": 79.79,
            "Solar_raw": 0,
            "Solar_cumulative": 0,
            "DG_raw": 61.47,
            "DG_cumulative": 194.15
          },
          {
            "x": "2026-01-25T21:30:00.000Z",
            "EB_raw": 25.35,
            "EB_cumulative": 86.13,
            "Solar_raw": 0,
            "Solar_cumulative": 0,
            "DG_raw": 78.32,
            "DG_cumulative": 213.73
          },
          {
            "x": "2026-01-25T21:45:00.000Z",
            "EB_raw": 29.32,
            "EB_cumulative": 93.46,
            "Solar_raw": 0,
            "Solar_cumulative": 0,
            "DG_raw": 26.18,
            "DG_cumulative": 220.28
          },
          {
            "x": "2026-01-25T22:00:00.000Z",
            "EB_raw": 23.28,
            "EB_cumulative": 99.28,
            "Solar_raw": 0,
            "Solar_cumulative": 0,
            "DG_raw": 93.68,
            "DG_cumulative": 243.7
          },
          {
            "x": "2026-01-25T22:15:00.000Z",
            "EB_raw": 23.18,
            "EB_cumulative": 105.07,
            "Solar_raw": 0,
            "Solar_cumulative": 0,
            "DG_raw": 52.91,
            "DG_cumulative": 256.92
          },
          {
            "x": "2026-01-25T22:30:00.000Z",
            "EB_raw": 22.91,
            "EB_cumulative": 110.8,
            "Solar_raw": 0,
            "Solar_cumulative": 0,
            "DG_raw": 42.62,
            "DG_cumulative": 267.58
          },
          {
            "x": "2026-01-25T22:45:00.000Z",
            "EB_raw": 20.4,
            "EB_cumulative": 115.9,
            "Solar_raw": 0,
            "Solar_cumulative": 0,
            "DG_raw": 64.52,
            "DG_cumulative": 283.71
          },
          {
            "x": "2026-01-25T23:00:00.000Z",
            "EB_raw": 22.15,
            "EB_cumulative": 121.44,
            "Solar_raw": 0,
            "Solar_cumulative": 0,
            "DG_raw": 32.53,
            "DG_cumulative": 291.84
          },
          {
            "x": "2026-01-25T23:15:00.000Z",
            "EB_raw": 27.04,
            "EB_cumulative": 128.2,
            "Solar_raw": 0,
            "Solar_cumulative": 0,
            "DG_raw": 69.01,
            "DG_cumulative": 309.09
          },
          {
            "x": "2026-01-25T23:30:00.000Z",
            "EB_raw": 23.94,
            "EB_cumulative": 134.18,
            "Solar_raw": 0,
            "Solar_cumulative": 0,
            "DG_raw": 32.92,
            "DG_cumulative": 317.32
          },
          {
            "x": "2026-01-25T23:45:00.000Z",
            "EB_raw": 20.09,
            "EB_cumulative": 139.21,
            "Solar_raw": 0,
            "Solar_cumulative": 0,
            "DG_raw": 1.48,
            "DG_cumulative": 317.69
          },
          {
            "x": "2026-01-26T00:00:00.000Z",
            "EB_raw": 21.12,
            "EB_cumulative": 144.49,
            "Solar_raw": 0,
            "Solar_cumulative": 0,
            "DG_raw": 14.51,
            "DG_cumulative": 321.32
          },
          {
            "x": "2026-01-26T00:15:00.000Z",
            "EB_raw": 24.15,
            "EB_cumulative": 150.52,
            "Solar_raw": 0,
            "Solar_cumulative": 0,
            "DG_raw": 44.9,
            "DG_cumulative": 332.55
          },
          {
            "x": "2026-01-26T00:30:00.000Z",
            "EB_raw": 23.16,
            "EB_cumulative": 156.31,
            "Solar_raw": 0,
            "Solar_cumulative": 0,
            "DG_raw": 89.61,
            "DG_cumulative": 354.95
          },
          {
            "x": "2026-01-26T00:45:00.000Z",
            "EB_raw": 29.86,
            "EB_cumulative": 163.78,
            "Solar_raw": 3.8,
            "Solar_cumulative": 0.95,
            "DG_raw": 10.37,
            "DG_cumulative": 357.54
          },
          {
            "x": "2026-01-26T01:00:00.000Z",
            "EB_raw": 29.3,
            "EB_cumulative": 171.11,
            "Solar_raw": 11.11,
            "Solar_cumulative": 3.73,
            "DG_raw": 67.38,
            "DG_cumulative": 374.39
          },
          {
            "x": "2026-01-26T01:15:00.000Z",
            "EB_raw": 28.04,
            "EB_cumulative": 178.12,
            "Solar_raw": 13.18,
            "Solar_cumulative": 7.02,
            "DG_raw": 99.03,
            "DG_cumulative": 399.14
          },
          {
            "x": "2026-01-26T01:30:00.000Z",
            "EB_raw": 25.99,
            "EB_cumulative": 184.61,
            "Solar_raw": 15.46,
            "Solar_cumulative": 10.89,
            "DG_raw": 67.1,
            "DG_cumulative": 415.92
          },
          {
            "x": "2026-01-26T01:45:00.000Z",
            "EB_raw": 22.94,
            "EB_cumulative": 190.35,
            "Solar_raw": 17.27,
            "Solar_cumulative": 15.21,
            "DG_raw": 38.33,
            "DG_cumulative": 425.5
          },
          {
            "x": "2026-01-26T02:00:00.000Z",
            "EB_raw": 20.9,
            "EB_cumulative": 195.57,
            "Solar_raw": 21.82,
            "Solar_cumulative": 20.66,
            "DG_raw": 34.63,
            "DG_cumulative": 434.16
          },
          {
            "x": "2026-01-26T02:15:00.000Z",
            "EB_raw": 21.89,
            "EB_cumulative": 201.05,
            "Solar_raw": 24.91,
            "Solar_cumulative": 26.89,
            "DG_raw": 8.66,
            "DG_cumulative": 436.32
          },
          {
            "x": "2026-01-26T02:30:00.000Z",
            "EB_raw": 26.01,
            "EB_cumulative": 207.55,
            "Solar_raw": 27.07,
            "Solar_cumulative": 33.66,
            "DG_raw": 63.68,
            "DG_cumulative": 452.24
          },
          {
            "x": "2026-01-26T02:45:00.000Z",
            "EB_raw": 104.88,
            "EB_cumulative": 233.77,
            "Solar_raw": 30.9,
            "Solar_cumulative": 41.38,
            "DG_raw": 87.14,
            "DG_cumulative": 474.03
          },
          {
            "x": "2026-01-26T03:00:00.000Z",
            "EB_raw": 107.77,
            "EB_cumulative": 260.71,
            "Solar_raw": 32.22,
            "Solar_cumulative": 49.44,
            "DG_raw": 40.62,
            "DG_cumulative": 484.18
          },
          {
            "x": "2026-01-26T03:15:00.000Z",
            "EB_raw": 106.44,
            "EB_cumulative": 287.32,
            "Solar_raw": 35.61,
            "Solar_cumulative": 58.34,
            "DG_raw": 93.17,
            "DG_cumulative": 507.47
          },
          {
            "x": "2026-01-26T03:30:00.000Z",
            "EB_raw": 104.47,
            "EB_cumulative": 313.44,
            "Solar_raw": 39.55,
            "Solar_cumulative": 68.23,
            "DG_raw": 88.08,
            "DG_cumulative": 529.49
          },
          {
            "x": "2026-01-26T03:45:00.000Z",
            "EB_raw": 102.19,
            "EB_cumulative": 338.99,
            "Solar_raw": 39.77,
            "Solar_cumulative": 78.17,
            "DG_raw": 78.61,
            "DG_cumulative": 549.15
          },
          {
            "x": "2026-01-26T04:00:00.000Z",
            "EB_raw": 108.11,
            "EB_cumulative": 366.01,
            "Solar_raw": 40.37,
            "Solar_cumulative": 88.26,
            "DG_raw": 93.39,
            "DG_cumulative": 572.49
          },
          {
            "x": "2026-01-26T04:15:00.000Z",
            "EB_raw": 102.14,
            "EB_cumulative": 391.55,
            "Solar_raw": 44.24,
            "Solar_cumulative": 99.33,
            "DG_raw": 69.07,
            "DG_cumulative": 589.76
          },
          {
            "x": "2026-01-26T04:30:00.000Z",
            "EB_raw": 102.44,
            "EB_cumulative": 417.16,
            "Solar_raw": 44.63,
            "Solar_cumulative": 110.48,
            "DG_raw": 34.76,
            "DG_cumulative": 598.45
          },
          {
            "x": "2026-01-26T04:45:00.000Z",
            "EB_raw": 102.2,
            "EB_cumulative": 442.71,
            "Solar_raw": 44.91,
            "Solar_cumulative": 121.71,
            "DG_raw": 91.43,
            "DG_cumulative": 621.31
          },
          {
            "x": "2026-01-26T05:00:00.000Z",
            "EB_raw": 103.61,
            "EB_cumulative": 468.61,
            "Solar_raw": 48.05,
            "Solar_cumulative": 133.72,
            "DG_raw": 33.47,
            "DG_cumulative": 629.68
          },
          {
            "x": "2026-01-26T05:15:00.000Z",
            "EB_raw": 106.26,
            "EB_cumulative": 495.17,
            "Solar_raw": 48.79,
            "Solar_cumulative": 145.92,
            "DG_raw": 30.73,
            "DG_cumulative": 637.36
          },
          {
            "x": "2026-01-26T05:30:00.000Z",
            "EB_raw": 109.28,
            "EB_cumulative": 522.49,
            "Solar_raw": 51.16,
            "Solar_cumulative": 158.71,
            "DG_raw": 40.9,
            "DG_cumulative": 647.59
          },
          {
            "x": "2026-01-26T05:45:00.000Z",
            "EB_raw": 102.87,
            "EB_cumulative": 548.21,
            "Solar_raw": 51.31,
            "Solar_cumulative": 171.54,
            "DG_raw": 73.69,
            "DG_cumulative": 666.01
          },
          {
            "x": "2026-01-26T06:00:00.000Z",
            "EB_raw": 104.66,
            "EB_cumulative": 574.38,
            "Solar_raw": 52.54,
            "Solar_cumulative": 184.67,
            "DG_raw": 18.86,
            "DG_cumulative": 670.73
          },
          {
            "x": "2026-01-26T06:15:00.000Z",
            "EB_raw": 101.77,
            "EB_cumulative": 599.82,
            "Solar_raw": 50.32,
            "Solar_cumulative": 197.25,
            "DG_raw": 50.9,
            "DG_cumulative": 683.45
          },
          {
            "x": "2026-01-26T06:30:00.000Z",
            "EB_raw": 109.88,
            "EB_cumulative": 627.29,
            "Solar_raw": 52.08,
            "Solar_cumulative": 210.27,
            "DG_raw": 22.26,
            "DG_cumulative": 689.02
          },
          {
            "x": "2026-01-26T06:45:00.000Z",
            "EB_raw": 107.65,
            "EB_cumulative": 654.2,
            "Solar_raw": 51.95,
            "Solar_cumulative": 223.26,
            "DG_raw": 40.58,
            "DG_cumulative": 699.16
          },
          {
            "x": "2026-01-26T07:00:00.000Z",
            "EB_raw": 103.88,
            "EB_cumulative": 680.17,
            "Solar_raw": 52.73,
            "Solar_cumulative": 236.44,
            "DG_raw": 40.25,
            "DG_cumulative": 709.22
          },
          {
            "x": "2026-01-26T07:15:00.000Z",
            "EB_raw": 102.89,
            "EB_cumulative": 705.89,
            "Solar_raw": 53.35,
            "Solar_cumulative": 249.78,
            "DG_raw": 27.98,
            "DG_cumulative": 716.22
          },
          {
            "x": "2026-01-26T07:30:00.000Z",
            "EB_raw": 103.81,
            "EB_cumulative": 731.85,
            "Solar_raw": 50.31,
            "Solar_cumulative": 262.36,
            "DG_raw": 6.34,
            "DG_cumulative": 717.8
          },
          {
            "x": "2026-01-26T07:45:00.000Z",
            "EB_raw": 108.7,
            "EB_cumulative": 759.02,
            "Solar_raw": 51.65,
            "Solar_cumulative": 275.27,
            "DG_raw": 14.72,
            "DG_cumulative": 721.48
          },
          {
            "x": "2026-01-26T08:00:00.000Z",
            "EB_raw": 102.41,
            "EB_cumulative": 784.62,
            "Solar_raw": 51.16,
            "Solar_cumulative": 288.06,
            "DG_raw": 89.38,
            "DG_cumulative": 743.83
          },
          {
            "x": "2026-01-26T08:15:00.000Z",
            "EB_raw": 102.36,
            "EB_cumulative": 810.21,
            "Solar_raw": 49.65,
            "Solar_cumulative": 300.47,
            "DG_raw": 29.48,
            "DG_cumulative": 751.2
          },
          {
            "x": "2026-01-26T08:30:00.000Z",
            "EB_raw": 102.7,
            "EB_cumulative": 835.89,
            "Solar_raw": 46.45,
            "Solar_cumulative": 312.09,
            "DG_raw": 94.99,
            "DG_cumulative": 774.95
          },
          {
            "x": "2026-01-26T08:45:00.000Z",
            "EB_raw": 106.27,
            "EB_cumulative": 862.46,
            "Solar_raw": 42.24,
            "Solar_cumulative": 322.65,
            "DG_raw": 34.03,
            "DG_cumulative": 783.45
          },
          {
            "x": "2026-01-26T09:00:00.000Z",
            "EB_raw": 102,
            "EB_cumulative": 887.96,
            "Solar_raw": 42.82,
            "Solar_cumulative": 333.35,
            "DG_raw": 4.87,
            "DG_cumulative": 784.67
          },
          {
            "x": "2026-01-26T09:15:00.000Z",
            "EB_raw": 103.3,
            "EB_cumulative": 913.78,
            "Solar_raw": 38.34,
            "Solar_cumulative": 342.94,
            "DG_raw": 93.17,
            "DG_cumulative": 807.96
          },
          {
            "x": "2026-01-26T09:30:00.000Z",
            "EB_raw": 103.29,
            "EB_cumulative": 939.6,
            "Solar_raw": 36.94,
            "Solar_cumulative": 352.17,
            "DG_raw": 82.56,
            "DG_cumulative": 828.6
          },
          {
            "x": "2026-01-26T09:45:00.000Z",
            "EB_raw": 107.22,
            "EB_cumulative": 966.41,
            "Solar_raw": 37.8,
            "Solar_cumulative": 361.62,
            "DG_raw": 81.43,
            "DG_cumulative": 848.96
          },
          {
            "x": "2026-01-26T10:00:00.000Z",
            "EB_raw": 107.11,
            "EB_cumulative": 993.18,
            "Solar_raw": 33.18,
            "Solar_cumulative": 369.92,
            "DG_raw": 76.65,
            "DG_cumulative": 868.12
          },
          {
            "x": "2026-01-26T10:15:00.000Z",
            "EB_raw": 107.58,
            "EB_cumulative": 1020.08,
            "Solar_raw": 27.98,
            "Solar_cumulative": 376.91,
            "DG_raw": 30.91,
            "DG_cumulative": 875.85
          },
          {
            "x": "2026-01-26T10:30:00.000Z",
            "EB_raw": 101.69,
            "EB_cumulative": 1045.5,
            "Solar_raw": 27.3,
            "Solar_cumulative": 383.74,
            "DG_raw": 77.33,
            "DG_cumulative": 895.18
          },
          {
            "x": "2026-01-26T10:45:00.000Z",
            "EB_raw": 104.2,
            "EB_cumulative": 1071.55,
            "Solar_raw": 24.24,
            "Solar_cumulative": 389.8,
            "DG_raw": 35.26,
            "DG_cumulative": 904
          },
          {
            "x": "2026-01-26T11:00:00.000Z",
            "EB_raw": 105.22,
            "EB_cumulative": 1097.85,
            "Solar_raw": 19.73,
            "Solar_cumulative": 394.73,
            "DG_raw": 83.28,
            "DG_cumulative": 924.82
          },
          {
            "x": "2026-01-26T11:15:00.000Z",
            "EB_raw": 103.05,
            "EB_cumulative": 1123.61,
            "Solar_raw": 16.77,
            "Solar_cumulative": 398.92,
            "DG_raw": 34.9,
            "DG_cumulative": 933.54
          },
          {
            "x": "2026-01-26T11:30:00.000Z",
            "EB_raw": 107.47,
            "EB_cumulative": 1150.48,
            "Solar_raw": 13.98,
            "Solar_cumulative": 402.42,
            "DG_raw": 7.41,
            "DG_cumulative": 935.4
          },
          {
            "x": "2026-01-26T11:45:00.000Z",
            "EB_raw": 102.89,
            "EB_cumulative": 1176.2,
            "Solar_raw": 12.91,
            "Solar_cumulative": 405.64,
            "DG_raw": 17.74,
            "DG_cumulative": 939.83
          },
          {
            "x": "2026-01-26T12:00:00.000Z",
            "EB_raw": 103.76,
            "EB_cumulative": 1202.15,
            "Solar_raw": 8.03,
            "Solar_cumulative": 407.65,
            "DG_raw": 33.64,
            "DG_cumulative": 948.24
          },
          {
            "x": "2026-01-26T12:15:00.000Z",
            "EB_raw": 109.04,
            "EB_cumulative": 1229.41,
            "Solar_raw": 3.83,
            "Solar_cumulative": 408.61,
            "DG_raw": 85.56,
            "DG_cumulative": 969.63
          },
          {
            "x": "2026-01-26T12:30:00.000Z",
            "EB_raw": 24.9,
            "EB_cumulative": 1235.63,
            "Solar_raw": 0,
            "Solar_cumulative": 408.61,
            "DG_raw": 3.66,
            "DG_cumulative": 970.55
          },
          {
            "x": "2026-01-26T12:45:00.000Z",
            "EB_raw": 26.72,
            "EB_cumulative": 1242.31,
            "Solar_raw": 0,
            "Solar_cumulative": 408.61,
            "DG_raw": 99.19,
            "DG_cumulative": 995.34
          },
          {
            "x": "2026-01-26T13:00:00.000Z",
            "EB_raw": 29.39,
            "EB_cumulative": 1249.66,
            "Solar_raw": 0,
            "Solar_cumulative": 408.61,
            "DG_raw": 96.03,
            "DG_cumulative": 1019.35
          },
          {
            "x": "2026-01-26T13:15:00.000Z",
            "EB_raw": 21.75,
            "EB_cumulative": 1255.1,
            "Solar_raw": 0,
            "Solar_cumulative": 408.61,
            "DG_raw": 90.49,
            "DG_cumulative": 1041.98
          },
          {
            "x": "2026-01-26T13:30:00.000Z",
            "EB_raw": 22.98,
            "EB_cumulative": 1260.84,
            "Solar_raw": 0,
            "Solar_cumulative": 408.61,
            "DG_raw": 79.64,
            "DG_cumulative": 1061.89
          },
          {
            "x": "2026-01-26T13:45:00.000Z",
            "EB_raw": 29.17,
            "EB_cumulative": 1268.13,
            "Solar_raw": 0,
            "Solar_cumulative": 408.61,
            "DG_raw": 2.57,
            "DG_cumulative": 1062.53
          },
          {
            "x": "2026-01-26T14:00:00.000Z",
            "EB_raw": 21.66,
            "EB_cumulative": 1273.55,
            "Solar_raw": 0,
            "Solar_cumulative": 408.61,
            "DG_raw": 39.78,
            "DG_cumulative": 1072.47
          }
        ],
        "timeRange": "1D",
        "selectedDate": "2026-01-26T14:10:23.261Z"
      },
      "performance-vs-baseline": {
        "config": {
          "title": "Performance vs Baseline",
          "subtitle": "Production Count \u2022 vs Last Week",
          "variant": "V4",
          "mode": "cumulative",
          "series": [
            {
              "id": "prod",
              "label": "This Week",
              "unit": "units",
              "color": "#171717"
            },
            {
              "id": "baseline",
              "label": "Last Week",
              "unit": "units",
              "color": "#a3a3a3",
              "isBaseline": true,
              "strokeDasharray": "4 4"
            }
          ]
        },
        "data": [
          {
            "x": "2026-01-25T18:30:00.000Z",
            "prod_raw": 9.09,
            "prod_cumulative": 2.27,
            "baseline_raw": 60,
            "baseline_cumulative": 15
          },
          {
            "x": "2026-01-25T18:45:00.000Z",
            "prod_raw": 43.27,
            "prod_cumulative": 13.09,
            "baseline_raw": 61.25,
            "baseline_cumulative": 30.31
          },
          {
            "x": "2026-01-25T19:00:00.000Z",
            "prod_raw": 72.97,
            "prod_cumulative": 31.33,
            "baseline_raw": 62.47,
            "baseline_cumulative": 45.93
          },
          {
            "x": "2026-01-25T19:15:00.000Z",
            "prod_raw": 8.48,
            "prod_cumulative": 33.45,
            "baseline_raw": 63.66,
            "baseline_cumulative": 61.85
          },
          {
            "x": "2026-01-25T19:30:00.000Z",
            "prod_raw": 30.28,
            "prod_cumulative": 41.02,
            "baseline_raw": 64.79,
            "baseline_cumulative": 78.04
          },
          {
            "x": "2026-01-25T19:45:00.000Z",
            "prod_raw": 22.48,
            "prod_cumulative": 46.64,
            "baseline_raw": 65.85,
            "baseline_cumulative": 94.51
          },
          {
            "x": "2026-01-25T20:00:00.000Z",
            "prod_raw": 7.15,
            "prod_cumulative": 48.43,
            "baseline_raw": 66.82,
            "baseline_cumulative": 111.21
          },
          {
            "x": "2026-01-25T20:15:00.000Z",
            "prod_raw": 2.77,
            "prod_cumulative": 49.12,
            "baseline_raw": 67.68,
            "baseline_cumulative": 128.13
          },
          {
            "x": "2026-01-25T20:30:00.000Z",
            "prod_raw": 43.07,
            "prod_cumulative": 59.89,
            "baseline_raw": 68.41,
            "baseline_cumulative": 145.23
          },
          {
            "x": "2026-01-25T20:45:00.000Z",
            "prod_raw": 53.42,
            "prod_cumulative": 73.24,
            "baseline_raw": 69.02,
            "baseline_cumulative": 162.49
          },
          {
            "x": "2026-01-25T21:00:00.000Z",
            "prod_raw": 67.14,
            "prod_cumulative": 90.03,
            "baseline_raw": 69.49,
            "baseline_cumulative": 179.86
          },
          {
            "x": "2026-01-25T21:15:00.000Z",
            "prod_raw": 66.25,
            "prod_cumulative": 106.59,
            "baseline_raw": 69.81,
            "baseline_cumulative": 197.31
          },
          {
            "x": "2026-01-25T21:30:00.000Z",
            "prod_raw": 18.88,
            "prod_cumulative": 111.31,
            "baseline_raw": 69.97,
            "baseline_cumulative": 214.81
          },
          {
            "x": "2026-01-25T21:45:00.000Z",
            "prod_raw": 26.66,
            "prod_cumulative": 117.97,
            "baseline_raw": 69.99,
            "baseline_cumulative": 232.3
          },
          {
            "x": "2026-01-25T22:00:00.000Z",
            "prod_raw": 1.04,
            "prod_cumulative": 118.23,
            "baseline_raw": 69.84,
            "baseline_cumulative": 249.76
          },
          {
            "x": "2026-01-25T22:15:00.000Z",
            "prod_raw": 14.93,
            "prod_cumulative": 121.97,
            "baseline_raw": 69.54,
            "baseline_cumulative": 267.15
          },
          {
            "x": "2026-01-25T22:30:00.000Z",
            "prod_raw": 74.74,
            "prod_cumulative": 140.65,
            "baseline_raw": 69.09,
            "baseline_cumulative": 284.42
          },
          {
            "x": "2026-01-25T22:45:00.000Z",
            "prod_raw": 6.12,
            "prod_cumulative": 142.18,
            "baseline_raw": 68.5,
            "baseline_cumulative": 301.55
          },
          {
            "x": "2026-01-25T23:00:00.000Z",
            "prod_raw": 58.54,
            "prod_cumulative": 156.82,
            "baseline_raw": 67.78,
            "baseline_cumulative": 318.49
          },
          {
            "x": "2026-01-25T23:15:00.000Z",
            "prod_raw": 26.7,
            "prod_cumulative": 163.49,
            "baseline_raw": 66.94,
            "baseline_cumulative": 335.23
          },
          {
            "x": "2026-01-25T23:30:00.000Z",
            "prod_raw": 7.99,
            "prod_cumulative": 165.49,
            "baseline_raw": 65.98,
            "baseline_cumulative": 351.72
          },
          {
            "x": "2026-01-25T23:45:00.000Z",
            "prod_raw": 97.57,
            "prod_cumulative": 189.88,
            "baseline_raw": 64.94,
            "baseline_cumulative": 367.96
          },
          {
            "x": "2026-01-26T00:00:00.000Z",
            "prod_raw": 71.85,
            "prod_cumulative": 207.84,
            "baseline_raw": 63.82,
            "baseline_cumulative": 383.91
          },
          {
            "x": "2026-01-26T00:15:00.000Z",
            "prod_raw": 13.36,
            "prod_cumulative": 211.18,
            "baseline_raw": 62.63,
            "baseline_cumulative": 399.57
          },
          {
            "x": "2026-01-26T00:30:00.000Z",
            "prod_raw": 26.11,
            "prod_cumulative": 217.71,
            "baseline_raw": 61.41,
            "baseline_cumulative": 414.92
          },
          {
            "x": "2026-01-26T00:45:00.000Z",
            "prod_raw": 68.52,
            "prod_cumulative": 234.84,
            "baseline_raw": 60.17,
            "baseline_cumulative": 429.97
          },
          {
            "x": "2026-01-26T01:00:00.000Z",
            "prod_raw": 32.86,
            "prod_cumulative": 243.05,
            "baseline_raw": 58.92,
            "baseline_cumulative": 444.7
          },
          {
            "x": "2026-01-26T01:15:00.000Z",
            "prod_raw": 7.11,
            "prod_cumulative": 244.83,
            "baseline_raw": 57.69,
            "baseline_cumulative": 459.12
          },
          {
            "x": "2026-01-26T01:30:00.000Z",
            "prod_raw": 4.52,
            "prod_cumulative": 245.96,
            "baseline_raw": 56.49,
            "baseline_cumulative": 473.24
          },
          {
            "x": "2026-01-26T01:45:00.000Z",
            "prod_raw": 12.13,
            "prod_cumulative": 248.99,
            "baseline_raw": 55.35,
            "baseline_cumulative": 487.08
          },
          {
            "x": "2026-01-26T02:00:00.000Z",
            "prod_raw": 24.96,
            "prod_cumulative": 255.23,
            "baseline_raw": 54.28,
            "baseline_cumulative": 500.65
          },
          {
            "x": "2026-01-26T02:15:00.000Z",
            "prod_raw": 38.12,
            "prod_cumulative": 264.76,
            "baseline_raw": 53.31,
            "baseline_cumulative": 513.98
          },
          {
            "x": "2026-01-26T02:30:00.000Z",
            "prod_raw": 78.93,
            "prod_cumulative": 284.5,
            "baseline_raw": 52.43,
            "baseline_cumulative": 527.08
          },
          {
            "x": "2026-01-26T02:45:00.000Z",
            "prod_raw": 19.52,
            "prod_cumulative": 289.37,
            "baseline_raw": 51.68,
            "baseline_cumulative": 540
          },
          {
            "x": "2026-01-26T03:00:00.000Z",
            "prod_raw": 0.93,
            "prod_cumulative": 289.61,
            "baseline_raw": 51.05,
            "baseline_cumulative": 552.77
          },
          {
            "x": "2026-01-26T03:15:00.000Z",
            "prod_raw": 49.95,
            "prod_cumulative": 302.09,
            "baseline_raw": 50.56,
            "baseline_cumulative": 565.41
          },
          {
            "x": "2026-01-26T03:30:00.000Z",
            "prod_raw": 23.11,
            "prod_cumulative": 307.87,
            "baseline_raw": 50.22,
            "baseline_cumulative": 577.96
          },
          {
            "x": "2026-01-26T03:45:00.000Z",
            "prod_raw": 34.14,
            "prod_cumulative": 316.41,
            "baseline_raw": 50.04,
            "baseline_cumulative": 590.47
          },
          {
            "x": "2026-01-26T04:00:00.000Z",
            "prod_raw": 22.46,
            "prod_cumulative": 322.02,
            "baseline_raw": 50.01,
            "baseline_cumulative": 602.97
          },
          {
            "x": "2026-01-26T04:15:00.000Z",
            "prod_raw": 56.29,
            "prod_cumulative": 336.09,
            "baseline_raw": 50.13,
            "baseline_cumulative": 615.51
          },
          {
            "x": "2026-01-26T04:30:00.000Z",
            "prod_raw": 81.03,
            "prod_cumulative": 356.35,
            "baseline_raw": 50.41,
            "baseline_cumulative": 628.11
          },
          {
            "x": "2026-01-26T04:45:00.000Z",
            "prod_raw": 83.74,
            "prod_cumulative": 377.29,
            "baseline_raw": 50.84,
            "baseline_cumulative": 640.82
          },
          {
            "x": "2026-01-26T05:00:00.000Z",
            "prod_raw": 88.7,
            "prod_cumulative": 399.46,
            "baseline_raw": 51.41,
            "baseline_cumulative": 653.67
          },
          {
            "x": "2026-01-26T05:15:00.000Z",
            "prod_raw": 10.47,
            "prod_cumulative": 402.08,
            "baseline_raw": 52.12,
            "baseline_cumulative": 666.7
          },
          {
            "x": "2026-01-26T05:30:00.000Z",
            "prod_raw": 14.04,
            "prod_cumulative": 405.59,
            "baseline_raw": 52.94,
            "baseline_cumulative": 679.94
          },
          {
            "x": "2026-01-26T05:45:00.000Z",
            "prod_raw": 64.19,
            "prod_cumulative": 421.64,
            "baseline_raw": 53.88,
            "baseline_cumulative": 693.41
          },
          {
            "x": "2026-01-26T06:00:00.000Z",
            "prod_raw": 89.01,
            "prod_cumulative": 443.89,
            "baseline_raw": 54.92,
            "baseline_cumulative": 707.14
          },
          {
            "x": "2026-01-26T06:15:00.000Z",
            "prod_raw": 49.37,
            "prod_cumulative": 456.23,
            "baseline_raw": 56.03,
            "baseline_cumulative": 721.14
          },
          {
            "x": "2026-01-26T06:30:00.000Z",
            "prod_raw": 42.46,
            "prod_cumulative": 466.85,
            "baseline_raw": 57.21,
            "baseline_cumulative": 735.45
          },
          {
            "x": "2026-01-26T06:45:00.000Z",
            "prod_raw": 6.65,
            "prod_cumulative": 468.51,
            "baseline_raw": 58.42,
            "baseline_cumulative": 750.05
          },
          {
            "x": "2026-01-26T07:00:00.000Z",
            "prod_raw": 3.71,
            "prod_cumulative": 469.44,
            "baseline_raw": 59.67,
            "baseline_cumulative": 764.97
          },
          {
            "x": "2026-01-26T07:15:00.000Z",
            "prod_raw": 22.08,
            "prod_cumulative": 474.96,
            "baseline_raw": 60.92,
            "baseline_cumulative": 780.2
          },
          {
            "x": "2026-01-26T07:30:00.000Z",
            "prod_raw": 32.07,
            "prod_cumulative": 482.97,
            "baseline_raw": 62.15,
            "baseline_cumulative": 795.74
          },
          {
            "x": "2026-01-26T07:45:00.000Z",
            "prod_raw": 58.77,
            "prod_cumulative": 497.67,
            "baseline_raw": 63.35,
            "baseline_cumulative": 811.57
          },
          {
            "x": "2026-01-26T08:00:00.000Z",
            "prod_raw": 35.77,
            "prod_cumulative": 506.61,
            "baseline_raw": 64.5,
            "baseline_cumulative": 827.7
          },
          {
            "x": "2026-01-26T08:15:00.000Z",
            "prod_raw": 75.88,
            "prod_cumulative": 525.58,
            "baseline_raw": 65.58,
            "baseline_cumulative": 844.09
          },
          {
            "x": "2026-01-26T08:30:00.000Z",
            "prod_raw": 5.44,
            "prod_cumulative": 526.94,
            "baseline_raw": 66.57,
            "baseline_cumulative": 860.74
          },
          {
            "x": "2026-01-26T08:45:00.000Z",
            "prod_raw": 75.2,
            "prod_cumulative": 545.74,
            "baseline_raw": 67.46,
            "baseline_cumulative": 877.6
          },
          {
            "x": "2026-01-26T09:00:00.000Z",
            "prod_raw": 71.49,
            "prod_cumulative": 563.61,
            "baseline_raw": 68.23,
            "baseline_cumulative": 894.66
          },
          {
            "x": "2026-01-26T09:15:00.000Z",
            "prod_raw": 29.58,
            "prod_cumulative": 571.01,
            "baseline_raw": 68.87,
            "baseline_cumulative": 911.88
          },
          {
            "x": "2026-01-26T09:30:00.000Z",
            "prod_raw": 49.92,
            "prod_cumulative": 583.49,
            "baseline_raw": 69.38,
            "baseline_cumulative": 929.22
          },
          {
            "x": "2026-01-26T09:45:00.000Z",
            "prod_raw": 41.65,
            "prod_cumulative": 593.9,
            "baseline_raw": 69.74,
            "baseline_cumulative": 946.66
          },
          {
            "x": "2026-01-26T10:00:00.000Z",
            "prod_raw": 96.94,
            "prod_cumulative": 618.14,
            "baseline_raw": 69.95,
            "baseline_cumulative": 964.14
          },
          {
            "x": "2026-01-26T10:15:00.000Z",
            "prod_raw": 8.74,
            "prod_cumulative": 620.32,
            "baseline_raw": 70,
            "baseline_cumulative": 981.64
          },
          {
            "x": "2026-01-26T10:30:00.000Z",
            "prod_raw": 90.45,
            "prod_cumulative": 642.93,
            "baseline_raw": 69.89,
            "baseline_cumulative": 999.12
          },
          {
            "x": "2026-01-26T10:45:00.000Z",
            "prod_raw": 32.82,
            "prod_cumulative": 651.14,
            "baseline_raw": 69.63,
            "baseline_cumulative": 1016.53
          },
          {
            "x": "2026-01-26T11:00:00.000Z",
            "prod_raw": 31.6,
            "prod_cumulative": 659.04,
            "baseline_raw": 69.23,
            "baseline_cumulative": 1033.83
          },
          {
            "x": "2026-01-26T11:15:00.000Z",
            "prod_raw": 24.79,
            "prod_cumulative": 665.24,
            "baseline_raw": 68.67,
            "baseline_cumulative": 1051
          },
          {
            "x": "2026-01-26T11:30:00.000Z",
            "prod_raw": 72.97,
            "prod_cumulative": 683.48,
            "baseline_raw": 67.98,
            "baseline_cumulative": 1068
          },
          {
            "x": "2026-01-26T11:45:00.000Z",
            "prod_raw": 48.12,
            "prod_cumulative": 695.51,
            "baseline_raw": 67.17,
            "baseline_cumulative": 1084.79
          },
          {
            "x": "2026-01-26T12:00:00.000Z",
            "prod_raw": 8.95,
            "prod_cumulative": 697.75,
            "baseline_raw": 99.37,
            "baseline_cumulative": 1109.63
          },
          {
            "x": "2026-01-26T12:15:00.000Z",
            "prod_raw": 22.76,
            "prod_cumulative": 703.44,
            "baseline_raw": 65.22,
            "baseline_cumulative": 1125.94
          },
          {
            "x": "2026-01-26T12:30:00.000Z",
            "prod_raw": 16.12,
            "prod_cumulative": 707.46,
            "baseline_raw": 64.12,
            "baseline_cumulative": 1141.97
          },
          {
            "x": "2026-01-26T12:45:00.000Z",
            "prod_raw": 34.48,
            "prod_cumulative": 716.08,
            "baseline_raw": 62.95,
            "baseline_cumulative": 1157.71
          },
          {
            "x": "2026-01-26T13:00:00.000Z",
            "prod_raw": 18.84,
            "prod_cumulative": 720.8,
            "baseline_raw": 61.74,
            "baseline_cumulative": 1173.14
          },
          {
            "x": "2026-01-26T13:15:00.000Z",
            "prod_raw": 91.33,
            "prod_cumulative": 743.63,
            "baseline_raw": 60.5,
            "baseline_cumulative": 1188.27
          },
          {
            "x": "2026-01-26T13:30:00.000Z",
            "prod_raw": 30.6,
            "prod_cumulative": 751.28,
            "baseline_raw": 59.25,
            "baseline_cumulative": 1203.08
          },
          {
            "x": "2026-01-26T13:45:00.000Z",
            "prod_raw": 34.33,
            "prod_cumulative": 759.86,
            "baseline_raw": 58.01,
            "baseline_cumulative": 1217.58
          },
          {
            "x": "2026-01-26T14:00:00.000Z",
            "prod_raw": 80.57,
            "prod_cumulative": 780,
            "baseline_raw": 56.8,
            "baseline_cumulative": 1231.78
          }
        ],
        "timeRange": "1D",
        "selectedDate": "2026-01-26T14:10:23.262Z"
      },
      "cost-vs-budget": {
        "config": {
          "title": "Cost vs Budget",
          "subtitle": "MTD OpEx \u2022 Utilities",
          "variant": "V5",
          "mode": "cumulative",
          "targetLine": 8000,
          "series": [
            {
              "id": "cost",
              "label": "Accrued Cost",
              "unit": "USD",
              "color": "#dc2626"
            }
          ]
        },
        "data": [
          {
            "x": "2026-01-25T18:30:00.000Z",
            "cost_raw": 44.75,
            "cost_cumulative": 11.19
          },
          {
            "x": "2026-01-25T18:45:00.000Z",
            "cost_raw": 40.47,
            "cost_cumulative": 21.31
          },
          {
            "x": "2026-01-25T19:00:00.000Z",
            "cost_raw": 52.53,
            "cost_cumulative": 34.44
          },
          {
            "x": "2026-01-25T19:15:00.000Z",
            "cost_raw": 50.2,
            "cost_cumulative": 46.99
          },
          {
            "x": "2026-01-25T19:30:00.000Z",
            "cost_raw": 84.47,
            "cost_cumulative": 68.11
          },
          {
            "x": "2026-01-25T19:45:00.000Z",
            "cost_raw": 51.69,
            "cost_cumulative": 81.03
          },
          {
            "x": "2026-01-25T20:00:00.000Z",
            "cost_raw": 92.43,
            "cost_cumulative": 104.14
          },
          {
            "x": "2026-01-25T20:15:00.000Z",
            "cost_raw": 71.79,
            "cost_cumulative": 122.08
          },
          {
            "x": "2026-01-25T20:30:00.000Z",
            "cost_raw": 40.14,
            "cost_cumulative": 132.12
          },
          {
            "x": "2026-01-25T20:45:00.000Z",
            "cost_raw": 70.95,
            "cost_cumulative": 149.86
          },
          {
            "x": "2026-01-25T21:00:00.000Z",
            "cost_raw": 99.68,
            "cost_cumulative": 174.78
          },
          {
            "x": "2026-01-25T21:15:00.000Z",
            "cost_raw": 75.84,
            "cost_cumulative": 193.74
          },
          {
            "x": "2026-01-25T21:30:00.000Z",
            "cost_raw": 74.59,
            "cost_cumulative": 212.38
          },
          {
            "x": "2026-01-25T21:45:00.000Z",
            "cost_raw": 33.7,
            "cost_cumulative": 220.81
          },
          {
            "x": "2026-01-25T22:00:00.000Z",
            "cost_raw": 69.88,
            "cost_cumulative": 238.28
          },
          {
            "x": "2026-01-25T22:15:00.000Z",
            "cost_raw": 40.03,
            "cost_cumulative": 248.29
          },
          {
            "x": "2026-01-25T22:30:00.000Z",
            "cost_raw": 27.64,
            "cost_cumulative": 255.2
          },
          {
            "x": "2026-01-25T22:45:00.000Z",
            "cost_raw": 75.44,
            "cost_cumulative": 274.06
          },
          {
            "x": "2026-01-25T23:00:00.000Z",
            "cost_raw": 0.5,
            "cost_cumulative": 274.18
          },
          {
            "x": "2026-01-25T23:15:00.000Z",
            "cost_raw": 96.89,
            "cost_cumulative": 298.41
          },
          {
            "x": "2026-01-25T23:30:00.000Z",
            "cost_raw": 90.14,
            "cost_cumulative": 320.94
          },
          {
            "x": "2026-01-25T23:45:00.000Z",
            "cost_raw": 52.84,
            "cost_cumulative": 334.15
          },
          {
            "x": "2026-01-26T00:00:00.000Z",
            "cost_raw": 24.5,
            "cost_cumulative": 340.27
          },
          {
            "x": "2026-01-26T00:15:00.000Z",
            "cost_raw": 35.66,
            "cost_cumulative": 349.19
          },
          {
            "x": "2026-01-26T00:30:00.000Z",
            "cost_raw": 8.05,
            "cost_cumulative": 351.2
          },
          {
            "x": "2026-01-26T00:45:00.000Z",
            "cost_raw": 22.29,
            "cost_cumulative": 356.78
          },
          {
            "x": "2026-01-26T01:00:00.000Z",
            "cost_raw": 81,
            "cost_cumulative": 377.03
          },
          {
            "x": "2026-01-26T01:15:00.000Z",
            "cost_raw": 96.54,
            "cost_cumulative": 401.16
          },
          {
            "x": "2026-01-26T01:30:00.000Z",
            "cost_raw": 10.34,
            "cost_cumulative": 403.75
          },
          {
            "x": "2026-01-26T01:45:00.000Z",
            "cost_raw": 77.47,
            "cost_cumulative": 423.11
          },
          {
            "x": "2026-01-26T02:00:00.000Z",
            "cost_raw": 123.48,
            "cost_cumulative": 453.98
          },
          {
            "x": "2026-01-26T02:15:00.000Z",
            "cost_raw": 80.27,
            "cost_cumulative": 474.05
          },
          {
            "x": "2026-01-26T02:30:00.000Z",
            "cost_raw": 43.08,
            "cost_cumulative": 484.82
          },
          {
            "x": "2026-01-26T02:45:00.000Z",
            "cost_raw": 77.75,
            "cost_cumulative": 504.26
          },
          {
            "x": "2026-01-26T03:00:00.000Z",
            "cost_raw": 39.28,
            "cost_cumulative": 514.08
          },
          {
            "x": "2026-01-26T03:15:00.000Z",
            "cost_raw": 13.66,
            "cost_cumulative": 517.49
          },
          {
            "x": "2026-01-26T03:30:00.000Z",
            "cost_raw": 58.9,
            "cost_cumulative": 532.22
          },
          {
            "x": "2026-01-26T03:45:00.000Z",
            "cost_raw": 53.34,
            "cost_cumulative": 545.55
          },
          {
            "x": "2026-01-26T04:00:00.000Z",
            "cost_raw": 13.12,
            "cost_cumulative": 548.83
          },
          {
            "x": "2026-01-26T04:15:00.000Z",
            "cost_raw": 17.59,
            "cost_cumulative": 553.23
          },
          {
            "x": "2026-01-26T04:30:00.000Z",
            "cost_raw": 42.1,
            "cost_cumulative": 563.76
          },
          {
            "x": "2026-01-26T04:45:00.000Z",
            "cost_raw": 58.8,
            "cost_cumulative": 578.46
          },
          {
            "x": "2026-01-26T05:00:00.000Z",
            "cost_raw": 97.72,
            "cost_cumulative": 602.89
          },
          {
            "x": "2026-01-26T05:15:00.000Z",
            "cost_raw": 66.32,
            "cost_cumulative": 619.47
          },
          {
            "x": "2026-01-26T05:30:00.000Z",
            "cost_raw": 71.74,
            "cost_cumulative": 637.4
          },
          {
            "x": "2026-01-26T05:45:00.000Z",
            "cost_raw": 99.01,
            "cost_cumulative": 662.15
          },
          {
            "x": "2026-01-26T06:00:00.000Z",
            "cost_raw": 76.01,
            "cost_cumulative": 681.16
          },
          {
            "x": "2026-01-26T06:15:00.000Z",
            "cost_raw": 19.95,
            "cost_cumulative": 686.15
          },
          {
            "x": "2026-01-26T06:30:00.000Z",
            "cost_raw": 46.68,
            "cost_cumulative": 697.82
          },
          {
            "x": "2026-01-26T06:45:00.000Z",
            "cost_raw": 65.63,
            "cost_cumulative": 714.22
          },
          {
            "x": "2026-01-26T07:00:00.000Z",
            "cost_raw": 22.78,
            "cost_cumulative": 719.92
          },
          {
            "x": "2026-01-26T07:15:00.000Z",
            "cost_raw": 32.9,
            "cost_cumulative": 728.14
          },
          {
            "x": "2026-01-26T07:30:00.000Z",
            "cost_raw": 69.61,
            "cost_cumulative": 745.55
          },
          {
            "x": "2026-01-26T07:45:00.000Z",
            "cost_raw": 24.58,
            "cost_cumulative": 751.69
          },
          {
            "x": "2026-01-26T08:00:00.000Z",
            "cost_raw": 29.81,
            "cost_cumulative": 759.15
          },
          {
            "x": "2026-01-26T08:15:00.000Z",
            "cost_raw": 7.95,
            "cost_cumulative": 761.14
          },
          {
            "x": "2026-01-26T08:30:00.000Z",
            "cost_raw": 92.93,
            "cost_cumulative": 784.37
          },
          {
            "x": "2026-01-26T08:45:00.000Z",
            "cost_raw": 59.8,
            "cost_cumulative": 799.32
          },
          {
            "x": "2026-01-26T09:00:00.000Z",
            "cost_raw": 26.36,
            "cost_cumulative": 805.91
          },
          {
            "x": "2026-01-26T09:15:00.000Z",
            "cost_raw": 75.87,
            "cost_cumulative": 824.87
          },
          {
            "x": "2026-01-26T09:30:00.000Z",
            "cost_raw": 62.3,
            "cost_cumulative": 840.45
          },
          {
            "x": "2026-01-26T09:45:00.000Z",
            "cost_raw": 43.6,
            "cost_cumulative": 851.35
          },
          {
            "x": "2026-01-26T10:00:00.000Z",
            "cost_raw": 63.68,
            "cost_cumulative": 867.27
          },
          {
            "x": "2026-01-26T10:15:00.000Z",
            "cost_raw": 11.14,
            "cost_cumulative": 870.06
          },
          {
            "x": "2026-01-26T10:30:00.000Z",
            "cost_raw": 96.96,
            "cost_cumulative": 894.3
          },
          {
            "x": "2026-01-26T10:45:00.000Z",
            "cost_raw": 39.83,
            "cost_cumulative": 904.25
          },
          {
            "x": "2026-01-26T11:00:00.000Z",
            "cost_raw": 87.66,
            "cost_cumulative": 926.17
          },
          {
            "x": "2026-01-26T11:15:00.000Z",
            "cost_raw": 33.76,
            "cost_cumulative": 934.61
          },
          {
            "x": "2026-01-26T11:30:00.000Z",
            "cost_raw": 24.48,
            "cost_cumulative": 940.73
          },
          {
            "x": "2026-01-26T11:45:00.000Z",
            "cost_raw": 54.96,
            "cost_cumulative": 954.47
          },
          {
            "x": "2026-01-26T12:00:00.000Z",
            "cost_raw": 46.61,
            "cost_cumulative": 966.12
          },
          {
            "x": "2026-01-26T12:15:00.000Z",
            "cost_raw": 40.45,
            "cost_cumulative": 976.23
          },
          {
            "x": "2026-01-26T12:30:00.000Z",
            "cost_raw": 91.12,
            "cost_cumulative": 999.01
          },
          {
            "x": "2026-01-26T12:45:00.000Z",
            "cost_raw": 56.62,
            "cost_cumulative": 1013.17
          },
          {
            "x": "2026-01-26T13:00:00.000Z",
            "cost_raw": 55.82,
            "cost_cumulative": 1027.12
          },
          {
            "x": "2026-01-26T13:15:00.000Z",
            "cost_raw": 84.86,
            "cost_cumulative": 1048.34
          },
          {
            "x": "2026-01-26T13:30:00.000Z",
            "cost_raw": 15.53,
            "cost_cumulative": 1052.22
          },
          {
            "x": "2026-01-26T13:45:00.000Z",
            "cost_raw": 48.58,
            "cost_cumulative": 1064.37
          },
          {
            "x": "2026-01-26T14:00:00.000Z",
            "cost_raw": 64.45,
            "cost_cumulative": 1080.48
          }
        ],
        "timeRange": "1D",
        "selectedDate": "2026-01-26T14:10:23.262Z"
      },
      "batch-production": {
        "config": {
          "title": "Batch Production",
          "subtitle": "Line 4 \u2022 Shift A",
          "variant": "V6",
          "mode": "step",
          "series": [
            {
              "id": "batch",
              "label": "Completed Units",
              "unit": "pcs",
              "color": "#4f46e5"
            }
          ]
        },
        "data": [
          {
            "x": "2026-01-25T18:30:00.000Z",
            "batch_raw": 86.91,
            "batch_cumulative": 21.73
          },
          {
            "x": "2026-01-25T18:45:00.000Z",
            "batch_raw": 62.5,
            "batch_cumulative": 37.35
          },
          {
            "x": "2026-01-25T19:00:00.000Z",
            "batch_raw": 107.02,
            "batch_cumulative": 64.11
          },
          {
            "x": "2026-01-25T19:15:00.000Z",
            "batch_raw": 20.59,
            "batch_cumulative": 69.25
          },
          {
            "x": "2026-01-25T19:30:00.000Z",
            "batch_raw": 71.17,
            "batch_cumulative": 87.05
          },
          {
            "x": "2026-01-25T19:45:00.000Z",
            "batch_raw": 35.86,
            "batch_cumulative": 96.01
          },
          {
            "x": "2026-01-25T20:00:00.000Z",
            "batch_raw": 66.97,
            "batch_cumulative": 112.76
          },
          {
            "x": "2026-01-25T20:15:00.000Z",
            "batch_raw": 90.24,
            "batch_cumulative": 135.32
          },
          {
            "x": "2026-01-25T20:30:00.000Z",
            "batch_raw": 52.52,
            "batch_cumulative": 148.45
          },
          {
            "x": "2026-01-25T20:45:00.000Z",
            "batch_raw": 85.31,
            "batch_cumulative": 169.77
          },
          {
            "x": "2026-01-25T21:00:00.000Z",
            "batch_raw": 57.23,
            "batch_cumulative": 184.08
          },
          {
            "x": "2026-01-25T21:15:00.000Z",
            "batch_raw": 50.63,
            "batch_cumulative": 196.74
          },
          {
            "x": "2026-01-25T21:30:00.000Z",
            "batch_raw": 13.44,
            "batch_cumulative": 200.1
          },
          {
            "x": "2026-01-25T21:45:00.000Z",
            "batch_raw": 87.18,
            "batch_cumulative": 221.89
          },
          {
            "x": "2026-01-25T22:00:00.000Z",
            "batch_raw": 38.28,
            "batch_cumulative": 231.46
          },
          {
            "x": "2026-01-25T22:15:00.000Z",
            "batch_raw": 44.25,
            "batch_cumulative": 242.52
          },
          {
            "x": "2026-01-25T22:30:00.000Z",
            "batch_raw": 64.89,
            "batch_cumulative": 258.75
          },
          {
            "x": "2026-01-25T22:45:00.000Z",
            "batch_raw": 95.35,
            "batch_cumulative": 282.58
          },
          {
            "x": "2026-01-25T23:00:00.000Z",
            "batch_raw": 88.2,
            "batch_cumulative": 304.63
          },
          {
            "x": "2026-01-25T23:15:00.000Z",
            "batch_raw": 91.2,
            "batch_cumulative": 327.43
          },
          {
            "x": "2026-01-25T23:30:00.000Z",
            "batch_raw": 149.41,
            "batch_cumulative": 364.79
          },
          {
            "x": "2026-01-25T23:45:00.000Z",
            "batch_raw": 25.32,
            "batch_cumulative": 371.12
          },
          {
            "x": "2026-01-26T00:00:00.000Z",
            "batch_raw": 1.1,
            "batch_cumulative": 371.39
          },
          {
            "x": "2026-01-26T00:15:00.000Z",
            "batch_raw": 36.35,
            "batch_cumulative": 380.48
          },
          {
            "x": "2026-01-26T00:30:00.000Z",
            "batch_raw": 31.91,
            "batch_cumulative": 388.46
          },
          {
            "x": "2026-01-26T00:45:00.000Z",
            "batch_raw": 69.66,
            "batch_cumulative": 405.87
          },
          {
            "x": "2026-01-26T01:00:00.000Z",
            "batch_raw": 44.56,
            "batch_cumulative": 417.01
          },
          {
            "x": "2026-01-26T01:15:00.000Z",
            "batch_raw": 47.11,
            "batch_cumulative": 428.79
          },
          {
            "x": "2026-01-26T01:30:00.000Z",
            "batch_raw": 45.7,
            "batch_cumulative": 440.21
          },
          {
            "x": "2026-01-26T01:45:00.000Z",
            "batch_raw": 2.97,
            "batch_cumulative": 440.96
          },
          {
            "x": "2026-01-26T02:00:00.000Z",
            "batch_raw": 78.21,
            "batch_cumulative": 460.51
          },
          {
            "x": "2026-01-26T02:15:00.000Z",
            "batch_raw": 81.92,
            "batch_cumulative": 480.99
          },
          {
            "x": "2026-01-26T02:30:00.000Z",
            "batch_raw": 74.45,
            "batch_cumulative": 499.6
          },
          {
            "x": "2026-01-26T02:45:00.000Z",
            "batch_raw": 7.92,
            "batch_cumulative": 501.58
          },
          {
            "x": "2026-01-26T03:00:00.000Z",
            "batch_raw": 43.66,
            "batch_cumulative": 512.5
          },
          {
            "x": "2026-01-26T03:15:00.000Z",
            "batch_raw": 8.01,
            "batch_cumulative": 514.5
          },
          {
            "x": "2026-01-26T03:30:00.000Z",
            "batch_raw": 7.07,
            "batch_cumulative": 516.27
          },
          {
            "x": "2026-01-26T03:45:00.000Z",
            "batch_raw": 18.06,
            "batch_cumulative": 520.78
          },
          {
            "x": "2026-01-26T04:00:00.000Z",
            "batch_raw": 66.21,
            "batch_cumulative": 537.33
          },
          {
            "x": "2026-01-26T04:15:00.000Z",
            "batch_raw": 41.56,
            "batch_cumulative": 547.72
          },
          {
            "x": "2026-01-26T04:30:00.000Z",
            "batch_raw": 55.33,
            "batch_cumulative": 561.55
          },
          {
            "x": "2026-01-26T04:45:00.000Z",
            "batch_raw": 16.93,
            "batch_cumulative": 565.79
          },
          {
            "x": "2026-01-26T05:00:00.000Z",
            "batch_raw": 47.66,
            "batch_cumulative": 577.7
          },
          {
            "x": "2026-01-26T05:15:00.000Z",
            "batch_raw": 53.69,
            "batch_cumulative": 591.12
          },
          {
            "x": "2026-01-26T05:30:00.000Z",
            "batch_raw": 52.6,
            "batch_cumulative": 604.27
          },
          {
            "x": "2026-01-26T05:45:00.000Z",
            "batch_raw": 41.62,
            "batch_cumulative": 614.68
          },
          {
            "x": "2026-01-26T06:00:00.000Z",
            "batch_raw": 18.88,
            "batch_cumulative": 619.4
          },
          {
            "x": "2026-01-26T06:15:00.000Z",
            "batch_raw": 28.34,
            "batch_cumulative": 626.48
          },
          {
            "x": "2026-01-26T06:30:00.000Z",
            "batch_raw": 47.99,
            "batch_cumulative": 638.48
          },
          {
            "x": "2026-01-26T06:45:00.000Z",
            "batch_raw": 82.28,
            "batch_cumulative": 659.05
          },
          {
            "x": "2026-01-26T07:00:00.000Z",
            "batch_raw": 28.15,
            "batch_cumulative": 666.09
          },
          {
            "x": "2026-01-26T07:15:00.000Z",
            "batch_raw": 97.39,
            "batch_cumulative": 690.43
          },
          {
            "x": "2026-01-26T07:30:00.000Z",
            "batch_raw": 69.56,
            "batch_cumulative": 707.82
          },
          {
            "x": "2026-01-26T07:45:00.000Z",
            "batch_raw": 69.08,
            "batch_cumulative": 725.09
          },
          {
            "x": "2026-01-26T08:00:00.000Z",
            "batch_raw": 26.59,
            "batch_cumulative": 731.74
          },
          {
            "x": "2026-01-26T08:15:00.000Z",
            "batch_raw": 51.34,
            "batch_cumulative": 744.58
          },
          {
            "x": "2026-01-26T08:30:00.000Z",
            "batch_raw": 93.24,
            "batch_cumulative": 767.89
          },
          {
            "x": "2026-01-26T08:45:00.000Z",
            "batch_raw": 50.06,
            "batch_cumulative": 780.4
          },
          {
            "x": "2026-01-26T09:00:00.000Z",
            "batch_raw": 20.62,
            "batch_cumulative": 785.56
          },
          {
            "x": "2026-01-26T09:15:00.000Z",
            "batch_raw": 26.87,
            "batch_cumulative": 792.28
          },
          {
            "x": "2026-01-26T09:30:00.000Z",
            "batch_raw": 12.71,
            "batch_cumulative": 795.46
          },
          {
            "x": "2026-01-26T09:45:00.000Z",
            "batch_raw": 46.08,
            "batch_cumulative": 806.97
          },
          {
            "x": "2026-01-26T10:00:00.000Z",
            "batch_raw": 0.12,
            "batch_cumulative": 807
          },
          {
            "x": "2026-01-26T10:15:00.000Z",
            "batch_raw": 57.01,
            "batch_cumulative": 821.26
          },
          {
            "x": "2026-01-26T10:30:00.000Z",
            "batch_raw": 70.5,
            "batch_cumulative": 838.88
          },
          {
            "x": "2026-01-26T10:45:00.000Z",
            "batch_raw": 73.54,
            "batch_cumulative": 857.27
          },
          {
            "x": "2026-01-26T11:00:00.000Z",
            "batch_raw": 39.05,
            "batch_cumulative": 867.03
          },
          {
            "x": "2026-01-26T11:15:00.000Z",
            "batch_raw": 98.44,
            "batch_cumulative": 891.64
          },
          {
            "x": "2026-01-26T11:30:00.000Z",
            "batch_raw": 33.42,
            "batch_cumulative": 899.99
          },
          {
            "x": "2026-01-26T11:45:00.000Z",
            "batch_raw": 36.27,
            "batch_cumulative": 909.06
          },
          {
            "x": "2026-01-26T12:00:00.000Z",
            "batch_raw": 76.83,
            "batch_cumulative": 928.27
          },
          {
            "x": "2026-01-26T12:15:00.000Z",
            "batch_raw": 69.08,
            "batch_cumulative": 945.54
          },
          {
            "x": "2026-01-26T12:30:00.000Z",
            "batch_raw": 76.28,
            "batch_cumulative": 964.61
          },
          {
            "x": "2026-01-26T12:45:00.000Z",
            "batch_raw": 95.66,
            "batch_cumulative": 988.52
          },
          {
            "x": "2026-01-26T13:00:00.000Z",
            "batch_raw": 99.59,
            "batch_cumulative": 1013.42
          },
          {
            "x": "2026-01-26T13:15:00.000Z",
            "batch_raw": 48.12,
            "batch_cumulative": 1025.45
          },
          {
            "x": "2026-01-26T13:30:00.000Z",
            "batch_raw": 60.17,
            "batch_cumulative": 1040.49
          },
          {
            "x": "2026-01-26T13:45:00.000Z",
            "batch_raw": 67.35,
            "batch_cumulative": 1057.33
          },
          {
            "x": "2026-01-26T14:00:00.000Z",
            "batch_raw": 67.72,
            "batch_cumulative": 1074.26
          }
        ],
        "timeRange": "1D",
        "selectedDate": "2026-01-26T14:10:23.262Z"
      }
    }
  }
};
