import type { LayoutJSON } from "@/types";

/**
 * Default dashboard layout for the "project engineer" persona.
 * Rendered on initial load before any voice commands.
 * 4 compact KPI tiles + an expanded trend chart + an expanded alerts panel.
 */
export const DEFAULT_LAYOUT: LayoutJSON = {
  heading: "Operations Dashboard",
  widgets: [
    {
      scenario: "kpi",
      fixture: "kpi_live-standard",
      size: "compact",
      position: null,
      relevance: 0.8,
      data_override: {
        demoData: { label: "Grid Voltage", value: "238.4", unit: "V", state: "normal" },
      },
    },
    {
      scenario: "kpi",
      fixture: "kpi_live-standard",
      size: "compact",
      position: null,
      relevance: 0.8,
      data_override: {
        demoData: { label: "Total Power", value: "1247", unit: "kW", state: "normal" },
      },
    },
    {
      scenario: "kpi",
      fixture: "kpi_live-standard",
      size: "compact",
      position: null,
      relevance: 0.7,
      data_override: {
        demoData: { label: "Equipment Online", value: "12", unit: "", state: "normal" },
      },
    },
    {
      scenario: "kpi",
      fixture: "kpi_alert-warning-state",
      size: "compact",
      position: null,
      relevance: 0.9,
      data_override: {
        demoData: { label: "Active Alerts", value: "3", unit: "", state: "warning" },
      },
    },
    {
      scenario: "trend",
      fixture: "trend_live-line",
      size: "expanded",
      position: null,
      relevance: 0.6,
      data_override: null,
    },
    {
      scenario: "alerts",
      fixture: "banner-energy-peak-threshold-exceeded",
      size: "expanded",
      position: null,
      relevance: 0.7,
      data_override: null,
    },
  ],
  transitions: {},
};
