// Widget Registry — maps scenario slugs to lazy-loaded React components
// All components extracted from Widgets/scenarios.sqlite3

import { ComponentType, lazy } from "react";

export interface WidgetComponentProps {
  data: Record<string, unknown>;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyComponent = ComponentType<any>;

// Lazy-load each widget component to keep initial bundle small.
// Widgets have varied prop signatures but all accept { data } at minimum.
const WIDGET_REGISTRY: Record<string, AnyComponent> = {
  "kpi": lazy(() => import("./widgets/kpi")),
  "alerts": lazy(() => import("./widgets/alerts")),
  "trend": lazy(() => import("./widgets/trend")),
  "trend-multi-line": lazy(() => import("./widgets/trend-multi-line")),
  "trends-cumulative": lazy(() => import("./widgets/trends-cumulative")),
  "distribution": lazy(() => import("./widgets/distribution")),
  "comparison": lazy(() => import("./widgets/comparison")),
  "composition": lazy(() => import("./widgets/composition")),
  "flow-sankey": lazy(() => import("./widgets/flow-sankey")),
  "matrix-heatmap": lazy(() => import("./widgets/matrix-heatmap")),
  "timeline": lazy(() => import("./widgets/timeline")),
  "eventlogstream": lazy(() => import("./widgets/eventlogstream")),
  "category-bar": lazy(() => import("./widgets/category-bar")),
  "edgedevicepanel": lazy(() => import("./widgets/edgedevicepanel")),
  "chatstream": lazy(() => import("./widgets/chatstream")),
  "peoplehexgrid": lazy(() => import("./widgets/peoplehexgrid")),
  "peoplenetwork": lazy(() => import("./widgets/peoplenetwork")),
  "peopleview": lazy(() => import("./widgets/peopleview")),
  "supplychainglobe": lazy(() => import("./widgets/supplychainglobe")),
};

export function getWidgetComponent(
  scenario: string
): AnyComponent | null {
  return WIDGET_REGISTRY[scenario] || null;
}

export function getAvailableScenarios(): string[] {
  return Object.keys(WIDGET_REGISTRY);
}

export default WIDGET_REGISTRY;
