"use client";

import React, { Suspense, Component, ReactNode } from "react";
import type { WidgetSize } from "@/types";
import WidgetToolbar from "./WidgetToolbar";
import { WidgetLifecycleProvider } from "./WidgetLifecycleContext";

// --- Error Boundary ---

interface ErrorBoundaryProps {
  scenario: string;
  children: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class WidgetErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="h-full flex items-center justify-center bg-red-950/30 border border-red-900/50 rounded-xl p-4">
          <div className="text-center">
            <p className="text-[10px] uppercase font-bold tracking-widest text-red-400 mb-1">
              Widget Error
            </p>
            <p className="text-xs text-red-400/70 font-mono">
              {this.props.scenario}: {this.state.error?.message}
            </p>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

// --- Loading Skeleton ---

function WidgetSkeleton() {
  return (
    <div className="h-full w-full animate-pulse rounded-xl bg-neutral-800 border border-neutral-700">
      <div className="p-4 space-y-3">
        <div className="h-2 w-20 rounded bg-neutral-700" />
        <div className="h-4 w-32 rounded bg-neutral-700" />
        <div className="h-2 w-full rounded bg-neutral-700" />
      </div>
    </div>
  );
}

// --- Size → CSS class mapping (responsive, matches Blob.tsx sizeClasses) ---

const SIZE_CLASSES: Record<WidgetSize, string> = {
  hero: "col-span-1 md:col-span-6 lg:col-span-12 row-span-4",
  expanded: "col-span-1 md:col-span-6 lg:col-span-6 row-span-2",
  normal: "col-span-1 md:col-span-3 lg:col-span-4 row-span-2",
  compact: "col-span-1 md:col-span-3 lg:col-span-3 row-span-2",
  hidden: "hidden",
};

// --- WidgetSlot ---

interface WidgetSlotProps {
  scenario: string;
  size: WidgetSize;
  children: ReactNode;
  /** When true, skip the outer grid-class div (parent handles grid placement). */
  noGrid?: boolean;
  /** Display title shown on hover (top-left). */
  title?: string;
  /** User-facing description shown at the bottom of the widget card. */
  description?: string;
  /** Widget toolbar props — passed when rendered from Blob with interactions. */
  widgetKey?: string;
  isPinned?: boolean;
  isFocused?: boolean;
  onPin?: () => void;
  onDismiss?: () => void;
  onResize?: (size: WidgetSize) => void;
  onFocus?: () => void;
  onUnfocus?: () => void;
  onSnapshot?: () => void;
  /** Drill-down callback — clicking widget body sends contextual query to AI. */
  onDrillDown?: (context: string) => void;
  /** When true, shows a visual "Demo Data" indicator on the widget. */
  isDemo?: boolean;
  /** Phase 2: Data is stale (older than freshness threshold). */
  isStale?: boolean;
  /** Phase 1: Contradiction message (e.g. KPI vs alert disagree). */
  conflictFlag?: string;
  /** Phase 2: Per-widget confidence (0-1). */
  widgetConfidence?: number;
}

export default function WidgetSlot({
  scenario,
  size,
  children,
  noGrid,
  title,
  description,
  widgetKey: wKey,
  isPinned = false,
  isFocused = false,
  onPin,
  onDismiss,
  onResize,
  onFocus,
  onUnfocus,
  onSnapshot,
  onDrillDown,
  isDemo = false,
  isStale = false,
  conflictFlag,
  widgetConfidence,
}: WidgetSlotProps) {
  if (size === "hidden") return null;

  const hasToolbar = !!(wKey && onPin && onDismiss && onResize && onFocus && onUnfocus && onSnapshot);

  const handleBodyClick = (e: React.MouseEvent) => {
    if (!onDrillDown || e.defaultPrevented) return;
    onDrillDown(title || scenario.replace(/-/g, " "));
  };

  const inner = (
    <div className="relative h-full w-full group rounded-xl border border-neutral-700/50 bg-neutral-900/80 backdrop-blur-sm shadow-lg overflow-hidden hover:border-neutral-600/50 transition-colors duration-200 flex flex-col">
      {/* Status badges — top-right stack */}
      <div className="absolute top-2 right-2 z-20 flex flex-col items-end gap-1 pointer-events-none">
        {isDemo && (
          <div className="rounded bg-amber-600/90 px-1.5 py-0.5" data-testid="demo-badge">
            <span className="text-[9px] font-bold uppercase tracking-wider text-white">Demo Data</span>
          </div>
        )}
        {isStale && (
          <div className="rounded bg-orange-600/80 px-1.5 py-0.5" data-testid="stale-badge">
            <span className="text-[9px] font-bold uppercase tracking-wider text-white">Stale</span>
          </div>
        )}
        {conflictFlag && (
          <div className="rounded bg-red-600/80 px-1.5 py-0.5 max-w-[140px]" data-testid="conflict-badge" title={conflictFlag}>
            <span className="text-[9px] font-bold uppercase tracking-wider text-white truncate block">Conflict</span>
          </div>
        )}
        {widgetConfidence !== undefined && widgetConfidence < 0.5 && (
          <div className="rounded bg-yellow-600/70 px-1.5 py-0.5" data-testid="low-confidence-badge">
            <span className="text-[9px] font-bold uppercase tracking-wider text-white">
              {Math.round(widgetConfidence * 100)}% conf
            </span>
          </div>
        )}
      </div>
      {/* Title label — top-left, appears on hover */}
      {title && (
        <div className="absolute top-2 left-2 z-10 rounded-lg bg-neutral-800/80 backdrop-blur-sm border border-neutral-700/50 px-2 py-0.5 opacity-0 group-hover:opacity-100 transition-opacity duration-150 pointer-events-none max-w-[60%] truncate">
          <span className="text-[11px] font-medium text-neutral-300">{title}</span>
        </div>
      )}
      {hasToolbar && (
        <WidgetToolbar
          scenario={scenario}
          widgetKey={wKey!}
          isPinned={isPinned}
          isFocused={isFocused}
          size={size}
          onPin={onPin!}
          onDismiss={onDismiss!}
          onResize={onResize!}
          onFocus={onFocus!}
          onUnfocus={onUnfocus!}
          onSnapshot={onSnapshot!}
        />
      )}
      <div
        className={`flex-1 min-h-0 overflow-y-auto overflow-x-hidden cc-scrollbar${onDrillDown ? " cursor-pointer" : ""}`}
        onClick={handleBodyClick}
      >
        <WidgetErrorBoundary scenario={scenario}>
          <Suspense fallback={<WidgetSkeleton />}>
            <WidgetLifecycleProvider>
              {children}
            </WidgetLifecycleProvider>
          </Suspense>
        </WidgetErrorBoundary>
      </div>
      {description && (
        <div className="shrink-0 px-3 py-1.5 border-t border-neutral-700/30 bg-neutral-900/50">
          <p className="text-[11px] leading-snug text-neutral-400 line-clamp-2">{description}</p>
        </div>
      )}
    </div>
  );

  if (noGrid) {
    return inner;
  }

  return (
    <div
      className={`${SIZE_CLASSES[size]} min-h-0`}
      data-scenario={scenario}
      data-size={size}
    >
      {inner}
    </div>
  );
}
