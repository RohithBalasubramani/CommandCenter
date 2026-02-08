"use client";

import { ReactNode } from "react";
import type { ConfidenceEnvelope } from "@/types";

interface BlobGridProps {
  children: ReactNode;
  heading?: string | null;
  confidence?: ConfidenceEnvelope;
}

/** Color for confidence level. */
function confidenceColor(score: number): string {
  if (score >= 0.75) return "bg-emerald-500";
  if (score >= 0.55) return "bg-yellow-500";
  if (score >= 0.35) return "bg-orange-500";
  return "bg-red-500";
}

/**
 * BlobGrid — 12-column CSS Grid container for widget layout.
 *
 * Outer container fills available viewport height; inner grid area
 * scrolls vertically when content exceeds the viewport (dense dashboards).
 * WidgetSlot components use col-span classes for width and row-span
 * classes for height within the auto-row grid.
 */
export default function BlobGrid({ children, heading, confidence }: BlobGridProps) {
  return (
    <div className="h-full w-full flex flex-col overflow-hidden">
      {/* Heading + confidence indicator */}
      {(heading || confidence) && (
        <div className="shrink-0 px-3 sm:px-5 pt-3 sm:pt-4 pb-1 sm:pb-2">
          <div className="flex items-center gap-3">
            {heading && (
              <h1 className="text-base sm:text-lg font-semibold text-neutral-100 tracking-tight flex-1 min-w-0">
                {heading}
              </h1>
            )}
            {confidence && confidence.action !== "full_dashboard" && (
              <div className="shrink-0 flex items-center gap-2" data-testid="confidence-indicator">
                <div className="w-16 h-1.5 rounded-full bg-neutral-700 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${confidenceColor(confidence.overall)}`}
                    style={{ width: `${Math.round(confidence.overall * 100)}%` }}
                  />
                </div>
                <span className="text-[10px] font-medium text-neutral-400">
                  {Math.round(confidence.overall * 100)}%
                </span>
              </div>
            )}
          </div>
          {confidence && confidence.caveats.length > 0 && confidence.action !== "full_dashboard" && (
            <p className="mt-1 text-[11px] text-neutral-500 line-clamp-1" data-testid="confidence-caveats">
              {confidence.caveats.slice(0, 2).join(" · ")}
            </p>
          )}
        </div>
      )}

      {/* Grid — fills remaining space, allow scroll for dense content */}
      <div className="flex-1 min-h-0 p-2 sm:p-3 xl:p-4 overflow-y-auto">
        <div
          className="grid grid-cols-1 md:grid-cols-6 lg:grid-cols-12 gap-2 md:gap-3 xl:gap-4 min-h-full"
          style={{ gridAutoRows: 'minmax(min(100px, 12vh), 1fr)', gridAutoFlow: 'dense' }}
        >
          {children}
        </div>
      </div>
    </div>
  );
}
