"use client";

import { ReactNode } from "react";

interface BlobGridProps {
  children: ReactNode;
  heading?: string | null;
}

/**
 * BlobGrid — 12-column CSS Grid container for widget layout.
 *
 * Outer container fills available viewport height; inner grid area
 * scrolls vertically when content exceeds the viewport (dense dashboards).
 * WidgetSlot components use col-span classes for width and row-span
 * classes for height within the auto-row grid.
 */
export default function BlobGrid({ children, heading }: BlobGridProps) {
  return (
    <div className="h-full w-full flex flex-col overflow-hidden">
      {/* Heading — gist of what user asked */}
      {heading && (
        <div className="shrink-0 px-5 pt-4 pb-2">
          <h1 className="text-lg font-semibold text-neutral-100 tracking-tight">
            {heading}
          </h1>
        </div>
      )}

      {/* Grid — fills remaining space, allow scroll for dense content */}
      <div className="flex-1 min-h-0 p-4 overflow-y-auto">
        <div className="grid grid-cols-12 gap-2 min-h-full" style={{ gridAutoRows: 'minmax(100px, auto)', gridAutoFlow: 'dense', alignContent: 'start' }}>
          {children}
        </div>
      </div>
    </div>
  );
}
