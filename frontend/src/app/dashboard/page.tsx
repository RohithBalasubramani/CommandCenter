"use client";

import Blob from "@/components/layer3/Blob";

/**
 * /dashboard — Test route for Layer 3 (Blob) + Layer 4 (Widgets).
 * Renders the default dashboard layout without the voice UI.
 */
export default function DashboardTestPage() {
  return (
    <div className="h-screen w-screen overflow-auto bg-[var(--cc-bg)]">
      <Blob />
    </div>
  );
}
