import { NextResponse } from "next/server";

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8100";

/**
 * Health check proxy — avoids browser HTTP/1.1 connection pool issues.
 * The browser's fetch to the backend can get queued behind slow orchestrate
 * requests sharing the same origin connection pool. This server-side proxy
 * uses its own connection, bypassing that limitation.
 */
export async function GET() {
  try {
    const res = await fetch(
      `${BACKEND_URL}/api/layer2/rag/industrial/health/`,
      { signal: AbortSignal.timeout(5000) }
    );
    if (res.ok) {
      const data = await res.json();
      return NextResponse.json(data);
    }
    return NextResponse.json(
      { error: `Backend returned ${res.status}` },
      { status: res.status }
    );
  } catch {
    return NextResponse.json(
      { error: "Backend unreachable" },
      { status: 503 }
    );
  }
}
