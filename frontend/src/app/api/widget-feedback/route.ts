import { NextRequest, NextResponse } from "next/server";
import { writeFile, mkdir } from "fs/promises";
import { join } from "path";

const FEEDBACK_DIR = join(process.cwd(), "..", "ref");
const FEEDBACK_FILE = join(FEEDBACK_DIR, "widget-feedback.json");

export async function POST(req: NextRequest) {
  try {
    const body = await req.text();
    await mkdir(FEEDBACK_DIR, { recursive: true });
    await writeFile(FEEDBACK_FILE, body, "utf-8");
    return NextResponse.json({ ok: true, path: FEEDBACK_FILE });
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : "unknown error";
    return NextResponse.json({ ok: false, error: msg }, { status: 500 });
  }
}

export async function GET() {
  try {
    const { readFile } = await import("fs/promises");
    const data = await readFile(FEEDBACK_FILE, "utf-8");
    return new NextResponse(data, {
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return NextResponse.json({ widgets: [], dashboards: [] });
  }
}
