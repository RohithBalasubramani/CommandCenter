/**
 * PersonaPlex/Moshi Binary WebSocket Protocol
 *
 * All messages are binary with a single-byte type prefix:
 *   0x00 = Handshake
 *   0x01 = Audio (Opus-encoded, bidirectional)
 *   0x02 = Text  (UTF-8, incremental from server)
 *   0x03 = Control
 *   0x04 = Metadata
 *   0x05 = Error
 *   0x06 = Ping
 */

// --- Types ---

export type WSMessage =
  | { type: "handshake"; version: number; model: number }
  | { type: "audio"; data: Uint8Array }
  | { type: "text"; data: string }
  | { type: "control"; action: ControlAction }
  | { type: "queueDrained" }
  | { type: "metadata"; data: unknown }
  | { type: "error"; data: string }
  | { type: "ping" };

export type ControlAction = "start" | "endTurn" | "pause" | "restart";

export type SocketStatus = "connected" | "disconnected" | "connecting";

// --- Maps ---

const CONTROL_MAP: Record<ControlAction, number> = {
  start: 0x00,
  endTurn: 0x01,
  pause: 0x02,
  restart: 0x03,
};

const CONTROL_REVERSE = Object.fromEntries(
  Object.entries(CONTROL_MAP).map(([k, v]) => [v, k]),
) as Record<number, ControlAction>;

// --- Helpers ---

function concat(prefix: number, payload: Uint8Array): Uint8Array {
  const out = new Uint8Array(1 + payload.length);
  out[0] = prefix;
  out.set(payload, 1);
  return out;
}

// --- Encode ---

export function encodeMessage(msg: WSMessage): Uint8Array {
  switch (msg.type) {
    case "handshake":
      return new Uint8Array([0x00, msg.version, msg.model]);
    case "audio":
      return concat(0x01, msg.data);
    case "text":
      return concat(0x02, new TextEncoder().encode(msg.data));
    case "control":
      return new Uint8Array([0x03, CONTROL_MAP[msg.action]]);
    case "metadata":
      return concat(0x04, new TextEncoder().encode(JSON.stringify(msg.data)));
    case "error":
      return concat(0x05, new TextEncoder().encode(msg.data));
    case "ping":
      return new Uint8Array([0x06]);
    case "queueDrained":
      return new Uint8Array([0x04]);
  }
}

// --- Decode ---

export function decodeMessage(data: Uint8Array): WSMessage {
  const type = data[0];
  const payload = data.slice(1);

  switch (type) {
    case 0x00:
      return { type: "handshake", version: payload[0] ?? 0, model: payload[1] ?? 0 };
    case 0x01:
      return { type: "audio", data: payload };
    case 0x02:
      return { type: "text", data: new TextDecoder().decode(payload) };
    case 0x03: {
      const action = CONTROL_REVERSE[payload[0]];
      if (!action) throw new Error(`Unknown control byte: ${payload[0]}`);
      return { type: "control", action };
    }
    case 0x04:
      // Empty payload from server = queue drained notification
      if (payload.length === 0) {
        return { type: "queueDrained" };
      }
      return { type: "metadata", data: JSON.parse(new TextDecoder().decode(payload)) };
    case 0x05:
      return { type: "error", data: new TextDecoder().decode(payload) };
    case 0x06:
      return { type: "ping" };
    default:
      throw new Error(`Unknown message type: 0x${type.toString(16)}`);
  }
}
