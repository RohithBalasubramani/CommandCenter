// ============================================================
// Command Center — Event Bus
// Inter-layer communication: Layer 1 → Layer 2 → Layer 3 → Layer 4
// ============================================================

import { CommandCenterEvent } from "@/types";

const LOG_PREFIX = "[EventBus]";

type EventHandler = (event: CommandCenterEvent) => void;

class EventBus {
  private handlers: Map<string, Set<EventHandler>> = new Map();
  private allHandlers: Set<EventHandler> = new Set();
  private eventCount = 0;

  /** Subscribe to a specific event type */
  on(eventType: CommandCenterEvent["type"], handler: EventHandler): () => void {
    if (!this.handlers.has(eventType)) {
      this.handlers.set(eventType, new Set());
    }
    this.handlers.get(eventType)!.add(handler);

    const listenerCount = this.handlers.get(eventType)!.size;
    console.debug(LOG_PREFIX, `Subscribed to "${eventType}" (${listenerCount} listener(s))`);

    // Return unsubscribe function
    return () => {
      this.handlers.get(eventType)?.delete(handler);
      console.debug(LOG_PREFIX, `Unsubscribed from "${eventType}"`);
    };
  }

  /** Subscribe to all events */
  onAny(handler: EventHandler): () => void {
    this.allHandlers.add(handler);
    console.debug(LOG_PREFIX, `Subscribed to ALL events (${this.allHandlers.size} catch-all listener(s))`);
    return () => {
      this.allHandlers.delete(handler);
      console.debug(LOG_PREFIX, `Unsubscribed from ALL events`);
    };
  }

  /** Emit an event to all subscribers */
  emit(event: CommandCenterEvent): void {
    this.eventCount++;

    const typeHandlers = this.handlers.get(event.type);
    const typeCount = typeHandlers?.size ?? 0;
    const allCount = this.allHandlers.size;

    console.debug(
      LOG_PREFIX,
      `#${this.eventCount} EMIT "${event.type}" → ${typeCount} type + ${allCount} catch-all = ${typeCount + allCount} handler(s)`,
      event
    );

    // Notify type-specific handlers
    if (typeHandlers) {
      typeHandlers.forEach((handler) => {
        try {
          handler(event);
        } catch (e) {
          console.error(LOG_PREFIX, `Handler error for "${event.type}":`, e);
        }
      });
    }

    // Notify catch-all handlers
    this.allHandlers.forEach((handler) => {
      try {
        handler(event);
      } catch (e) {
        console.error(LOG_PREFIX, `Catch-all handler error for "${event.type}":`, e);
      }
    });
  }
}

// Singleton event bus for the entire Command Center.
// Use globalThis to guarantee a single instance across all Next.js chunks —
// prevents module-level singletons from being duplicated in different bundles.
const GLOBAL_KEY = "__commandCenterBus__";
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const commandCenterBus: EventBus =
  (globalThis as any)[GLOBAL_KEY] ??
  ((globalThis as any)[GLOBAL_KEY] = new EventBus());
