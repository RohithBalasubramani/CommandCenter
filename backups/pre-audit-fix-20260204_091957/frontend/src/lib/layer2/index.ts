/**
 * Layer 2 — AI + RAG Module
 *
 * Exports the Layer 2 client for integration with Layer 1.
 */

export {
  orchestrate,
  getFiller,
  checkProactiveTrigger,
  Layer2Service,
  getLayer2Service,
} from "./client";

export type {
  Layer2Intent,
  Layer2RAGResult,
  Layer2WidgetCommand,
  Layer2LayoutJSON,
  Layer2Response,
  ProactiveTrigger,
} from "./client";
