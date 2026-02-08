# Interactive Widget Mode — Implementation Plan

## Context
When a widget is selected, it goes **full-screen** and all subsequent conversation becomes contextual to that widget's equipment/metric. Follow-ups bring full multi-widget dashboards scoped to that equipment. Conversation history is retained so the AI can reference prior Q&A.

## User Flow
1. User asks "pump 4 vibration" → dashboard shows KPI + trend + edge panel
2. User clicks **Focus** on the KPI widget → **enters interactive mode**
3. Widget goes full-screen with context bar: `"Pump 4 — Vibration"` + **Back** button
4. Auto-query fires: "Tell me more about Pump 4 Vibration"
5. Backend returns full multi-widget dashboard scoped to pump_004
6. User speaks: "show me maintenance history" → backend resolves to pump_004 → returns new widgets
7. Each follow-up brings fresh multi-widget dashboards, all context-locked to pump_004
8. Conversation history retained — AI can reference prior answers
9. User presses **Escape** or **Back** → exits interactive mode, returns to original dashboard

## Implementation (7 files)

### Frontend

#### 1. `frontend/src/types/index.ts` — Event Types
```ts
| { type: "WIDGET_INTERACTIVE_ENTER"; scenario: string; label: string; equipment: string; metric: string }
| { type: "WIDGET_INTERACTIVE_EXIT" }
```

#### 2. `frontend/src/components/layer3/useLayoutState.ts` — Interactive State
- `interactiveCtx` state: `{ key, scenario, label, equipment, metric } | null`
- `enterInteractive(key, instruction)` — extract equipment/metric from data_override, save layout, emit event
- `exitInteractive()` — restore layout, clear context, emit event
- Escape key exits interactive mode
- `LAYOUT_UPDATE` during interactive: accept new widgets, keep context active

#### 3. `frontend/src/components/layer3/Blob.tsx` — Full-Screen Rendering
- When interactive: context bar on top + BlobGrid below
- Context bar: Back button + label + equipment badge
- Follow-up responses render full multi-widget dashboards in normal grid
- When not interactive: existing BlobGrid (no changes)

#### 4. `frontend/src/components/layer1/useVoicePipeline.ts` — Context + History
- `interactiveHistoryRef` accumulates Q&A pairs
- On enter: reset history, update Layer2Service context with widget_context
- On each exchange: push user/ai messages to history, update context
- On exit: clear history and widget_context
- Last 10 exchanges sent to backend

### Backend

#### 5. `backend/layer2/orchestrator.py` — Pass widget_context
- Extract `widget_context` from session context
- Pass to intent_parser.parse() and widget_selector
- Pass to voice response generator

#### 6. `backend/layer2/intent_parser.py` — Entity Merge
- Accept `widget_context` parameter
- If query has no devices, inject equipment from context
- Add context hint to LLM parse prompt

#### 7. `backend/layer2/widget_selector.py` — Context Prompt
- Append INTERACTIVE CONTEXT + CONVERSATION HISTORY to prompt
- Instruct LLM to build full multi-widget dashboards scoped to equipment
- Format conversation_history as Q&A pairs, truncate to ~2000 tokens

## Verification
1. Ask "pump 4 vibration" → dashboard renders
2. Click Focus → interactive mode with context bar
3. Say "show me maintenance history" → resolves to pump_004 automatically
4. Say "you mentioned vibration earlier" → AI references conversation history
5. Press Escape → returns to dashboard, context cleared
