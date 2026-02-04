# Command Center — What We Built So Far

## What Is This

Command Center is a voice-driven industrial operations dashboard. You speak to it, an AI figures out what you need, retrieves real data from equipment databases, and renders the right widgets on screen — KPI tiles, trend charts, alert panels, device tables, Sankey flows, heatmaps, 3D globes, and more.

Four layers, each doing one job:

```
Layer 1 (Voice I/O)     →  Captures speech, speaks responses
Layer 2 (AI + RAG)       →  Understands intent, retrieves data, decides what to show
Layer 3 (Blob)           →  Executes layout — places widgets on a CSS Grid
Layer 4 (Widgets)        →  23 visual components from a widget database
```

---

## The Stack

| Component | Tech | Port | What It Does |
|-----------|------|------|--------------|
| Frontend | Next.js 14, TypeScript, Tailwind | 3100 | Web UI, widgets, voice controls |
| Backend | Django 5.2, Gunicorn | 8100 | REST API, orchestration, RAG |
| STT Server | FastAPI, Parakeet/Whisper | 8890 | Speech-to-text |
| TTS Server | Kokoro (Docker) | 8880 | Text-to-speech |
| Vector DB | ChromaDB | disk | Equipment/alert RAG index |
| LLM | Ollama (phi4) | 11434 | Natural language generation |
| Database | SQLite | disk | Sessions, equipment, transcripts |

Deployed via systemd user services on LAN (192.168.1.20). HTTPS with self-signed cert. UFW firewall configured for LAN access.

---

## Layer 1 — Voice I/O

### Two Implementations

**V1: PersonaPlex full-duplex** (`VoiceInterface.tsx` + `usePersonaPlex.ts`)
- Uses PersonaPlex-7B over WebSocket — a full-duplex voice model
- Listens while speaking, handles interruptions, generates natural filler
- Opus 24kHz audio, binary Moshi protocol
- GPU-intensive (17.6GB VRAM)
- More natural but more complex

**V2: Server STT/TTS pipeline** (`VoiceInterfaceV2.tsx` + `useVoicePipeline.ts`)
- Separates speech input and output into two server calls
- STT: Parakeet (NVIDIA NeMo, best accuracy) or Whisper (fallback) at port 8890
- TTS: Kokoro neural voice (Docker) at port 8880
- Cleaner pipeline, lower latency, configurable models
- No filler speech — queues messages if user speaks during AI response
- **This is the active interface**

### How V2 Works

```
User speaks
    → Microphone captures audio (MediaRecorder, 500ms chunks)
    → Accumulates chunks, sends to STT server every 3s
    → STT returns full transcript (cumulative re-transcription)
    → Pipeline extracts delta (new words since last send)
    → Waits for silence (2 identical consecutive transcripts)
    → Sends final transcript to Layer 2 backend
    → Layer 2 returns voice_response + layout_json
    → TTS speaks the voice_response
    → layout_json emitted on event bus for Blob
```

### V2 UI Layout

VoiceInterfaceV2 is currently the full-page interface:
- **Top bar**: Pipeline status chips (STT, AI/RAG, TTS) with green/yellow/red dots
- **Left panel**: Conversation transcript (user blue, AI purple, interim dashed)
- **Right sidebar**: Performance stats — session uptime, query count, STT latency, TTS latency, Layer 2 processing time, RAG domain execution times
- **Bottom bar**: Device selector, Speak/Stop buttons, error banners
- **Settings** (Ctrl+M): Switch STT model, TTS engine, toggle interim transcripts
- **Hotkey**: Ctrl+Space to toggle voice

### Supporting Components

| File | What It Does |
|------|-------------|
| `useSTT.ts` | STT hook — Parakeet/Whisper/Web Speech API, cumulative transcription, stats |
| `useKokoroTTS.ts` | TTS hook — Kokoro/Piper/Browser fallback, speak() with onEnd callback |
| `usePersonaPlex.ts` | V1 full-duplex hook — WebSocket, Opus audio, text injection |
| `useSpeechRecognition.ts` | Browser Web Speech API wrapper |
| `useSpeechSynthesis.ts` | Browser Speech Synthesis wrapper |
| `TranscriptPanel.tsx` | Small floating overlay (top-right) showing recent transcripts |
| `ConversationTranscript.tsx` | Message bubbles with timestamps |
| `AudioVisualizer.tsx` | Real-time waveform canvas |
| `MetricsCollector.ts` | Telemetry aggregation (latency percentiles, audio quality, errors) |
| `MetricsDashboard.tsx` | Performance stats panel with sparkline graphs |
| `ConnectionStatus.tsx` | WebSocket connection state indicator |
| `DeviceSelector.tsx` | Audio input/output device picker |

### STT Server (port 8890)

Standalone FastAPI server in `backend/stt/server.py`:
- Loads Parakeet TDT 0.6B (NVIDIA NeMo) as primary model
- Falls back to Faster-Whisper large-v3 if Parakeet fails at runtime
- Hot-swap between models via `/v1/stt/switch`
- Accepts WAV, FLAC, OGG, WebM/Opus audio
- Auto-resamples to 16kHz mono

### TTS Server (port 8880)

Docker container running Kokoro FastAPI:
- Image: `ghcr.io/remsky/kokoro-fastapi-cpu:latest` (CPU variant — GPU requires nvidia-container-toolkit)
- OpenAI-compatible API at `/v1/audio/speech`
- Returns MP3 audio

---

## Layer 2 — AI + RAG Orchestration

### Backend Orchestrator (`backend/layer2/orchestrator.py`)

Three pipeline stages, executed on every voice command:

**Stage 2A: Intent Parsing**
- Keyword-based (no ML model needed, fast)
- Detects intent type: `query`, `action`, `greeting`, `conversation`, `out_of_scope`
- Detects relevant domains: `industrial`, `alerts`, `supply`, `people`, `tasks` (50+ keywords per domain)
- Extracts entities: numbers, device references (pump_1, motor_3), time spans
- Scope guard: unknown queries get "I can help with operations" response

**Stage 2B: Parallel RAG Queries**
- For each detected domain, spawns a thread (ThreadPoolExecutor, 5 workers)
- Industrial + Alerts domains → real RAG pipeline (ChromaDB + Ollama LLM)
- Supply, People, Tasks → stub data (for now)
- All domains queried simultaneously to minimize latency
- Each domain returns: retrieved documents, LLM response, sources, execution time

**Stage 2C: Response Generation + Layout**
- Generates natural language response for TTS (2-3 sentences, conversational)
- Maps RAG results to widget instructions (scenario slug + fixture variant + data override)
- Maps domain → widget type:
  - Industrial → KPI tiles (device metrics), trend charts, device panels
  - Alerts → alert banner, alert count KPI
  - Supply → category bar chart
  - People → people view
  - Tasks → task count KPI

### RAG Pipeline (`backend/layer2/rag_pipeline.py`)

```
User question
    → Embed with sentence-transformers (all-MiniLM-L6-v2, 384-dim)
    → Search ChromaDB for similar documents (top 5)
    → Build context from retrieved docs
    → Send context + question to Ollama LLM (phi4)
    → Return LLM response + sources
```

Three ChromaDB collections:
- `industrial_equipment` — equipment catalog (Transformer, Pump, Motor, etc.)
- `industrial_alerts` — active alerts with severity
- `maintenance_records` — service history

Each equipment record becomes a rich text document like:
```
TX-001 (Transformer) | Type: distribution | Location: Building A | Status: running
Health Score: 95% | Criticality: critical | Capacity: 315 kVA | Load: 75%
Oil Temp: 65C | Winding Temp: 78C
```

### Frontend Layer 2 Client (`frontend/src/lib/layer2/client.ts`)

- `Layer2Service` class wraps the backend API
- Calls `POST /api/layer2/orchestrate/` with transcript
- Returns: `voice_response`, `filler_text`, `layout_json`, `intent`, `rag_results`, `processing_time_ms`
- Provides callbacks: `onResponse()`, `onLayout()`, `onFiller()`

### API Endpoints

| Endpoint | Method | What It Does |
|----------|--------|-------------|
| `/api/layer2/orchestrate/` | POST | Main pipeline: transcript → AI response + widget layout |
| `/api/layer2/filler/` | POST | Get filler text while RAG processes |
| `/api/layer2/proactive/` | POST | Get proactive triggers based on system state |
| `/api/layer2/rag/industrial/` | POST | Direct industrial RAG query |
| `/api/layer2/rag/industrial/health/` | GET | RAG health + index stats |
| `/api/layer1/sessions/` | GET/POST | Voice session management |
| `/api/layer1/transcripts/` | GET/POST | Transcript logging |

---

## Layer 3 — Blob (Layout Executor)

Blob is the layout executor. It receives a `LayoutJSON` from Layer 2 (via the event bus) and renders widgets in a CSS Grid.

### Files

| File | What It Does |
|------|-------------|
| `Blob.tsx` | Main component — subscribes to layout state, resolves widget instructions to components, merges fixture data with real data overrides, renders in BlobGrid |
| `BlobGrid.tsx` | 12-column CSS Grid container with gap-4, auto-rows |
| `WidgetSlot.tsx` | Per-widget wrapper — error boundary, loading skeleton (Suspense), size-based col-span classes |
| `useLayoutState.ts` | Hook that holds current LayoutJSON, subscribes to LAYOUT_UPDATE events from event bus |
| `defaultLayout.ts` | Initial dashboard for "project engineer" persona — 4 KPI tiles + trend chart + alert panel |

### Size → Grid Mapping

| Size | Grid Columns | Min Height |
|------|-------------|-----------|
| hero | span 12 (full width) | 60vh |
| expanded | span 6 (half) | 40vh |
| normal | span 4 (third) | 30vh |
| compact | span 3 (quarter) | 15vh |
| hidden | not rendered | — |

### Data Flow

```
Layer 2 returns layout_json
    → useVoicePipeline emits LAYOUT_UPDATE on event bus
    → useLayoutState receives it, updates state
    → Blob re-renders with new widget list
    → For each widget instruction:
        1. Look up React component in widgetRegistry by scenario slug
        2. Look up fixture data by scenario + fixture slug
        3. Merge fixture defaults with data_override from Layer 2
        4. Render component inside WidgetSlot (with Suspense + ErrorBoundary)
```

### WidgetInstruction Format

```typescript
{
  scenario: "kpi",                    // which widget component
  fixture: "kpi_live-standard",       // which fixture variant for defaults
  size: "compact",                    // grid sizing
  position: null,                     // positioning hint (unused for now)
  relevance: 0.9,                     // sort order (higher = rendered first)
  data_override: {                    // real data from RAG (overrides fixture)
    demoData: {
      label: "Grid Voltage",
      value: "238.4",
      unit: "V",
      state: "normal"
    }
  }
}
```

### Current Status

Blob is built and compiles but is **not wired into page.tsx yet**. VoiceInterfaceV2 is still the full-page UI. The next step is to either:
- Make Blob the primary view with voice controls as a compact floating element, or
- Embed Blob inside VoiceInterfaceV2's main content area

---

## Layer 4 — Widget Components

23 React components extracted from `Widgets/scenarios.sqlite3`. Each exports `ScenarioComponent({ data })` and renders a self-contained visualization.

### Widget Registry (`widgetRegistry.ts`)

Maps scenario slugs to lazy-loaded components:
```typescript
const WIDGET_REGISTRY = {
  "kpi": lazy(() => import("./widgets/kpi")),
  "alerts": lazy(() => import("./widgets/alerts")),
  // ... all 23
};
```

### Fixture Data (`fixtureData.ts`)

84 fixture variants across 23 scenarios. Each fixture contains visual config + demo data that the widget can render standalone.

### All 23 Widgets

**Data Visualization**

| Slug | Component | What It Shows |
|------|-----------|-------------|
| `kpi` | KPI tiles | Single metric with label, value, unit, state color (9 fixtures: live, alert-warning, alert-critical, accumulated, lifecycle, gauge, badge, offline) |
| `trend` | Trend line | Single time-series line chart |
| `trend-multi-line` | Multi-line trend | Overlaid series with dual Y-axes, thresholds, legend |
| `trends-cumulative` | Cumulative area | Stacked area chart for accumulation over time |
| `distribution` | Distribution | Histogram / distribution chart |
| `comparison` | Comparison | Side-by-side bar chart |
| `composition` | Composition | Pie/donut + treemap + stacked bar (multiple sub-visualizations) |
| `flow-sankey` | Sankey diagram | Energy/material flow from sources to loads |
| `matrix-heatmap` | Heatmap grid | Equipment health by location/time |
| `category-bar` | Category bars | Horizontal/vertical bar chart (7 variants: vertical, horizontal, stacked, stacked-percent, grouped, diverging, dense) |

**Data Streams**

| Slug | Component | What It Shows |
|------|-----------|-------------|
| `timeline` | Event timeline | Operational events on a time axis |
| `eventlogstream` | Event log | Real-time scrolling log feed (like tail -f) |
| `chatstream` | Chat interface | AI chat bubbles with message history |
| `alerts` | Alert panel | Alert list with severity filtering, actions (5 fixtures by alert type) |

**Domain Panels**

| Slug | Component | What It Shows |
|------|-----------|-------------|
| `edgedevicepanel` | Device table | Equipment list with status, health, location, actions |
| `agentsview` | Agent dashboard | AI agent registry with status |
| `peoplehexgrid` | Hex grid | Personnel visualization as hexagons |
| `peoplenetwork` | Network graph | Relationship/org chart as nodes + edges |
| `peopleview` | People directory | HR view with shifts, attendance |
| `pulseview` | Pulse indicator | Animated system health heartbeat |
| `supplychainglobe` | 3D Globe | Supply chain geography on a Three.js globe |
| `helpview` | Help panel | Documentation and support links |
| `vaultview` | Data vault | Secure document archive browser |

### Widget Database (`Widgets/scenarios.sqlite3`)

Source of truth for all widgets. Contains:
- 23 scenarios with slugs, names, tags
- 84 fixture variants with visual config + demo data JSON
- Full React component source code for each scenario
- Control groups (sliders, toggles for interactive preview)
- Component references and extraction metadata

---

## Industrial Equipment Database

Django models in `backend/industrial/models.py`. 11 equipment types covering a real industrial facility:

**Electrical**: Transformer, DieselGenerator, ElectricalPanel (MCC/PCC/APFC/VFD/DB), UPS
**HVAC**: Chiller (air/water-cooled), AHU (fresh/return/mixed air), CoolingTower
**Mechanical**: Pump (chilled water/condenser/fire/sump), Compressor (screw/reciprocating/centrifugal), Motor (induction/synchronous/DC/servo)
**Metering**: EnergyMeter (main/sub/feeder/DG/solar)
**Operations**: Alert (severity levels), MaintenanceRecord (preventive/corrective/breakdown/predictive)

Populated via `python manage.py populate_industrial_db` — creates 600+ equipment records, 50+ alerts, 200+ maintenance records.

Indexed into ChromaDB via `python manage.py index_rag`.

---

## Canvas, StatusBar, Spot, Debug

### Canvas (`components/canvas/Canvas.tsx`)
Root container. 100vh x 100vw, no scroll. Two zones:
- `<main>` (flex-1): where Blob/VoiceInterface renders
- `<footer>` (48px): StatusBar

### StatusBar (`components/status-bar/StatusBar.tsx`)
Fixed bottom dock:
- Left: Spot orb + state label ("Listening...", "Speaking...", "Ready")
- Center: Capability chips (placeholder)
- Right: Ledger info (placeholder)

### Spot (`components/spot/Spot.tsx` + `SpotWalk.ts`)
Animated particle orb showing AI state:
- Idle: slow calm particles, small
- Listening: expanded, waveform-reactive, blue
- Speaking: pulsing, purple
- Processing: animated swirl, yellow
- Error: red shake

Canvas-based animation with particle physics (gravity, damping, chaos). SpotWalk is the state machine that drives transitions based on events from the event bus.

### Debug Panel (`components/debug/DebugPanel.tsx`)
Toggle with Ctrl+D. Shows:
- Layer 1 status (PersonaPlex connection, audio I/O)
- Layer 2 status (backend health, Ollama availability, RAG index stats)
- Query logs with timing
- Test query input

---

## Event Bus (`lib/events.ts`)

Singleton `commandCenterBus` for inter-layer communication:
```typescript
commandCenterBus.on("LAYOUT_UPDATE", handler)  // subscribe
commandCenterBus.emit({ type: "LAYOUT_UPDATE", layout })  // broadcast
```

Event types:
- `PERSONAPLEX_STATE_CHANGE` — Spot and StatusBar react
- `TRANSCRIPT_UPDATE` — TranscriptPanel displays
- `VOICE_INPUT_START` / `VOICE_INPUT_STOP` — Spot reacts
- `RAG_RESULT` — Debug panel logs
- `LAYOUT_UPDATE` — Blob re-renders widgets
- `SPOT_STATE_CHANGE` — Spot visual updates

---

## Config (`lib/config.ts`)

All server URLs read from environment or defaults:
```
API base:      NEXT_PUBLIC_API_URL or http://localhost:8100
PersonaPlex:   http://localhost:8998
STT server:    http://localhost:8890
TTS server:    http://localhost:8880
LLM (Ollama):  http://localhost:11434, model: phi4
```

---

## Deployment

### Services (systemd user units)

| Service | Port | What It Runs |
|---------|------|-------------|
| cc-backend | 8100 | Gunicorn + Django (3 workers, 120s timeout) |
| cc-frontend | 3100 | Node.js custom HTTPS server (server.js) |
| cc-stt | 8890 | Python FastAPI (Parakeet + Whisper) |
| cc-tts | 8880 | Docker: Kokoro FastAPI CPU |

All configured with `Restart=always`, log to `logs/` directory, enabled with linger for persistence.

### Deploy Script (`scripts/deploy.sh`)
```bash
./scripts/deploy.sh          # Full deploy
./scripts/deploy.sh --stop   # Stop all
./scripts/deploy.sh --status # Check status
./scripts/deploy.sh --logs   # Tail all logs
```

### Setup Script (`scripts/setup.sh`)
```bash
./scripts/setup.sh           # Full setup (venv, deps, migrations, RAG index)
./scripts/setup.sh --backend # Backend only
./scripts/setup.sh --rag     # RAG indexing only
```

### SSL
Self-signed cert in `certs/` covering localhost + 192.168.1.20 (10-year validity). Used by `frontend/server.js` for HTTPS on LAN.

---

## What Works End-to-End

1. User opens `https://192.168.1.20:3100` on any LAN device
2. VoiceInterfaceV2 loads with pipeline status chips
3. User clicks Speak (or Ctrl+Space) and says "What's the status of the pumps?"
4. STT server (Parakeet) transcribes in 3s cycles
5. Pipeline detects silence, sends transcript to Layer 2
6. Layer 2 parses intent (query, industrial domain), spawns RAG query
7. ChromaDB retrieves pump documents, Ollama generates response
8. Backend returns voice response + layout JSON
9. TTS speaks: "3 of 4 pumps are running normally. Pump 3 shows elevated temperature."
10. Layout JSON emitted on event bus (ready for Blob to render widgets)

---

## What's Not Wired Yet

- **Blob is not rendering on page** — VoiceInterfaceV2 is the full-page UI. Blob compiles and works but needs to be integrated as the main content area with voice controls floating on top or embedded compactly.
- **Supply, People, Tasks RAG** — return stub data, not connected to real databases yet.
- **Widget data refresh** — widgets render static data from Layer 2 response. No live polling or WebSocket updates.
- **Transitions/animations** — defined in types but not implemented in BlobGrid.
- **PersonaPlex V1** — exists but not active. V2 is the current interface.

---

## File Tree (Key Files Only)

```
CommandCenter/
  backend/
    command_center/        Django project settings
    layer1/                Voice session models + API
    layer2/
      orchestrator.py      2A/2B/2C pipeline (intent → RAG → response + layout)
      rag_pipeline.py      ChromaDB + Ollama RAG system
      views.py             API endpoints
    industrial/
      models.py            11 equipment types + alerts + maintenance
    stt/
      server.py            Parakeet/Whisper STT server (port 8890)
    chroma_db/             Vector store persistence

  frontend/
    src/
      app/page.tsx         Main page (Canvas + VoiceInterfaceV2)
      components/
        layer1/
          VoiceInterfaceV2.tsx    Active voice UI
          VoiceInterface.tsx      Legacy V1 (PersonaPlex)
          useVoicePipeline.ts     V2 pipeline hook
          useSTT.ts               STT hook
          useKokoroTTS.ts         TTS hook
          usePersonaPlex.ts       V1 full-duplex hook
        layer2/
          orchestrator.ts         Frontend orchestrator stub
        layer3/
          Blob.tsx                Layout executor
          BlobGrid.tsx            12-column CSS Grid
          WidgetSlot.tsx          Widget wrapper + error boundary
          useLayoutState.ts       Layout state + event bus
          defaultLayout.ts        Default dashboard
        layer4/
          widgetRegistry.ts       23 widgets, lazy-loaded
          fixtureData.ts          84 fixture variants
          widgets/*.tsx            23 widget components
        canvas/Canvas.tsx         100vh root container
        spot/Spot.tsx             Particle AI orb
        status-bar/StatusBar.tsx  Bottom dock
        debug/DebugPanel.tsx      Pipeline debug (Ctrl+D)
      lib/
        config.ts                Server URLs + config
        events.ts                Event bus (commandCenterBus)
        layer2/client.ts         Layer 2 API client
    server.js                    HTTPS production server

  scripts/
    deploy.sh                    Full-stack deploy
    setup.sh                     Initial setup
    systemd/*.service            4 service definitions

  Widgets/
    scenarios.sqlite3            Widget database (23 scenarios, 84 fixtures)

  certs/
    lan.crt, lan.key             Self-signed SSL cert
```

---

## Dependencies Installed

**Backend (Python):** Django, DRF, django-cors-headers, chromadb, sentence-transformers, requests, gunicorn, nemo_toolkit, faster-whisper

**Frontend (npm):** next, react, recharts, @mui/material, @mui/icons-material, @emotion/react, @emotion/styled, lucide-react, three, @react-three/fiber, @react-three/drei, react-window, date-fns, opus-recorder, tailwindcss

**Infrastructure:** Docker (Kokoro TTS), Ollama (phi4 LLM), systemd (4 services), self-signed SSL, UFW firewall rules
