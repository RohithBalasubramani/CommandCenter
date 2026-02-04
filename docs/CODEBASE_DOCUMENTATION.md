# Command Center - Complete Codebase Documentation

**A Line-by-Line Guide to Understanding Every File**

This document explains every file in the Command Center codebase in simple, everyday language. Think of Command Center as a smart assistant for factory workers - they can talk to it, ask questions about their equipment, and see visual dashboards that answer their questions.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [How the System Works](#how-the-system-works)
3. [Backend Files](#backend-files)
   - [Django Configuration](#django-configuration)
   - [Layer 1 - Voice Input/Output](#layer-1---voice-inputoutput)
   - [Layer 2 - The AI Brain](#layer-2---the-ai-brain)
   - [Actions Module](#actions-module)
   - [Feedback Module](#feedback-module)
   - [Industrial Module](#industrial-module)
   - [Reinforcement Learning](#reinforcement-learning)
4. [Frontend Files](#frontend-files)
   - [Library Utilities](#library-utilities)
   - [Layer 1 Components](#layer-1-components)
   - [Layer 3 Components](#layer-3-components)
   - [Layer 4 Components](#layer-4-components)
5. [Scripts and Automation](#scripts-and-automation)
6. [Tests](#tests)

---

## Project Overview

Command Center is like having a smart assistant in a factory. Imagine you're a factory worker and you want to know "How is pump 3 doing?" Instead of walking to a computer, finding the right screen, and clicking through menus, you just ask out loud. The system:

1. **Hears you** (Speech-to-Text)
2. **Understands what you want** (AI Brain)
3. **Finds the information** (Database Search)
4. **Shows you visual charts and numbers** (Dashboard)
5. **Tells you the answer out loud** (Text-to-Speech)

The code is organized into **4 layers**:
- **Layer 1**: Handles hearing and speaking (voice input/output)
- **Layer 2**: The brain that understands questions and finds answers
- **Layer 3**: Arranges visual widgets on screen
- **Layer 4**: The actual charts, graphs, and numbers you see

---

## How the System Works

Here's what happens when you ask "What's the pump status?":

```
YOU SPEAK: "What's the pump status?"
     ↓
[LAYER 1: Voice] Your voice is converted to text
     ↓
[LAYER 2: Brain] The AI understands you want pump information
     ↓
[LAYER 2: Search] Searches the database for pump data
     ↓
[LAYER 2: Widgets] Decides to show a trend chart + status + alerts
     ↓
[LAYER 3: Layout] Arranges those widgets nicely on screen
     ↓
[LAYER 4: Display] Actually draws the charts and numbers
     ↓
[LAYER 1: Voice] Says "Pump 3 is running at 85% capacity"
     ↓
YOU SEE AND HEAR THE ANSWER
```

---

## Backend Files

The backend is written in Python using Django (a popular web framework). It runs on the server and handles all the "thinking" - understanding questions, searching databases, and deciding what to show.

---

### Django Configuration

These files set up how the Django web server works.

---

#### `backend/command_center/settings.py`

**What it does:** This is the main configuration file for the entire backend. Think of it as the "control panel" that tells Django how to behave.

**In simple terms:** Just like your phone has settings for Wi-Fi, notifications, and display, this file has settings for the web server.

**Key sections explained:**

```python
SECRET_KEY = 'django-insecure-x@47$...'
```
**What this means:** A password that Django uses internally to keep things secure. The "insecure" prefix warns that this shouldn't be used in a real production system - it's fine for testing.

```python
DEBUG = True
```
**What this means:** When something goes wrong, show detailed error messages. This is like "developer mode" - helpful for finding problems, but you'd turn it off before real users see the system.

```python
ALLOWED_HOSTS = ["*"]
```
**What this means:** Accept requests from any computer. The `*` means "anyone" - in a real system, you'd list specific allowed addresses.

```python
INSTALLED_APPS = [
    'django.contrib.admin',    # Built-in admin dashboard
    'rest_framework',          # Tools for building APIs
    'corsheaders',             # Allows frontend to talk to backend
    'layer1',                  # Our voice handling code
    'layer2',                  # Our AI brain code
    'industrial',              # Factory equipment data
    'actions',                 # Voice commands like "start pump"
    'feedback',                # User ratings and feedback
]
```
**What this means:** A list of all the "apps" (code modules) that are part of this system. Each app handles a specific job.

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
```
**What this means:** Where to store data. SQLite is a simple database stored in a single file called `db.sqlite3`. It's like a spreadsheet file that holds all our data.

```python
CORS_ALLOWED_ORIGINS = [
    'http://localhost:3100',      # Development computer
    'http://192.168.1.20:3100',   # Office network
]
```
**What this means:** Which websites are allowed to talk to this backend. The frontend (user interface) runs on a different address, so we need to explicitly allow it.

---

#### `backend/command_center/urls.py`

**What it does:** This file is like a phone directory - it tells Django where to send different requests.

**In simple terms:** When someone visits a web address, this file decides which code should handle it.

```python
urlpatterns = [
    path("admin/", admin.site.urls),           # /admin/ goes to Django's admin panel
    path("api/layer1/", include("layer1.urls")),  # /api/layer1/... goes to voice code
    path("api/layer2/", include("layer2.urls")),  # /api/layer2/... goes to AI brain code
    path("api/actions/", include("actions.urls")), # /api/actions/... goes to action code
    path("api/", include("feedback.urls")),        # /api/... goes to feedback code
]
```
**What this means:**
- If someone visits `/admin/`, show the admin dashboard
- If someone visits `/api/layer1/something`, let the Layer 1 code handle it
- And so on...

---

#### `backend/command_center/wsgi.py`

**What it does:** This is the "entry point" for running the backend in production.

**In simple terms:** When you want to run the server for real users (not just testing), this file tells the computer how to start it.

```python
application = get_wsgi_application()
```
**What this means:** Creates an "application" object that web servers (like Gunicorn) can use to run Django.

---

### Layer 1 - Voice Input/Output

Layer 1 handles everything related to voice - hearing what users say and speaking responses back.

---

#### `backend/layer1/models.py`

**What it does:** Defines the database tables for storing voice-related information.

**In simple terms:** This creates "spreadsheet templates" for storing conversation data.

**The VoiceSession model:**
```python
class VoiceSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(choices=[("active", "Active"), ("completed", "Completed"), ("error", "Error")])
```
**What this means:** Each conversation gets its own "session" record. It tracks:
- `id`: A unique identifier (like a ticket number)
- `started_at`: When the conversation began
- `ended_at`: When it finished (blank if still going)
- `status`: Whether it's active, completed, or had an error

**The Transcript model:**
```python
class Transcript(models.Model):
    session = models.ForeignKey(VoiceSession, on_delete=models.CASCADE)
    role = models.CharField(choices=[("user", "User"), ("assistant", "Assistant")])
    text = models.TextField()
    is_final = models.BooleanField(default=True)
```
**What this means:** Each thing someone says (or the AI says) gets recorded:
- `session`: Which conversation this belongs to
- `role`: Who said it - the human user or the AI assistant
- `text`: The actual words
- `is_final`: Whether this is the final version (speech recognition sometimes updates as you speak)

**The PersonaPlexConfig model:**
```python
class PersonaPlexConfig(models.Model):
    server_url = models.URLField(default="http://localhost:8090")
    model = models.CharField(default="personaplex-7b-v1")
    voice = models.CharField(default="NATF0")
    always_on = models.BooleanField(default=False)
```
**What this means:** Settings for the voice AI system:
- `server_url`: Where the voice AI server is running
- `model`: Which AI model to use
- `voice`: Which voice style (NATF0 = Natural Female voice)
- `always_on`: Whether to keep the connection open all the time

---

#### `backend/layer1/views.py`

**What it does:** Handles web requests related to voice sessions and transcripts.

**In simple terms:** When the frontend wants to save a transcript or start a session, this code handles those requests.

```python
class VoiceSessionViewSet(viewsets.ModelViewSet):
    queryset = VoiceSession.objects.all()
    serializer_class = VoiceSessionSerializer
```
**What this means:** Automatically creates these API endpoints:
- `GET /api/layer1/sessions/` - List all voice sessions
- `POST /api/layer1/sessions/` - Start a new session
- `GET /api/layer1/sessions/123/` - Get details of session 123
- `DELETE /api/layer1/sessions/123/` - Delete session 123

```python
@action(detail=True, methods=["post"])
def end(self, request, pk=None):
    session = self.get_object()
    session.status = "completed"
    session.ended_at = timezone.now()
    session.save()
```
**What this means:** A custom action to mark a session as finished. When you call `POST /api/layer1/sessions/123/end/`, it sets the status to "completed" and records the end time.

---

### Layer 2 - The AI Brain

This is the most important part - it understands what users want and decides what to show them.

---

#### `backend/layer2/models.py`

**What it does:** Database tables for tracking AI queries and user memory.

**The RAGPipeline model:**
```python
class RAGPipeline(models.Model):
    DOMAIN_CHOICES = [
        ("industrial", "Industrial"),
        ("supply", "Supply Chain"),
        ("people", "People"),
        ("tasks", "Tasks"),
        ("alerts", "Alerts"),
    ]
    domain = models.CharField(choices=DOMAIN_CHOICES, unique=True)
    enabled = models.BooleanField(default=True)
    endpoint_url = models.URLField()
```
**What this means:** The system has different "knowledge areas" (domains):
- **Industrial**: Equipment, sensors, energy data
- **Supply**: Inventory, vendors, shipments
- **People**: Employees, shifts, schedules
- **Tasks**: Work orders, tickets
- **Alerts**: Alarms and warnings

Each domain can be enabled/disabled and has its own data source.

**The UserMemory model:**
```python
class UserMemory(models.Model):
    user_id = models.CharField(max_length=100)
    query = models.CharField(max_length=500)
    primary_characteristic = models.CharField(max_length=50)
    domains = models.JSONField(default=list)
    scenarios_used = models.JSONField(default=list)
```
**What this means:** The system remembers what each user has asked:
- `user_id`: Who asked
- `query`: What they asked
- `primary_characteristic`: What type of question (trend, comparison, etc.)
- `domains`: Which knowledge areas were used
- `scenarios_used`: Which widgets were shown

This helps the AI avoid showing the same things repeatedly and maintain conversation context.

---

#### `backend/layer2/orchestrator.py`

**What it does:** The central coordinator that processes user questions and generates responses.

**In simple terms:** This is the "brain" - it receives a question, figures out what it means, finds the answer, and decides how to display it.

**Key constants:**

```python
DOMAIN_KEYWORDS = {
    "industrial": ["pump", "motor", "temperature", "voltage", "power", ...],
    "supply": ["inventory", "stock", "vendor", "shipment", ...],
    "people": ["employee", "shift", "attendance", ...],
    "tasks": ["task", "work order", "ticket", ...],
    "alerts": ["alert", "alarm", "warning", ...],
}
```
**What this means:** Words that help identify which knowledge area a question belongs to. If someone says "pump", it's probably about industrial equipment.

```python
SCENARIO_HEIGHT_HINTS = {
    "kpi": "short",           # A single number - doesn't need much space
    "alerts": "medium",        # A list of alerts - moderate space
    "trend": "tall",          # A chart showing history - needs more space
    "flow-sankey": "x-tall",  # A complex flow diagram - needs lots of space
}
```
**What this means:** How much vertical space each widget type needs on screen.

```python
FILLER_TEMPLATES = {
    "checking": ["Let me check that for you.", "One moment while I look that up."],
    "processing": ["Processing your request.", "Analyzing the data."],
}
```
**What this means:** Things the AI says while it's working on your question, so you know it heard you.

```python
OUT_OF_SCOPE_MESSAGE = (
    "That's outside what I can help with. "
    "I'm your industrial operations assistant — I can help with "
    "equipment monitoring, alerts, maintenance, supply chain, "
    "workforce management, and task tracking."
)
```
**What this means:** What to say when someone asks about something the system can't help with (like "What's the weather?").

**The main processing function:**

```python
def process_transcript(self, transcript: str, session_context: dict = None,
                       user_id: str = "default_user") -> OrchestratorResponse:
```
**What this means:** This is THE main function. You give it what the user said (`transcript`), and it returns:
- `voice_response`: What to say back
- `layout_json`: What widgets to show
- `processing_time_ms`: How long it took
- `query_id`: A unique ID for tracking feedback

**The processing stages:**

1. **Intent Parsing** (understanding what the user wants):
```python
parsed = self._intent_parser.parse(transcript)
```
Figures out: Is this a question? A command? A greeting? What topics does it involve?

2. **Short-circuit for simple cases:**
```python
if parsed.type == "out_of_scope":
    return OrchestratorResponse(voice_response=OUT_OF_SCOPE_MESSAGE, ...)

if parsed.type == "greeting":
    return OrchestratorResponse(voice_response=self._generate_greeting(), ...)
```
Some questions don't need complex processing - greetings, out-of-scope questions, etc.

3. **Widget Selection** (choosing what to show):
```python
widget_plan = self._widget_selector.select(parsed, data_summary, user_context)
```
Based on the question, choose which charts/graphs/numbers to display.

4. **Data Collection** (getting the actual data):
```python
widget_data = self._data_collector.collect_all(widget_plan.widgets, transcript)
```
Fetch the actual numbers and information for each widget.

5. **Voice Response Generation:**
```python
voice_response = self._generate_voice_response_v2(parsed, layout_json, transcript)
```
Create a natural language answer to speak back.

6. **Save to Memory:**
```python
memory_mgr.record(user_id, transcript, parsed, scenarios_used)
```
Remember this interaction for context in future questions.

---

#### `backend/layer2/intent_parser.py`

**What it does:** Understands what type of question the user is asking.

**In simple terms:** Like a secretary who reads your message and figures out "this is a question about equipment" or "this is asking to send a message."

**Intent Types:**
```python
INTENT_TYPES = [
    "query",              # Asking for information
    "action_reminder",    # Set a reminder
    "action_message",     # Send a message to someone
    "action_control",     # Start/stop equipment
    "action_task",        # Create a work order
    "conversation",       # Small talk ("how are you")
    "out_of_scope",       # Not related to factory operations
]
```

**Characteristics** (what kind of visualization is needed):
```python
CHARACTERISTICS = [
    "comparison",    # Comparing two things
    "trend",         # How something changes over time
    "distribution",  # Breaking down into categories
    "maintenance",   # Repair/service information
    "energy",        # Power consumption
    "alerts",        # Warnings and alarms
    # ... and more
]
```

**How it works:**
```python
def parse(self, transcript: str) -> ParsedIntent:
    # Try the smart AI method first
    try:
        result = self._parse_with_llm(transcript)
        if result is not None:
            return result
    except Exception:
        pass

    # If AI fails, use simple pattern matching
    return self._parse_with_regex(transcript)
```
**What this means:**
1. First, try using a smart AI model to understand the question
2. If that doesn't work (server down, etc.), fall back to simpler pattern matching

**The AI parsing prompt:**
```python
PARSE_PROMPT_TEMPLATE = """Classify this user message for an industrial command center.

Intent types:
- "query": asking for information, status, data
- "action_reminder": set a reminder
- "action_control": start/stop equipment
...

User message: "{transcript}"

JSON:"""
```
**What this means:** We tell the AI exactly what categories exist and ask it to classify the user's message.

---

#### `backend/layer2/widget_selector.py`

**What it does:** Chooses which visual widgets to show based on the user's question.

**In simple terms:** Like a TV director who decides "for this story, we should show a graph, three numbers, and a list of alerts."

**Configuration:**
```python
MAX_HEIGHT_UNITS = 24   # Total "space budget" for the dashboard
MAX_WIDGETS = 10        # Maximum number of widgets
MAX_KPIS = 4            # Maximum simple number displays (avoid clutter)
BANNED_SCENARIOS = {"helpview", "pulseview"}  # Never show these
```

**The fast selection prompt:**
```python
FAST_SELECT_PROMPT = '''Select 8 widgets for this industrial operations query.

## WIDGET CATALOG
{catalog}

## QUERY
"{query}"

## SIZING RULES
Hero-capable widgets (use for first/main answer):
  trend, comparison, flow-sankey, matrix-heatmap...

Small widgets (NOT hero, use for supporting info):
  kpi: compact or normal only
  alerts: normal or expanded only

## RULES
1. First widget MUST be hero-capable with size="hero"
2. Use EXACT scenario names
3. Include diverse widget types
4. 8 widgets total

## OUTPUT (JSON only)
{{"heading": "<title>", "widgets": [...]}}'''
```
**What this means:** We give the AI:
- A catalog of all available widgets
- The user's question
- Rules about sizing and selection
- And ask it to pick 8 appropriate widgets

**Template descriptions:**
```python
WHY_TEMPLATES = {
    "kpi": "Shows the current value and status of the key metric at a glance.",
    "trend": "Displays how this metric has changed over time to identify patterns.",
    "alerts": "Lists active alerts and warnings that need attention.",
    "flow-sankey": "Visualizes how energy or resources flow through the system.",
    # ... more
}
```
**What this means:** Pre-written descriptions for each widget type. Instead of asking the AI to write these (slow), we use these templates.

**Validation:**
```python
def _validate_and_build_plan(self, data: dict, method: str = "llm") -> WidgetPlan:
    for w in raw_widgets:
        scenario = w.get("scenario", "").lower()

        # Is this a known widget type?
        if scenario not in VALID_SCENARIOS:
            continue  # Skip unknown types

        # Is it banned?
        if scenario in BANNED_SCENARIOS:
            continue  # Skip banned types

        # Would it exceed our budget?
        if budget - height < 0:
            continue  # Skip if no room

        # Is the size valid for this widget?
        if size not in allowed_sizes:
            size = next(s for s in ["hero", "expanded", "normal", "compact"] if s in allowed_sizes)
```
**What this means:** Even after the AI picks widgets, we double-check everything:
- Only allow known widget types
- Skip banned widgets
- Don't exceed space budget
- Fix invalid size selections

---

#### `backend/layer2/widget_catalog.py`

**What it does:** A registry of all available widget types with their properties.

**In simple terms:** Like a menu at a restaurant - it lists all the "dishes" (widgets) available and describes each one.

```python
WIDGET_CATALOG = [
    {
        "scenario": "kpi",
        "description": "Single metric display — shows one number with label, unit, and optional state (warning/critical).",
        "good_for": ["single metric", "status", "live reading", "count", "percentage"],
        "sizes": ["compact", "normal"],
        "height_units": 1,
    },
    {
        "scenario": "trend",
        "description": "Time series line/area chart for a single metric. Shows how a value changes over time.",
        "good_for": ["trend", "history", "over time", "monitoring", "last 24 hours"],
        "sizes": ["expanded", "hero"],
        "height_units": 3,
    },
    {
        "scenario": "flow-sankey",
        "description": "Sankey diagram showing flow from sources to destinations — energy flows, losses.",
        "good_for": ["flow", "energy balance", "losses", "where does it go"],
        "sizes": ["hero"],
        "height_units": 4,
    },
    # ... 20 more widget types
]
```
**What this means:** Each widget has:
- `scenario`: Its name/ID
- `description`: What it shows
- `good_for`: Keywords that suggest when to use it
- `sizes`: What sizes it can be displayed at
- `height_units`: How much vertical space it needs (1=small, 4=large)

---

#### `backend/layer2/rag_pipeline.py`

**What it does:** Searches through documents and data to find relevant information.

**In simple terms:** Like a librarian who can quickly find relevant books based on your question.

**RAG = Retrieval-Augmented Generation:**
1. **Retrieval**: Find relevant documents/data
2. **Augmented**: Add that context to the AI's knowledge
3. **Generation**: Generate an answer using that context

**Embedding Service:**
```python
class EmbeddingService:
    def embed(self, text: str) -> list[float]:
        # Convert text into numbers (a "vector")
        return self._model.encode(text).tolist()
```
**What this means:** Converts text into a list of numbers. Similar texts have similar numbers, which helps find related content.

**Vector Store:**
```python
class VectorStoreService:
    def search(self, collection: str, query: str, k: int = 5):
        # Find the k most similar documents to the query
        query_embedding = self.embedding_service.embed(query)
        results = collection.query(query_embeddings=[query_embedding], n_results=k)
        return results
```
**What this means:** Searches a collection of documents to find the ones most similar to your question.

**LLM Service:**
```python
class LLMService:
    def generate_json(self, prompt: str, system_prompt: str = None):
        # Ask the AI model and get JSON back
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[...],
            format="json"
        )
        return json.loads(response)
```
**What this means:** Sends a question to the AI model (like ChatGPT) and gets a structured answer back.

---

### Actions Module

Handles voice commands that DO something (not just ask questions).

---

#### `backend/actions/models.py`

**What it does:** Database tables for storing voice-triggered actions.

**Reminder Model:**
```python
class Reminder(models.Model):
    message = models.TextField()              # "Check pump 3 pressure"
    trigger_time = models.DateTimeField()     # When to remind
    recurring = models.CharField()             # "daily", "weekly", or blank
    status = models.CharField(choices=[
        ("pending", "Pending"),
        ("triggered", "Triggered"),
        ("dismissed", "Dismissed"),
    ])
    entity = models.CharField()               # Related equipment name
```
**What this means:** Stores reminders that users set via voice, like "Remind me to check pump 3 in 30 minutes."

**Message Model:**
```python
class Message(models.Model):
    recipient = models.CharField()   # "maintenance team"
    content = models.TextField()     # "The chiller needs inspection"
    channel = models.CharField(choices=[
        ("internal", "Internal"),
        ("sms", "SMS"),
        ("email", "Email"),
    ])
    status = models.CharField()      # "queued", "sent", "failed"
```
**What this means:** Stores messages to be sent, like "Send a message to maintenance about the chiller issue."

**DeviceCommand Model:**
```python
class DeviceCommand(models.Model):
    device_type = models.CharField()        # "pump"
    device_name = models.CharField()        # "pump_3"
    command = models.CharField()            # "start", "stop", "set_parameter"
    parameters = models.JSONField()         # {"speed": 75}
    requires_confirmation = models.BooleanField(default=True)
    confirmed = models.BooleanField(default=False)
    status = models.CharField(choices=[
        ("pending", "Pending"),
        ("confirmed", "Confirmed"),
        ("executed", "Executed"),
        ("failed", "Failed"),
        ("rejected", "Rejected"),
    ])
```
**What this means:** Stores equipment control commands like "Start pump 3." These require confirmation for safety - you don't want accidental "stop everything" commands!

---

#### `backend/actions/handlers.py`

**What it does:** Executes voice commands and saves them to the database.

```python
class ActionHandler:
    def execute(self, intent: ParsedIntent) -> ActionResult:
        handlers = {
            "action_reminder": self._create_reminder,
            "action_message": self._send_message,
            "action_control": self._device_command,
            "action_task": self._create_task,
        }
        handler = handlers.get(intent.type)
        return handler(intent)
```
**What this means:** Routes each action type to its handler function.

**Creating a reminder:**
```python
def _create_reminder(self, intent: ParsedIntent) -> ActionResult:
    trigger_time = self._parse_trigger_time(params)

    reminder = Reminder.objects.create(
        message=message,
        trigger_time=trigger_time,
        recurring=params.get("recurring", ""),
    )

    return ActionResult(
        success=True,
        voice_response=f"Reminder set for {time_str}: {message[:60]}",
    )
```
**What this means:** Creates a reminder in the database and confirms it to the user.

**Device command (with safety):**
```python
def _device_command(self, intent: ParsedIntent) -> ActionResult:
    cmd = DeviceCommand.objects.create(
        device_name=device_name,
        command=command,
        requires_confirmation=True,  # Safety first!
    )

    return ActionResult(
        success=True,
        voice_response=(
            f"Command '{command}' for {device_name} is pending confirmation. "
            f"Please confirm to proceed."
        ),
    )
```
**What this means:** Equipment commands don't run immediately - they wait for confirmation. This prevents accidents.

---

### Feedback Module

Collects user feedback to improve the system over time.

---

#### `backend/feedback/models.py`

**What it does:** Stores user ratings and feedback on widgets.

```python
class WidgetRating(models.Model):
    entry_id = models.CharField()      # Which widget was rated
    rating = models.CharField(choices=[("up", "Up"), ("down", "Down")])
    tags = models.JSONField()          # ["wrong data", "confusing"]
    notes = models.TextField()         # Free-text feedback
    device_id = models.CharField()     # Browser fingerprint
```
**What this means:** When users click thumbs up/down on a widget, it's stored here. This helps the system learn what works and what doesn't.

```python
class WidgetFeedback(models.Model):
    scenario = models.CharField()      # "trend"
    variant = models.CharField()       # "trend_line-standard"
    feedback_type = models.CharField(choices=[
        ("size", "Size Adjustment"),
        ("issue", "Issue Report"),
    ])
    data = models.JSONField()          # Full feedback details
```
**What this means:** More detailed feedback about specific widgets - whether they're the wrong size, showing wrong data, etc.

---

#### `backend/feedback/views.py`

**What it does:** API endpoints for submitting and retrieving ratings.

**In simple terms:** The "doors" that the frontend uses to send thumbs up/down and get feedback data.

**Single Rating Endpoint:**
```python
@api_view(["GET", "POST"])
def ratings_list(request):
    if request.method == "GET":
        qs = WidgetRating.objects.all()
        return Response(WidgetRatingSerializer(qs, many=True).data)

    # POST — upsert (update or create)
    obj, created = WidgetRating.objects.update_or_create(
        entry_id=entry_id,
        device_id=device_id,
        defaults={
            "rating": serializer.validated_data["rating"],
            "tags": serializer.validated_data.get("tags", []),
            "notes": serializer.validated_data.get("notes", ""),
        },
    )
```
**What this means:**
- GET: Returns all ratings
- POST: Creates or updates a rating (if you rate the same widget twice, it updates instead of creating duplicate)

**Bulk Rating Sync:**
```python
@api_view(["POST"])
def ratings_bulk(request):
    ratings_map = request.data.get("ratings", {})
    device_id = request.data.get("device_id", "")

    for entry_id, payload in ratings_map.items():
        WidgetRating.objects.update_or_create(
            entry_id=entry_id,
            device_id=device_id,
            defaults={...}
        )

    return Response({"created": created_count, "updated": updated_count})
```
**What this means:** The frontend stores ratings in the browser first (localStorage), then syncs them all at once when coming online. This handles offline usage.

---

#### `backend/feedback/signals.py`

**What it does:** Connects feedback to the learning system automatically.

**In simple terms:** Whenever someone rates a widget, this code automatically tells the AI learning system about it.

```python
@receiver(post_save, sender=WidgetRating)
def on_rating_saved(sender, instance, created, **kwargs):
    """Notify online learner of new feedback."""
    if not created:
        return  # Only process new ratings

    # Add to online learner buffer
    feedback = {
        "entry_id": instance.entry_id,
        "rating": instance.rating,
        "tags": instance.tags or [],
        "notes": instance.notes or "",
    }

    should_retrain = online_learner.add_feedback(feedback)

    if should_retrain:
        # Queue async retraining task
        async_task("rl.online_learner.trigger_retrain")
```
**What this means:**
1. Django "signals" automatically call this function when a rating is saved
2. The rating is added to the learning buffer
3. If enough ratings have accumulated, trigger a retraining job

This is how the system automatically improves over time without manual intervention.

---

### Industrial Module

Defines all the factory equipment the system can monitor.

---

#### `backend/industrial/models.py`

**What it does:** Database models for every type of industrial equipment.

**In simple terms:** Templates for storing information about pumps, motors, generators, chillers, and all other factory equipment.

**Base Equipment (common fields for all equipment):**
```python
class BaseEquipment(models.Model):
    equipment_id = models.CharField(unique=True)     # "PUMP-001"
    name = models.CharField()                        # "Main Coolant Pump"
    location = models.CharField()                    # "Building A, Floor 2"
    status = models.CharField(choices=[
        ("running", "Running"),
        ("stopped", "Stopped"),
        ("maintenance", "Under Maintenance"),
        ("fault", "Fault"),
    ])
    health_score = models.IntegerField()             # 0-100
    last_maintenance = models.DateTimeField()
    running_hours = models.FloatField()
```
**What this means:** Every piece of equipment has these basic properties.

**Transformer (electrical power):**
```python
class Transformer(BaseEquipment):
    capacity_kva = models.FloatField()       # Power rating
    primary_voltage = models.FloatField()    # Input voltage
    secondary_voltage = models.FloatField()  # Output voltage
    load_percent = models.FloatField()       # Current load (0-100%)
    oil_temperature = models.FloatField()    # For cooling
    winding_temperature = models.FloatField()
```

**Diesel Generator (backup power):**
```python
class DieselGenerator(BaseEquipment):
    capacity_kw = models.FloatField()        # Power output
    fuel_level_percent = models.FloatField() # How much fuel left
    coolant_temperature = models.FloatField()
    oil_pressure = models.FloatField()
    total_run_hours = models.FloatField()
```

**Chiller (cooling equipment):**
```python
class Chiller(BaseEquipment):
    capacity_tr = models.FloatField()              # Cooling capacity (tons)
    refrigerant_type = models.CharField()          # "R134a", "R410A"
    chilled_water_supply_temp = models.FloatField()
    chilled_water_return_temp = models.FloatField()
    power_consumption_kw = models.FloatField()
```

**Pump:**
```python
class Pump(BaseEquipment):
    flow_rate = models.FloatField()          # m³/hr
    motor_kw = models.FloatField()           # Motor power
    discharge_pressure = models.FloatField() # Output pressure
    vibration = models.FloatField()          # Vibration level (high = problem)
    bearing_temperature = models.FloatField()
```

**Alert (alarms and warnings):**
```python
class Alert(models.Model):
    equipment_id = models.CharField()
    severity = models.CharField(choices=[
        ("critical", "Critical"),
        ("high", "High"),
        ("medium", "Medium"),
        ("low", "Low"),
    ])
    alert_type = models.CharField(choices=[
        ("threshold", "Threshold Breach"),
        ("fault", "Equipment Fault"),
        ("maintenance", "Maintenance Due"),
    ])
    message = models.TextField()             # "Temperature exceeded 80°C"
    acknowledged = models.BooleanField()     # Has someone seen this?
    resolved = models.BooleanField()         # Is it fixed?
```
**What this means:** Alerts are created when something goes wrong - temperature too high, equipment failure, maintenance needed, etc.

---

### Reinforcement Learning

The system learns and improves based on user feedback.

---

#### `backend/rl/continuous.py`

**What it does:** Coordinates the continuous learning system.

**In simple terms:** Like a student who keeps learning - every time a user says "good" or "bad," the system remembers and tries to do better next time.

```python
class ContinuousRL:
    def record_experience(self, query_id, transcript, parsed_intent, widget_plan, ...):
        """Called after every question - stores what happened."""
        experience = Experience(
            query_id=query_id,
            transcript=transcript,
            parsed_intent=parsed_intent,
            widget_plan=widget_plan,
            ...
        )
        self.buffer.add(experience)
```
**What this means:** Every time someone asks a question, we record:
- What they asked
- What we understood
- What we showed them
- How long it took

```python
    def update_feedback(self, query_id, rating=None, interactions=None):
        """Called when user gives feedback (thumbs up/down)."""
        self.buffer.update_feedback(query_id, {"rating": rating})

        # Compute reward and learn
        exp = self.buffer.get_by_query_id(query_id)
        exp.computed_reward = self.reward_aggregator.compute_reward(exp)
```
**What this means:** When a user clicks thumbs up/down:
1. Find the experience we recorded earlier
2. Add the feedback to it
3. Calculate a "reward score" (positive for thumbs up, negative for down)
4. Use this to improve future selections

---

#### `backend/rl/online_learner.py`

**What it does:** Manages continuous learning from production feedback.

**In simple terms:** A system that accumulates user feedback and automatically retrains the AI when enough data has been collected.

**The FeedbackSample dataclass:**
```python
@dataclass
class FeedbackSample:
    entry_id: str           # Which widget was rated
    rating: str             # "up" or "down"
    tags: list[str]         # ["wrong data", "too small"]
    notes: str              # Free text feedback
    timestamp: datetime     # When the rating happened
    scenario: str           # "trend", "kpi", etc.
    fixture: str            # "trend_line-standard"
    query: str              # Original user question
```
**What this means:** Each piece of feedback contains everything needed to learn from it.

**The OnlineLearner class:**
```python
class OnlineLearner:
    def __init__(self, min_samples=50, max_buffer_size=1000, retrain_interval_hours=24):
        self.min_samples = min_samples           # Need at least 50 ratings before training
        self.max_buffer_size = max_buffer_size   # Keep at most 1000 samples
        self.retrain_interval_hours = retrain_interval_hours  # Train at most daily

        self.feedback_buffer = deque(maxlen=max_buffer_size)  # Ring buffer
        self.last_train_time = None
        self.is_training = False
```
**What this means:** Configuration for when to trigger training:
- Wait until 50 ratings accumulate
- Don't keep more than 1000 ratings (old ones fall off)
- Don't train more than once per day

**Adding feedback:**
```python
def add_feedback(self, feedback: dict) -> bool:
    sample = FeedbackSample(
        entry_id=feedback["entry_id"],
        rating=feedback["rating"],
        ...
    )

    with self.lock:  # Thread-safe
        self.feedback_buffer.append(sample)
        self._save_buffer()  # Persist to disk

    return self.should_retrain()  # Returns True if training should start
```
**What this means:** Thread-safe because Django handles multiple requests at once.

**Triggering training:**
```python
def _run_training(self) -> bool:
    # Snapshot and clear buffer
    with self.lock:
        samples = list(self.feedback_buffer)
        self.feedback_buffer.clear()

    # Build training dataset from samples
    widget_pairs = build_widget_dpo_pairs(entries, all_scenarios)
    fixture_pairs = build_fixture_dpo_pairs(entries, fixture_descriptions)
    dataset = pairs_to_hf_dataset(all_pairs)

    # Run DPO training
    trainer = CommandCenterDPOTrainer()
    trainer.load_base_model()
    result = trainer.train(train_dataset=train_data)

    # Export and deploy new model
    if self.auto_export:
        export_to_ollama(checkpoint_path=result.checkpoint_path)

    self.last_train_time = datetime.now()
```
**What this means:** The full training pipeline:
1. Take all accumulated feedback
2. Convert to training pairs (good vs bad choices)
3. Fine-tune the AI model using DPO (Direct Preference Optimization)
4. Export the improved model for use

---

#### `backend/rl/data_formatter.py`

**What it does:** Converts user ratings into training data for the AI.

**In simple terms:** Turns "thumbs up on widget A, thumbs down on widget B" into "when asked X, prefer A over B."

**DPO Training Pairs:**
```python
@dataclass
class DPOPair:
    prompt: str      # The user's question + context
    chosen: str      # The response that got thumbs up
    rejected: str    # The response that got thumbs down
    question_id: str # Links back to original interaction
```
**What this means:** DPO (Direct Preference Optimization) trains the AI by showing it pairs of choices and which one humans preferred.

**Building widget selection pairs:**
```python
def build_widget_dpo_pairs(entries: list[dict], all_scenarios: list[str]) -> list[DPOPair]:
    # Group ratings by question
    by_question = defaultdict(lambda: {
        "liked_scenarios": set(),
        "disliked_scenarios": set(),
    })

    for entry in entries:
        qid = entry.get("question_id")
        if entry["rating"] == "up":
            by_question[qid]["liked_scenarios"].add(entry["scenario"])
        else:
            by_question[qid]["disliked_scenarios"].add(entry["scenario"])

    # Create DPO pairs
    pairs = []
    for qid, data in by_question.items():
        prompt = format_widget_selection_prompt(query, all_scenarios)
        chosen = format_widget_selection_response(data["liked_scenarios"])
        rejected = format_widget_selection_response(data["disliked_scenarios"])

        pairs.append(DPOPair(prompt=prompt, chosen=chosen, rejected=rejected))

    return pairs
```
**What this means:** For each question where users rated some widgets good and others bad:
1. Group all the "liked" widgets together
2. Group all the "disliked" widgets together
3. Create a training example: "For this question, choose these (liked) not those (disliked)"

**Example training pair:**
```
Prompt: "User query: What's the pump status?
         Available scenarios: [kpi, trend, alerts, comparison, ...]
         Select the most appropriate widgets."

Chosen: "kpi (compact), trend (hero), alerts (normal)"
Rejected: "distribution (expanded), matrix-heatmap (hero)"
```

---

#### `backend/rl/experience_buffer.py`

**What it does:** Stores recent interactions in memory for learning.

```python
@dataclass
class Experience:
    query_id: str                    # Unique identifier
    timestamp: datetime              # When it happened
    transcript: str                  # What user asked
    parsed_intent: dict              # How we understood it
    widget_plan: dict                # What widgets we chose
    fixtures: dict                   # Which visual variants
    explicit_feedback: dict = None   # Thumbs up/down
    implicit_signals: dict = None    # Other behavior signals
    computed_reward: float = 0.0     # Final reward value
```
**What this means:** Each experience captures everything about one interaction.

```python
class ExperienceBuffer:
    def add(self, experience: Experience):
        """Add new experience to buffer."""
        self._buffer.append(experience)
        # Keep buffer from growing too large
        if len(self._buffer) > self._max_size:
            self._buffer.pop(0)  # Remove oldest
```
**What this means:** A "ring buffer" that keeps the most recent interactions. Old ones get removed when new ones come in.

---

#### `backend/rl/reward_signals.py`

**What it does:** Calculates how "good" each interaction was.

```python
class RewardSignalAggregator:
    def compute_reward(self, experience: Experience) -> float:
        reward = 0.0

        # Explicit feedback (thumbs up/down)
        if experience.explicit_feedback:
            rating = experience.explicit_feedback.get("rating")
            if rating == "up":
                reward += 1.0
            elif rating == "down":
                reward -= 1.0

        # Implicit signals (user behavior)
        if experience.follow_up_type == "drill_down":
            reward += 0.5  # User wanted more detail = good
        elif experience.follow_up_type == "refinement":
            reward -= 0.3  # User had to rephrase = not great

        return reward
```
**What this means:**
- Thumbs up = +1.0 (good)
- Thumbs down = -1.0 (bad)
- User drilled down for more = +0.5 (interested = good)
- User had to rephrase = -0.3 (we didn't understand well)

---

## Frontend Files

The frontend is built with React/Next.js and runs in the user's browser. It handles the user interface - what users see and interact with.

---

### Library Utilities

Helper code used throughout the frontend.

---

#### `frontend/src/lib/events.ts`

**What it does:** A messaging system that lets different parts of the app talk to each other.

**In simple terms:** Like a radio station - some parts "broadcast" messages, others "tune in" to hear them.

```typescript
class EventBus {
  private handlers: Map<string, Set<EventHandler>> = new Map();
```
**What this means:** A dictionary that maps event names to lists of functions that want to hear about them.

```typescript
  on(eventType: string, handler: EventHandler): () => void {
    if (!this.handlers.has(eventType)) {
      this.handlers.set(eventType, new Set());
    }
    this.handlers.get(eventType)!.add(handler);

    // Return unsubscribe function
    return () => {
      this.handlers.get(eventType)?.delete(handler);
    };
  }
```
**What this means:** "I want to listen for events of type X, and here's what to do when one happens." Returns a function to stop listening.

```typescript
  emit(event: CommandCenterEvent): void {
    const typeHandlers = this.handlers.get(event.type);
    typeHandlers?.forEach((handler) => {
      try {
        handler(event);
      } catch (e) {
        console.error("Handler error:", e);
      }
    });
  }
```
**What this means:** "Here's an event - tell everyone who's listening."

**Usage example:**
```typescript
// Component A: "I want to know when layouts update"
commandCenterBus.on("LAYOUT_UPDATE", (event) => {
  setWidgets(event.layout.widgets);
});

// Component B: "New layout ready!"
commandCenterBus.emit({ type: "LAYOUT_UPDATE", layout: {...} });
```

---

#### `frontend/src/lib/config.ts`

**What it does:** Stores configuration settings for the frontend.

```typescript
export const config = {
  api: {
    baseUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8100',
  },
  stt: {
    serverUrl: process.env.NEXT_PUBLIC_STT_SERVER_URL || 'http://localhost:8890',
  },
  tts: {
    serverUrl: process.env.NEXT_PUBLIC_TTS_SERVER_URL || 'http://localhost:8880',
  },
};
```
**What this means:** URLs for different services. Can be overridden with environment variables.

---

#### `frontend/src/lib/layer2/client.ts`

**What it does:** Talks to the Layer 2 backend API.

```typescript
class Layer2Service {
  async processTranscript(text: string): Promise<Layer2Response> {
    const response = await fetch(`${this.baseUrl}/api/layer2/orchestrate/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        transcript: text,
        session_id: this.sessionId,
        context: {},
      }),
    });
    return response.json();
  }
}
```
**What this means:** Sends the user's words to the backend and gets back what to show and say.

---

### Layer 1 Components

Voice input/output components.

---

#### `frontend/src/components/layer1/useVoicePipeline.ts`

**What it does:** The main voice processing hook - coordinates speech-to-text, backend calls, and text-to-speech.

**In simple terms:** The "conductor" that makes sure everything happens in the right order when you speak.

**State management:**
```typescript
const [state, setState] = useState<VoicePipelineState>("idle");
// Possible states: "idle" | "listening" | "speaking" | "error"

const [messages, setMessages] = useState<ConversationMessage[]>([]);
// Conversation history
```

**The main flow:**
```typescript
// When transcript changes (user is speaking)
useEffect(() => {
  if (!userTranscript) return;

  // Check if speech has stabilized (user stopped talking)
  if (userTranscript === lastTranscriptRef.current) {
    stableCountRef.current++;

    if (stableCountRef.current >= requiredStableCount) {
      // User stopped - send to backend
      const delta = userTranscript.slice(lastSentLenRef.current).trim();
      processOneTranscript(delta);
    }
  }
}, [userTranscript]);
```
**What this means:**
1. Track what the user is saying
2. When the transcript stops changing (user stopped talking)
3. Send the new part to the backend

**Processing a transcript:**
```typescript
const processOneTranscript = useCallback((text: string) => {
  console.info(`[VoicePipeline] Processing: "${text}"`);
  setState("processing");

  layer2Service.processTranscript(text)
    .then(() => {
      setState("ready");
    })
    .catch((err) => {
      setError(err.message);
      setState("ready");
    });
}, [layer2Service]);
```

**Handling the response:**
```typescript
const handleResponse = (response: Layer2Response) => {
  const voiceResponse = response.voice_response;

  // Add to conversation
  addMessage("ai", voiceResponse, "response");

  // Update the dashboard
  commandCenterBus.emit({
    type: "LAYOUT_UPDATE",
    layout: response.layout_json,
  });

  // Speak the response
  setState("speaking");
  speak(voiceResponse, {
    onEnd: () => {
      setState("listening");  // Ready for next question
    },
  });
};
```
**What this means:**
1. Get the voice response text
2. Add it to the conversation display
3. Tell the dashboard to update (broadcast LAYOUT_UPDATE)
4. Speak the response out loud
5. When done speaking, go back to listening

**VAD (Voice Activity Detection):**
```typescript
const { start: startVAD, isSpeaking: vadIsSpeaking } = useVAD({
  onSpeechEnd: () => {
    // User stopped talking - process immediately
    vadSpeechEndRef.current = true;
  },
});
```
**What this means:** VAD detects when someone stops speaking, so we can send the transcript immediately instead of waiting for a timeout.

---

#### `frontend/src/components/layer1/useSTT.ts`

**What it does:** Converts speech to text using a server or browser API.

```typescript
export function useSTT(deviceId?: string) {
  const [transcript, setTranscript] = useState("");
  const [interimTranscript, setInterimTranscript] = useState("");
  const [isListening, setIsListening] = useState(false);
```

**Server-based STT:**
```typescript
// Send audio to server for transcription
const transcribeAudio = async (audioData: Blob) => {
  const formData = new FormData();
  formData.append("audio", audioData);

  const response = await fetch(`${sttServerUrl}/transcribe`, {
    method: "POST",
    body: formData,
  });

  const result = await response.json();
  setTranscript(result.text);
};
```

**Browser fallback (Web Speech API):**
```typescript
// If server isn't available, use browser's built-in speech recognition
const recognition = new webkitSpeechRecognition();
recognition.continuous = true;
recognition.interimResults = true;

recognition.onresult = (event) => {
  let interim = "";
  let final = "";

  for (let i = event.resultIndex; i < event.results.length; i++) {
    if (event.results[i].isFinal) {
      final += event.results[i][0].transcript;
    } else {
      interim += event.results[i][0].transcript;
    }
  }

  setTranscript(prev => prev + final);
  setInterimTranscript(interim);
};
```
**What this means:** While you're speaking, we show "interim" results (might change). When you pause, we finalize them.

---

#### `frontend/src/components/layer1/useKokoroTTS.ts`

**What it does:** Converts text to speech using the Kokoro engine.

```typescript
export function useKokoroTTS() {
  const [isSpeaking, setIsSpeaking] = useState(false);

  const speak = useCallback(async (text: string, options?: TTSOptions) => {
    setIsSpeaking(true);

    try {
      // Request audio from TTS server
      const response = await fetch(`${ttsServerUrl}/synthesize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, voice: "NATF2" }),
      });

      // Play the audio
      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);

      audio.onended = () => {
        setIsSpeaking(false);
        options?.onEnd?.();
      };

      audio.play();
    } catch (error) {
      setIsSpeaking(false);
      options?.onError?.(error);
    }
  }, [ttsServerUrl]);

  return { speak, isSpeaking, stop };
}
```
**What this means:**
1. Send text to the TTS server
2. Get audio back
3. Play it through the browser
4. Call callbacks when done or on error

---

### Layer 3 Components

The layout system that arranges widgets on screen.

---

#### `frontend/src/components/layer3/Blob.tsx`

**What it does:** The main layout component that renders all widgets.

**In simple terms:** Like a grid where you can place different sized tiles (widgets).

```typescript
export default function Blob() {
  const { layout, pinnedKeys, dismissWidget, ... } = useLayoutState();
```
**What this means:** Get the current layout and functions to modify it.

**Sorting widgets:**
```typescript
const sortedWidgets = useMemo(() => {
  return layout.widgets.filter((w) => w.size !== "hidden");
}, [layout.widgets]);
```
**What this means:** Take all widgets from the layout, but hide any marked as "hidden."

**Rendering widgets:**
```typescript
return (
  <BlobGrid heading={layout.heading}>
    <AnimatePresence mode="popLayout">
      {sortedWidgets.map((instruction, index) => {
        const WidgetComponent = getWidgetComponent(instruction.scenario);

        return (
          <motion.div
            key={key}
            initial={{ opacity: 0, x: 60 }}    // Start invisible, to the right
            animate={{ opacity: 1, x: 0 }}      // Fade in, slide to position
            exit={{ opacity: 0, x: -60 }}       // Fade out, slide left
            className={sizeClasses(instruction.size, instruction.heightHint)}
          >
            <WidgetSlot scenario={instruction.scenario} size={instruction.size}>
              <WidgetComponent data={resolveWidgetData(instruction)} />
            </WidgetSlot>
          </motion.div>
        );
      })}
    </AnimatePresence>
  </BlobGrid>
);
```
**What this means:**
1. Create a grid container
2. For each widget instruction:
   - Get the React component for that widget type
   - Wrap it in an animation container
   - Apply size classes (hero, expanded, normal, compact)
   - Pass it the data to display

**Size classes:**
```typescript
function sizeClasses(size: string, heightHint?: WidgetHeightHint): string {
  switch (size) {
    case "hero":    return "col-span-12 row-span-4";  // Full width, 4 rows tall
    case "expanded": return "col-span-6";              // Half width
    case "normal":   return "col-span-4";              // Third width
    case "compact":  return "col-span-3";              // Quarter width
    default:         return "col-span-4";
  }
}
```
**What this means:** Maps size names to CSS grid classes. The grid has 12 columns.

---

#### `frontend/src/components/layer3/useLayoutState.ts`

**What it does:** Manages the state of the dashboard layout.

```typescript
export function useLayoutState() {
  const [layout, setLayout] = useState<LayoutJSON>(defaultLayout);
  const [pinnedKeys, setPinnedKeys] = useState<Set<string>>(new Set());
  const [focusedKey, setFocusedKey] = useState<string | null>(null);
```

**Listening for layout updates:**
```typescript
useEffect(() => {
  const unsubscribe = commandCenterBus.on("LAYOUT_UPDATE", (event) => {
    if (event.type === "LAYOUT_UPDATE") {
      // Merge new layout with pinned widgets
      const newWidgets = [...event.layout.widgets];

      // Keep pinned widgets from previous layout
      pinnedKeys.forEach((key) => {
        const pinned = layout.widgets.find((w) => widgetKey(w) === key);
        if (pinned && !newWidgets.some((w) => widgetKey(w) === key)) {
          newWidgets.push(pinned);
        }
      });

      setLayout({ ...event.layout, widgets: newWidgets });
    }
  });

  return unsubscribe;
}, [layout, pinnedKeys]);
```
**What this means:** When a new layout arrives:
1. Start with the new widgets
2. Add back any widgets the user "pinned" (wanted to keep)
3. Update the state

**Widget actions:**
```typescript
const pinWidget = (key: string) => {
  setPinnedKeys((prev) => {
    const next = new Set(prev);
    if (next.has(key)) next.delete(key);  // Toggle off
    else next.add(key);                    // Toggle on
    return next;
  });
};

const dismissWidget = (key: string) => {
  setLayout((prev) => ({
    ...prev,
    widgets: prev.widgets.map((w) =>
      widgetKey(w) === key ? { ...w, size: "hidden" } : w
    ),
  }));
  setPinnedKeys((prev) => {
    const next = new Set(prev);
    next.delete(key);
    return next;
  });
};
```
**What this means:**
- **Pin**: Keep this widget even when the layout changes
- **Dismiss**: Hide this widget

---

#### `frontend/src/components/layer3/WidgetSlot.tsx`

**What it does:** A wrapper component that provides consistent styling and behavior for all widgets.

**In simple terms:** Think of it as a "picture frame" that every widget sits inside - it provides the border, title, toolbar, and error handling.

**Error Boundary:**
```typescript
class WidgetErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="h-full flex items-center justify-center bg-red-950/30 border border-red-900/50 rounded-xl p-4">
          <p className="text-xs text-red-400/70 font-mono">
            {this.props.scenario}: {this.state.error?.message}
          </p>
        </div>
      );
    }
    return this.props.children;
  }
}
```
**What this means:** If a widget crashes (bad data, coding error), instead of crashing the whole dashboard, it shows a red error box just for that widget. The rest of the dashboard keeps working.

**Size Classes:**
```typescript
const SIZE_CLASSES: Record<WidgetSize, string> = {
  hero: "col-span-12 row-span-2",    // Full width (12/12 columns), tall
  expanded: "col-span-6",             // Half width (6/12 columns)
  normal: "col-span-4",               // Third width (4/12 columns)
  compact: "col-span-3",              // Quarter width (3/12 columns)
  hidden: "hidden",                   // Not shown
};
```
**What this means:** The dashboard uses a 12-column grid (like a newspaper). Each size takes different number of columns.

**The Slot Component:**
```typescript
export default function WidgetSlot({
  scenario,      // Widget type ("kpi", "trend", etc.)
  size,          // How big to display it
  children,      // The actual widget content
  title,         // Label shown on hover
  description,   // Text at the bottom
  onPin,         // Callback when user pins widget
  onDismiss,     // Callback when user dismisses widget
  onDrillDown,   // Callback when user clicks for more detail
  ...
}: WidgetSlotProps) {
  return (
    <div className="relative h-full w-full group rounded-xl border border-neutral-700/50 bg-neutral-900/80">
      {/* Title - appears on hover */}
      {title && (
        <div className="absolute top-2 left-2 opacity-0 group-hover:opacity-100">
          <span className="text-[11px]">{title}</span>
        </div>
      )}

      {/* Toolbar - pin, resize, dismiss buttons */}
      {hasToolbar && <WidgetToolbar ... />}

      {/* Widget content with error handling */}
      <div className="flex-1" onClick={handleBodyClick}>
        <WidgetErrorBoundary scenario={scenario}>
          <Suspense fallback={<WidgetSkeleton />}>
            {children}
          </Suspense>
        </WidgetErrorBoundary>
      </div>

      {/* Description footer */}
      {description && (
        <div className="px-3 py-1.5 border-t">
          <p className="text-[11px] text-neutral-400">{description}</p>
        </div>
      )}
    </div>
  );
}
```
**What this means:** Every widget gets:
- A semi-transparent dark background with rounded corners
- Title on hover (top-left)
- Toolbar with pin/resize/dismiss buttons (top-right)
- Error handling (shows error message if widget crashes)
- Loading skeleton while content loads
- Description footer (optional)

---

### Layer 4 Components

The actual widget components that display data. There are 21+ different widget types, each designed for specific data visualization needs.

---

#### `frontend/src/components/layer4/widgetRegistry.ts`

**What it does:** Maps widget scenario names to their React components.

```typescript
const registry: Record<string, React.FC<WidgetProps>> = {
  "kpi": KPIWidget,
  "trend": TrendWidget,
  "alerts": AlertsWidget,
  "comparison": ComparisonWidget,
  "flow-sankey": FlowSankeyWidget,
  // ... more widgets
};

export function getWidgetComponent(scenario: string): React.FC<WidgetProps> | null {
  return registry[scenario] || null;
}
```
**What this means:** A lookup table to find the right component for each widget type.

---

#### `frontend/src/components/layer4/widgets/kpi.tsx`

**What it does:** Displays a single metric (Key Performance Indicator).

**In simple terms:** A box showing one number with a label - like "Temperature: 72°F" or "Pump Status: Running."

**Design Tokens:**
```typescript
const COLORS = {
  success: '#16a34a',    // Green for good values
  warning: '#d97706',    // Orange for caution
  critical: '#ef4444',   // Red for problems
  accent: '#2563eb',     // Blue for neutral highlights
};
```

**The KPI Renderer:**
```typescript
const KpiRenderer: React.FC<KpiRendererProps> = ({ spec }) => {
  const { layout, visual, demoData, variant } = spec;

  // Helper to determine text colors based on state
  const valueColor = demoData.state === 'critical' ? 'text-red-600' :
                     demoData.state === 'warning' ? 'text-amber-600' :
                     'text-neutral-900';

  return (
    <div className="p-4 rounded-lg border flex flex-col justify-between h-full">
      {/* Label - what this metric is */}
      <div className="flex justify-between items-start">
        <span className="text-[10px] uppercase tracking-widest text-neutral-400">
          {demoData.label}
        </span>
        {/* Show warning icon if state is warning or critical */}
        {demoData.state === 'critical' && <AlertTriangle className="text-red-500" />}
      </div>

      {/* Value - the actual number */}
      <div className="flex items-baseline gap-1.5">
        <span className={`text-xl font-bold ${valueColor}`}>
          {demoData.value}
        </span>
        <span className="text-xs text-neutral-400">{demoData.unit}</span>
      </div>

      {/* Progress bar for lifecycle KPIs */}
      {variant === 'KPI_LIFECYCLE' && demoData.max && (
        <div className="mt-1.5 w-full bg-neutral-100 rounded-full h-1">
          <div
            className="h-full rounded-full bg-blue-600"
            style={{ width: `${(demoData.value / demoData.max) * 100}%` }}
          />
        </div>
      )}
    </div>
  );
};
```
**What this means:** Shows a compact card with:
- Label at top (e.g., "Temperature")
- Large value in the middle (e.g., "72")
- Unit next to value (e.g., "°F")
- Color changes based on state (red for critical, orange for warning)
- Optional progress bar for showing percentage values

---

#### `frontend/src/components/layer4/widgets/trend.tsx`

**What it does:** Displays a time-series chart showing how values change over time.

**In simple terms:** A line graph like "Power consumption over the last 24 hours."

**Using Recharts library:**
```typescript
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
```
**What this means:** We use the Recharts library to draw professional-looking charts.

**Chart Rendering:**
```typescript
const TrendRenderer: React.FC<TrendRendererProps> = ({ spec }) => {
  const { demoData, variant, representation } = spec;
  const mainColor = visual.colors?.[0] || '#2563eb';

  // Different chart types based on representation
  if (representation === 'Area') {
    return (
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={demoData.timeSeries}>
          <defs>
            <linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={mainColor} stopOpacity={0.3}/>
              <stop offset="95%" stopColor={mainColor} stopOpacity={0}/>
            </linearGradient>
          </defs>
          <XAxis dataKey="time" />
          <YAxis />
          <Tooltip />
          <Area type="monotone" dataKey="value" stroke={mainColor} fill="url(#grad)" />
        </AreaChart>
      </ResponsiveContainer>
    );
  }

  // Default: Line chart
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={demoData.timeSeries}>
        <XAxis dataKey="time" />
        <YAxis />
        <Tooltip />
        <Line type="monotone" dataKey="value" stroke={mainColor} dot={false} />
        {/* Critical threshold line for alerts */}
        {variant === 'TREND_ALERT_CONTEXT' && (
          <ReferenceLine y={85} stroke="#ef4444" strokeDasharray="3 3" label="Crit" />
        )}
      </LineChart>
    </ResponsiveContainer>
  );
};
```
**What this means:** The chart:
- Automatically sizes to fit its container
- Shows time on X-axis, values on Y-axis
- Can be a line chart or filled area chart
- Shows a tooltip when you hover over data points
- Can show a critical threshold line (red dashed) if configured

**Live Indicator:**
```typescript
{variant === 'TREND_LIVE' && (
  <div className="absolute top-0 right-0 flex items-center gap-1.5 bg-red-500/10 px-2 py-0.5 rounded-full">
    <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse"></span>
    <span className="text-[9px] font-bold text-red-500">LIVE</span>
  </div>
)}
```
**What this means:** For live-updating charts, shows a pulsing red "LIVE" badge.

---

#### `frontend/src/components/layer4/widgets/alerts.tsx`

**What it does:** Displays alert notifications with severity levels and actions.

**In simple terms:** A list of problems or warnings like "Pump 3 temperature high - needs attention."

**Severity Configuration:**
```typescript
const SEVERITY_CONFIG = {
  critical: {
    color: 'text-red-600',
    bg: 'bg-red-50',
    border: 'border-red-500',
    icon: XCircle
  },
  warning: {
    color: 'text-amber-600',
    bg: 'bg-amber-50',
    border: 'border-amber-500',
    icon: AlertTriangle
  },
  low: {
    color: 'text-blue-600',
    bg: 'bg-blue-50',
    border: 'border-blue-500',
    icon: Info
  },
  success: {
    color: 'text-green-600',
    bg: 'bg-green-50',
    border: 'border-green-500',
    icon: CheckCircle2
  },
};
```
**What this means:** Different severity levels get different colors and icons:
- Critical = Red with X icon
- Warning = Orange with triangle icon
- Low = Blue with info icon
- Success = Green with checkmark

**State Configuration:**
```typescript
const STATE_CONFIG = {
  new: { icon: ZapIcon, label: 'New', style: 'text-blue-600 bg-blue-50' },
  acknowledged: { icon: Check, label: 'Ack', style: 'text-neutral-600 bg-neutral-100' },
  in_progress: { icon: PlayCircle, label: 'Active', style: 'text-purple-600 bg-purple-50' },
  resolved: { icon: CheckCircle2, label: 'Resolved', style: 'text-green-600 bg-green-50' },
  escalated: { icon: ArrowUpCircle, label: 'Escalated', style: 'text-orange-600 bg-orange-50' },
};
```
**What this means:** Alerts have workflow states showing where they are in the resolution process.

**Multiple Display Variants:**
The alert component can display in several ways:
- **Badge**: Compact pill showing severity and count
- **Toast**: Popup notification that slides in
- **Banner**: Full-width inline message
- **Modal**: Blocking popup for critical alerts
- **Card**: Full details with actions (default)

**The Card Variant (default):**
```typescript
return (
  <div className="bg-white rounded-xl border p-4 hover:shadow-md transition-all">
    {/* Severity Strip on left edge */}
    <div className={`absolute left-0 top-0 bottom-0 w-1 ${severityCfg.accent}`} />

    {/* Header: Category, Source, State Badge, Time */}
    <div className="flex items-center justify-between">
      <span className="text-[10px] uppercase">{data.category} • {data.source}</span>
      <span className="text-xs text-neutral-400">{formatTime(data.timestamp)}</span>
    </div>

    {/* Content: Title, Message, Evidence */}
    <h4 className="text-sm font-bold">{data.title}</h4>
    <p className="text-xs text-neutral-500">{data.message}</p>

    {/* Primary Evidence (sensor reading that triggered alert) */}
    {data.evidence && (
      <div className="bg-neutral-50 rounded px-2 py-1.5">
        <span className="text-[9px]">{data.evidence.label}</span>
        <span className="text-sm font-mono">{data.evidence.value}{data.evidence.unit}</span>
      </div>
    )}

    {/* Action Buttons (shown on hover) */}
    <div className="opacity-0 group-hover:opacity-100 transition-all">
      {data.actions.map((action) => (
        <button onClick={() => onAction?.(data.id, action.intent)}>
          {action.label}
        </button>
      ))}
    </div>
  </div>
);
```
**What this means:** Each alert card shows:
- Color-coded severity strip on the left edge
- Category and source (e.g., "HVAC • Chiller-1")
- State badge (New, Acknowledged, etc.)
- Time since alert (e.g., "5m ago")
- Title and detailed message
- Evidence that triggered the alert (e.g., "Temperature: 92°C")
- Action buttons on hover (Acknowledge, View Details, etc.)

---

#### `frontend/src/components/layer4/fixtureData.ts`

**What it does:** Demo data and metadata for each widget type.

```typescript
export const FIXTURES: Record<string, FixtureMeta> = {
  "kpi": {
    name: "KPI",
    icon: "gauge",
    description: "Single metric display",
    defaultFixture: "kpi_live-standard",
    variants: {
      "kpi_live-standard": {
        demoData: { label: "Temperature", value: 72, unit: "°F", state: "normal" }
      },
      "kpi_live-alert-critical": {
        demoData: { label: "Pressure", value: 150, unit: "psi", state: "critical" }
      },
      // more variants
    }
  },
  "trend": {
    name: "Trend Chart",
    variants: {
      "trend_line-standard": {
        demoData: {
          label: "Power Consumption",
          unit: "kW",
          data: [
            { time: "08:00", value: 120 },
            { time: "09:00", value: 135 },
            // more data points
          ]
        }
      }
    }
  },
  // more widget types
};
```
**What this means:** Each widget type has:
- Display name and icon
- Multiple visual variants (styles)
- Demo data for each variant (used when real data isn't available)

---

## Scripts and Automation

Helper scripts for development and deployment.

---

#### `scripts/dev.sh`

**What it does:** Starts all development servers with one command.

**In simple terms:** Instead of opening 4 terminal windows and typing 4 commands, run one script.

```bash
# Start Django backend
cd backend && source venv/bin/activate && python manage.py runserver 0.0.0.0:8100 &

# Start Next.js frontend
cd frontend && npm run dev &

# Start STT server
python stt/server.py &

# Start TTS server
python tts/server.py &
```

**Usage:**
```bash
./scripts/dev.sh         # Start everything
./scripts/dev.sh --setup # Install dependencies first
```

---

#### `scripts/deploy.sh`

**What it does:** Deploys the system for production use.

```bash
# Build the frontend
cd frontend && npm run build

# Install systemd services
cp scripts/systemd/*.service ~/.config/systemd/user/
systemctl --user daemon-reload

# Start services
systemctl --user start cc-backend cc-frontend cc-stt cc-tts
```
**What this means:** Creates proper "services" that start automatically and stay running.

---

## Tests

Automated tests to make sure everything works correctly.

---

#### `tests/test_intent_parser.py`

**What it does:** Tests that the intent parser correctly understands different types of questions.

```python
def test_query_intent():
    parser = IntentParser()
    result = parser.parse("What is the pump status?")
    assert result.type == "query"
    assert "industrial" in result.domains

def test_greeting_intent():
    result = parser.parse("Hello")
    assert result.type == "greeting" or result.type == "conversation"

def test_out_of_scope():
    result = parser.parse("What's the weather like?")
    assert result.type == "out_of_scope"
```

---

#### `tests/test_widget_selector.py`

**What it does:** Tests that the widget selector chooses appropriate widgets.

```python
def test_hero_first():
    """First widget should be hero-sized."""
    selector = WidgetSelector()
    intent = ParsedIntent(type="query", domains=["industrial"], raw_text="Show pump trends")
    plan = selector.select(intent)

    assert plan.widgets[0].size == "hero"

def test_valid_scenarios():
    """All selected scenarios should be valid."""
    plan = selector.select(intent)
    for widget in plan.widgets:
        assert widget.scenario in VALID_SCENARIOS
```

---

#### `tests/ai_accuracy_test.py`

**What it does:** Measures how accurate the AI is at understanding questions and selecting widgets.

```python
TEST_CASES = [
    {
        "query": "What's the status of pump 3?",
        "expected_domains": ["industrial"],
        "expected_scenarios": ["kpi", "trend"],
    },
    {
        "query": "Show me energy consumption trends",
        "expected_domains": ["industrial"],
        "expected_scenarios": ["trend", "trends-cumulative"],
    },
]

def test_accuracy():
    correct = 0
    total = len(TEST_CASES)

    for case in TEST_CASES:
        result = orchestrator.process_transcript(case["query"])
        if matches_expected(result, case):
            correct += 1

    accuracy = correct / total
    assert accuracy >= 0.90, f"Accuracy {accuracy:.1%} below 90% threshold"
```

---

## Summary

Command Center is a voice-controlled industrial dashboard with these key parts:

1. **Voice Pipeline**: Converts speech to text, processes it, and speaks responses
2. **Intent Parser**: Understands what the user is asking for
3. **Widget Selector**: Chooses which visualizations to show
4. **RAG Pipeline**: Searches databases for relevant information
5. **Blob Layout**: Arranges widgets on screen with animations
6. **Widget Components**: Actually display the charts, numbers, and data

The system learns over time through reinforcement learning - user feedback (thumbs up/down) helps it make better choices in the future.

All the pieces communicate through:
- **Backend API**: REST endpoints for the frontend to call
- **Event Bus**: Frontend components broadcasting messages to each other
- **Database**: Persistent storage for equipment data, sessions, and feedback

---

*This documentation was generated by analyzing every file in the Command Center codebase.*
