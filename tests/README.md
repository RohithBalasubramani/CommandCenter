# Command Center Test Suite

Comprehensive test suite for the Command Center industrial operations platform.

## Test Structure

```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── pytest.ini               # Pytest settings
├── requirements-test.txt    # Test dependencies
├── README.md                # This file
│
├── test_intent_parser.py    # Intent parser unit tests
├── test_rag_pipeline.py     # RAG pipeline unit tests
├── test_api_endpoints.py    # API endpoint integration tests
├── test_models.py           # Django model tests
│
├── ai_accuracy_test.py      # Standalone AI accuracy tests
└── ai_speed_test.py         # Standalone performance tests
```

## Quick Start

### Install Dependencies

```bash
cd tests
pip install -r requirements-test.txt
```

### Run All Pytest Tests

```bash
# From project root
cd backend
pytest ../tests/ -v

# Or from tests directory
cd tests
pytest -v
```

### Run Specific Test Files

```bash
pytest test_intent_parser.py -v
pytest test_rag_pipeline.py -v
pytest test_api_endpoints.py -v
pytest test_models.py -v
```

### Run Standalone Tests

```bash
# AI Accuracy Tests
python ai_accuracy_test.py

# AI Performance Tests
python ai_speed_test.py

# Run both
python ai_accuracy_test.py && python ai_speed_test.py
```

## Test Categories

### 1. Intent Parser Tests (`test_intent_parser.py`)
Tests the Layer 2 intent parser regex fallback:
- Intent type detection (query, action, greeting, etc.)
- Domain detection (industrial, alerts, people, supply, tasks)
- Entity extraction (devices, numbers, time references)
- Characteristic detection (trend, comparison, distribution)
- Out-of-scope query handling
- Confidence scoring

### 2. RAG Pipeline Tests (`test_rag_pipeline.py`)
Tests the RAG (Retrieval Augmented Generation) system:
- Document storage and retrieval
- Embedding service
- Vector search accuracy
- LLM service integration
- Pipeline statistics

### 3. API Endpoint Tests (`test_api_endpoints.py`)
Tests the Django REST API endpoints:
- `/api/layer2/orchestrate/` - Main orchestration
- `/api/layer2/filler/` - Filler text generation
- `/api/layer2/rag/industrial/` - Industrial RAG queries
- `/api/layer2/rag/industrial/health/` - Health checks
- `/api/layer2/proactive/` - Proactive triggers

### 4. Model Tests (`test_models.py`)
Tests Django ORM models:
- Layer 2 models (RAGPipeline, RAGQuery, RAGResult, UserMemory)
- Industrial equipment models (Transformer, Pump, Chiller, etc.)
- Alert and maintenance models

### 5. AI Accuracy Tests (`ai_accuracy_test.py`)
Standalone accuracy benchmark:
- Intent classification accuracy
- Domain detection accuracy
- Entity extraction accuracy
- Characteristic detection accuracy
- RAG retrieval accuracy
- Out-of-scope detection

### 6. AI Performance Tests (`ai_speed_test.py`)
Standalone performance benchmark:
- Intent parser response time
- RAG vector search speed
- Embedding generation speed
- Database query speed
- Concurrent request handling
- Memory usage under load
- LLM response time

## Test Fixtures

Common fixtures available in `conftest.py`:

```python
@pytest.fixture
def api_client():
    """Django REST framework test client."""

@pytest.fixture
def sample_transcript():
    """Sample voice transcript for testing."""

@pytest.fixture
def mock_ollama():
    """Mock Ollama LLM service."""

@pytest.fixture
def mock_chromadb():
    """Mock ChromaDB for tests without vector store."""

@pytest.fixture
def sample_equipment_data():
    """Sample equipment data for testing."""
```

## Running Tests with Coverage

```bash
pytest --cov=layer2 --cov=industrial --cov-report=html
```

## Pass/Fail Criteria

### Accuracy Tests
| Test | Pass Threshold |
|------|----------------|
| Intent Classification | >= 70% accuracy |
| Domain Detection | >= 60% accuracy |
| Entity Extraction | >= 50% accuracy |
| Characteristic Detection | >= 60% accuracy |
| Out-of-Scope Detection | >= 60% accuracy |

### Performance Tests
| Test | Pass Threshold |
|------|----------------|
| Intent Parser | < 10ms average |
| RAG Search | < 200ms average |
| Embedding | < 100ms average |
| Database Query | < 50ms average |
| Concurrent (10 workers) | > 100 req/sec |
| Memory (500 ops) | < 50MB used |
| LLM Response | < 5000ms average |

## Test Reports

Standalone tests generate JSON reports:
- `cc_accuracy_report_YYYYMMDD_HHMMSS.json`
- `cc_speed_report_YYYYMMDD_HHMMSS.json`

## Continuous Integration

Add to your CI pipeline:

```yaml
# GitHub Actions example
test:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install -r backend/requirements.txt
        pip install -r tests/requirements-test.txt
    - name: Run tests
      run: |
        cd backend
        pytest ../tests/ -v --tb=short
```

## Troubleshooting

### Import Errors
Ensure you're running from the correct directory:
```bash
cd backend
pytest ../tests/ -v
```

### Database Errors
Run migrations first:
```bash
cd backend
python manage.py migrate
```

### LLM Tests Skipped
If Ollama is not running, LLM tests will pass with a warning.
Start Ollama to run full LLM tests:
```bash
ollama serve
```

### ChromaDB Errors
Ensure ChromaDB is installed:
```bash
pip install chromadb
```

## Adding New Tests

1. Create test file in `tests/` directory
2. Use `Test` prefix for classes, `test_` prefix for functions
3. Import fixtures from `conftest.py`
4. Use `@pytest.mark.django_db` for database tests

Example:
```python
import pytest

class TestNewFeature:
    @pytest.mark.django_db
    def test_feature_works(self, api_client):
        response = api_client.get("/api/new-endpoint/")
        assert response.status_code == 200
```
