#!/bin/bash
#
# Command Center - Setup Script
#
# Initializes the full Command Center environment:
# - Backend database migrations
# - Industrial equipment data population
# - RAG index creation
# - Frontend dependencies
#
# Usage:
#   ./scripts/setup.sh              # Full setup
#   ./scripts/setup.sh --backend    # Backend only
#   ./scripts/setup.sh --frontend   # Frontend only
#   ./scripts/setup.sh --rag        # RAG indexing only
#   ./scripts/setup.sh --reset      # Reset and rebuild everything
#

set -e

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                             ║
# ╚════════════════════════════════════════════════════════════════════════════╝

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"
PERSONAPLEX_DIR="$PROJECT_DIR/personaplex"

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  COLORS                                                                    ║
# ╚════════════════════════════════════════════════════════════════════════════╝

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
DIM='\033[0;90m'
NC='\033[0m'
BOLD='\033[1m'

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  LOGGING                                                                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

log() {
    local level="$1"
    local message="$2"
    local color="$NC"

    case "$level" in
        INFO)    color="$GREEN" ;;
        WARN)    color="$YELLOW" ;;
        ERROR)   color="$RED" ;;
        STEP)    color="$CYAN" ;;
        SUCCESS) color="$GREEN" ;;
    esac

    echo -e "${color}[$level]${NC} $message"
}

print_banner() {
    echo -e "${BOLD}${MAGENTA}"
    echo "    +================================================================+"
    echo "    |                                                                |"
    echo "    |      COMMAND CENTER - SETUP                                    |"
    echo "    |                                                                |"
    echo "    |      Initializing Layer 1 (Voice) + Layer 2 (AI + RAG)        |"
    echo "    |                                                                |"
    echo "    +================================================================+"
    echo -e "${NC}"
}

print_section() {
    local title="$1"
    echo ""
    echo -e "${BOLD}${BLUE}=== $title ===${NC}"
    echo ""
}

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  SETUP FUNCTIONS                                                           ║
# ╚════════════════════════════════════════════════════════════════════════════╝

check_python() {
    if command -v python3 &> /dev/null; then
        log "INFO" "Python3: $(python3 --version)"
        return 0
    else
        log "ERROR" "Python3 not found!"
        return 1
    fi
}

check_node() {
    if command -v node &> /dev/null; then
        log "INFO" "Node.js: $(node --version)"
        return 0
    else
        log "ERROR" "Node.js not found!"
        return 1
    fi
}

check_ollama() {
    if command -v ollama &> /dev/null; then
        log "INFO" "Ollama: installed"
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            log "INFO" "Ollama server: running"
            return 0
        else
            log "WARN" "Ollama server not running. Start with: ollama serve"
            return 1
        fi
    else
        log "WARN" "Ollama not installed. RAG will use fallback responses."
        log "INFO" "Install Ollama: curl -fsSL https://ollama.com/install.sh | sh"
        return 1
    fi
}

setup_backend_venv() {
    print_section "Backend Virtual Environment"

    cd "$BACKEND_DIR"

    if [ -d "venv" ]; then
        log "INFO" "Virtual environment exists"
        source venv/bin/activate
    elif [ -d ".venv" ]; then
        log "INFO" "Virtual environment exists (.venv)"
        source .venv/bin/activate
    else
        log "STEP" "Creating virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
        log "SUCCESS" "Virtual environment created"
    fi

    # Upgrade pip
    log "STEP" "Upgrading pip..."
    pip install --upgrade pip -q

    # Install Django dependencies
    log "STEP" "Installing Django dependencies..."
    pip install django djangorestframework django-cors-headers -q

    # Install RAG dependencies
    log "STEP" "Installing RAG dependencies (chromadb, sentence-transformers)..."
    pip install chromadb sentence-transformers requests -q

    log "SUCCESS" "Backend dependencies installed"
}

setup_backend_database() {
    print_section "Database Setup"

    cd "$BACKEND_DIR"

    # Activate venv
    if [ -d "venv" ]; then
        source venv/bin/activate
    elif [ -d ".venv" ]; then
        source .venv/bin/activate
    fi

    # Run migrations
    log "STEP" "Running database migrations..."
    python manage.py migrate --run-syncdb

    log "SUCCESS" "Database migrations complete"
}

populate_industrial_data() {
    print_section "Industrial Equipment Data"

    cd "$BACKEND_DIR"

    # Activate venv
    if [ -d "venv" ]; then
        source venv/bin/activate
    elif [ -d ".venv" ]; then
        source .venv/bin/activate
    fi

    # Clear and repopulate - ensures idempotent setup
    log "STEP" "Populating industrial equipment database..."
    log "INFO" "  Creating 500+ equipment records (transformers, generators, chillers, etc.)"

    python manage.py populate_industrial_db --clear

    log "SUCCESS" "Industrial data populated"
}

index_rag_data() {
    print_section "RAG Index"

    cd "$BACKEND_DIR"

    # Activate venv
    if [ -d "venv" ]; then
        source venv/bin/activate
    elif [ -d ".venv" ]; then
        source .venv/bin/activate
    fi

    log "STEP" "Indexing equipment data for RAG retrieval..."
    log "INFO" "  This may take a few minutes on first run (downloading embedding model)"

    python manage.py index_rag --clear

    log "SUCCESS" "RAG index created"

    # Show stats
    log "INFO" "RAG Index Statistics:"
    python manage.py index_rag --stats
}

setup_frontend() {
    print_section "Frontend Setup"

    cd "$FRONTEND_DIR"

    if [ -d "node_modules" ]; then
        log "INFO" "node_modules exists"
    else
        log "STEP" "Installing npm dependencies..."
        npm install
        log "SUCCESS" "Frontend dependencies installed"
    fi
}

setup_ollama_model() {
    print_section "LLM Setup (Ollama)"

    if ! command -v ollama &> /dev/null; then
        log "WARN" "Ollama not installed - skipping LLM setup"
        log "INFO" "Install Ollama: curl -fsSL https://ollama.com/install.sh | sh"
        return
    fi

    # Check if model exists
    if ollama list 2>/dev/null | grep -q "qwen3:7b\|phi4\|llama3"; then
        log "INFO" "LLM model already available"
        ollama list 2>/dev/null | grep -E "qwen3|phi4|llama3" | head -3
    else
        log "STEP" "Pulling Qwen3-7B model (recommended for RAG)..."
        log "INFO" "  This will download ~4GB"
        ollama pull qwen3:7b || {
            log "WARN" "Failed to pull qwen3:7b, trying phi4..."
            ollama pull phi4 || log "WARN" "Could not pull LLM model"
        }
    fi
}

reset_all() {
    print_section "Reset"

    cd "$BACKEND_DIR"

    log "WARN" "This will delete all data and recreate from scratch!"
    read -p "Are you sure? (y/N) " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        log "INFO" "Cancelled"
        return
    fi

    # Activate venv
    if [ -d "venv" ]; then
        source venv/bin/activate
    elif [ -d ".venv" ]; then
        source .venv/bin/activate
    fi

    # Delete database
    log "STEP" "Removing database..."
    rm -f db.sqlite3

    # Delete RAG index
    log "STEP" "Removing RAG index..."
    rm -rf chroma_db

    # Delete migrations (keep __init__.py)
    log "STEP" "Removing migrations..."
    find . -path "*/migrations/*.py" -not -name "__init__.py" -delete 2>/dev/null || true
    find . -path "*/migrations/*.pyc" -delete 2>/dev/null || true

    log "SUCCESS" "Reset complete"
}

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                      ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# Parse arguments
DO_BACKEND=false
DO_FRONTEND=false
DO_RAG=false
DO_RESET=false
DO_ALL=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --backend)
            DO_BACKEND=true
            DO_ALL=false
            shift
            ;;
        --frontend)
            DO_FRONTEND=true
            DO_ALL=false
            shift
            ;;
        --rag)
            DO_RAG=true
            DO_ALL=false
            shift
            ;;
        --reset)
            DO_RESET=true
            DO_ALL=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --backend    Backend setup only (venv, migrations, data)"
            echo "  --frontend   Frontend setup only (npm install)"
            echo "  --rag        RAG indexing only"
            echo "  --reset      Reset and rebuild everything"
            echo "  --help       Show this help"
            echo ""
            echo "Without options, runs full setup."
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# Print banner
print_banner

# Check prerequisites
print_section "Prerequisites"
check_python
check_node
check_ollama || true  # Don't fail if ollama not available

# Run setup based on flags
if [ "$DO_RESET" = true ]; then
    reset_all
    DO_ALL=true  # After reset, do full setup
fi

if [ "$DO_ALL" = true ]; then
    setup_backend_venv
    setup_backend_database
    populate_industrial_data
    index_rag_data
    setup_frontend
    setup_ollama_model || true
elif [ "$DO_BACKEND" = true ]; then
    setup_backend_venv
    setup_backend_database
    populate_industrial_data
elif [ "$DO_FRONTEND" = true ]; then
    setup_frontend
elif [ "$DO_RAG" = true ]; then
    index_rag_data
fi

# Final summary
print_section "Setup Complete"
echo -e "${GREEN}"
echo "    +------------------------------------------------------------------+"
echo "    |                                                                  |"
echo "    |    COMMAND CENTER READY                                          |"
echo "    |                                                                  |"
echo "    |    To start the development server:                              |"
echo "    |                                                                  |"
echo "    |        ./scripts/dev.sh                                          |"
echo "    |                                                                  |"
echo "    |    Or start components individually:                             |"
echo "    |                                                                  |"
echo "    |        Backend:  cd backend && source venv/bin/activate          |"
echo "    |                  python manage.py runserver 8100                 |"
echo "    |                                                                  |"
echo "    |        Frontend: cd frontend && npm run dev                      |"
echo "    |                                                                  |"
echo "    |        Ollama:   ollama serve  (for LLM responses)               |"
echo "    |                                                                  |"
echo "    +------------------------------------------------------------------+"
echo -e "${NC}"
