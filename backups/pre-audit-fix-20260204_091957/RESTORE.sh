#!/bin/bash
# Restore script for pre-audit-fix backup
# Created: 2026-02-04
#
# Usage: ./RESTORE.sh [--full]
#   --full: Restore from full backup (backend_full, frontend_full)
#   default: Restore from individual file backups

BACKUP_DIR="$(dirname "$0")"
PROJECT_ROOT="$(dirname "$(dirname "$BACKUP_DIR")")"

echo "=== RESTORE SCRIPT ==="
echo "Backup location: $BACKUP_DIR"
echo "Project root: $PROJECT_ROOT"
echo ""

# Check for full restore mode
FULL_MODE=false
if [[ "$1" == "--full" ]]; then
    FULL_MODE=true
    echo "Mode: FULL RESTORE"
else
    echo "Mode: Individual files (use --full for complete restore)"
fi
echo ""

read -p "This will overwrite current files with backup versions. Continue? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Restore cancelled."
    exit 1
fi

echo "Restoring files..."

if $FULL_MODE; then
    # Full restore from rsync backups
    echo "Restoring full backend..."
    rsync -av --exclude='__pycache__' "$BACKUP_DIR/backend_full/" "$PROJECT_ROOT/backend/"

    echo "Restoring full frontend..."
    rsync -av "$BACKUP_DIR/frontend_full/" "$PROJECT_ROOT/frontend/"
else
    # Individual file restore
    echo "Restoring Backend Layer 1..."
    cp "$BACKUP_DIR/backend/layer1/models.py" "$PROJECT_ROOT/backend/layer1/" 2>/dev/null || true
    cp "$BACKUP_DIR/backend/layer1/serializers.py" "$PROJECT_ROOT/backend/layer1/" 2>/dev/null || true
    cp "$BACKUP_DIR/backend/layer1/views.py" "$PROJECT_ROOT/backend/layer1/" 2>/dev/null || true
    cp "$BACKUP_DIR/backend/layer1/urls.py" "$PROJECT_ROOT/backend/layer1/" 2>/dev/null || true
    cp "$BACKUP_DIR/backend/layer1/tests.py" "$PROJECT_ROOT/backend/layer1/" 2>/dev/null || true

    echo "Restoring Backend Layer 2..."
    cp "$BACKUP_DIR/backend/layer2/widget_selector.py" "$PROJECT_ROOT/backend/layer2/"
    cp "$BACKUP_DIR/backend/layer2/data_collector.py" "$PROJECT_ROOT/backend/layer2/"
    cp "$BACKUP_DIR/backend/layer2/orchestrator.py" "$PROJECT_ROOT/backend/layer2/"
    cp "$BACKUP_DIR/backend/layer2/tests.py" "$PROJECT_ROOT/backend/layer2/"
    cp "$BACKUP_DIR/backend/layer2/intent_parser.py" "$PROJECT_ROOT/backend/layer2/"
    cp "$BACKUP_DIR/backend/layer2/rag_pipeline.py" "$PROJECT_ROOT/backend/layer2/"
    cp "$BACKUP_DIR/backend/layer2/widget_catalog.py" "$PROJECT_ROOT/backend/layer2/"
    cp "$BACKUP_DIR/backend/layer2/widget_schemas.py" "$PROJECT_ROOT/backend/layer2/"
    cp "$BACKUP_DIR/backend/layer2/views.py" "$PROJECT_ROOT/backend/layer2/"

    echo "Restoring Backend STT..."
    cp "$BACKUP_DIR/backend/stt/server.py" "$PROJECT_ROOT/backend/stt/"

    echo "Restoring Frontend Layer 1..."
    cp "$BACKUP_DIR/frontend/src/components/layer1/"*.tsx "$PROJECT_ROOT/frontend/src/components/layer1/"
    cp "$BACKUP_DIR/frontend/src/components/layer1/"*.ts "$PROJECT_ROOT/frontend/src/components/layer1/"

    echo "Restoring Frontend Layer 3..."
    cp "$BACKUP_DIR/frontend/src/components/layer3/"*.tsx "$PROJECT_ROOT/frontend/src/components/layer3/"
    cp "$BACKUP_DIR/frontend/src/components/layer3/"*.ts "$PROJECT_ROOT/frontend/src/components/layer3/"

    echo "Restoring Frontend Layer 4..."
    cp "$BACKUP_DIR/frontend/src/components/layer4/"*.ts "$PROJECT_ROOT/frontend/src/components/layer4/"
    cp -r "$BACKUP_DIR/frontend/src/components/layer4/widgets" "$PROJECT_ROOT/frontend/src/components/layer4/"

    echo "Restoring Frontend Lib..."
    cp "$BACKUP_DIR/frontend/src/lib/"*.ts "$PROJECT_ROOT/frontend/src/lib/"
    cp "$BACKUP_DIR/frontend/src/lib/layer2/"*.ts "$PROJECT_ROOT/frontend/src/lib/layer2/"
    cp "$BACKUP_DIR/frontend/src/lib/personaplex/"*.ts "$PROJECT_ROOT/frontend/src/lib/personaplex/" 2>/dev/null || true

    echo "Restoring Frontend Types..."
    cp "$BACKUP_DIR/frontend/src/types/"*.ts "$PROJECT_ROOT/frontend/src/types/"

    echo "Restoring Settings..."
    cp "$BACKUP_DIR/backend/settings.py" "$PROJECT_ROOT/backend/command_center/settings.py" 2>/dev/null || true
    cp "$BACKUP_DIR/frontend/.env.production.local" "$PROJECT_ROOT/frontend/" 2>/dev/null || true
fi

echo ""
echo "=== RESTORE COMPLETE ==="
echo "All files have been restored to their pre-audit-fix state."
echo ""
echo "Next steps:"
echo "  1. Restart backend: cd backend && source venv/bin/activate && python manage.py runserver 8100"
echo "  2. Restart frontend: cd frontend && npm run dev"
echo ""
echo "To verify: python manage.py check"
