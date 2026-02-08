#!/usr/bin/env python3
"""
Manual Tier 3 Training Trigger

Run this to manually trigger SFT training on accumulated traces.
"""

import sys
from pathlib import Path

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from rl.tier3_integration import check_and_trigger_training

if __name__ == "__main__":
    print("Checking if Tier 3 training should trigger...")
    success = check_and_trigger_training()

    if success:
        print("✓ Training completed successfully!")
        sys.exit(0)
    else:
        print("✗ Training did not run (not enough traces or failed)")
        sys.exit(1)
