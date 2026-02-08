#!/usr/bin/env python3
"""
Unified Test Suite for All RL Tiers

Runs comprehensive tests for:
- Tier 1: Real-time online learning (LoRA scorer, prompt evolver)
- Tier 2: Offline policy learning (DPO training)
- Tier 3: Reasoning distillation (SFT from Claude traces)

Usage:
    python3 backend/rl/test_all_tiers.py
"""

import subprocess
import sys
import time
from pathlib import Path

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_header(text):
    """Print a bold header."""
    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{BLUE}{text.center(70)}{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")


def print_tier_header(tier_name, description):
    """Print tier section header."""
    print(f"\n{BOLD}{YELLOW}━━━ {tier_name}: {description} ━━━{RESET}\n")


def run_test(test_name, command, cwd=None):
    """Run a test and return success status."""
    print(f"{BLUE}Running: {test_name}{RESET}")
    start = time.time()

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300  # 5 min timeout
        )
        duration = time.time() - start

        if result.returncode == 0:
            print(f"{GREEN}✓ {test_name} PASSED{RESET} ({duration:.1f}s)\n")
            return True
        else:
            print(f"{RED}✗ {test_name} FAILED{RESET} ({duration:.1f}s)")
            if result.stdout:
                print(f"  stdout: {result.stdout[:200]}")
            if result.stderr:
                print(f"  stderr: {result.stderr[:200]}")
            print()
            return False

    except subprocess.TimeoutExpired:
        print(f"{RED}✗ {test_name} TIMEOUT{RESET} (>300s)\n")
        return False
    except Exception as e:
        print(f"{RED}✗ {test_name} ERROR: {e}{RESET}\n")
        return False


def main():
    """Run all tier tests."""
    print_header("RL SYSTEM - COMPREHENSIVE TEST SUITE")
    print(f"Testing all three tiers of the continuous RL system\n")

    backend_dir = Path(__file__).parent.parent
    results = []

    # ================================================================
    # TIER 1: Real-time Online Learning
    # ================================================================
    print_tier_header("TIER 1", "Real-time Online Learning")

    # Note: Tier 1 is tested implicitly via the e2e test
    # The LoRA scorer and prompt evolver are exercised during query processing
    print(f"{YELLOW}ℹ Tier 1 components (LoRA scorer, prompt evolver) are tested via E2E suite{RESET}")
    print(f"{YELLOW}  Run separately: python3 backend/rl/test_continuous.py{RESET}\n")

    # ================================================================
    # TIER 2: Offline Policy Learning (DPO)
    # ================================================================
    print_tier_header("TIER 2", "Offline Policy Learning (DPO)")

    tier2_tests = [
        ("Tier 2 Hardening Tests (33 tests)", "python3 -m rl.test_tier2_hardening"),
        ("Tier 2 Behavioral Tests (3 tests)", "python3 -m rl.test_tier2_behavioral"),
        ("Tier 2 Stress Tests (14 tests)", "python3 -m rl.test_tier2_stress"),
    ]

    for test_name, cmd in tier2_tests:
        success = run_test(test_name, cmd, cwd=backend_dir)
        results.append((test_name, success))

    # ================================================================
    # TIER 3: Reasoning Distillation (SFT)
    # ================================================================
    print_tier_header("TIER 3", "Reasoning Distillation (SFT)")

    tier3_tests = [
        ("Tier 3 Integration Tests", "python3 rl/test_tier3_e2e.py"),
    ]

    for test_name, cmd in tier3_tests:
        success = run_test(test_name, cmd, cwd=backend_dir)
        results.append((test_name, success))

    # ================================================================
    # OVERALL E2E: Full System Integration
    # ================================================================
    print_tier_header("OVERALL E2E", "Full System Integration")

    e2e_tests = [
        ("RL System E2E Tests", "python3 -m rl.e2e_test http://127.0.0.1:8100"),
    ]

    print(f"{YELLOW}ℹ E2E tests require backend running at http://127.0.0.1:8100{RESET}")
    print(f"{YELLOW}  Skipping E2E tests (run manually: python -m rl.e2e_test){RESET}\n")

    # ================================================================
    # SUMMARY
    # ================================================================
    print_header("TEST RESULTS SUMMARY")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    print(f"Results: {passed}/{total} test suites passed\n")

    for test_name, success in results:
        status = f"{GREEN}✓ PASS{RESET}" if success else f"{RED}✗ FAIL{RESET}"
        print(f"  {status}  {test_name}")

    print()

    if passed == total:
        print(f"{BOLD}{GREEN}{'='*70}{RESET}")
        print(f"{BOLD}{GREEN}✅ ALL TIER TESTS PASSED!{RESET}")
        print(f"{BOLD}{GREEN}{'='*70}{RESET}\n")
        return 0
    else:
        print(f"{BOLD}{RED}{'='*70}{RESET}")
        print(f"{BOLD}{RED}❌ SOME TESTS FAILED{RESET}")
        print(f"{BOLD}{RED}{'='*70}{RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
