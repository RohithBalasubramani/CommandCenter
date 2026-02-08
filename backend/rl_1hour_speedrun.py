#!/usr/bin/env python3
"""
1-Hour RL Training Speedrun

Aggressively generates queries, evaluations, and triggers training
to see measurable improvement in just 1 hour.

Usage:
    python rl_1hour_speedrun.py
"""

import requests
import time
import subprocess
import json
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Configuration
API_URL = "http://127.0.0.1:8100/api/layer2/orchestrate/"
STATUS_URL = "http://127.0.0.1:8100/api/layer2/rl-status/"
FEEDBACK_URL = "http://127.0.0.1:8100/api/layer2/feedback/"
EVAL_SCRIPT = Path(__file__).parent / "auto_evaluate_responses.py"
VENV_PYTHON = Path(__file__).parent / "venv/bin/python"

# Industrial test queries (diverse set)
TEST_QUERIES = [
    # Status queries
    "Show me all critical alerts",
    "What equipment is offline right now?",
    "Display current production status",
    "Which pumps are running?",
    "Show me equipment health scores",

    # Maintenance queries
    "What equipment needs maintenance?",
    "Show pumps with high vibration",
    "Which assets are degraded?",
    "Display maintenance schedule",
    "What's overdue for service?",

    # Trend/Historical queries
    "Show temperature trends for the last hour",
    "Display production rate over time",
    "What's the pressure trend?",
    "Show energy consumption history",
    "Display efficiency trends",

    # Specific equipment queries
    "What's the status of pump P-101?",
    "Show me Tank T-201 level",
    "Display motor M-305 metrics",
    "What's the valve V-410 position?",
    "Show compressor C-150 status",

    # Operational queries
    "Show me top 5 alerts by severity",
    "Which tanks are above 90% full?",
    "Display all open alarms",
    "What sensors are in alarm state?",
    "Show equipment with abnormal readings",

    # Performance queries
    "Display KPIs for production",
    "Show overall equipment effectiveness",
    "What's the current throughput?",
    "Display energy efficiency metrics",
    "Show production vs target",

    # Comparison queries
    "Compare pump performance",
    "Show all motor temperatures",
    "Display tank levels",
    "Compare production lines",
    "Show pressure across all zones",
]

def print_banner(text):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def get_rl_status():
    """Get current RL system status."""
    try:
        resp = requests.get(STATUS_URL, timeout=5)
        return resp.json() if resp.status_code == 200 else None
    except:
        return None

def send_query(query):
    """Send a single query to the orchestrator."""
    try:
        resp = requests.post(API_URL,
                           json={"transcript": query, "user_id": "speedrun", "device_id": "test"},
                           timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "success": True,
                "query_id": data.get("query_id"),
                "widgets": len(data.get("widget_plan", {}).get("widgets", []))
            }
        return {"success": False}
    except Exception as e:
        return {"success": False, "error": str(e)}

def generate_queries_batch(num_cycles=3, parallel=5):
    """Generate multiple query cycles in parallel."""
    print_banner("PHASE 1: GENERATING TEST QUERIES")

    total_sent = 0
    total_failed = 0

    for cycle in range(num_cycles):
        print(f"\n[Cycle {cycle+1}/{num_cycles}] Generating {len(TEST_QUERIES)} queries...")

        with ThreadPoolExecutor(max_workers=parallel) as executor:
            results = list(executor.map(send_query, TEST_QUERIES))

        successful = sum(1 for r in results if r.get("success"))
        total_sent += successful
        total_failed += len(results) - successful

        print(f"  ✓ Sent: {successful}/{len(TEST_QUERIES)}")

        if cycle < num_cycles - 1:
            print(f"  ⏳ Waiting 10 seconds before next cycle...")
            time.sleep(10)

    print(f"\n✅ Phase 1 Complete:")
    print(f"   Total queries: {total_sent}")
    print(f"   Failed: {total_failed}")

    return total_sent

def run_evaluations_aggressive(duration_minutes=10):
    """Run continuous evaluation for specified duration."""
    print_banner("PHASE 2: AGGRESSIVE AUTO-EVALUATION")
    print(f"Duration: {duration_minutes} minutes")
    print(f"Interval: 30 seconds (aggressive)")

    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)

    total_evaluated = 0
    iteration = 0

    while time.time() < end_time:
        iteration += 1
        remaining = int((end_time - time.time()) / 60)

        print(f"\n[Iteration {iteration}] Running evaluator (Time remaining: {remaining}m)...")

        try:
            # Run evaluator batch
            result = subprocess.run(
                [str(VENV_PYTHON), str(EVAL_SCRIPT), "--batch-size", "20"],
                capture_output=True,
                text=True,
                timeout=120
            )

            # Parse output for stats
            if "Evaluated:" in result.stdout:
                for line in result.stdout.split('\n'):
                    if "Evaluated:" in line:
                        try:
                            num = int(line.split("Evaluated:")[1].strip().split()[0])
                            total_evaluated += num
                            print(f"  ✓ Batch evaluated: {num} (Total: {total_evaluated})")
                        except:
                            pass

        except subprocess.TimeoutExpired:
            print(f"  ⚠ Evaluation timeout (continuing...)")
        except Exception as e:
            print(f"  ✗ Error: {e}")

        if time.time() < end_time:
            wait_time = min(30, int(end_time - time.time()))
            if wait_time > 0:
                print(f"  ⏳ Waiting {wait_time}s...")
                time.sleep(wait_time)

    print(f"\n✅ Phase 2 Complete:")
    print(f"   Total evaluated: {total_evaluated}")

    return total_evaluated

def trigger_tier2_training():
    """Force trigger Tier 2 training."""
    print_banner("PHASE 3: TRIGGERING TIER 2 TRAINING")

    try:
        resp = requests.post("http://127.0.0.1:8100/api/layer2/approve-training/", timeout=5)
        if resp.status_code == 200:
            print("✓ Tier 2 training approval file created")
            print("⏳ Waiting for background trainer to start (up to 60s)...")

            # Wait for training to start
            for i in range(12):  # Wait up to 2 minutes
                time.sleep(10)
                status = get_rl_status()
                if status:
                    tier2 = status.get("trainer", {}).get("tier2_lora", {})
                    if tier2.get("training_in_progress"):
                        print("✓ Tier 2 training STARTED!")
                        return True
                    print(f"  [{i*10}s] Still waiting...")

            print("⚠ Training didn't start in expected time (may start later)")
            return False
        else:
            print(f"✗ Failed to trigger training: {resp.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def monitor_progress(duration_minutes=5):
    """Monitor RL progress for specified duration."""
    print_banner("PHASE 4: MONITORING PROGRESS")
    print(f"Duration: {duration_minutes} minutes")

    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)

    initial_status = get_rl_status()
    if not initial_status:
        print("✗ Could not get initial status")
        return

    initial_tier1_steps = initial_status.get("trainer", {}).get("tier1_scorer", {}).get("training_steps", 0)

    print(f"\nInitial state:")
    print(f"  Tier 1 steps: {initial_tier1_steps:,}")

    while time.time() < end_time:
        time.sleep(30)

        status = get_rl_status()
        if status:
            tier1 = status.get("trainer", {}).get("tier1_scorer", {})
            tier2 = status.get("trainer", {}).get("tier2_lora", {})
            buffer = status.get("buffer", {})

            steps_gained = tier1.get("training_steps", 0) - initial_tier1_steps

            print(f"\n[{int((time.time() - start_time) / 60)}m] Progress:")
            print(f"  Tier 1: {tier1.get('training_steps', 0):,} steps (+{steps_gained:,})")
            print(f"  Loss: {tier1.get('avg_loss', 0):.6f}")
            print(f"  Tier 2: {'TRAINING' if tier2.get('training_in_progress') else 'Ready'} | Pairs: {tier2.get('pending_pairs', 0)}")
            print(f"  Buffer: {buffer.get('total_experiences', 0)} experiences")

    print(f"\n✅ Phase 4 Complete")

    # Final status
    final_status = get_rl_status()
    if final_status:
        final_tier1_steps = final_status.get("trainer", {}).get("tier1_scorer", {}).get("training_steps", 0)
        total_gain = final_tier1_steps - initial_tier1_steps

        print(f"\n📊 Final Results:")
        print(f"   Tier 1 steps gained: {total_gain:,}")
        print(f"   Tier 1 final steps: {final_tier1_steps:,}")

def main():
    print_banner("RL 1-HOUR SPEEDRUN - STARTING")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target: Measurable improvement in 1 hour")

    overall_start = time.time()

    # Check initial status
    initial_status = get_rl_status()
    if not initial_status or not initial_status.get("running"):
        print("✗ RL system not running! Start backend first.")
        return 1

    print(f"\n✓ RL System operational")
    print(f"✓ Initial Tier 1 steps: {initial_status['trainer']['tier1_scorer']['training_steps']:,}")

    # Execute speedrun
    try:
        # Phase 1: Generate 100+ queries (5 minutes)
        queries_sent = generate_queries_batch(num_cycles=3, parallel=5)
        time.sleep(5)  # Let queries settle

        # Phase 2: Aggressive evaluation (20 minutes)
        evaluated = run_evaluations_aggressive(duration_minutes=20)

        # Phase 3: Trigger Tier 2 (if ready)
        status = get_rl_status()
        if status:
            tier2 = status.get("trainer", {}).get("tier2_lora", {})
            if tier2.get("pending_pairs", 0) >= 10:  # Lower threshold for speedrun
                trigger_tier2_training()
            else:
                print("\n⚠ Not enough DPO pairs yet for Tier 2 training")

        # Phase 4: Monitor progress (remaining time)
        monitor_progress(duration_minutes=10)

    except KeyboardInterrupt:
        print("\n\n⚠ Speedrun interrupted by user")

    # Final summary
    elapsed = time.time() - overall_start
    final_status = get_rl_status()

    print_banner("1-HOUR SPEEDRUN COMPLETE")
    print(f"Total time: {elapsed/60:.1f} minutes")

    if final_status and initial_status:
        initial_steps = initial_status['trainer']['tier1_scorer']['training_steps']
        final_steps = final_status['trainer']['tier1_scorer']['training_steps']
        steps_gain = final_steps - initial_steps

        initial_loss = initial_status['trainer']['tier1_scorer']['avg_loss']
        final_loss = final_status['trainer']['tier1_scorer']['avg_loss']
        loss_improvement = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0

        print(f"\n📊 Performance Gains:")
        print(f"   Queries generated: {queries_sent}")
        print(f"   Experiences evaluated: {evaluated}")
        print(f"   Tier 1 steps: {initial_steps:,} → {final_steps:,} (+{steps_gain:,})")
        print(f"   Loss: {initial_loss:.6f} → {final_loss:.6f} ({loss_improvement:+.2f}%)")
        print(f"   Buffer: {final_status['buffer']['total_experiences']} experiences")

        if steps_gain > 1000:
            print(f"\n✅ SUCCESS: {steps_gain:,} training steps in {elapsed/60:.1f} minutes!")
        else:
            print(f"\n⚠ Limited progress - may need more queries or time")

    print(f"\n🎯 System is now continuously learning!")
    print(f"   Come back in 2-3 days for even better results")

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
