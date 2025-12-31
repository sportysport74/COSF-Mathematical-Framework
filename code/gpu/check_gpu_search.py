"""Quick GPU search progress checker"""
import time
from pathlib import Path

print("\nChecking GPU search progress...\n")

# Check for results
results_dir = Path("results/gpu_search")
if results_dir.exists():
    results = list(results_dir.glob("*.csv"))
    if results:
        latest = max(results, key=lambda p: p.stat().st_mtime)
        print(f"Latest results: {latest.name}")
        print(f"File size: {latest.stat().st_size / 1024:.1f} KB")
        print(f"Modified: {time.ctime(latest.stat().st_mtime)}")

        # Count lines
        with open(latest, 'r') as f:
            lines = sum(1 for _ in f) - 1  # Subtract header
        print(f"Convergences found: {lines}")
    else:
        print("No results yet...")
else:
    print("No results directory yet...")

# Check for checkpoints
checkpoint_dir = Path("results/gpu_search_checkpoints")
if checkpoint_dir.exists():
    checkpoints = list(checkpoint_dir.glob("*.csv"))
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        print(f"\nLatest checkpoint: {latest_checkpoint.name}")
        print(f"Modified: {time.ctime(latest_checkpoint.stat().st_mtime)}")
