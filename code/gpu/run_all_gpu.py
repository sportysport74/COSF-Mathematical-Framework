"""
Master GPU Pipeline Runner for COSF Framework
Executes all GPU-accelerated tools in sequence

Author: Sportysport
Date: December 31, 2025
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import time


def run_script(script_name, description):
    """Run a Python script and report results"""
    print("\n" + "="*80)
    print(f"🚀 RUNNING: {description}")
    print("="*80)
    print(f"Script: {script_name}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )

        elapsed = time.time() - start_time

        print(f"\n✅ SUCCESS: {description}")
        print(f"Elapsed time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")

        return True, elapsed

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time

        print(f"\n❌ FAILED: {description}")
        print(f"Error: {e}")
        print(f"Elapsed time: {elapsed:.1f} seconds")

        return False, elapsed


def main():
    """Run complete GPU pipeline"""

    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                                                                          ║
    ║          🔥 COSF GPU PIPELINE - MASTER RUNNER 🔥                        ║
    ║                                                                          ║
    ║          Running complete GPU-accelerated analysis                      ║
    ║          Hardware: RTX 5090 (32GB VRAM, 21,760 CUDA cores)             ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)

    gpu_dir = Path(__file__).parent

    # Pipeline stages
    pipeline = [
        {
            'script': gpu_dir / 'cuda_convergence_search.py',
            'name': 'CUDA Convergence Search (n ≤ 10,000)',
            'description': 'Exhaustive GPU search for φⁿ/eᵐ convergences'
        },
        {
            'script': gpu_dir / 'neural_pattern_finder.py',
            'name': 'Neural Pattern Finder',
            'description': 'Deep learning for convergence pattern discovery'
        },
        {
            'script': gpu_dir / 'monte_carlo_validator.py',
            'name': 'Monte Carlo Validator (1B samples)',
            'description': 'Statistical validation with billions of samples'
        },
        {
            'script': gpu_dir / 'raytraced_visualization.py',
            'name': 'GPU Ray-Traced Visualization',
            'description': 'Interactive 3D visuals and 4K renders'
        }
    ]

    # Execution summary
    results = []
    total_start = time.time()

    print(f"\n📋 PIPELINE STAGES:")
    for i, stage in enumerate(pipeline, 1):
        print(f"   {i}. {stage['name']}")

    print(f"\n⏰ Pipeline started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run each stage
    for i, stage in enumerate(pipeline, 1):
        print(f"\n{'#'*80}")
        print(f"STAGE {i}/{len(pipeline)}")
        print(f"{'#'*80}")

        success, elapsed = run_script(stage['script'], stage['name'])

        results.append({
            'stage': stage['name'],
            'success': success,
            'elapsed': elapsed
        })

        if not success:
            print(f"\n⚠️  Stage {i} failed. Continue anyway? (y/n)")
            response = input().strip().lower()
            if response != 'y':
                print("\n🛑 Pipeline aborted by user.")
                break

    # Final summary
    total_elapsed = time.time() - total_start

    print("\n" + "="*80)
    print("🏁 PIPELINE COMPLETE")
    print("="*80)
    print(f"Total time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\n📊 STAGE RESULTS:")
    for i, result in enumerate(results, 1):
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"   {i}. {result['stage']}: {status} ({result['elapsed']:.1f}s)")

    # Success count
    success_count = sum(1 for r in results if r['success'])
    print(f"\n🎯 Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.0f}%)")

    if success_count == len(results):
        print("\n" + "🔥"*40)
        print("🔥 LEGENDARY STATUS ACHIEVED - ALL STAGES COMPLETE! 🔥")
        print("🔥"*40 + "\n")

    # Output summary
    print(f"\n📁 OUTPUT LOCATIONS:")
    print(f"   - GPU search results: results/gpu_search/")
    print(f"   - Neural patterns: results/neural_patterns/")
    print(f"   - Monte Carlo validation: results/monte_carlo/")
    print(f"   - Visualizations: images/toroidal_geometry/, images/convergence_landscapes/")
    print(f"   - 4K renders: images/4k_renders/")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
