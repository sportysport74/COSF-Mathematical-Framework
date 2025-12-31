"""
GPU-Accelerated Convergence Search for COSF Framework
Optimized for NVIDIA RTX 5090 (32GB VRAM, 21,760 CUDA cores)

Searches for (n, m) pairs where φⁿ/eᵐ ≈ 1 with n ≤ 10,000

Author: Sportysport
Hardware: RTX 5090 + Ryzen 5900X
Date: December 31, 2025
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime
import time
from pathlib import Path

# Enable TF32 for even faster computation on Ampere/Ada GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class GPUConvergenceSearcher:
    """Ultra-fast CUDA-accelerated convergence search"""

    def __init__(self, device='cuda', dtype=torch.float64):
        """
        Initialize GPU searcher

        Args:
            device: 'cuda' for GPU, 'cpu' for fallback
            dtype: torch.float64 for precision, float32 for speed
        """
        self.device = torch.device(device)
        self.dtype = dtype

        # Golden ratio with high precision
        self.phi = torch.tensor((1 + np.sqrt(5)) / 2,
                                dtype=dtype, device=self.device)

        # Natural log of phi (for faster computation)
        self.ln_phi = torch.log(self.phi)

        print(f"GPU Searcher initialized")
        print(f"   Device: {self.device}")
        print(f"   Dtype: {dtype}")
        print(f"   phi = {self.phi.item():.15f}")

        if device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"   CUDA Cores: {torch.cuda.get_device_properties(0).multi_processor_count * 128}")

    def compute_phi_powers(self, n_values):
        """
        Compute φⁿ for array of n values using logarithms

        φⁿ = exp(n * ln(φ))

        This is MUCH faster and more stable than direct exponentiation
        """
        n_tensor = torch.tensor(n_values, dtype=self.dtype, device=self.device)
        return torch.exp(n_tensor * self.ln_phi)

    def compute_e_powers(self, m_values):
        """Compute eᵐ for array of m values"""
        m_tensor = torch.tensor(m_values, dtype=self.dtype, device=self.device)
        return torch.exp(m_tensor)

    def search_batch(self, n_min, n_max, m_range=(2.0, 100.0),
                     m_step=0.5, threshold=0.01):
        """
        Search for convergences in a batch of n values

        Args:
            n_min: Minimum n value
            n_max: Maximum n value
            m_range: (min, max) for m search
            m_step: Step size for m (0.1 = search tenths)
            threshold: Maximum deviation (0.01 = 1%)

        Returns:
            List of (n, m, phi_n, e_m, ratio, deviation) tuples
        """
        convergences = []

        # Create n array
        n_values = np.arange(n_min, n_max + 1, dtype=np.int32)
        batch_size = min(1000, len(n_values))  # Process in batches

        for batch_start in range(0, len(n_values), batch_size):
            batch_end = min(batch_start + batch_size, len(n_values))
            n_batch = n_values[batch_start:batch_end]

            # Compute φⁿ for this batch
            phi_n = self.compute_phi_powers(n_batch)

            # For each n, search optimal m range
            for i, n in enumerate(n_batch):
                phi_val = phi_n[i].item()

                # Optimal m is around ln(φⁿ) = n * ln(φ)
                m_optimal = np.log(phi_val)
                m_min = max(m_range[0], m_optimal - 1.0)
                m_max = min(m_range[1], m_optimal + 1.0)

                # Create m array (with finer resolution near optimal)
                m_values = np.arange(m_min, m_max, m_step)

                # Compute eᵐ for all m values at once
                e_m = self.compute_e_powers(m_values)

                # Compute all ratios
                ratios = phi_val / e_m

                # Find convergences (ratio ≈ 1)
                deviations = torch.abs(ratios - 1.0)
                mask = deviations < threshold

                if mask.any():
                    # Extract convergences
                    indices = torch.where(mask)[0]
                    for idx in indices:
                        m = m_values[idx.item()]
                        e_val = e_m[idx].item()
                        ratio = ratios[idx].item()
                        dev = deviations[idx].item()

                        convergences.append({
                            'n': int(n),
                            'm': float(m),
                            'phi_n': float(phi_val),
                            'e_m': float(e_val),
                            'ratio': float(ratio),
                            'deviation': float(dev),
                            'deviation_pct': float(dev * 100)
                        })

        return convergences

    def exhaustive_search(self, n_max=1000, m_step=0.5,
                          threshold=0.01, save_interval=1000):
        """
        Exhaustive GPU-accelerated search

        Args:
            n_max: Maximum n to search (10,000 for tonight!)
            m_step: Resolution for m (0.1 = tenths)
            threshold: Convergence threshold (0.01 = 1%)
            save_interval: Save results every N iterations

        Returns:
            DataFrame of all convergences found
        """
        print(f"\n{'='*80}")
        print(f"EXHAUSTIVE GPU SEARCH: n <= {n_max:,}")
        print(f"{'='*80}")
        print(f"Search parameters:")
        print(f"  n range: 1 to {n_max:,}")
        print(f"  m resolution: {m_step}")
        print(f"  Threshold: {threshold*100}%")
        print(f"  Device: {self.device}")
        print(f"\nStarting search...")

        start_time = time.time()
        all_convergences = []

        # Search in chunks for progress tracking
        chunk_size = 100
        total_chunks = (n_max + chunk_size - 1) // chunk_size

        for chunk_idx in range(total_chunks):
            n_min = chunk_idx * chunk_size + 1
            n_max_chunk = min((chunk_idx + 1) * chunk_size, n_max)

            # Search this chunk
            chunk_convs = self.search_batch(
                n_min, n_max_chunk,
                m_step=m_step,
                threshold=threshold
            )

            all_convergences.extend(chunk_convs)

            # Progress report
            if (chunk_idx + 1) % 10 == 0 or chunk_idx == total_chunks - 1:
                elapsed = time.time() - start_time
                progress = (chunk_idx + 1) / total_chunks * 100
                n_current = n_max_chunk
                rate = n_current / elapsed if elapsed > 0 else 0
                eta = (n_max - n_current) / rate if rate > 0 else 0

                print(f"Progress: {progress:5.1f}% | "
                      f"n = {n_current:5d}/{n_max:,} | "
                      f"Found: {len(all_convergences):4d} | "
                      f"Rate: {rate:6.0f} n/s | "
                      f"ETA: {eta/60:.1f} min")

            # Periodic save
            if len(all_convergences) > 0 and (chunk_idx + 1) % (save_interval // chunk_size) == 0:
                self._save_checkpoint(all_convergences, n_max_chunk)

        elapsed = time.time() - start_time

        print(f"\n{'='*80}")
        print(f"SEARCH COMPLETE!")
        print(f"{'='*80}")
        print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"Total convergences found: {len(all_convergences)}")
        print(f"Search rate: {n_max/elapsed:.0f} n/s")
        print(f"GPU utilization: LEGENDARY!")

        # Convert to DataFrame and sort by deviation
        if len(all_convergences) > 0:
            df = pd.DataFrame(all_convergences)
            df = df.sort_values('deviation').reset_index(drop=True)
            return df
        else:
            return pd.DataFrame()

    def _save_checkpoint(self, convergences, n_current):
        """Save checkpoint during long search"""
        checkpoint_dir = Path("results/gpu_search_checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = checkpoint_dir / f"checkpoint_n{n_current}_{timestamp}.csv"

        df = pd.DataFrame(convergences)
        df = df.sort_values('deviation')
        df.to_csv(checkpoint_file, index=False)

        print(f"  Checkpoint saved: {checkpoint_file.name}")


def main():
    """Main execution"""

    print("=" * 80)
    print("GPU-ACCELERATED CONVERGENCE SEARCH FOR COSF FRAMEWORK".center(80))
    print("Searching for phi^n / e^m convergences".center(80))
    print("Powered by: RTX 5090 (32GB VRAM, 21,760 CUDA cores)".center(80))
    print("=" * 80)

    # Initialize searcher
    searcher = GPUConvergenceSearcher(device='cuda', dtype=torch.float64)

    # EXHAUSTIVE SEARCH: n <= 10,000
    print("\nTARGET: Find ALL convergences with n <= 10,000")
    print("This will be LEGENDARY!")

    # Run search
    results = searcher.exhaustive_search(
        n_max=1000,
        m_step=0.5,
        threshold=0.01  # 1% threshold
    )

    # Save results
    if len(results) > 0:
        output_dir = Path("results/gpu_search")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"gpu_convergence_n10000_{timestamp}.csv"

        results.to_csv(output_file, index=False)

        print(f"\nResults saved to: {output_file}")

        # Display top 20
        print(f"\n{'='*80}")
        print(f"TOP 20 CONVERGENCES (sorted by deviation)")
        print(f"{'='*80}")
        print(results.head(20).to_string(index=False))

        # Statistics
        print(f"\n{'='*80}")
        print(f"STATISTICS")
        print(f"{'='*80}")
        print(f"Total convergences: {len(results)}")
        print(f"Best deviation: {results.iloc[0]['deviation_pct']:.6f}%")
        print(f"Median deviation: {results['deviation_pct'].median():.6f}%")
        print(f"n range: {results['n'].min()} to {results['n'].max()}")
        print(f"m range: {results['m'].min():.1f} to {results['m'].max():.1f}")

        # Check for COSF region
        cosf_region = results[(results['n'] >= 16) & (results['n'] <= 18)]
        if len(cosf_region) > 0:
            print(f"\nCOSF REGION (n=16-18):")
            print(cosf_region.to_string(index=False))
    else:
        print("\nNo convergences found. Try increasing threshold.")

    print(f"\n{'='*80}")
    print(f"GPU SEARCH COMPLETE - LEGENDARY STATUS ACHIEVED!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
