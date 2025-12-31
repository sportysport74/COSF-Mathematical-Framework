"""
Find REAL convergences between φⁿ and eᵐ
WHERE m is a "nice" number (integer, half-integer, or simple fraction)
NOT the trivial m = ln(φⁿ)
"""

import numpy as np
import pandas as pd
from math import sqrt, log, exp

# Constants
PHI = (1 + sqrt(5)) / 2
COSF = 5465.701174755326

def find_real_convergences():
    """
    Search for convergences where m is restricted to nice values
    """
    results = []

    # Search parameters
    n_range = range(1, 51)  # Powers of φ

    # Nice m values: integers, half-integers, thirds
    nice_m_values = []
    for i in range(1, 101):  # m from 0.1 to 50
        nice_m_values.append(i * 0.1)  # Tenths
        nice_m_values.append(i * 0.5)  # Halves
        nice_m_values.append(i / 3.0)  # Thirds

    nice_m_values = sorted(set(nice_m_values))

    print(f"Searching {len(n_range)} powers of φ against {len(nice_m_values)} nice m values...")
    print(f"Total combinations: {len(n_range) * len(nice_m_values):,}\n")

    for n in n_range:
        phi_n = PHI ** n

        for m in nice_m_values:
            e_m = exp(m)

            # Calculate deviation
            if e_m > 0:
                deviation = abs(phi_n - e_m) / e_m
            else:
                continue

            # Store if reasonably close (within 5%)
            if deviation < 0.05:
                results.append({
                    'n': n,
                    'm': m,
                    'phi_n': phi_n,
                    'e_m': e_m,
                    'deviation_pct': deviation * 100,
                    'ratio': phi_n / e_m,
                    'm_type': classify_m(m)
                })

    return results

def classify_m(m):
    """Classify what kind of 'nice' number m is"""
    if abs(m - round(m)) < 0.001:
        return "integer"
    elif abs(m - round(m * 2) / 2) < 0.001:
        return "half-integer"
    elif abs(m - round(m * 3) / 3) < 0.001:
        return "third"
    elif abs(m - round(m * 10) / 10) < 0.001:
        return "tenth"
    else:
        return "other"

def main():
    print("=" * 70)
    print("REAL CONVERGENCE FINDER")
    print("Finding φⁿ ≈ eᵐ where m is a 'nice' number")
    print("=" * 70)
    print()

    results = find_real_convergences()

    if not results:
        print("No convergences found!")
        return

    # Convert to DataFrame and sort
    df = pd.DataFrame(results)
    df = df.sort_values('deviation_pct')

    print(f"\n{'=' * 70}")
    print(f"FOUND {len(df)} REAL CONVERGENCES (within 5% deviation)")
    print(f"{'=' * 70}\n")

    # Show top 20 closest convergences
    print("TOP 20 CLOSEST CONVERGENCES:")
    print("-" * 70)
    for idx, row in df.head(20).iterrows():
        print(f"n={row['n']:2d}, m={row['m']:6.2f} ({row['m_type']:12s}): "
              f"φ^{row['n']} = {row['phi_n']:12.2f}, "
              f"e^{row['m']:.2f} = {row['e_m']:12.2f}, "
              f"dev = {row['deviation_pct']:6.3f}%")

    # Special check: Does COSF appear?
    print(f"\n{'=' * 70}")
    print("CHECKING COSF POSITION...")
    print(f"{'=' * 70}")
    print(f"COSF = {COSF:.2f}")

    for n in range(1, 51):
        phi_n = PHI ** n
        if abs(phi_n - COSF) / COSF < 0.5:  # Within 50%
            print(f"  φ^{n} = {phi_n:.2f} (deviation from COSF: {abs(phi_n - COSF)/COSF*100:.2f}%)")

    # Group by m_type
    print(f"\n{'=' * 70}")
    print("CONVERGENCES BY M TYPE:")
    print(f"{'=' * 70}")
    for m_type in ['integer', 'half-integer', 'third', 'tenth']:
        subset = df[df['m_type'] == m_type]
        if len(subset) > 0:
            print(f"\n{m_type.upper()}: {len(subset)} convergences")
            print("-" * 70)
            for idx, row in subset.head(5).iterrows():
                print(f"  n={row['n']:2d}, m={row['m']:6.2f}: "
                      f"φ^{row['n']} = {row['phi_n']:12.2f}, "
                      f"e^{row['m']:.2f} = {row['e_m']:12.2f}, "
                      f"dev = {row['deviation_pct']:6.3f}%")

    # Save results
    output_file = 'real_convergences.csv'
    df.to_csv(output_file, index=False)
    print(f"\n{'=' * 70}")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()
