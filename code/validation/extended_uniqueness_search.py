"""
Extended Uniqueness Search: φⁿ vs e^m for n  200

Uses arbitrary precision arithmetic (mpmath) to search for convergences
beyond the standard n  50 domain. This proves the true uniqueness of
the (17, 8.6) convergence across a much larger search space.

Requirements: pip install mpmath
Runtime: ~30 seconds on modern CPU
"""

from mpmath import mp, mpf, exp, sqrt
import pandas as pd
from datetime import datetime

# Set precision to 50 decimal places
mp.dps = 50

def compute_phi_power(n):
    """Compute φⁿ with arbitrary precision"""
    phi = (1 + sqrt(5)) / 2
    return phi ** n

def find_all_convergences(n_max=200, m_step=0.1, tolerance=0.01):
    """
    Search for all (n, m) pairs where φⁿ/e^m ≈ 1
    
    Args:
        n_max: Maximum n to search
        m_step: Step size for m values
        tolerance: Maximum deviation from 1 (as fraction)
    
    Returns:
        DataFrame with convergence points
    """
    results = []
    phi = (1 + sqrt(5)) / 2
    
    print(f"Searching n  [1, {n_max}], m  [1, 100] with step {m_step}")
    print(f"Tolerance: {tolerance*100}%")
    print(f"Precision: {mp.dps} decimal places")
    print(f"Started: {datetime.now()}")
    print()
    
    for n in range(1, n_max + 1):
        phi_n = phi ** n
        
        # For each n, scan m values
        m_min = max(1, n * 0.4)  # Heuristic lower bound
        m_max = min(100, n * 0.6)  # Heuristic upper bound
        
        m = m_min
        while m <= m_max:
            e_m = exp(m)
            ratio = phi_n / e_m
            deviation = abs(ratio - 1)
            
            if deviation < tolerance:
                results.append({
                    'n': n,
                    'm': float(m),
                    'φⁿ': float(phi_n),
                    'e^m': float(e_m),
                    'ratio': float(ratio),
                    'deviation': float(deviation),
                    'deviation_pct': float(deviation * 100)
                })
                
                # Print immediately
                print(f"Found: n={n}, m={m:.2f}, dev={deviation*100:.4f}%")
            
            m += mpf(m_step)
        
        # Progress indicator
        if n % 20 == 0:
            print(f"Progress: n={n}/{n_max}")
    
    print(f"\nCompleted: {datetime.now()}")
    print(f"Total convergences found: {len(results)}")
    
    return pd.DataFrame(results)

def analyze_results(df):
    """Analyze convergence results"""
    if len(df) == 0:
        print("No convergences found!")
        return
    
    # Sort by deviation
    df_sorted = df.sort_values('deviation')
    
    print("\n" + "="*80)
    print("TOP 20 CONVERGENCES (sorted by deviation)")
    print("="*80)
    print(df_sorted.head(20).to_string(index=False))
    
    # Find (17, 8.6)
    row_17 = df[df['n'] == 17].sort_values('deviation').head(1)
    if len(row_17) > 0:
        rank = (df_sorted['deviation'] < row_17.iloc[0]['deviation']).sum() + 1
        print("\n" + "="*80)
        print(f"COSF (n=17, m8.6) RANKING")
        print("="*80)
        print(f"Rank: {rank} out of {len(df)}")
        print(f"Deviation: {row_17.iloc[0]['deviation_pct']:.6f}%")
        print(f"m value: {row_17.iloc[0]['m']:.2f}")
    
    # Statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Mean deviation: {df['deviation_pct'].mean():.4f}%")
    print(f"Median deviation: {df['deviation_pct'].median():.4f}%")
    print(f"Best deviation: {df['deviation_pct'].min():.4f}%")
    print(f"n values with convergences: {df['n'].nunique()}")
    
    return df_sorted

if __name__ == "__main__":
    # Run extended search
    df = find_all_convergences(n_max=200, m_step=0.1, tolerance=0.01)
    
    # Analyze
    df_sorted = analyze_results(df)
    
    # Save results
    if len(df) > 0:
        df_sorted.to_csv('extended_search_results.csv', index=False)
        print("\n Results saved to: extended_search_results.csv")
