"""
Extended Uniqueness Search: phi^n vs e^m for n <= 200

Uses arbitrary precision arithmetic (mpmath) to search for convergences
beyond the standard n <= 50 domain.
"""

from mpmath import mp, mpf, exp, sqrt
import pandas as pd
from datetime import datetime

# Set precision to 50 decimal places
mp.dps = 50

def compute_phi_power(n):
    """Compute phi^n with arbitrary precision"""
    phi = (1 + sqrt(5)) / 2
    return phi ** n

def find_all_convergences(n_max=200, m_step=0.1, tolerance=0.01):
    """Search for all (n, m) pairs where phi^n/e^m  1"""
    results = []
    phi = (1 + sqrt(5)) / 2
    
    print(f"Searching n in [1, {n_max}], m in [1, 100] with step {m_step}")
    print(f"Tolerance: {tolerance*100}%")
    print(f"Precision: {mp.dps} decimal places")
    print(f"Started: {datetime.now()}")
    print()
    
    for n in range(1, n_max + 1):
        phi_n = phi ** n
        
        m_min = max(1, n * 0.4)
        m_max = min(100, n * 0.6)
        
        m = m_min
        while m <= m_max:
            e_m = exp(m)
            ratio = phi_n / e_m
            deviation = abs(ratio - 1)
            
            if deviation < tolerance:
                results.append({
                    'n': n,
                    'm': float(m),
                    'phi_n': float(phi_n),
                    'e_m': float(e_m),
                    'ratio': float(ratio),
                    'deviation': float(deviation),
                    'deviation_pct': float(deviation * 100)
                })
                
                print(f"Found: n={n}, m={float(m):.2f}, dev={float(deviation)*100:.4f}%")
            
            m += mpf(m_step)
        
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
    
    df_sorted = df.sort_values('deviation')
    
    print("\n" + "="*80)
    print("TOP 20 CONVERGENCES (sorted by deviation)")
    print("="*80)
    print(df_sorted.head(20).to_string(index=False))
    
    row_17 = df[df['n'] == 17].sort_values('deviation').head(1)
    if len(row_17) > 0:
        rank = (df_sorted['deviation'] < row_17.iloc[0]['deviation']).sum() + 1
        print("\n" + "="*80)
        print(f"COSF (n=17, m~8.6) RANKING")
        print("="*80)
        print(f"Rank: {rank} out of {len(df)}")
        print(f"Deviation: {row_17.iloc[0]['deviation_pct']:.6f}%")
        print(f"m value: {row_17.iloc[0]['m']:.2f}")
    
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Mean deviation: {df['deviation_pct'].mean():.4f}%")
    print(f"Median deviation: {df['deviation_pct'].median():.4f}%")
    print(f"Best deviation: {df['deviation_pct'].min():.4f}%")
    print(f"n values with convergences: {df['n'].nunique()}")
    
    return df_sorted

if __name__ == "__main__":
    df = find_all_convergences(n_max=200, m_step=0.1, tolerance=0.01)
    df_sorted = analyze_results(df)
    
    if len(df) > 0:
        df_sorted.to_csv('../../extended_search_results.csv', index=False)
        print("\nResults saved to: extended_search_results.csv")
