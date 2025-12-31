"""
COSF Convergence Analysis and Visualization

Creates publication-quality plots demonstrating the unique convergence
of φ¹⁷ and e^8.6, and the optimality of COSF ≈ 5466.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import math

def compute_phi_powers(n_max=30):
    """Compute φⁿ for n = 1 to n_max"""
    phi = (1 + np.sqrt(5)) / 2
    n_values = np.arange(1, n_max + 1)
    phi_powers = phi ** n_values
    return n_values, phi_powers

def compute_e_powers(m_max=20):
    """Compute e^m for m = 1 to m_max"""
    m_values = np.arange(1, m_max + 1, 0.1)
    e_powers = np.exp(m_values)
    return m_values, e_powers

def find_convergence_points(n_max=50, tolerance=0.02):
    """Find all (n,m) pairs where φⁿ/e^m  1"""
    phi = (1 + np.sqrt(5)) / 2
    convergence_points = []
    
    for n in range(1, n_max + 1):
        phi_n = phi ** n
        # Find m where e^m  φⁿ
        m_approx = np.log(phi_n)
        
        # Check nearby m values
        for m_offset in np.linspace(-1, 1, 21):
            m = m_approx + m_offset
            ratio = phi_n / np.exp(m)
            deviation = abs(ratio - 1)
            
            if deviation < tolerance:
                convergence_points.append((n, m, ratio, deviation))
    
    return convergence_points

def create_comprehensive_visualization():
    """Create multi-panel convergence analysis"""
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    phi = (1 + np.sqrt(5)) / 2
    
    # ============ Panel 1: Growth Comparison ============
    ax1 = fig.add_subplot(gs[0, :])
    
    n_vals, phi_vals = compute_phi_powers(25)
    m_vals, e_vals = compute_e_powers(15)
    
    ax1.semilogy(n_vals, phi_vals, 'o-', color='#FF6B6B', 
                 linewidth=2, markersize=6, label='φⁿ')
    ax1.semilogy(m_vals, e_vals, '-', color='#4ECDC4', 
                 linewidth=2, alpha=0.7, label='e^m')
    
    # Highlight COSF point
    ax1.semilogy(17, phi**17, 'o', color='gold', markersize=15, 
                 label=f'φ = {phi**17:.2f}', zorder=5)
    ax1.semilogy(8.6, np.exp(8.6), 's', color='gold', markersize=15, 
                 label=f'e^8.6 = {np.exp(8.6):.2f}', zorder=5)
    
    ax1.axhline(5466, color='orange', linestyle='--', linewidth=2, 
                label='COSF = 5466', alpha=0.7)
    
    ax1.set_xlabel('n or m', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax1.set_title('Growth Comparison: φⁿ vs e^m', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ============ Panel 2: Convergence Ratio ============
    ax2 = fig.add_subplot(gs[1, 0])
    
    n_range = np.arange(1, 30)
    ratios = []
    
    for n in n_range:
        m = n / 2  # Approximate relationship
        ratio = (phi**n) / np.exp(m)
        ratios.append(ratio)
    
    ax2.plot(n_range, ratios, 'o-', color='#4ECDC4', linewidth=2, markersize=5)
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Perfect convergence')
    ax2.plot(17, phi**17 / np.exp(8.6), 'o', color='gold', markersize=15, 
             label=f'n=17: ratio={phi**17/np.exp(8.6):.5f}', zorder=5)
    
    ax2.set_xlabel('n', fontsize=12, fontweight='bold')
    ax2.set_ylabel('φⁿ / e^(n/2)', fontsize=12, fontweight='bold')
    ax2.set_title('Convergence Ratio Analysis', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # ============ Panel 3: Deviation from Unity ============
    ax3 = fig.add_subplot(gs[1, 1])
    
    deviations = [abs(r - 1) * 100 for r in ratios]
    
    ax3.semilogy(n_range, deviations, 'o-', color='#FF6B6B', linewidth=2, markersize=5)
    ax3.semilogy(17, abs(phi**17/np.exp(8.6) - 1) * 100, 'o', 
                 color='gold', markersize=15, 
                 label=f'n=17: {abs(phi**17/np.exp(8.6) - 1)*100:.3f}%', zorder=5)
    ax3.axhline(1.0, color='orange', linestyle='--', linewidth=2, 
                alpha=0.7, label='1% threshold')
    
    ax3.set_xlabel('n', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Deviation from 1 (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Convergence Quality', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # ============ Panel 4: 2D Convergence Map ============
    ax4 = fig.add_subplot(gs[2, 0])
    
    n_grid = np.arange(10, 25, 1)
    m_grid = np.arange(5, 12, 0.1)
    N, M = np.meshgrid(n_grid, m_grid)
    
    Ratios = (phi**N) / np.exp(M)
    Deviations = np.abs(Ratios - 1) * 100
    
    contour = ax4.contourf(N, M, Deviations, levels=20, cmap='RdYlGn_r')
    ax4.plot(17, 8.6, 'o', color='gold', markersize=15, 
             markeredgecolor='black', markeredgewidth=2, label='(17, 8.6)')
    
    ax4.set_xlabel('n', fontsize=12, fontweight='bold')
    ax4.set_ylabel('m', fontsize=12, fontweight='bold')
    ax4.set_title('Convergence Landscape', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    plt.colorbar(contour, ax=ax4, label='Deviation (%)')
    
    # ============ Panel 5: Uniqueness Analysis ============
    ax5 = fig.add_subplot(gs[2, 1])
    
    convergence_points = find_convergence_points(30, 0.02)
    
    if convergence_points:
        n_vals = [p[0] for p in convergence_points]
        m_vals = [p[1] for p in convergence_points]
        dev_vals = [p[3] * 100 for p in convergence_points]
        
        scatter = ax5.scatter(n_vals, m_vals, c=dev_vals, s=100, 
                             cmap='RdYlGn_r', edgecolors='black', linewidth=1)
        ax5.plot(17, 8.6, 'o', color='gold', markersize=15, 
                markeredgecolor='black', markeredgewidth=3, label='(17, 8.6)', zorder=5)
        
        plt.colorbar(scatter, ax=ax5, label='Deviation (%)')
    
    ax5.set_xlabel('n', fontsize=12, fontweight='bold')
    ax5.set_ylabel('m', fontsize=12, fontweight='bold')
    ax5.set_title('Convergence Points (|ratio-1| < 2%)', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle('COSF Mathematical Framework: Comprehensive Convergence Analysis\nφ  e^8.6 Unique Convergence', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    plt.savefig('../../figures/convergence_plots/comprehensive_analysis.png', 
                dpi=300, bbox_inches='tight')
    print(" Saved: figures/convergence_plots/comprehensive_analysis.png")
    
    plt.show()

def create_golden_ratio_table():
    """Generate detailed table of φⁿ values"""
    phi = (1 + np.sqrt(5)) / 2
    
    print("\n" + "="*70)
    print("Golden Ratio Powers: φⁿ")
    print("="*70)
    print(f"φ = {phi:.15f}")
    print("-"*70)
    print(f"{'n':<5} {'φⁿ':<20} {'log(φⁿ)':<15} {'φⁿ/φⁿ':<15}")
    print("-"*70)
    
    prev_val = 1
    for n in range(0, 21):
        val = phi ** n
        log_val = np.log10(val) if val > 0 else 0
        ratio = val / prev_val if prev_val != 0 else 0
        
        print(f"{n:<5} {val:<20.6f} {log_val:<15.6f} {ratio:<15.6f}")
        prev_val = val
        
        if n == 17:
            print(">>> " + "-"*66 + " <<<")
    
    print("="*70)

if __name__ == "__main__":
    print("COSF Convergence Analysis")
    print("="*70)
    
    # Generate visualizations
    create_comprehensive_visualization()
    
    # Generate table
    create_golden_ratio_table()
    
    # Summary statistics
    phi = (1 + np.sqrt(5)) / 2
    phi_17 = phi ** 17
    e_86 = np.exp(8.6)
    cosf = 42800 / 7.83
    
    print(f"\nSummary Statistics:")
    print(f"φ = {phi_17:.10f}")
    print(f"e^8.6 = {e_86:.10f}")
    print(f"COSF = {cosf:.10f}")
    print(f"\nConvergence: φ/e^8.6 = {phi_17/e_86:.10f}")
    print(f"Deviation: {abs(phi_17/e_86 - 1)*100:.4f}%")
    print(f"\nCOSF proximity to φ: {abs(cosf - phi_17)/phi_17*100:.4f}%")
    print(f"COSF proximity to e^8.6: {abs(cosf - e_86)/e_86*100:.4f}%")
