"""
Julia Set Stability Visualizer

Visualizes the Julia set stability regions for COSF phase coherence.
Requires: matplotlib, numpy
"""

import numpy as np
import matplotlib.pyplot as plt

def julia_set(c, max_iter=100, bound=2):
    """
    Compute Julia set for parameter c
    Returns stability map (True = stable, False = escapes)
    """
    # Create grid
    x = np.linspace(-2, 2, 800)
    y = np.linspace(-2, 2, 800)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y
    
    # Initialize output
    output = np.zeros(Z.shape, dtype=bool)
    
    # Iterate
    for i in range(max_iter):
        mask = np.abs(Z) <= bound
        Z[mask] = Z[mask]**2 + c
        output |= ~mask
    
    return ~output

def visualize_cosf_stability():
    """Visualize Julia set for COSF golden angle parameter"""
    # Golden angle in radians
    golden_angle = 2 * np.pi * (2 - (1 + np.sqrt(5))/2)
    
    # COSF parameter: c = 0.3 * e^(i * 137.5)
    c = 0.3 * np.exp(1j * np.radians(137.5))
    
    print(f"Computing Julia set for c = {c}")
    print(f"Golden angle = {np.degrees(golden_angle):.2f}")
    
    # Compute Julia set
    stable = julia_set(c)
    
    # Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(stable, extent=[-2, 2, -2, 2], cmap='twilight', origin='lower')
    plt.title(f'Julia Set Stability for COSF\nc = {c:.3f}', fontsize=14)
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    
    # Add stability boundary circle
    circle = plt.Circle((0, 0), 2, fill=False, color='red', 
                       linestyle='--', linewidth=2, label='|z| = 2 boundary')
    plt.gca().add_patch(circle)
    
    plt.legend()
    plt.colorbar(label='Stable (blue) / Unstable (yellow)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    
    # Save
    plt.savefig('../../figures/julia_set_stability.png', dpi=150)
    print(" Saved to figures/julia_set_stability.png")
    plt.show()

if __name__ == "__main__":
    try:
        visualize_cosf_stability()
    except ImportError:
        print("Error: This script requires matplotlib and numpy")
        print("Install with: pip install matplotlib numpy")
