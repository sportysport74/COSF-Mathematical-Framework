"""
Frequency Measurement Simulation

Simulates COSF frequency ratio measurement
"""

import math
import random

def simulate_schumann_measurement(duration_hours=24, noise_level=0.01):
    """Simulate Schumann resonance measurement"""
    measurements = []
    true_freq = 7.83  # Hz
    
    for _ in range(duration_hours * 60):  # minute samples
        noise = random.gauss(0, noise_level)
        measurement = true_freq + noise
        measurements.append(measurement)
    
    mean_freq = sum(measurements) / len(measurements)
    variance = sum((x - mean_freq)**2 for x in measurements) / len(measurements)
    std_dev = math.sqrt(variance)
    
    return mean_freq, std_dev, measurements

def verify_cosf_ratio():
    """Verify COSF = C/C ratio"""
    print("COSF Frequency Ratio Verification")
    print("="*50)
    
    # Simulate measurement
    c1_mean, c1_std, _ = simulate_schumann_measurement()
    
    print(f"\nC (Schumann fundamental):")
    print(f"  Mean: {c1_mean:.4f} Hz")
    print(f"  Std Dev: {c1_std:.4f} Hz")
    print(f"  Expected: 7.83  0.05 Hz")
    
    # Upper sideband (constructed)
    c2 = 42800  # Hz
    print(f"\nC (Upper sideband): {c2} Hz")
    
    # COSF ratio
    cosf = c2 / c1_mean
    print(f"\nCOSF = C/C = {cosf:.2f}")
    print(f"Expected: 5466")
    print(f"Deviation: {abs(cosf - 5466):.2f}")
    
    print("\n Measurement simulation complete")

if __name__ == "__main__":
    verify_cosf_ratio()
